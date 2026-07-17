#TODO the structure of the summary dict has changed!
#TODO refactor_glu for summary_dataset?

from argparse import ArgumentParser
import os
import pickle

import torch

#make sure HF_HUB_CACHE is set to 1 if necessary, before loading datasets
if "SLURM_JOBID" in os.environ:
    os.environ["HF_HUB_OFFLINE"]='1'
import datasets

from transformer_lens.model_bridge import TransformerBridge

from a_dataset import tokenize_dataset
import recompute
from c_neuron_vis import neuron_vis_full, update_pagelist
import utils
from utils import _move_to, load_data

parser = ArgumentParser()
#dataset arguments
parser.add_argument('--dataset', default='dolma-small')
parser.add_argument('--add_bos_token', type=bool, default=True,
                    help="add bos token to every example")
parser.add_argument('--max_length', type=int, default=1024,
                    help="length of example token blocks")
parser.add_argument('--return_overflowing_tokens', type=bool, default=False,
                    help="""Make additional training examples with overflowing tokens.
                    In this case it is currently not possible to keep the ids and metadata.""")
parser.add_argument('--padding', type=bool, default=False,
                    help="pad examples to args.max_length")
#model arguments
parser.add_argument('--model', default='allenai/OLMo-1B-hf')
parser.add_argument(
    '--refactor_glu',
    action='store_true',
    help='whether to refactor the weights such that cos(w_gate,w_in)>=0'
)
#directories
parser.add_argument('--datasets_dir', default='datasets')
parser.add_argument('--results_dir', default='GLUScope-results')
parser.add_argument('--site_dir', default='gluscope.github.io' if "WORK" in os.environ else 'docs')
parser.add_argument('--save_to', default=None)
#other
parser.add_argument('--from_scratch', type=bool, default=True)
parser.add_argument('--neurons',
    nargs='+',
    default=[],
    help='one or several neurons denoted as layer.neuron, or "all", or "zeros" (neuron 0 from each layer)',
)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

print("preparation...")
RUN_CODE = utils.get_run_code(args)
#the id of the b_activations.py run
SAVE_PATH = utils.make_save_path(args.results_dir, RUN_CODE)
VIS_PATH = utils.make_save_path(args.site_dir, RUN_CODE)
with open("html_boilerplate/head.html", "r", encoding="utf-8") as read_file:
    HEAD_AND_TITLE = read_file.read()+f"\n<body>\n<h1>Model: <b>{args.model}</b></h1>\n"
with open("html_boilerplate/script.html", "r", encoding="utf-8") as read_file:
    TAIL = read_file.read()+"\n</body>\n</html>\n"

torch.set_grad_enabled(False)

#load summary if exists:
SUMMARY_FILE = f'{SAVE_PATH}/summary{"_refactored" if args.refactor_glu else""}'
summary_dict = None
summary_dataset = None
if refactored_already:= os.path.exists(f"{SUMMARY_FILE}.pt") or os.path.exists(f"{SUMMARY_FILE}.pickle"):
    MY_FILE = SUMMARY_FILE
else:
    MY_FILE = f'{SAVE_PATH}/summary'
if os.path.exists(f"{MY_FILE}.pt"):
    summary_dict = torch.load(f"{MY_FILE}.pt", map_location='cuda')
elif os.path.exists(f"{MY_FILE}.pickle"):
    with open(f"{MY_FILE}.pickle", 'rb') as read_file:
        summary_dict = _move_to(pickle.load(read_file), 'cuda')
else:
    if os.path.exists(f'{SAVE_PATH}/activation_dataset'):
        summary_dataset = datasets.load_from_disk(f'{SAVE_PATH}/activation_dataset')
    else:
        assert args.dataset=='dolma-small'
        assert args.model=='allenai/OLMo-7B-0424-hf'
        assert args.refactor_glu
        print("HF_HUB_OFFLINE =", os.environ["HF_HUB_OFFLINE"])
        #if you get an offline error in the following line, just run this line once while online, then it should also work offline
        summary_dataset = datasets.load_dataset('sjgerstner/OLMo-7B-0424-hf_neuron-activations')['train']
    assert isinstance(summary_dataset, datasets.Dataset)

text_dataset = load_data(args)
assert isinstance(text_dataset, datasets.Dataset)
if not args.model.startswith('allenai/OLMo-1B') and not args.model.startswith('allenai/OLMo-7B'):
        text_dataset, _tokenizer = tokenize_dataset(args, text_dataset)

need_to_refactor = summary_dict and args.refactor_glu and not refactored_already
model = TransformerBridge.boot_transformers(
    args.model,
    device='cpu' if need_to_refactor else 'cuda',
)
if not args.refactor_glu or need_to_refactor:
    model.enable_compatibility_mode(no_processing=True)
else:
    try:
        model.enable_compatibility_mode(
            refactor_glu=True,
            fold_ln=False, center_writing_weights=False, center_unembed=False, fold_value_biases=False,
        )
    except torch.cuda.OutOfMemoryError:
        model.enable_compatibility_mode(no_processing=True)
        sign_to_adapt=utils.compute_sign_to_adapt(model)
        model = utils.refactor_glu_model(model=model, sign_to_adapt=sign_to_adapt)
utils.add_actfn(model)
#assert model.W_gate is not None

tokenizer = model.tokenizer
N_LAYERS, N_NEURONS = model.cfg.n_layers, model.cfg.d_mlp

if need_to_refactor:
    #first we detect which neurons to refactor and then we update the model
    sign_to_adapt=utils.compute_sign_to_adapt(model)
    if "summary_mean" in summary_dict:#legacy
        summary_dict = utils.refactor_glu_old(summary_dict, sign_to_adapt)
    else:
        summary_dict = utils.refactor_glu(summary_dict, sign_to_adapt)
    torch.save(summary_dict, f"{SUMMARY_FILE}.pt")
    model = model.cuda()
    model.enable_compatibility_mode(no_processing=True)
    utils.add_actfn(model)
    model = utils.refactor_glu_model(model=model, sign_to_adapt=sign_to_adapt)
else:
    sign_to_adapt = torch.ones(size=(N_LAYERS, N_NEURONS), dtype=torch.int)

if args.neurons==['all']:
    layer_neuron_list = [range(N_NEURONS) for _layer in range(N_LAYERS)]
elif args.neurons==['zeros']:
    layer_neuron_list = [[0] for _layer in range(N_LAYERS)]
elif args.neurons:
    layer_neuron_list = [[] for _layer in range(N_LAYERS)]
    for ln_str in args.neurons:
        layer, neuron = tuple(int(n) for n in ln_str.split('.'))
        layer_neuron_list[layer].append(neuron)
elif args.test:
    layer_neuron_list = [[0]]

if summary_dict:
    maxmin_keys = [key for key in summary_dict.keys() if key[-1] in ['max','min']]
else:
    maxmin_keys = []#dummy

for layer,neuron_list in enumerate(layer_neuron_list):
    print(f'processing layer {layer}...')
    layer_dir = f"{SAVE_PATH}/L{layer}"
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)

    for neuron in neuron_list:
        print(f'> processing neuron {neuron}...')
        neuron_vis_dir = f"{layer_dir}/N{neuron}"
        if not os.path.exists(neuron_vis_dir):
            os.mkdir(neuron_vis_dir)

        #recomputing neuron activations on max and min examples
        print('>> gathering/recomputing data...')
        kwargs = {
            "neuron_dir": neuron_vis_dir,
            "model": model, "layer": layer, "neuron": neuron,
            "save_path": SAVE_PATH,
            "text_dataset": text_dataset,
            "single_sign_to_adapt": int(sign_to_adapt[layer,neuron]),
        }
        #print(kwargs)
        if summary_dict:
            activation_data = recompute.neuron_data_from_dict(
                args=args,
                summary_dict=summary_dict,
                maxmin_keys=maxmin_keys,
                **kwargs,
            )
        else:
            activation_data = recompute.neuron_data_from_dataset(
                args=args,
                activation_dataset=summary_dataset,
                **kwargs,
            )
        #visualisation
        print('>> updating page list if necessary...')
        if not args.test:
            update_pagelist(run_code=RUN_CODE, layer=layer, neuron=neuron)
        print('>> creating html page...')
        neuron_vis_dir = f'{VIS_PATH}/L{layer}/N{neuron}'
        if not os.path.exists(neuron_vis_dir):
            os.makedirs(neuron_vis_dir)
        # We add some text to tell us what layer and neuron we're looking at
        heading = f"<h2>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron}</b></h2>\n"
        HTML = HEAD_AND_TITLE + heading + neuron_vis_full(
                activation_data=activation_data,
                dataset=text_dataset,
                model=model,
                neuron_dir=neuron_vis_dir,
        ) + TAIL
        with open(f'{neuron_vis_dir}/vis.html', 'w', encoding="utf-8") as read_file:
            read_file.write(HTML)
print('done!')
