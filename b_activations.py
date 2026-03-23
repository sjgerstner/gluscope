"""
Compute max activations for all neurons on a given dataset.
The code was written with batch size 1 in mind.
Other batch sizes will almost certainly lead to bugs.
"""

#TODO (also other files) pathlib

from argparse import ArgumentParser
import json
import os
import pickle
import random
from tqdm import tqdm

import torch
#from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import einops
#from transformers import DataCollatorWithPadding
import datasets

import utils

HOOKS_TO_CACHE = ['ln2.hook_normalized', 'mlp.hook_post', 'mlp.hook_pre', 'mlp.hook_pre_linear']
REDUCTIONS = ['max', 'sum']
EXPERIMENTS = REDUCTIONS + ['sample']

def _get_reduce_and_arg(cache_item, reduction, k=1, to_device='cpu')-> dict[str,torch.Tensor]:
    if reduction not in ('max', 'min', 'top', 'bottom'):
        raise NotImplementedError(f"reduction {reduction} not implemented")

    #... ? layer neuron -> ... k layer neuron
    myred = torch.topk(
        cache_item,
        dim=-3,
        k=k,
        largest=reduction in ('max','top'),
    )
    vi_dict = {
        'values': myred.values.to(to_device),
        'indices':myred.indices.to(dtype=torch.int, device=to_device),
    }
    if k==1:
        for key,tensor in vi_dict.items():
            vi_dict[key] = einops.rearrange(
                tensor, '... 1 layer neuron -> ... layer neuron'
            )
    return vi_dict

def _get_reduce(
    cache_item, reduction, arg=False, use_cuda=True, to_device='cpu', k=1,
)->dict[str,torch.Tensor]|torch.Tensor:
    if use_cuda and torch.cuda.is_available():
        cache_item = cache_item.cuda()

    if arg:
        return _get_reduce_and_arg(cache_item, reduction, k=k, to_device=to_device)
    return einops.reduce(
            cache_item,
            '... layer neuron -> layer neuron',
            reduction
            ).to(to_device)

def _compute_reductions_on_single_batch(
    cache,
    intermediate:dict[str|tuple[str,str]|tuple[str,str,str], torch.Tensor|dict[str,torch.Tensor]],
    zero_one:torch.Tensor,
    case:str,
    reductions:list[str]|None=None,
) -> dict[str|tuple[str,str]|tuple[str,str,str], torch.Tensor|dict[str,torch.Tensor]]:
    if reductions is None:
        reductions=REDUCTIONS
    for key_to_summarise in utils.VALUES_TO_SUMMARISE:
        if key_to_summarise.startswith('hook'):
            values = cache[f'mlp.{key_to_summarise}'].cuda()
        elif key_to_summarise=='swish':
            values = model.actfn(cache['mlp.hook_pre'].cuda())
        else:
            continue
        values *= zero_one
        # print(relevant_values.shape)
        for reduction in reductions:
            if key_to_summarise=='swish' and case.startswith('gate+') and reduction=='max':
                continue
            intermediate[(case, key_to_summarise, reduction)] = _get_reduce(
                values if reduction=="sum" else torch.abs(values),
                reduction=reduction,
                arg=(reduction!="sum"),
            )
            #batch pos layer neuron -> {'values': batch layer neuron, 'indices': batch layer neuron}
            if reduction=='max':
                # print((case, key_to_summarise, reduction))
                # print(intermediate[(case, key_to_summarise, reduction)]['values'].shape)
                intermediate[(case, key_to_summarise, reduction)]['values'] *= utils.RELEVANT_SIGNS[case][key_to_summarise]
                # print(intermediate[(case, key_to_summarise, reduction)]['values'].shape)
    return intermediate

def _init_out_dict(intermediate):
    initial_dict={}
    for key,value in intermediate.items():
        if key[-1] in ['sum', 'freq']:
            initial_dict[key]=value
        elif key[-1] in ['max', 'min']:
            initial_dict[key] = {
                'values':value['values'],
                'indices':torch.stack([
                    torch.full((model.cfg.n_layers,model.cfg.d_mlp), counter)
                    for counter in range(args.batch_size)
                ])
            }
    return initial_dict

def _update_out_dict(args, dict_to_update, update_values, i):
    for key,value in dict_to_update.items():
        if key[-1] in ['sum', 'freq']:
            dict_to_update[key] = _get_reduce(
                torch.stack([value, update_values[key]]),
                'sum'
                )#batch layer neuron -> layer neuron
        elif key[-1] in ['max', 'min']:
            #print(key)
            dict_to_update[key] = {
                'values': torch.cat(
                    [
                        value['values'],
                        update_values[key]['values']
                        ]
                    ),
                'indices':torch.cat(
                    [
                        value['indices'],
                        torch.stack([
                                torch.full(
                                    (model.cfg.n_layers,model.cfg.d_mlp),
                                    i*args.batch_size+counter
                                )
                                for counter in range(args.batch_size)
                            ])
                        ]
                    )
            }#both entries: sample layer neuron
            # print(out_dict[key]['values'].shape)
            # print(out_dict[key]['indices'][:,:2,:2])
            #running topk computation
            #print(out_dict[key]['values'].shape) #should be: k layer neuron
            vi = _get_reduce(
                dict_to_update[key]['values'] * utils.RELEVANT_SIGNS[key[0]][key[1]],
                reduction=key[-1],
                arg=True,
                k=min(dict_to_update[key]['values'].shape[0], args.examples_per_neuron),
                )#k+1 layer neuron -> k layer neuron
            # print(vi['indices'][:,:2,:2])
            # if args.test:
            #     print(out_dict[key]['indices'].shape)
            #     print(vi['indices'].shape)
            dict_to_update[key]['values'] = vi['values'] * utils.RELEVANT_SIGNS[key[0]][key[1]]
            dict_to_update[key]['indices'] = torch.gather(
                dict_to_update[key]['indices'], dim=0, index=vi['indices']
            )
            #original dataset indices!
            #I want:
            #new_out_dict[key]['indices'][i,layer,neuron] =
            # out_dict[key]['indices'][vi['indices'][i,layer,neuron],layer,neuron]
            #hence the above line of code
    return dict_to_update

def _update_sample(
    sample_to_update:dict[str,list],
    sampled_positions:list, sampled_activations:dict[str,torch.Tensor]
):
    sample_to_update["sampled_positions"].extend(sampled_positions)
    for key,value in sampled_activations.items():
        sample_to_update[key].append(value)
    return sample_to_update

def _precompute_neuron_acts(
    model:utils.ModelWrapper,
    ids_and_mask,
    batch_size,
    names_filter,
    sampled_positions:torch.Tensor|None=None
) -> tuple[dict,dict]:
    _logits, raw_cache = model.run_with_cache(
        ids_and_mask['input_ids'],
        attention_mask=ids_and_mask['attention_mask'],
        names_filter=names_filter,
        #return_type=None,
    )
    #ActivationCache
    # with keys 'blocks.layer.mlp.hook_post' etc
    # and entries mostly with shape (batch pos neuron)
    del _logits

    mask = einops.rearrange(ids_and_mask['attention_mask'], 'batch pos -> batch pos 1 1').cpu()
    #batch pos neuron
    cache={}
    sampled_activations = {}
    for key_to_summarise in HOOKS_TO_CACHE:
        # print(key_to_summarise)
        # print(raw_cache[f'blocks.0.{key_to_summarise}'].shape)
        cache[key_to_summarise] = torch.stack(
            [#only load it to gpu when needed:
                raw_cache[f'blocks.{layer}.{key_to_summarise}'].cpu()
                for layer in range(model.cfg.n_layers)
            ],
            dim=-2,#batch pos neuron/d_model -> batch pos layer neuron/d_model
        )
        # print(cache[key_to_summarise].shape)
        cache[key_to_summarise] *= mask
        #cache[key_to_summarise] = cache[key_to_summarise].cpu()
        if sampled_positions is not None and sampled_positions.numel()!=0 and key_to_summarise.startswith('mlp'):
            assert "sample" in args.experiments
            sampled_activations[key_to_summarise] = cache[key_to_summarise][
                torch.arange(batch_size), sampled_positions, :,:
                ].cpu()
    del raw_cache

    return cache, sampled_activations

def _finalize_sample(sample_data):
    assert len(sample_data["sampled_positions"]) > 0, "No positions were sampled!"
    for key,value in sample_data.items():
        if key!="sampled_positions":
            assert isinstance(value, list), f"Entry for {key} should be a list, but is a {type(value)}"
            assert len(value)>0, f"Entry for {key} is empty"
            assert isinstance(value[0], torch.Tensor), f"The list stored at key {key} should contain tensors, but contains {type(value[0])}"
            sample_data[key] = torch.cat(value, dim=0)#concatenate along batch dimension
    torch.save(sample_data, f"{SAVE_PATH}/sample{REFACTOR_STR}.pt")

def _get_all_neuron_acts(
    args, model, ids_and_mask, max_seq_len=1024,
    **kwargs,
) -> tuple[dict,dict]:
    #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
    intermediate : dict[str|tuple[str,str]|tuple[str,str,str], torch.Tensor|dict[str,torch.Tensor]] = {}

    batch_size = len(ids_and_mask['input_ids'])
    seq_len = max(len(ids) for ids in ids_and_mask['input_ids'])

    cache, sampled_activations = _precompute_neuron_acts(
        model=model,
        ids_and_mask=ids_and_mask,
        batch_size=batch_size,
        **kwargs,
    )
    #ln_cache: initialise with zeros (batch pos layer d_model)
    intermediate['ln_cache'] = torch.zeros(
        (batch_size, max_seq_len, model.cfg.n_layers, model.cfg.d_model)
        )
    #fill in
    intermediate['ln_cache'][:, :seq_len, :] = cache['ln2.hook_normalized'].cpu()

    #prepare the loop
    reductions = [s for s in REDUCTIONS if s in args.experiments]

    #summary keys (mean and frequencies)
    #layer neuron
    bins=utils.detect_cases(
        gate_values=cache['mlp.hook_pre'], in_values=cache['mlp.hook_pre_linear']
    )
    for case,zero_one in bins.items():
        zero_one = zero_one.cuda()
        intermediate[(case, 'freq')] = _get_reduce(zero_one, 'sum')
        if reductions:
            intermediate = _compute_reductions_on_single_batch(
                cache=cache,
                intermediate=intermediate,
                zero_one=zero_one,
                case=case,
                reductions=reductions,
            )
        del zero_one
    return intermediate, sampled_activations

def get_all_neuron_acts_on_dataset(
    args, model, dataset:datasets.Dataset, path=None
):
    """Get all neuron activations on dataset.

    Args:
        args (Namespace): The argparse arguments
        model (HookedTransformer): The model to run
        dataset (Dataset): A Huggingface-style dataset to run the model on
        path (str, optional): The path to save the data.
            Within this path we will have a subdirectory activation_cache.
            Defaults to None (i.e., current directory).

    Returns:
        dict[Tensor]: a dict of tensors with all the relevant information
            (cached activations and summary statistics).
        Keys are those in the KEYS constant.
    """
    #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb
    if path is None:
        path = '.'

    batched_dataset = dataset.batch(
        batch_size=args.batch_size,
        drop_last_batch=False
        ) #preserves order
    #each row is one batch, represented as a dict[str, list[Tensor]],
    #where the str is 'input_ids' or 'attention_mask' and the list has args.batch_size elements.

    names_filter = [
        f"blocks.{layer}.{hook}"
        for layer in range(model.cfg.n_layers)
        for hook in HOOKS_TO_CACHE
    ]

    if not args.no_cache and not os.path.exists(f'{path}/activation_cache'):
        os.mkdir(f'{path}/activation_cache')
    previous_batch_size = 0
    if os.path.exists(f'{path}/activation_cache/batch_size.txt'):
        with open(f'{path}/activation_cache/batch_size.txt', 'r', encoding='utf-8') as file:
            previous_batch_size = int(file.read())
    #print(previous_batch_size, args.batch_size)
    batch_size_unchanged = previous_batch_size==args.batch_size
    if not args.no_cache and not batch_size_unchanged:
        with open(f'{path}/activation_cache/batch_size.txt', 'w', encoding='utf-8') as file:
            file.write(str(args.batch_size))
    sample_data = {"sampled_positions":[]}
    for key_to_summarise in HOOKS_TO_CACHE:
        if key_to_summarise.startswith('mlp'):
            sample_data[key_to_summarise] = []
    n_batches_to_sample = args.sample_size // args.batch_size
    random.seed(43)
    torch.manual_seed(43)
    for i, batch in tqdm(enumerate(batched_dataset)):
        batch_file = f"{path}/activation_cache/batch{i}"
        if "sample" in args.experiments:
            if i<=n_batches_to_sample:
                sampled_positions = [random.randrange(seq.size(dim=0)) for seq in batch['input_ids']]
            else:
                sampled_positions = []
                _finalize_sample(sample_data)
        else:
            if batch_size_unchanged and os.path.exists(f"{batch_file}.pt"):
                intermediate = torch.load(f"{batch_file}.pt")
                continue
            if batch_size_unchanged and os.path.exists(f"{batch_file}.pickle"):
                with open(f"{batch_file}.pickle", 'rb') as file:
                    intermediate = utils._move_to(pickle.load(file), device='cuda')
                continue
            sampled_positions=[]
        batch = {
            'input_ids': pad_sequence(
                batch['input_ids'],
                padding_value=model.tokenizer.pad_token_type_id,
                batch_first=True,
            ), #tensor of shape batch x pos
            'attention_mask': pad_sequence(
                batch['attention_mask'],
                batch_first=True,
            )
        }
        intermediate, sampled_activations = _get_all_neuron_acts(
            args=args,
            model=model, ids_and_mask=batch, names_filter=names_filter, max_seq_len=dataset.max_seq_len,
            #experiments=args.experiments,
            sampled_positions=torch.tensor(sampled_positions),
        )
        if not args.no_cache:
            torch.save(intermediate, f"{batch_file}.pt")
        del intermediate['ln_cache']
        if sampled_activations:
            sample_data = _update_sample(
                sample_data, sampled_positions=sampled_positions, sampled_activations=sampled_activations
            )
        if i==0:
            my_out_dict = _init_out_dict(intermediate)
        else:
            my_out_dict = _update_out_dict(
                dict_to_update=my_out_dict, update_values=intermediate,
                args=args, i=i,
            )
    if "sample" in args.experiments and sampled_positions:#second condition checks that sample_data was not finalized
        _finalize_sample(sample_data)
    for key in my_out_dict:
        if key[-1] in ('sum', 'freq'):
            my_out_dict[key] = my_out_dict[key].to(torch.float)
    for key in my_out_dict:
        if key[-1]=='sum':
            #for the moment frequencies are still absolute numbers so we can do this
            my_out_dict[key] /= my_out_dict[(key[0],'freq')]
            #now the 'sum' entry is actually a mean!
    for key in my_out_dict:
        if key[-1]=='freq':
            my_out_dict[key] /= float(dataset.n_tokens)

    return my_out_dict#, sample_data

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='dolma-small')
    parser.add_argument('--model', default='allenai/OLMo-1B-hf')
    parser.add_argument(
        '--refactor_glu',
        action='store_true',
        help='whether to refactor the weights such that cos(w_gate,w_in)>=0'
    )
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--examples_per_neuron', default=16, type=int)
    #parser.add_argument('--resume_from', default=0)
    parser.add_argument('--datasets_dir', default='datasets')
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--save_to', default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_cache', action='store_true')
    parser.add_argument('--sample_size', default=20000, help="only relevant if 'sample' in args.experiments")
    parser.add_argument('--experiments', nargs='+', default=EXPERIMENTS)
    args = parser.parse_args()

    RUN_CODE = utils.get_run_code(args)

    SAVE_PATH = f"{args.results_dir}/{RUN_CODE}"
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if not args.test:
        with open("docs/pages.json", "r", encoding="utf-8") as f:
            page_list = json.load(f)
        model_present = False
        for d in page_list:
            if d["title"]==RUN_CODE:
                model_present=True
                break
        if not model_present:
            page_list.append({"title": RUN_CODE, "children":[]})
            with open("docs/pages.json", "w", encoding="utf-8") as f:
                json.dump(page_list, f, indent=True)

    torch.set_grad_enabled(False)

    model = utils.ModelWrapper.from_pretrained(args.model, refactor_glu=args.refactor_glu)

    dataset = utils.load_data(args)
    assert isinstance(dataset, datasets.Dataset)
    if args.test:
        dataset = dataset.select(range(8))
    utils.add_properties(dataset)
    # dataset = dataset.with_format(
    #     type="torch",
    #     columns=["input_ids", "attention_mask"],
    #     pad=True,                # <-- enable automatic padding
    #     padding_value=model.tokenizer.pad_token_type_id,         # match your model's pad token
    #     pad_to_multiple_of=None
    # )

    print('computing activations...')
    REFACTOR_STR = "_refactored" if args.refactor_glu else""
    SUMMARY_FILE = f'{SAVE_PATH}/summary{REFACTOR_STR}'
    if not os.path.exists(f'{SUMMARY_FILE}.pickle') and not os.path.exists(f'{SUMMARY_FILE}.pt'):
        out_dict = get_all_neuron_acts_on_dataset(
            args=args,
            model=model,
            dataset=dataset,
            path=SAVE_PATH,
        )
        torch.save(out_dict, f'{SUMMARY_FILE}.pt')
        #torch.save(sample, f"{SAVE_PATH}/sample{REFACTOR_STR}.pt")
    print('done!')
