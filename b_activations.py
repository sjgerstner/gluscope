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
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
import einops

#make sure HF_HUB_CACHE is set to 1 if necessary, before loading datasets
if "SLURM_JOBID" in os.environ:
    os.environ["HF_HUB_OFFLINE"]='1'
import datasets

from transformer_lens.model_bridge import TransformerBridge

import utils

HOOKS_TO_CACHE = ['ln2.hook_normalized', 'mlp.hook_post', 'mlp.hook_pre', 'mlp.hook_pre_linear']
REDUCTIONS = ['max', 'sum']
EXPERIMENTS = REDUCTIONS + ['sample']

def lists_to_tensors(example:dict[str,list|Any])->dict[str,list|torch.Tensor|Any]:
    for key, value in example.items():
        if isinstance(value, list):
            if isinstance(value[0], int):
                example[key] = torch.tensor(value)
            elif isinstance(value[0], list) and isinstance(value[0][0], int):
                example[key] = [torch.tensor(sublist) for sublist in value]
    return example

def _get_reduce_and_arg(
        cache_item:torch.Tensor, reduction:str, k=1, to_device='cpu',
        tensors_to_write:dict[str,torch.Tensor]|None=None, layer:int|None=None,
        with_layer_dim:bool=True,
    )-> dict[str,torch.Tensor]:
    if reduction not in ('max', 'min', 'top', 'bottom'):
        raise NotImplementedError(f"reduction {reduction} not implemented")

    #... ? layer neuron -> ... k layer neuron
    myred = torch.topk(
        cache_item,
        dim=-3 if with_layer_dim else -2,
        k=k,
        largest=reduction in ('max','top'),
    )
    if tensors_to_write is None:
        tensors_to_write = {
            'values': myred.values.to(to_device),
            'indices':myred.indices.to(dtype=torch.int, device=to_device),
        }
        if k==1:
            for key,tensor in tensors_to_write.items():
                tensors_to_write[key] = einops.rearrange(
                    tensor, '... 1 layer neuron -> ... layer neuron'
                )
    else:
        assert layer is not None
        try:
            tensors_to_write['values'][...,layer:layer+1,:] = myred.values.to(to_device)
            tensors_to_write['indices'][...,layer:layer+1,:] = myred.indices.to(to_device)
        except RuntimeError as e:
            print("tried to compute reduction", reduction, "with k=", k)
            if layer is not None:
                print("computed on a single layer")
            if not with_layer_dim:
                print("tried not to keep layer dim")
            print("desired shape of values and indices:", tensors_to_write['values'].shape, tensors_to_write['indices'].shape)
            print("actual computed shapes:", myred.values.shape, myred.indices.shape)
            raise e

    return tensors_to_write

def _get_reduce(
    cache_item:torch.Tensor, reduction:str,
    arg=False, use_cuda=True, to_device='cpu', k=1,
    tensors_to_write:torch.Tensor|dict[str,torch.Tensor]|None=None, layer:int|None=None,
    with_layer_dim:bool=True,
)->dict[str,torch.Tensor]|torch.Tensor:
    if use_cuda and torch.cuda.is_available():
        cache_item = cache_item.cuda()

    if arg:
        return _get_reduce_and_arg(
            cache_item=cache_item,
            reduction=reduction,
            k=k, to_device=to_device,
            tensors_to_write=tensors_to_write, layer=layer,
            with_layer_dim=with_layer_dim,
        )
    answer = einops.reduce(
        cache_item,
        '... layer neuron -> layer neuron' if with_layer_dim else '... neuron -> neuron',
        reduction
    ).to(to_device)
    if isinstance(tensors_to_write, torch.Tensor):
        assert layer is not None
        tensors_to_write[layer:layer+1,:] = answer
        return tensors_to_write
    else:
        return answer

def _compute_reductions_on_single_batch(
    cache,
    intermediate:dict[str|tuple[str,str]|tuple[str,str,str], torch.Tensor|dict[str,torch.Tensor]],
    zero_one:torch.Tensor,
    case:str,
    reductions:list[str]|None=None,
    layer:int|None=None,
) -> dict[str|tuple[str,str]|tuple[str,str,str], torch.Tensor|dict[str,torch.Tensor]]:
    if reductions is None:
        reductions=REDUCTIONS
    for key_to_summarise in utils.VALUES_TO_SUMMARISE:
        if key_to_summarise.startswith('hook'):
            raw_values = cache[f'mlp.{key_to_summarise}'].cuda() if layer is None else cache[f'blocks.{layer}.mlp.{key_to_summarise}'].cuda()
        elif key_to_summarise=='swish':
            raw_values = model.actfn(cache['mlp.hook_pre'].cuda()) if layer is None else model.actfn(cache[f'blocks.{layer}.mlp.hook_pre'].cuda())
        else:
            continue
        values = raw_values * zero_one
        # print(relevant_values.shape)
        for reduction in reductions:
            if key_to_summarise=='swish' and case.startswith('gate+') and reduction=='max':
                continue
            intermediate[(case, key_to_summarise, reduction)] = _get_reduce(
                values if reduction=="sum" else torch.abs(values),
                reduction=reduction,
                arg=(reduction!="sum"),
                tensors_to_write=intermediate[(case, key_to_summarise, reduction)] if (case, key_to_summarise, reduction) in intermediate else None,
                layer=layer,
                with_layer_dim=layer is None,
            )
            #batch pos layer neuron -> {'values': batch layer neuron, 'indices': batch layer neuron}
            if reduction=="max":#undo the abs
                intermediate[(case, key_to_summarise, reduction)]['values'] *= utils.RELEVANT_SIGNS[case][key_to_summarise]
    return intermediate

def _init_out_dict(intermediate):
    initial_dict={}
    for key,value in intermediate.items():
        if key[-1] in ['sum', 'freq']:
            initial_dict[key]=value
        elif key[-1] in ['max', 'min']:
            initial_dict[key] = {
                'values': value['values'],
                'indices': torch.arange(
                        0, args.batch_size, device='cpu'
                    ).view(
                        -1, 1, 1
                    ).expand(
                        -1, model.cfg.n_layers, model.cfg.d_mlp
                    ).contiguous()
                # torch.stack([
                #     torch.full((model.cfg.n_layers,model.cfg.d_mlp), counter)
                #     for counter in range(args.batch_size)
                # ])
            }
    return initial_dict

def _update_out_dict(args, dict_to_update, update_values, i):
    for key,value in dict_to_update.items():
        if key[-1] in ['sum', 'freq']:
            dict_to_update[key] = _get_reduce(
                torch.stack([value.cuda(), update_values[key].cuda()]),
                'sum'
            ).cpu()#batch layer neuron -> layer neuron
        elif key[-1] in ['max', 'min']:
            assert key[-1]=='max', """key 'min' should not appear anymore in this version of the code.
            This code uses the key 'max' even if it's technically a min."""
            dict_to_update[key] = {
                'values': torch.cat(
                    [
                        value['values'].cpu(),
                        update_values[key]['values'].cpu()#[mask_new]
                        ]
                    ),
                'indices':torch.cat(
                    [
                        value['indices'].cpu(),
                        torch.arange(
                            i*args.batch_size, (i+1)*args.batch_size, device='cpu'
                        ).view(
                            -1, 1, 1
                        ).expand(
                            -1, model.cfg.n_layers, model.cfg.d_mlp
                        ).contiguous()#[mask_new]
                        # torch.stack([
                        #         torch.full(
                        #             (model.cfg.n_layers,model.cfg.d_mlp),
                        #             i*args.batch_size+counter
                        #         )
                        #         for counter in range(args.batch_size)
                        #     ])
                        ]
                    )
            }#both entries: sample layer neuron
            # print(out_dict[key]['values'].shape)
            # print(out_dict[key]['indices'][:,:2,:2])
            #running topk computation
            #print(out_dict[key]['values'].shape) #should be: k layer neuron
            vi = _get_reduce(
                cache_item=torch.abs(dict_to_update[key]['values']),
                reduction="max",
                arg=True,
                k=min(dict_to_update[key]['values'].shape[0], args.examples_per_neuron),
                )#k+1 layer neuron -> k layer neuron
            # print(vi['indices'][:,:2,:2])
            # if args.test:
            #     print(out_dict[key]['indices'].shape)
            #     print(vi['indices'].shape)
            dict_to_update[key]['values'] = vi['values'] * utils.RELEVANT_SIGNS[key[0]][key[1]]#undo the abs
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
    sample_to_update:dict[str,list[int|torch.Tensor]],
    sampled_positions:list[int], sampled_activations:dict[str,torch.Tensor]
) -> dict[str,list[int|torch.Tensor]]:
    sample_to_update["sampled_positions"].extend(sampled_positions)
    for key,value in sampled_activations.items():
        sample_to_update[key].append(value)
    return sample_to_update

def _precompute_neuron_acts(
    model:TransformerBridge,
    ids_and_mask,
    batch_size,
    names_filter,
    sampled_positions:torch.Tensor|None=None,
    hooks_to_cache:list[str]=HOOKS_TO_CACHE,
) -> tuple[dict,dict]:
    _logits, raw_cache = model.run_with_cache(
        ids_and_mask['input_ids'],
        attention_mask=ids_and_mask['attention_mask'],
        names_filter=names_filter,
        #return_type=None,
        #device='cpu'#moves the cache, not the model. Avoids OOM errors.
    )
    # raw_cache is an ActivationCache
    # with keys 'blocks.layer.mlp.hook_post' etc
    # and entries mostly with shape (batch pos neuron)
    del _logits

    #mask = einops.rearrange(ids_and_mask['attention_mask'], 'batch pos -> batch pos 1')#.cpu()
    for layer in range(model.cfg.n_layers):
        for key_to_summarise in hooks_to_cache:
            raw_cache.cache_dict[f'blocks.{layer}.{key_to_summarise}'] *= ids_and_mask['attention_mask'].unsqueeze(-1)#this should just be a view of mask

    sampled_activations = {}
    if sampled_positions is not None and sampled_positions.numel()!=0:
        assert "sample" in args.experiments
        for key_to_summarise in hooks_to_cache:
            if not key_to_summarise.startswith('mlp'):
                continue
            sampled_activations[key_to_summarise] = torch.stack(
                [
                    raw_cache[f'blocks.{layer}.{key_to_summarise}'][torch.arange(batch_size), sampled_positions]
                    for layer in range(model.cfg.n_layers)
                ],
                dim=-2,#batch pos neuron -> batch 1 layer neuron
            ).cpu()

    #alternative:
    # raw_cache = raw_cache.to('cpu')

    return raw_cache, sampled_activations

def _finalize_sample(sample_data:dict[str,list[torch.Tensor|int]]):
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
    batch_file:str|None=None,
    **kwargs,
) -> tuple[dict,dict]:
    #loosely inspired by
    #https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb

    #prepare the loop
    reductions = [s for s in REDUCTIONS if s in args.experiments]#currently ['max', 'sum']

    batch_size = len(ids_and_mask['input_ids'])
    seq_len = max(len(ids) for ids in ids_and_mask['input_ids'])

    #prepare space for "intermediate" dictionary:
    intermediate : dict[str|tuple[str,str]|tuple[str,str,str], torch.Tensor|dict[str,torch.Tensor]] = {}
    for case in utils.CASES:
        intermediate[(case, 'freq')] = torch.empty((model.cfg.n_layers, model.cfg.d_mlp), device='cuda')
        for key_to_summarise in utils.VALUES_TO_SUMMARISE:
            for reduction in reductions:
                if key_to_summarise=='swish' and case.startswith('gate+') and reduction=='max':
                    continue
                if reduction=='sum':
                    intermediate[(case, key_to_summarise, 'sum')] = torch.empty((model.cfg.n_layers, model.cfg.d_mlp), device='cuda')
                elif reduction=='max':
                    intermediate[(case, key_to_summarise, 'max')] = {
                        "values": torch.empty((batch_size, model.cfg.n_layers, model.cfg.d_mlp), device='cuda'),
                        "indices": torch.empty((batch_size, model.cfg.n_layers, model.cfg.d_mlp), device='cuda', dtype=torch.int)
                    }
                else:
                    raise NotImplementedError(f"reduction must be 'max' or 'sum', but specified {reduction}")

    cache, sampled_activations = _precompute_neuron_acts(
        model=model,
        ids_and_mask=ids_and_mask,
        batch_size=batch_size,
        **kwargs,
    )
    if not args.no_cache:#TODO instead of this, directly stack tensors on GPU and save it from there?
        assert batch_file is not None
        #ln_cache: initialise with zeros (batch pos layer d_model)
        ln_cache = torch.zeros(
            (batch_size, max_seq_len, model.cfg.n_layers, model.cfg.d_model)
            )
        #fill in
        for layer in range(model.cfg.n_layers):
            ln_cache[:, :seq_len, layer, :] = cache[f'blocks.{layer}.ln2.hook_normalized'].cpu()
        torch.save(ln_cache, f"{batch_file}.pt")
        del ln_cache

    #summary keys (mean and frequencies)
    #layer neuron
    for layer in range(model.cfg.n_layers):
        layerwise_intermediate = {}
        bins=utils.detect_cases(
            gate_values=cache[f'blocks.{layer}.mlp.hook_pre'], in_values=cache[f'blocks.{layer}.mlp.hook_pre_linear'],
            to_device='cuda'
        )
        for case,zero_one in bins.items():
            #zero_one = einops.rearrange(zero_one, 'batch pos neuron -> batch pos 1 neuron')
            intermediate[(case, 'freq')][layer:layer+1, :] = _get_reduce(zero_one, 'sum', to_device='cuda', with_layer_dim=False)
            if reductions:
                intermediate = _compute_reductions_on_single_batch(
                    cache=cache,
                    intermediate=intermediate,
                    zero_one=zero_one,
                    case=case,
                    reductions=reductions,
                    layer=layer,
                )
            del zero_one
            torch.cuda.empty_cache()
    if "to_device" in kwargs and kwargs["to_device"]=='cpu':
        for key, value in intermediate.items():
            if isinstance(value, torch.Tensor):
                intermediate[key] = value.cpu()
                del value
            elif isinstance(value, dict):
                for skey, svalue in value.items():
                    value[skey] = svalue.cpu()
                    del svalue
    return intermediate, sampled_activations

def get_all_neuron_acts_on_dataset(
    args,
    model,
    dataset:datasets.Dataset,
    path=None,
):
    """Get all neuron activations on dataset.

    Args:
        args (Namespace): The argparse arguments
        model (HookedTransformer|TransformerBridge): The model to run
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

    hooks_to_cache = HOOKS_TO_CACHE.copy()
    if args.no_cache:
        hooks_to_cache.remove('ln2.hook_normalized')
    names_filter = [
        f"blocks.{layer}.{hook}"
        for layer in range(model.cfg.n_layers)
        for hook in hooks_to_cache
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
    for key_to_summarise in hooks_to_cache:
        if key_to_summarise.startswith('mlp'):
            sample_data[key_to_summarise] = []
    n_batches_to_sample = args.sample_size // args.batch_size
    sample_finalized = False
    if os.path.exists(f'{path}/checkpoints') and os.listdir(f'{path}/checkpoints'):
        last_checkpoint_number = max(
            int(s.split('-')[1].split('.')[0])
            for s in os.listdir(f'{path}/checkpoints')
        )
        my_out_dict = torch.load(f'{path}/checkpoints/checkpoint-{last_checkpoint_number}.pt')
    else:
        last_checkpoint_number=-1
    number_of_batches_to_skip=last_checkpoint_number//args.batch_size
    random.seed(43)
    torch.manual_seed(43)
    for i, batch in tqdm(enumerate(batched_dataset)):
        if i<=number_of_batches_to_skip:
            continue
        batch_file = f"{path}/activation_cache/batch{i}"
        batch = {
            'input_ids': pad_sequence(
                batch['input_ids'],
                padding_value=model.tokenizer.pad_token_type_id,
                batch_first=True,
            ).to(model.device), #tensor of shape batch x pos
            'attention_mask': pad_sequence(
                batch['attention_mask'],
                batch_first=True,
            ).to(model.device)
        }
        #print('batch shape:', batch['input_ids'].shape)
        if "sample" in args.experiments:
            if i<=n_batches_to_sample:
                sampled_positions = [random.randrange(seq.size(dim=0)) for seq in batch['input_ids']]
            else:
                sampled_positions = []
                if not sample_finalized:
                    _finalize_sample(sample_data)
                    sample_finalized = True
        else:
            if batch_size_unchanged and os.path.exists(f"{batch_file}.pt"):
                intermediate = torch.load(f"{batch_file}.pt")
                #continue
            if batch_size_unchanged and os.path.exists(f"{batch_file}.pickle"):
                with open(f"{batch_file}.pickle", 'rb') as file:
                    intermediate = utils._move_to(pickle.load(file), device='cuda')
                #continue
            sampled_positions=[]

        intermediate, sampled_activations = _get_all_neuron_acts(
            args=args,
            model=model, ids_and_mask=batch, names_filter=names_filter, max_seq_len=dataset.max_seq_len,
            #experiments=args.experiments,
            sampled_positions=torch.tensor(sampled_positions),
            hooks_to_cache=hooks_to_cache,
            batch_file=batch_file,
        )
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
            if i%args.save_every==0:
                if not os.path.exists(f'{path}/checkpoints'):
                    os.makedirs(f'{path}/checkpoints')
                torch.save(my_out_dict, f'{path}/checkpoints/checkpoint-{i*args.batch_size}.pt')
                if os.path.exists(f'{path}/checkpoints/checkpoint-{(i-args.save_every)*args.batch_size}.pt'):
                    os.remove(f'{path}/checkpoints/checkpoint-{(i-args.save_every)*args.batch_size}.pt')
    if "sample" in args.experiments and not sample_finalized:
        _finalize_sample(sample_data)
    for key in my_out_dict:
        if key[-1] in ('sum', 'freq'):
            my_out_dict[key] = my_out_dict[key].to(torch.float)
    for key in my_out_dict:
        if key[-1]=='sum':
            #for the moment frequencies are still absolute numbers so we can do this
            freq = my_out_dict[(key[0],'freq')]
            my_out_dict[key] = torch.where(
                freq > 0,
                my_out_dict[key] / freq,
                torch.zeros_like(my_out_dict[key])
            )
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
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--examples_per_neuron', default=16, type=int)
    #parser.add_argument('--resume_from', default=0)
    parser.add_argument('--datasets_dir', default='datasets')
    parser.add_argument('--results_dir', default='GLUScope-results')
    parser.add_argument('--save_to', default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_cache', type=bool, default=True)
    parser.add_argument('--sample_size', default=7000, help="only relevant if 'sample' in args.experiments", type=int)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--experiments', nargs='+', default=EXPERIMENTS)
    args = parser.parse_args()

    RUN_CODE = utils.get_run_code(args)
    SAVE_PATH = utils.make_save_path(args.results_dir, RUN_CODE)
    if not args.test:
        page_file = utils.get_page_file_path()
        if os.path.exists(page_file):
            with open(page_file, "r", encoding="utf-8") as f:
                page_list = json.load(f)
            model_present = False
            for d in page_list:
                if d["title"]==RUN_CODE:
                    model_present=True
                    break
            if not model_present:
                page_list.append({"title": RUN_CODE, "children":[]})
                with open(page_file, "w", encoding="utf-8") as f:
                    json.dump(page_list, f, indent=True)

    torch.set_grad_enabled(False)

    print('loading dataset...')
    dataset = utils.load_data(args)
    assert isinstance(dataset, datasets.Dataset)
    if args.test:
        dataset = dataset.select(range(33))
    dataset = dataset.with_format('torch')
    utils.add_properties(dataset)
    # dataset = dataset.with_format(
    #     type="torch",
    #     columns=["input_ids", "attention_mask"],
    #     pad=True,                # <-- enable automatic padding
    #     padding_value=model.tokenizer.pad_token_type_id,         # match your model's pad token
    #     pad_to_multiple_of=None
    # )

    print('loading model...')
    print(torch.cuda.device_count(), "available GPUs")
    model = TransformerBridge.boot_transformers(
        args.model,
        n_devices=torch.cuda.device_count(),
    )
    model.enable_compatibility_mode(
        refactor_glu=args.refactor_glu,
        fold_ln=False, center_writing_weights=False, center_unembed=False,
    )
    utils.add_actfn(model)

    print('computing activations...')
    REFACTOR_STR = "_refactored" if args.refactor_glu else""
    SUMMARY_FILE = f'{SAVE_PATH}/summary{REFACTOR_STR}'
    if args.test and os.path.exists(SAVE_PATH):
        if os.path.exists(f'{SUMMARY_FILE}.pt'):
            os.remove(f'{SUMMARY_FILE}.pt')#make sure we can recompute and save stuff
        if os.path.exists(os.path.join(SAVE_PATH, 'checkpoints')):
            for f in os.listdir(os.path.join(SAVE_PATH, 'checkpoints')):
                os.remove(os.path.join(SAVE_PATH, 'checkpoints', f))
    if not os.path.exists(f'{SUMMARY_FILE}.pickle') and not os.path.exists(f'{SUMMARY_FILE}.pt'):
        out_dict = get_all_neuron_acts_on_dataset(
            args=args,
            model=model,
            dataset=dataset,
            path=SAVE_PATH,
        )
        torch.save(out_dict, f'{SUMMARY_FILE}.pt')
    print('done!')
