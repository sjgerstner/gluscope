"""Functions to visualise a neuron once all the data is computed"""
from json import load, dump
import os

from torch import allclose, zeros_like, Tensor
#from circuitsvis.tokens import colored_tokens_multi

#make sure HF_HUB_CACHE is set to 1 if necessary, before loading datasets
if "SLURM_JOBID" in os.environ:
    os.environ["HF_HUB_OFFLINE"]='1'
from datasets import Dataset
from transformer_lens.model_bridge import TransformerBridge

from utils import CASES, get_act_type_keys, VALUES_TO_SUMMARISE, get_page_file_path

def update_pagelist(run_code, layer, neuron):
    page_file = get_page_file_path()
    modified_json=False
    if os.path.exists(page_file):
        with open(page_file, "r", encoding="utf-8") as read_file:
            page_list = load(read_file)
    else:
        page_list=[
            {"title":run_code, "children":[
                {"title": f"L{layer}", "children": [
                    {"title": f"N{neuron}", "children":[]}
                ]}
            ]}
        ]
        modified_json=True
    model_present = False
    for model_dict in page_list:
        if model_dict["title"]==run_code:
            model_present=True
            break
    if not model_present:
        page_list.append({"title": run_code, "children":[]})
        model_dict = page_list[-1]
        page_list = sorted(page_list, key=lambda d: d['title'])
        modified_json = True
    layer_present=False
    for layer_dict in model_dict["children"]:
        if layer_dict["title"]==f"L{layer}":
            layer_present=True
            break
    if not layer_present:
        model_dict["children"].append({"title": f"L{layer}", "children":[]})
        layer_dict=model_dict["children"][-1]
        model_dict["children"]=sorted(model_dict["children"], key=lambda d:int(d["title"][1:]))
        modified_json=True
    neuron_present=False
    for neuron_dict in layer_dict["children"]:
        if neuron_dict["title"]==f"N{neuron}":
            neuron_present=True
            break
    if not neuron_present:
        layer_dict["children"].append({"title": f"N{neuron}", "url": f"{run_code}/L{layer}/N{neuron}/vis.html"})
        layer_dict["children"]=sorted(layer_dict["children"], key=lambda d:int(d["title"][1:]))
        modified_json=True
    if modified_json:
        with open(page_file, "w", encoding="utf-8") as write_file:
            dump(page_list, write_file, indent=4)

def _vis_example(
        i, indices, acts, dataset,
        model:TransformerBridge,
        key, neuron_dir,
        stop_tokens=None,
        verbose=False,
    ):
    index = int(indices[i])
    #print(dataset[index]['input_ids'])#tensor of ints
    tokens = model.to_str_tokens(
        dataset[index]['text']
    )
    if stop_tokens is not None:
        #TODO truncate beginning
        #TODO option to show full example
        tokens = tokens[:stop_tokens[i]]
    relevant_acts = acts[i,:len(tokens),:]#batch, pos, act_type
    if verbose:
        print(relevant_acts[:,0])
    data_url = f"{'_'.join(key[:2])}_example_{i}.json"
    data_dict = {
        "tokens": tokens,
        "values": relevant_acts.tolist(),
        "labels": get_act_type_keys(key),
    }
    if verbose:
        print(data_dict["values"])
    with open(f"{neuron_dir}/{data_url}", "w", encoding="utf-8") as f:
        dump(data_dict, f, indent=4)
    #TODO source of dataset example
        # colored_tokens_multi(
        #     tokens=tokens,
        #     values=relevant_acts,
        #     labels=get_act_type_keys(key),
        # )
    return f"""<details>
            <summary><h4>Example {i}</h4></summary>
            <div class="circuit-viz" data-url="./{data_url}">
            </div>
        </details>"""

def _vis_examples(activation_data, dataset, model:TransformerBridge, neuron_dir):
    htmls = []
    for case in CASES:
        htmls.append(f'<details>\n<summary><h2>Prototypical activations for case {case}</h2></summary>')
        for act_type in VALUES_TO_SUMMARISE:
            key = (case, act_type, 'max')
            if key in activation_data and 'all_acts' in activation_data[key] and activation_data[key]['values'][0]!=0:
                htmls.append(f'<details>\n<summary><h3>Extreme {act_type} activations</h3></summary>')
                for i in range(activation_data[key]['indices'].shape[0]):
                    first_acts=activation_data[key]['all_acts'][i,:,0]#sample pos act_type
                    #ignore samples in which no token satisfies the condition:
                    if allclose(first_acts, zeros_like(first_acts), atol=1e-7):
                        break
                    htmls.append(
                        _vis_example(
                            i=i,
                            indices=activation_data[key]['indices'],
                            acts=activation_data[key]['all_acts'],
                            stop_tokens=activation_data[key]['position_indices']+3,
                            dataset=dataset,
                            model=model,
                            key=key,
                            neuron_dir=neuron_dir,
                            #verbose = key==('gate-_in-', 'hook_pre', 'max') and i==0,
                            )
                    )
            htmls.append('</details>\n<hr>')
        htmls.append('</details>\n<hr>')
    return '\n'.join(htmls)

def _vis_stats(activation_data, actfn):
    # We add a kind of
    # "table": cases are main columns,
    # and within that we have paragraphs with frequency,
    # and max/min/mean (all within one paragraph)
    # of gate/swish/in/post (separate paragraphs)
    htmls = []
    htmls.append('<table><tr>')
    htmls.extend([f"<td><h4>{case}</h4></td>" for case in CASES])
    htmls.append('</tr><tr>')
    htmls.extend(
        [f"<td>Frequency: <b>{activation_data[(case, 'freq')]:.2%}</b>.</td>" for case in CASES]
    )
    htmls.append('</tr>')
    extreme_values = {}
    maxima = {}
    minima = {}
    for act_type in VALUES_TO_SUMMARISE:
        for case in CASES:
            if (case,act_type,'max') in activation_data.keys():
                extreme_values[(case, act_type)] = activation_data[(case,act_type,'max')]['values'][0]
            elif act_type=='swish':
                extreme_values[(case, act_type)] = actfn(extreme_values[(case, 'hook_pre')])
            maxima[(case, act_type)] = extreme_values[(case, act_type)] if extreme_values[(case, act_type)]>0 else 0
            minima[(case, act_type)] = extreme_values[(case, act_type)] if extreme_values[(case, act_type)]<0 else -0
        htmls.append('<tr>')
        avgs = {
            case: (
                activation_data[(case, act_type, 'sum')]
                if (case, act_type, 'sum') in activation_data
                else activation_data[(case, act_type, 'mean')]
            )
            for case in CASES
        }
        for case, avg in avgs.items():
            if isinstance(avg, Tensor):
                assert avg.numel()==1, f"avg is not a singleton, but has shape {avg.shape}"
                avgs[case] = avg.item()
        htmls.extend(
            [f"""<td>
            <b>{act_type}</b>:<br>
            Max: <b>{maxima[(case, act_type)]:.2f}</b>;<br>
            Min: <b>{minima[(case, act_type)]:.2f}</b>;<br>
            Avg: <b>{avgs[case]:.2f}</b>.
            </td>
            """
            for case in CASES
            ]
        )
        htmls.append('</tr>')
    htmls.append('</table>')
    return "\n".join(htmls)

def neuron_vis_full(activation_data:dict, dataset:Dataset, model:TransformerBridge, neuron_dir:str):
    """Full neuron visualisation for a given neuron.
    Args:
        activation_data (dict): contains summary statistics and data on max/min activations
        dataset (datasets.Dataset)
        model (TransformerBridge)
        neuron_dir (str)
    Returns:
        html string
    """
    htmls = []
    # # We first add the style to make each token element have a nice border
    # htmls = [style_string]
    #TODO weight-based analysis (topk tokens + RW functionality)
    htmls.append(_vis_stats(
        activation_data=activation_data, actfn=model.actfn
    ))
    htmls.append(_vis_examples(
        activation_data=activation_data, dataset=dataset, model=model,
        neuron_dir=neuron_dir,
    ))
    return "\n".join(htmls)
