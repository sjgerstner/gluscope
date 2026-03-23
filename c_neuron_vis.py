"""Functions to visualise a neuron once all the data is computed"""
from json import dump

from torch import allclose, zeros_like
#from circuitsvis.tokens import colored_tokens_multi

from utils import CASES, get_act_type_keys, VALUES_TO_SUMMARISE

def _vis_example(i, indices, acts, dataset, tokenizer, key, neuron_dir, stop_tokens=None):
    index = int(indices[i])
    #print(dataset[index]['input_ids'])#tensor of ints
    tokens = tokenizer.batch_decode(#or convert_ids_to_tokens
        dataset[index]['input_ids']
    )
    if stop_tokens is not None:
        #TODO truncate beginning
        #TODO option to show full example
        tokens = tokens[:stop_tokens[i]]
    relevant_acts = acts[i,:len(tokens),:]#batch, pos, act_type
    data_url = f"{"_".join(key[:2])}_example_{i}.json"
    data_dict = {
        "tokens": tokens,
        "values": relevant_acts.tolist(),
        "labels": get_act_type_keys(key),
    }
    with open(f"{neuron_dir}/{data_url}", "w", encoding="utf-8") as f:
        dump(data_dict, f)
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

def _vis_examples(activation_data, dataset, tokenizer, neuron_dir):
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
                            tokenizer=tokenizer,
                            key=key,
                            neuron_dir=neuron_dir
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

def neuron_vis_full(activation_data, dataset, model, neuron_dir):
    """Full neuron visualisation for a given neuron.
    Args:
        neuron_data (dict): contains summary statistics and data on max/min activations
        dataset (datasets.Dataset)
        tokenizer (Huggingface tokenizer)
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
        activation_data=activation_data, dataset=dataset, tokenizer=model.tokenizer,
        neuron_dir=neuron_dir,
    ))
    return "\n".join(htmls)
