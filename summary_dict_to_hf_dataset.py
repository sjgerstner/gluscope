from argparse import ArgumentParser
import os

import torch
from einops import rearrange

import datasets
from huggingface_hub import whoami, login

parser = ArgumentParser()
parser.add_argument('--model', default='OLMo-7B-0424-hf')
parser.add_argument('--file', default='summary_refactored.pt')
parser.add_argument('--push_to_hub', action='store_true', help="Make sure to set the right HF_TOKEN first.")
args = parser.parse_args()

summary_dict = torch.load(f'results/{args.model}/{args.file}')

n_layers, n_neurons = summary_dict['gate+_in+', 'freq'].shape

new_dict = {
    "layer": [l for l in range(n_layers) for _n in range(n_neurons)],
    "neuron": list(range(n_neurons))*n_layers,
}
for key,value in summary_dict.items():
    str_key = "_".join(key).replace("sum", "mean")
    if key[-1]!='max':
        new_dict[str_key] = rearrange(value, 'l n -> (l n)').detach().cpu().numpy()
    else:
        new_dict[f'{str_key}_values'] = rearrange(
            value['values'], 'topk l n -> (l n) topk'
        ).detach().cpu().numpy()
        new_dict[f'{str_key}_indices'] = rearrange(
            value['indices'], 'topk l n -> (l n) topk'
        ).detach().cpu().numpy()

dataset = datasets.Dataset.from_dict(new_dict)
if args.push_to_hub:
    username = whoami(token=os.environ['HF_TOKEN'])['name']
    login(token=os.environ['HF_TOKEN'])
    dataset.push_to_hub(f"{username}/{args.model}_neuron-activations")
else:
    dataset.save_to_disk(f'results/{args.model}/activation_dataset')
