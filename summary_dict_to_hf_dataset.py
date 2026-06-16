from argparse import ArgumentParser
import os

import torch
from einops import rearrange

#make sure HF_HUB_CACHE is set to 1 if necessary, before loading datasets
if "SLURM_JOBID" in os.environ:
    os.environ["HF_HUB_OFFLINE"]='1'
import datasets

parser = ArgumentParser()
parser.add_argument('--dir', default='results/7B_new')
parser.add_argument('--file', default='summary_refactored.pt')
args = parser.parse_args()

summary_dict = torch.load(f'{args.dir}/{args.file}')

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
dataset.save_to_disk(f'{args.dir}/activation_dataset')
