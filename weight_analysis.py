import torch
from einops import rearrange
from transformer_lens import HookedTransformer

#TODO idea:
# sample means W_E or W_U,
# first dimension "layers" means w_gate, w_in or w_out

def standard_data(model:HookedTransformer):
    return {
        "tokens": [model.to_str_tokens(list(range(model.cfg.d_vocab)))],#officially samples x tokens, here just one "sample"
        "thirdDimensionName": "",#officially neurons, just numbered
        "firstDimensionLabels": [""],
        "firstDimensionName": "",#officially layers
        "sampleLabels": [""]#zeroth dimension
    }

def topk_token_data(logits:torch.Tensor, k=16):
    topk = torch.topk(logits, k, largest=True)
    bottomk = torch.topk(logits, k, largest=False)
    return {#officially dimensions are samples, k, layers, neurons
        "topkVals": rearrange(topk.values, 'k -> 1 1 k 1').tolist(),
        "topkIdxs": rearrange(topk.indices, 'k -> 1 1 k 1').tolist(),
        "bottomkVals": rearrange(bottomk.values, 'k -> 1 1 k 1').tolist(),
        "bottomkIdxs": rearrange(bottomk.indices, 'k -> 1 1 k 1').tolist(),
    }

def full_data(model:HookedTransformer, logits:torch.Tensor, k=16):
    mydict = standard_data(model)
    mydict.update(topk_token_data(logits=logits, k=k))
    return mydict
