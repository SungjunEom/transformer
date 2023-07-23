import torch

def expand_mask(mask):
    if mask.ndim == 1:
        mask = mask.repeat(mask.shape[-1],1)
    return mask