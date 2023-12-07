import torch

def weighted_mse(pred, true, weights=None):
    if weights is not None:
        return torch.mean(((pred - true) ** 2) * weights)
    else:
        return torch.mean(((pred - true) ** 2))