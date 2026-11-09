"""Channel perturbation of a decoder tap for the unlabeled consistency branch."""
from contextlib import contextmanager

import torch
from torch import nn


TAP_FIRST_HEAD = {"d3": 1, "d2": 2}   # deep-supervision outputs produced from each tap onward


def channel_dropout(feature, drop_probability):
    """Zero whole channels per sample with the given probability (scalar or (B, C)).

    Survivors are rescaled per sample so the number of active channels is preserved on
    average (CHAP's rescaling), which also keeps random and score-guided masks comparable.
    """
    probability = torch.as_tensor(drop_probability, dtype=feature.dtype, device=feature.device)
    keep = torch.bernoulli((1 - probability).expand(feature.shape[:2]))
    scale = keep.shape[1] / keep.sum(dim=1, keepdim=True).clamp_min(1)
    return feature * (keep * scale)[..., None, None, None]


@contextmanager
def frozen_bn_statistics(module):
    """Train-mode BN (batch statistics) without updating running statistics."""
    norms = [layer for layer in module.modules() if isinstance(layer, nn.modules.batchnorm._BatchNorm)]
    saved = [(layer.momentum, layer.num_batches_tracked.clone()) for layer in norms]
    for layer in norms:
        layer.momentum = 0.0
    try:
        yield
    finally:
        for layer, (momentum, tracked) in zip(norms, saved):
            layer.momentum = momentum
            layer.num_batches_tracked.copy_(tracked)
