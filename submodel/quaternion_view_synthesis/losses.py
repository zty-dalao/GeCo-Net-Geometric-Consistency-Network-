"""Regularisers for quaternion view synthesis.

The synthesiser is a **residual** model: ``b = warp(a) + g * residual``.  The
plane-induced warp alone already reaches ~35 dB, so every pixel the network
changes has to be justified by the data.  Elastic Net on the residual encodes
exactly that prior:

* L1 part  -> sparsity: most pixels are left untouched (only occlusion and
  cone-beam parallax need correcting);
* L2 part  -> shrinkage: forbids a few large hallucinated patches.

``l1_ratio = 1`` is pure L1 (LASSO), ``0`` is pure L2 (ridge).
"""

import torch


def elastic_net(tensor, l1_ratio=0.5):
    """Elastic Net penalty of a tensor, averaged over all elements."""
    l1_ratio = float(l1_ratio)
    l1 = tensor.abs().mean()
    l2 = tensor.pow(2).mean()
    return l1_ratio * l1 + (1.0 - l1_ratio) * l2


def parameter_elastic_net(parameters, l1_ratio=0.5):
    """Elastic Net over the flattened parameters (weight decay variant)."""
    l1_ratio = float(l1_ratio)
    count = 0
    l1 = torch.zeros((), device=next(iter(parameters)).device)
    l2 = torch.zeros_like(l1)
    for parameter in parameters:
        l1 = l1 + parameter.abs().sum()
        l2 = l2 + parameter.pow(2).sum()
        count += parameter.numel()
    if count == 0:
        return l1
    return (l1_ratio * l1 + (1.0 - l1_ratio) * l2) / count
