"""Quaternion view synthesis: 20 measured views -> many synthetic views -> FDK.

Public API::

    QuaternionViewSynthesizer   single-view synthesiser F(a, q_{s->t}) -> b
    QuaternionSoftmaxEnsemble   quaternion-softmax fusion of the per-view outputs
    geometry                    homography / quaternion geometry helpers
    qconv                       Hamilton-product QConv / QLinear / QNorm
    fdk                         weighted cone-beam FDK reconstruction
"""

from .model import QuaternionSoftmaxEnsemble, QuaternionViewSynthesizer, build_model

__all__ = [
    "QuaternionViewSynthesizer",
    "QuaternionSoftmaxEnsemble",
    "build_model",
]
