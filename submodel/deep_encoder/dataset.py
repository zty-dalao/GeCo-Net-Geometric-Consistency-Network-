"""Dataset used by deep prior-encoder pretraining.

The implementation is deliberately shared with ``submodel.decoder`` so both
experiments use exactly the same subjects, clamping, and XYZ/ZYX conversion.
"""

from submodel.decoder.dataset import DentalVolumeDataset

__all__ = ["DentalVolumeDataset"]
