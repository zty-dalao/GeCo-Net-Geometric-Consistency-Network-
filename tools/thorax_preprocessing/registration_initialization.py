"""Initialization strategies for thorax CT-to-CBCT registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class InitializationResult:
    """Rigid initialization together with centers used for audit logging."""

    transform: sitk.Euler3DTransform
    method: str
    fixed_center_mm: tuple[float, float, float]
    moving_center_mm: tuple[float, float, float]


def _mask_center_of_mass(mask: sitk.Image) -> tuple[float, float, float]:
    statistics = sitk.LabelShapeStatisticsImageFilter()
    binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
    statistics.Execute(binary)
    if not statistics.HasLabel(1):
        raise ValueError("Cannot initialize registration from an empty body mask")
    return tuple(float(value) for value in statistics.GetCentroid(1))


def _image_physical_center(image: sitk.Image) -> tuple[float, float, float]:
    continuous_index = tuple((np.asarray(image.GetSize(), dtype=float) - 1.0) / 2.0)
    return tuple(float(value) for value in image.TransformContinuousIndexToPhysicalPoint(continuous_index))


def initialize_rigid_transform(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    method: str = "geometry",
) -> InitializationResult:
    """Create a CT-to-CBCT registration initializer.

    ``geometry`` reproduces the original image-grid-center initialization.
    ``moments`` uses binary body masks so couch, truncation and cross-modality
    intensity differences have less influence than raw-image intensity moments.
    """
    if method == "geometry":
        transform = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        fixed_center = _image_physical_center(fixed)
        moving_center = _image_physical_center(moving)
    elif method == "moments":
        fixed_binary = sitk.Cast(fixed_mask > 0, sitk.sitkFloat32)
        moving_binary = sitk.Cast(moving_mask > 0, sitk.sitkFloat32)
        fixed_center = _mask_center_of_mass(fixed_mask)
        moving_center = _mask_center_of_mass(moving_mask)
        transform = sitk.CenteredTransformInitializer(
            fixed_binary,
            moving_binary,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )
    else:
        raise ValueError(f"Unknown initialization method: {method}")

    return InitializationResult(
        transform=sitk.Euler3DTransform(transform),
        method=method,
        fixed_center_mm=fixed_center,
        moving_center_mm=moving_center,
    )
