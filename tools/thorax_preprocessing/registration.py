"""Automatically register the planning CT to the Varian CBCT.

The CBCT is the fixed image and the planning CT is the moving image.  The
resulting transform therefore maps CBCT physical points into planning-CT
physical space and can be passed directly to ``sitk.Resample(moving, fixed,
transform, ...)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from .dicom_io import (
    discover_series,
    hu_to_mu,
    image_with_array,
    load_hu,
    select_series,
)


def preprocess_hu(image: sitk.Image) -> sitk.Image:
    """Clamp extreme HU values and map them to [0, 1] for registration."""
    return sitk.IntensityWindowing(
        sitk.Cast(image, sitk.sitkFloat32),
        windowMinimum=-1000.0,
        windowMaximum=1500.0,
        outputMinimum=0.0,
        outputMaximum=1.0,
    )


def largest_body_mask(image_hu: sitk.Image) -> sitk.Image:
    """Extract a conservative body mask while excluding exterior air/table."""
    mask = sitk.Cast(image_hu > -700.0, sitk.sitkUInt8)
    mask = sitk.BinaryMorphologicalClosing(mask, [4, 4, 1])
    mask = sitk.BinaryFillhole(mask)
    components = sitk.ConnectedComponent(mask)
    components = sitk.RelabelComponent(components, sortByObjectSize=True)
    return sitk.Cast(components == 1, sitk.sitkUInt8)


def geometry_initializer(fixed: sitk.Image, moving: sitk.Image) -> sitk.Euler3DTransform:
    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    return sitk.Euler3DTransform(transform)


def _shrink(image: sitk.Image, factor: int, is_mask: bool = False) -> sitk.Image:
    if factor == 1:
        return image
    result = sitk.Shrink(image, [factor] * 3)
    return sitk.Cast(result > 0, sitk.sitkUInt8) if is_mask else result


def coarse_longitudinal_search(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    initial: sitk.Euler3DTransform,
    search_range_mm: float,
    step_mm: float,
) -> tuple[sitk.Euler3DTransform, list[dict[str, float]]]:
    """Choose a robust initial superior/inferior translation using mutual information."""
    factor = 4
    fixed_low = _shrink(fixed, factor)
    moving_low = _shrink(moving, factor)
    fixed_mask_low = _shrink(fixed_mask, factor, is_mask=True)
    moving_mask_low = _shrink(moving_mask, factor, is_mask=True)

    metric = sitk.ImageRegistrationMethod()
    metric.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    metric.SetMetricFixedMask(fixed_mask_low)
    metric.SetMetricMovingMask(moving_mask_low)
    offsets = np.arange(-search_range_mm, search_range_mm + step_mm * 0.5, step_mm)
    base_translation = np.asarray(initial.GetTranslation(), dtype=float)
    # The third direction column is the DICOM slice-normal direction.
    direction = np.asarray(moving.GetDirection(), dtype=float).reshape(3, 3)
    slice_normal = direction[:, 2]
    records: list[dict[str, float]] = []
    best_metric = float("inf")
    best = sitk.Euler3DTransform(initial)
    for offset in offsets:
        candidate = sitk.Euler3DTransform(initial)
        candidate.SetTranslation(tuple(base_translation + offset * slice_normal))
        metric.SetInitialTransform(candidate)
        try:
            value = float(metric.MetricEvaluate(fixed_low, moving_low))
        except RuntimeError:
            value = float("inf")
        records.append({"offset_mm": float(offset), "metric": value})
        if np.isfinite(value) and value < best_metric:
            best_metric = value
            best = candidate
    if not np.isfinite(best_metric):
        raise RuntimeError("All coarse longitudinal registration candidates failed")
    return best, records


def configure_registration(
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    sampling: float,
    iterations: int,
    learning_rate: float,
) -> sitk.ImageRegistrationMethod:
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(sampling, seed=20260916)
    method.SetMetricFixedMask(fixed_mask)
    method.SetMetricMovingMask(moving_mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=learning_rate,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=12,
        lineSearchLowerLimit=0.0,
        lineSearchUpperLimit=2.0,
        lineSearchEpsilon=0.2,
        lineSearchMaximumIterations=15,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2, 1, 0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method


def register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    initial: sitk.Euler3DTransform,
    sampling: float,
    iterations: int,
) -> tuple[sitk.Euler3DTransform, dict[str, object]]:
    method = configure_registration(fixed_mask, moving_mask, sampling, iterations, 1.0)
    transform = sitk.Euler3DTransform(initial)
    method.SetInitialTransform(transform, inPlace=True)
    method.Execute(fixed, moving)
    return transform, {
        "metric": float(method.GetMetricValue()),
        "iterations": int(method.GetOptimizerIteration()),
        "stop_condition": method.GetOptimizerStopConditionDescription(),
    }


def register_affine(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    rigid: sitk.Euler3DTransform,
    sampling: float,
    iterations: int,
) -> tuple[sitk.AffineTransform, dict[str, object]]:
    transform = sitk.AffineTransform(3)
    transform.SetCenter(rigid.GetCenter())
    transform.SetMatrix(rigid.GetMatrix())
    transform.SetTranslation(rigid.GetTranslation())
    method = configure_registration(fixed_mask, moving_mask, sampling, iterations, 0.25)
    method.SetInitialTransform(transform, inPlace=True)
    method.Execute(fixed, moving)
    return transform, {
        "metric": float(method.GetMetricValue()),
        "iterations": int(method.GetOptimizerIteration()),
        "stop_condition": method.GetOptimizerStopConditionDescription(),
    }


def resample_to_fixed(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
    interpolator: int = sitk.sitkLinear,
    default_value: float = -1000.0,
) -> sitk.Image:
    return sitk.Resample(moving, fixed, transform, interpolator, default_value, moving.GetPixelID())


def centered_training_reference(
    image: sitk.Image, target_spacing_mm: float, size_multiple: int
) -> sitk.Image:
    """Create an isotropic, centered grid that covers the complete CBCT FOV."""
    old_size = np.asarray(image.GetSize(), dtype=float)
    old_spacing = np.asarray(image.GetSpacing(), dtype=float)
    target_spacing = np.full(3, float(target_spacing_mm))
    target_size = np.ceil(old_size * old_spacing / target_spacing).astype(int)
    target_size = ((target_size + size_multiple - 1) // size_multiple) * size_multiple
    old_center = np.asarray(
        image.TransformContinuousIndexToPhysicalPoint(tuple((old_size - 1.0) / 2.0))
    )
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    half_extent = direction @ ((target_size - 1.0) * target_spacing / 2.0)
    origin = old_center - half_extent
    reference = sitk.Image([int(value) for value in target_size], sitk.sitkFloat32)
    reference.SetSpacing(tuple(float(value) for value in target_spacing))
    reference.SetDirection(image.GetDirection())
    reference.SetOrigin(tuple(float(value) for value in origin))
    return reference


def dice(mask_a: sitk.Image, mask_b: sitk.Image) -> float:
    overlap = sitk.LabelOverlapMeasuresImageFilter()
    overlap.Execute(sitk.Cast(mask_a, sitk.sitkUInt8), sitk.Cast(mask_b, sitk.sitkUInt8))
    return float(overlap.GetDiceCoefficient())


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.clip(a[mask], -1000, 1500)
    y = np.clip(b[mask], -1000, 1500)
    if x.size > 1_000_000:
        indices = np.linspace(0, x.size - 1, 1_000_000, dtype=int)
        x, y = x[indices], y[indices]
    histogram, _, _ = np.histogram2d(x, y, bins=64, range=[[-1000, 1500], [-1000, 1500]])
    probability = histogram / max(histogram.sum(), 1.0)
    px = probability.sum(axis=1)
    py = probability.sum(axis=0)

    def entropy(values: np.ndarray) -> float:
        values = values[values > 0]
        return float(-(values * np.log(values)).sum())

    joint_entropy = entropy(probability.reshape(-1))
    return (entropy(px) + entropy(py)) / joint_entropy if joint_entropy > 0 else 0.0


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    x = np.clip(a[mask], -1000, 1500).astype(np.float64)
    y = np.clip(b[mask], -1000, 1500).astype(np.float64)
    if x.size == 0 or x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def transform_summary(transform: sitk.AffineTransform | sitk.Euler3DTransform) -> dict[str, object]:
    matrix = np.asarray(transform.GetMatrix()).reshape(3, 3)
    result: dict[str, object] = {
        "center_mm": list(transform.GetCenter()),
        "translation_mm": list(transform.GetTranslation()),
        "matrix": matrix.tolist(),
        "matrix_determinant": float(np.linalg.det(matrix)),
    }
    if isinstance(transform, sitk.Euler3DTransform):
        result["rotation_degrees_xyz"] = [
            float(np.rad2deg(transform.GetAngleX())),
            float(np.rad2deg(transform.GetAngleY())),
            float(np.rad2deg(transform.GetAngleZ())),
        ]
    return result


def _display_slice(array: np.ndarray, axis: int) -> np.ndarray:
    index = array.shape[axis] // 2
    image = np.take(array, index, axis=axis)
    if axis in (1, 2):
        image = np.flipud(image)
    return image


def save_qa_figure(fixed_hu: np.ndarray, registered_hu: np.ndarray, path: Path) -> None:
    names_and_axes: Iterable[tuple[str, int]] = (("Axial", 0), ("Coronal", 1), ("Sagittal", 2))
    figure, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for row, (name, axis) in enumerate(names_and_axes):
        fixed_slice = _display_slice(fixed_hu, axis)
        moving_slice = _display_slice(registered_hu, axis)
        fixed_norm = np.clip((fixed_slice + 1000.0) / 2000.0, 0.0, 1.0)
        moving_norm = np.clip((moving_slice + 1000.0) / 2000.0, 0.0, 1.0)
        overlay = np.stack((moving_norm, fixed_norm, moving_norm), axis=-1)
        axes[row, 0].imshow(fixed_slice, cmap="gray", vmin=-1000, vmax=1000)
        axes[row, 1].imshow(moving_slice, cmap="gray", vmin=-1000, vmax=1000)
        axes[row, 2].imshow(overlay)
        axes[row, 0].set_title(f"{name}: CBCT (fixed)")
        axes[row, 1].set_title(f"{name}: registered CT")
        axes[row, 2].set_title(f"{name}: CT magenta / CBCT green")
        for column in range(3):
            axes[row, column].axis("off")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/thorax"))
    parser.add_argument("--output", type=Path, default=Path("dataset/thorax/registration/current"))
    parser.add_argument("--ct-series-uid", default=None)
    parser.add_argument("--cbct-series-uid", default=None)
    parser.add_argument("--sampling", type=float, default=0.15)
    parser.add_argument("--rigid-iterations", type=int, default=180)
    parser.add_argument("--affine-iterations", type=int, default=140)
    parser.add_argument("--coarse-z-range-mm", type=float, default=240.0)
    parser.add_argument("--coarse-z-step-mm", type=float, default=30.0)
    parser.add_argument("--rigid-only", action="store_true")
    parser.add_argument(
        "--target-spacing-mm",
        type=float,
        default=2.0,
        help="Isotropic spacing of the saved training volumes",
    )
    parser.add_argument(
        "--size-multiple",
        type=int,
        default=4,
        help="Round each output dimension up to this model-compatible multiple",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.sampling <= 1:
        parser.error("--sampling must be in (0, 1]")
    if args.target_spacing_mm <= 0 or args.size_multiple <= 0:
        parser.error("--target-spacing-mm and --size-multiple must be positive")

    image_root = args.root / "image"
    all_series = discover_series(image_root)
    if not all_series:
        print(f"[SKIP] No CT DICOM slices found below {image_root}")
        return
    try:
        ct_series = select_series(all_series, "ct", args.ct_series_uid)
    except ValueError as error:
        print(f"[SKIP] Missing or empty CT data below {image_root}: {error}")
        return
    try:
        cbct_series = select_series(all_series, "cbct", args.cbct_series_uid)
    except ValueError as error:
        print(f"[SKIP] Missing or empty CBCT data below {image_root}: {error}")
        return
    if not ct_series.paths:
        print(f"[SKIP] Empty CT DICOM series below {image_root}")
        return
    if not cbct_series.paths:
        print(f"[SKIP] Empty CBCT DICOM series below {image_root}")
        return

    # Read both inputs successfully before an existing output may be replaced.
    ct_hu, ct_image = load_hu(ct_series.paths)
    cbct_hu, cbct_image = load_hu(cbct_series.paths)
    del ct_hu, cbct_hu

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")
        if args.output.resolve() in (args.root.resolve(), args.root.resolve().parent):
            raise ValueError(f"Unsafe output path: {args.output}")
        marker = args.output / ".thorax_registration_output"
        known_output = (args.output / "registration_metrics.json").exists() and (
            args.output / "registered_ct_hu.nii.gz"
        ).exists()
        if not marker.exists() and not known_output:
            raise ValueError(f"Refusing to overwrite an unrecognized directory: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    (args.output / ".thorax_registration_output").write_text(
        "Generated by tools.thorax_preprocessing.registration\n", encoding="utf-8"
    )

    fixed = preprocess_hu(cbct_image)
    moving = preprocess_hu(ct_image)
    fixed_mask = largest_body_mask(cbct_image)
    moving_mask = largest_body_mask(ct_image)

    center_initial = geometry_initializer(fixed, moving)
    coarse_initial, coarse_records = coarse_longitudinal_search(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        center_initial,
        args.coarse_z_range_mm,
        args.coarse_z_step_mm,
    )
    print(
        "Coarse z offset:",
        min(coarse_records, key=lambda item: item["metric"])["offset_mm"],
        "mm",
    )
    rigid, rigid_stats = register_rigid(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        coarse_initial,
        args.sampling,
        args.rigid_iterations,
    )
    print("Rigid:", rigid_stats["metric"], rigid_stats["stop_condition"])
    if args.rigid_only:
        final_transform: sitk.Transform = rigid
        affine_stats = None
    else:
        affine, affine_stats = register_affine(
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            rigid,
            args.sampling,
            args.affine_iterations,
        )
        final_transform = affine
        print("Affine:", affine_stats["metric"], affine_stats["stop_condition"])

    training_reference = centered_training_reference(
        cbct_image, args.target_spacing_mm, args.size_multiple
    )
    identity = sitk.Transform(3, sitk.sitkIdentity)
    standardized_cbct = resample_to_fixed(
        cbct_image, training_reference, identity, sitk.sitkLinear, -1000.0
    )
    fixed_mask_standard = resample_to_fixed(
        fixed_mask, training_reference, identity, sitk.sitkNearestNeighbor, 0.0
    )
    registered = resample_to_fixed(ct_image, training_reference, final_transform)
    registered_mask = resample_to_fixed(
        moving_mask, training_reference, final_transform, sitk.sitkNearestNeighbor, 0.0
    )
    initial_resampled = resample_to_fixed(ct_image, training_reference, center_initial)
    registered_hu = sitk.GetArrayFromImage(registered).astype(np.float32)
    initial_hu = sitk.GetArrayFromImage(initial_resampled).astype(np.float32)
    fixed_hu = sitk.GetArrayFromImage(standardized_cbct).astype(np.float32)
    fixed_mask_array = sitk.GetArrayFromImage(fixed_mask_standard).astype(bool)
    registered_mask_array = sitk.GetArrayFromImage(registered_mask).astype(bool)
    comparison_mask = fixed_mask_array & registered_mask_array
    metrics = {
        "ct_series_uid": ct_series.uid,
        "cbct_series_uid": cbct_series.uid,
        "ct_frame_of_reference_uid": ct_series.frame_of_reference_uid,
        "cbct_frame_of_reference_uid": cbct_series.frame_of_reference_uid,
        "coarse_search": coarse_records,
        "center_initial_transform": transform_summary(center_initial),
        "rigid_transform": transform_summary(rigid),
        "rigid_optimizer": rigid_stats,
        "affine_transform": None if args.rigid_only else transform_summary(final_transform),
        "affine_optimizer": affine_stats,
        "body_mask_dice": dice(fixed_mask_standard, registered_mask),
        "initial_nmi": normalized_mutual_information(fixed_hu, initial_hu, fixed_mask_array),
        "registered_nmi": normalized_mutual_information(fixed_hu, registered_hu, comparison_mask),
        "registered_correlation": correlation(fixed_hu, registered_hu, comparison_mask),
        "comparison_voxels": int(comparison_mask.sum()),
        "native_cbct_grid": {
            "size": list(cbct_image.GetSize()),
            "spacing": list(cbct_image.GetSpacing()),
            "origin": list(cbct_image.GetOrigin()),
            "direction": list(cbct_image.GetDirection()),
        },
        "training_grid": {
            "size": list(training_reference.GetSize()),
            "spacing": list(training_reference.GetSpacing()),
            "origin": list(training_reference.GetOrigin()),
            "direction": list(training_reference.GetDirection()),
            "size_multiple": args.size_multiple,
        },
    }
    nmi_improvement = metrics["registered_nmi"] - metrics["initial_nmi"]
    determinant = float(np.linalg.det(np.asarray(final_transform.GetMatrix()).reshape(3, 3)))
    quality_flags = []
    if metrics["body_mask_dice"] < 0.85:
        quality_flags.append("body_mask_dice_below_0.85")
    if nmi_improvement <= 0:
        quality_flags.append("mutual_information_did_not_improve")
    if metrics["registered_correlation"] < 0.5:
        quality_flags.append("correlation_below_0.5")
    if not 0.85 <= determinant <= 1.15:
        quality_flags.append("affine_determinant_outside_0.85_to_1.15")
    metrics["nmi_improvement"] = nmi_improvement
    metrics["quality_pass"] = not quality_flags
    metrics["quality_flags"] = quality_flags
    metrics["transform_semantics"] = {
        "resample_cbct_to_ct.tfm": "output/fixed CBCT points -> input/moving CT points; pass to sitk.Resample",
        "ct_to_cbct.tfm": "forward physical transform from moving CT points -> fixed CBCT points",
    }

    sitk.WriteImage(standardized_cbct, str(args.output / "fixed_cbct_hu.nii.gz"), useCompression=True)
    sitk.WriteImage(registered, str(args.output / "registered_ct_hu.nii.gz"), useCompression=True)
    registered_mu = image_with_array(hu_to_mu(registered_hu), registered)
    sitk.WriteImage(registered_mu, str(args.output / "registered_ct_mu.nii.gz"), useCompression=True)
    sitk.WriteTransform(final_transform, str(args.output / "resample_cbct_to_ct.tfm"))
    sitk.WriteTransform(final_transform.GetInverse(), str(args.output / "ct_to_cbct.tfm"))
    (args.output / "registration_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_qa_figure(fixed_hu, registered_hu, args.output / "registration_qa.png")
    print(json.dumps({key: value for key, value in metrics.items() if key in (
        "body_mask_dice", "initial_nmi", "registered_nmi", "registered_correlation", "quality_pass"
    )}, indent=2))
    print("Wrote registration outputs to", args.output)


if __name__ == "__main__":
    main()
