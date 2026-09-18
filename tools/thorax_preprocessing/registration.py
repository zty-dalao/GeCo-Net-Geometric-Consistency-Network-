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
import sys
import time
from typing import Callable, Iterable

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
from .registration_initialization import initialize_rigid_transform


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
    progress: Callable[[int, int, float, float], None] | None = None,
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
    total = len(offsets)
    for index, offset in enumerate(offsets, start=1):
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
        if progress is not None:
            progress(index, total, float(offset), value)
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


def attach_iteration_logger(
    method: sitk.ImageRegistrationMethod,
    label: str,
    log: Callable[[str], None] | None,
    report_every_seconds: float = 20.0,
) -> None:
    """Report optimizer progress so a long registration never looks frozen.

    ``GradientDescentLineSearch`` fires the iteration event once per line-search
    trial, so the optimizer's own counter is not a reliable progress numerator;
    report elapsed time and the current metric instead.
    """
    if log is None:
        return
    state = {"started": time.perf_counter(), "reported": time.perf_counter()}

    def on_iteration() -> None:
        now = time.perf_counter()
        if now - state["reported"] < report_every_seconds:
            return
        state["reported"] = now
        log(
            f"    {label} 进行中 {now - state['started']:.0f}s"
            f"  metric={method.GetMetricValue():.5f}"
        )

    method.AddCommand(sitk.sitkIterationEvent, on_iteration)


def register_rigid(
    fixed: sitk.Image,
    moving: sitk.Image,
    fixed_mask: sitk.Image,
    moving_mask: sitk.Image,
    initial: sitk.Euler3DTransform,
    sampling: float,
    iterations: int,
    log: Callable[[str], None] | None = None,
) -> tuple[sitk.Euler3DTransform, dict[str, object]]:
    method = configure_registration(fixed_mask, moving_mask, sampling, iterations, 1.0)
    transform = sitk.Euler3DTransform(initial)
    method.SetInitialTransform(transform, inPlace=True)
    attach_iteration_logger(method, "刚性", log)
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
    log: Callable[[str], None] | None = None,
) -> tuple[sitk.AffineTransform, dict[str, object]]:
    transform = sitk.AffineTransform(3)
    transform.SetCenter(rigid.GetCenter())
    transform.SetMatrix(rigid.GetMatrix())
    transform.SetTranslation(rigid.GetTranslation())
    method = configure_registration(fixed_mask, moving_mask, sampling, iterations, 0.25)
    method.SetInitialTransform(transform, inPlace=True)
    attach_iteration_logger(method, "仿射", log)
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


def discover_case_dirs(image_root: Path) -> list[Path]:
    """Return every patient folder below ``image`` that contains DICOM files."""
    if not image_root.is_dir():
        print(f"[SKIP] Image root does not exist: {image_root}", flush=True)
        return []
    cases: list[Path] = []
    for path in sorted(image_root.iterdir()):
        if not path.is_dir():
            continue
        if any(path.rglob("*.dcm")):
            cases.append(path)
        else:
            print(f"[SKIP] No DICOM below {path}", flush=True)
    return cases


def projection_case_names(root: Path) -> list[str]:
    """List ``projection/<case>`` folder names for the closing cross-check."""
    projection_root = root / "projection"
    if not projection_root.is_dir():
        return []
    return sorted(path.name for path in projection_root.iterdir() if path.is_dir())


def register_case(
    case_dir: Path,
    output_root: Path,
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> dict[str, object]:
    """Register one patient folder; report problems as a record instead of raising."""
    name = case_dir.name
    started = time.perf_counter()
    case_series = discover_series(case_dir)
    if not case_series:
        return {"case": name, "status": "skipped", "reason": f"{case_dir} 下没有 CT DICOM 文件"}
    try:
        ct_series = select_series(case_series, "ct", args.ct_series_uid)
    except ValueError as error:
        return {"case": name, "status": "skipped", "reason": f"缺少计划 CT 序列（{error}）"}
    try:
        cbct_series = select_series(case_series, "cbct", args.cbct_series_uid)
    except ValueError as error:
        return {"case": name, "status": "skipped", "reason": f"缺少 Varian CBCT 序列（{error}）"}
    if not ct_series.paths:
        return {"case": name, "status": "skipped", "reason": "计划 CT 序列为空"}
    if not cbct_series.paths:
        return {"case": name, "status": "skipped", "reason": "CBCT 序列为空"}

    output = output_root / name
    if output.resolve().parent != output_root.resolve():
        return {"case": name, "status": "failed", "reason": f"不安全的输出路径: {output}"}
    if output.exists():
        if not args.overwrite:
            return {
                "case": name,
                "status": "existing",
                "reason": f"输出已存在: {output}（加 --overwrite 可覆盖）",
            }
        marker = output / ".thorax_registration_output"
        known_output = (output / "registration_metrics.json").exists() and (
            output / "registered_ct_hu.nii.gz"
        ).exists()
        if not marker.exists() and not known_output:
            return {"case": name, "status": "failed", "reason": f"拒绝覆盖无法识别的目录: {output}"}
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / ".thorax_registration_output").write_text(
        "Generated by tools.thorax_preprocessing.registration\n", encoding="utf-8"
    )

    log(f"计划 CT   : {len(ct_series.paths)} 层 {ct_series.columns}x{ct_series.rows}")
    log(f"Varian CBCT: {len(cbct_series.paths)} 层 {cbct_series.columns}x{cbct_series.rows}")
    log("读取 DICOM 体数据 ...")
    ct_hu, ct_image = load_hu(ct_series.paths)
    cbct_hu, cbct_image = load_hu(cbct_series.paths)
    del ct_hu, cbct_hu

    log("提取身体掩膜 ...")
    fixed = preprocess_hu(cbct_image)
    moving = preprocess_hu(ct_image)
    fixed_mask = largest_body_mask(cbct_image)
    moving_mask = largest_body_mask(ct_image)

    initialization = initialize_rigid_transform(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        method=args.initializer,
    )
    center_initial = initialization.transform
    log(
        f"初始化 {initialization.method}: "
        f"CBCT中心={tuple(round(value, 2) for value in initialization.fixed_center_mm)} mm, "
        f"CT中心={tuple(round(value, 2) for value in initialization.moving_center_mm)} mm"
    )
    log(
        f"轴向粗搜索 z 偏移（±{args.coarse_z_range_mm:g} mm，步长 {args.coarse_z_step_mm:g} mm）..."
    )

    def coarse_progress(index: int, count: int, offset: float, value: float) -> None:
        if index % 5 == 0 or index == count:
            log(f"    粗搜索 {index}/{count}: z={offset:+.0f} mm metric={value:.5f}")

    coarse_initial, coarse_records = coarse_longitudinal_search(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        center_initial,
        args.coarse_z_range_mm,
        args.coarse_z_step_mm,
        progress=coarse_progress,
    )
    log(f"  选定 z 偏移 {min(coarse_records, key=lambda item: item['metric'])['offset_mm']:+.0f} mm")
    log(f"刚性配准（{args.rigid_iterations} 次迭代，采样率 {args.sampling:g}）...")
    rigid, rigid_stats = register_rigid(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        coarse_initial,
        args.sampling,
        args.rigid_iterations,
        log=log,
    )
    log(f"  刚性完成 metric={rigid_stats['metric']:.5f} 迭代={rigid_stats['iterations']}")
    if args.rigid_only:
        final_transform: sitk.Transform = rigid
        affine_stats = None
    else:
        log(f"仿射配准（{args.affine_iterations} 次迭代）...")
        affine, affine_stats = register_affine(
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            rigid,
            args.sampling,
            args.affine_iterations,
            log=log,
        )
        final_transform = affine
        log(f"  仿射完成 metric={affine_stats['metric']:.5f} 迭代={affine_stats['iterations']}")

    log(f"重采样到 {args.target_spacing_mm:g} mm 训练网格并写盘 ...")
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
        "initialization": {
            "method": initialization.method,
            "fixed_center_mm": list(initialization.fixed_center_mm),
            "moving_center_mm": list(initialization.moving_center_mm),
        },
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

    metrics["case"] = name
    sitk.WriteImage(standardized_cbct, str(output / "fixed_cbct_hu.nii.gz"), useCompression=True)
    sitk.WriteImage(registered, str(output / "registered_ct_hu.nii.gz"), useCompression=True)
    registered_mu = image_with_array(hu_to_mu(registered_hu), registered)
    sitk.WriteImage(registered_mu, str(output / "registered_ct_mu.nii.gz"), useCompression=True)
    sitk.WriteTransform(final_transform, str(output / "resample_cbct_to_ct.tfm"))
    sitk.WriteTransform(final_transform.GetInverse(), str(output / "ct_to_cbct.tfm"))
    (output / "registration_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_qa_figure(fixed_hu, registered_hu, output / "registration_qa.png")

    elapsed = time.perf_counter() - started
    log(
        f"完成 {elapsed:.1f}s  Dice={metrics['body_mask_dice']:.3f} "
        f"NMI {metrics['initial_nmi']:.3f}->{metrics['registered_nmi']:.3f} "
        f"相关={metrics['registered_correlation']:.3f} "
        f"质控={'通过' if metrics['quality_pass'] else '不通过(' + ','.join(quality_flags) + ')'}"
    )
    return {
        "case": name,
        "status": "ok",
        "output": str(output),
        "initializer": initialization.method,
        "quality_pass": bool(metrics["quality_pass"]),
        "quality_flags": quality_flags,
        "body_mask_dice": metrics["body_mask_dice"],
        "initial_nmi": metrics["initial_nmi"],
        "registered_nmi": metrics["registered_nmi"],
        "registered_correlation": metrics["registered_correlation"],
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        # Long batches must never die on an unencodable progress character.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/thorax"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/thorax/registration/current"),
        help="Parent folder holding one output sub-folder per patient",
    )
    parser.add_argument(
        "--ct-series-uid",
        default=None,
        help="Force a planning-CT series UID; only meaningful for a single patient folder",
    )
    parser.add_argument(
        "--cbct-series-uid",
        default=None,
        help="Force a CBCT series UID; only meaningful for a single patient folder",
    )
    parser.add_argument("--sampling", type=float, default=0.15)
    parser.add_argument("--rigid-iterations", type=int, default=180)
    parser.add_argument("--affine-iterations", type=int, default=140)
    parser.add_argument("--coarse-z-range-mm", type=float, default=240.0)
    parser.add_argument("--coarse-z-step-mm", type=float, default=30.0)
    parser.add_argument(
        "--initializer",
        choices=("geometry", "moments"),
        default="geometry",
        help=(
            "Initial rigid alignment: geometry uses image-grid centers; "
            "moments uses CT/CBCT body-mask centers of mass"
        ),
    )
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
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N patient folders (0 = all); handy for a smoke test",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.sampling <= 1:
        parser.error("--sampling must be in (0, 1]")
    if args.target_spacing_mm <= 0 or args.size_multiple <= 0:
        parser.error("--target-spacing-mm and --size-multiple must be positive")
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    image_root = args.root / "image"
    all_case_dirs = discover_case_dirs(image_root)
    if not all_case_dirs:
        print(f"[SKIP] No patient folders with DICOM found below {image_root}", flush=True)
        return
    discovered = len(all_case_dirs)
    case_dirs = all_case_dirs[: args.limit] if args.limit else all_case_dirs
    if args.output.resolve() in (args.root.resolve(), image_root.resolve()):
        parser.error(f"--output must not be the dataset root or its image folder: {args.output}")
    if args.ct_series_uid or args.cbct_series_uid:
        print(
            "警告: 指定了 --ct-series-uid/--cbct-series-uid，会对每个病例套用同一 UID，"
            "只有单病例目录下才有意义",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    total = len(case_dirs)
    if total < discovered:
        print(
            f"在 {image_root} 下发现 {discovered} 个病人文件夹，"
            f"--limit {args.limit} 只处理前 {total} 个",
            flush=True,
        )
    else:
        print(f"在 {image_root} 下发现 {total} 个病人文件夹", flush=True)
    print(f"输出目录 {args.output}", flush=True)
    batch_started = time.perf_counter()
    results: list[dict[str, object]] = []
    for index, case_dir in enumerate(case_dirs, start=1):
        print(f"\n[{index}/{total}] {case_dir.name}", flush=True)

        def log(message: str) -> None:
            print(f"    {message}", flush=True)

        try:
            result = register_case(case_dir, args.output, args, log)
        except Exception as error:  # noqa: BLE001 - 批处理必须继续处理后续病人
            result = {
                "case": case_dir.name,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
            print(f"    [FAIL] {result['reason']}", flush=True)
        if result["status"] == "skipped":
            print(f"    [SKIP] {result['reason']}", flush=True)
        elif result["status"] == "existing":
            print(f"    [SKIP] {result['reason']}", flush=True)
        results.append(result)

    succeeded = [item for item in results if item["status"] == "ok"]
    skipped = [item for item in results if item["status"] == "skipped"]
    existing = [item for item in results if item["status"] == "existing"]
    failed = [item for item in results if item["status"] == "failed"]
    quality_failed = [item for item in succeeded if not item["quality_pass"]]
    image_names = {path.name for path in all_case_dirs}
    projection_names = projection_case_names(args.root)
    missing_image = [name for name in projection_names if name not in image_names]
    missing_projection = [name for name in sorted(image_names) if name not in set(projection_names)]
    elapsed_minutes = (time.perf_counter() - batch_started) / 60.0

    summary = {
        "image_root": str(image_root),
        "output_root": str(args.output),
        "initializer": args.initializer,
        "discovered_cases": discovered,
        "total_cases": total,
        "succeeded": len(succeeded),
        "skipped_missing_data": len(skipped),
        "skipped_existing_output": len(existing),
        "failed": len(failed),
        "quality_failed": [item["case"] for item in quality_failed],
        "skipped_detail": {item["case"]: item["reason"] for item in skipped},
        "failed_detail": {item["case"]: item["reason"] for item in failed},
        "projection_without_image_case": missing_image,
        "image_case_without_projection": missing_projection,
        "elapsed_minutes": round(elapsed_minutes, 2),
        "cases": results,
    }
    (args.output / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 72, flush=True)
    print(
        f"批处理结束：共 {total} 例，成功 {len(succeeded)}，"
        f"缺数据跳过 {len(skipped)}，已有输出跳过 {len(existing)}，失败 {len(failed)}；"
        f"总用时 {elapsed_minutes:.1f} 分钟",
        flush=True,
    )
    if skipped:
        print("\n缺失数据（缺 CT / 缺 CBCT / 无 DICOM）的文件夹：", flush=True)
        for item in skipped:
            print(f"  - {item['case']}: {item['reason']}", flush=True)
    if failed:
        print("\n处理失败的文件夹：", flush=True)
        for item in failed:
            print(f"  - {item['case']}: {item['reason']}", flush=True)
    if existing:
        print("\n已有输出被跳过的文件夹（需 --overwrite 才会重跑）：", flush=True)
        for item in existing:
            print(f"  - {item['case']}", flush=True)
    if quality_failed:
        print("\n配准质控不通过（建议人工复核）：", flush=True)
        for item in quality_failed:
            print(
                f"  - {item['case']}: {','.join(item['quality_flags'])}"
                f" (Dice={item['body_mask_dice']:.3f}, 相关={item['registered_correlation']:.3f})",
                flush=True,
            )
    if missing_image:
        print("\nprojection 有目录但 image 缺文件夹：", flush=True)
        for name in missing_image:
            print(f"  - {name}", flush=True)
    if not missing_image:
        print("\nprojection 与 image 的病人文件夹一一对应，无缺失。", flush=True)
    if missing_projection:
        print("\nimage 有文件夹但 projection 缺失（无法生成投影，后续不能训练）：", flush=True)
        for name in missing_projection:
            print(f"  - {name}", flush=True)
    print(f"\n完整汇总 JSON: {args.output / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
