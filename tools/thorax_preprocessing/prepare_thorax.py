"""Convert mixed DICOM and Varian XIM thorax data into the training layout.

Batch entry point: every patient folder below ``<root>/image`` is converted
independently, using that patient's own CT/CBCT series and, for
``--gt-source registered-ct``, that patient's own registered CT.  A patient that
lacks data is recorded in ``batch_summary.json`` and the batch continues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Callable

import numpy as np
import SimpleITK as sitk

from .dicom_io import (
    crop_paths,
    discover_series,
    hu_to_mu,
    image_with_array,
    load_hu,
    select_series,
)
from .projection_io import (
    convert_projections,
    find_acquisition,
    find_air_frames,
    read_scan_geometry,
    write_transforms,
)


def parse_slice_range(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        start, end = (int(part) for part in value.split(":"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("slice range must have form START:END") from error
    return start, end


def centered_geometry(image: sitk.Image) -> dict[str, list[float]]:
    resolution = np.asarray(image.GetSize(), dtype=int)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    physical = resolution * spacing
    origin = -physical / 2.0
    return {
        "volume_resolution": resolution.tolist(),
        "volume_spacing": spacing.tolist(),
        "volume_origin": origin.tolist(),
        "volume_phy": physical.tolist(),
    }


def make_starter_split(case_names: list[str]) -> dict[str, list[str]]:
    """Make every stage non-empty because ``train.py`` constructs all four loaders.

    With fewer than three cases, overlap is unavoidable and the result is useful
    only for an end-to-end/overfitting check.  The README calls this out clearly.
    """
    if not case_names:
        raise ValueError("Cannot create a split without cases")
    if len(case_names) == 1:
        train = val = test = case_names[:]
    elif len(case_names) == 2:
        train, val, test = case_names[:1], case_names[1:], case_names[1:]
    else:
        n_test = max(1, round(len(case_names) * 0.1))
        n_val = max(1, round(len(case_names) * 0.1))
        test = case_names[-n_test:]
        val = case_names[-(n_test + n_val) : -n_test]
        train = case_names[: -(n_test + n_val)]
    return {"train": train, "val": val, "test": test, "visual": test[:1]}


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


def projection_folder_names(root: Path) -> list[str]:
    """List ``projection/<case>`` folder names for the closing cross-check."""
    projection_root = root / "projection"
    if not projection_root.is_dir():
        return []
    return sorted(path.name for path in projection_root.iterdir() if path.is_dir())


def resolve_registered_ct(args: argparse.Namespace, name: str) -> Path | None:
    """Resolve the registered CT of one patient.

    ``--registered-ct-root`` is the batch form (one sub-folder per patient, as
    written by ``registration.py``); ``--registered-ct-mu`` remains available for
    a single explicit file.
    """
    if args.registered_ct_root is not None:
        return args.registered_ct_root / name / "registered_ct_mu.nii.gz"
    return args.registered_ct_mu


def convert_case(
    case_dir: Path,
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> dict[str, object]:
    """Convert one patient folder; report problems as a record instead of raising.

    Every path is resolved from this patient's own folders, so a batch run can
    never mix one patient's projections with another patient's CT/CBCT.
    """
    name = case_dir.name
    started = time.perf_counter()

    projection_case = args.root / "projection" / name
    if not projection_case.is_dir():
        return {"case": name, "status": "skipped", "reason": f"缺少投影文件夹: {projection_case}"}
    if not (projection_case / "Scan.xml").is_file():
        return {
            "case": name,
            "status": "skipped",
            "reason": f"投影文件夹缺少 Scan.xml: {projection_case}",
        }
    try:
        acquisition = find_acquisition(projection_case)
    except ValueError as error:
        return {"case": name, "status": "skipped", "reason": str(error)}
    if not any(acquisition.glob("Proj_*.xim")):
        return {"case": name, "status": "skipped", "reason": "投影文件夹没有 Proj_*.xim 帧"}

    air_frames = find_air_frames(projection_case)
    if args.projection_mode == "log" and not air_frames:
        return {
            "case": name,
            "status": "skipped",
            "reason": (
                "缺少 Calibrations 空气/蝴蝶结校准帧，无法做空气校正"
                "（--projection-mode raw 可绕过，但投影分布与其余病例不一致）"
            ),
        }

    destination = args.output / name
    if destination.resolve().parent != args.output.resolve():
        return {"case": name, "status": "failed", "reason": f"不安全的输出路径: {destination}"}
    if destination.exists():
        if not args.overwrite:
            return {
                "case": name,
                "status": "existing",
                "reason": f"输出已存在: {destination}（加 --overwrite 可覆盖）",
            }
        marker = destination / ".thorax_prepare_output"
        known_output = (destination / "transforms.json").exists() and (
            destination / "gt_volume.nii.gz"
        ).exists()
        if not marker.exists() and not known_output:
            return {
                "case": name,
                "status": "failed",
                "reason": f"拒绝覆盖无法识别的目录: {destination}",
            }
        shutil.rmtree(destination)

    case_series = discover_series(case_dir)
    if not case_series:
        return {"case": name, "status": "skipped", "reason": f"{case_dir} 下没有 CT DICOM 文件"}
    try:
        cbct_series = select_series(case_series, "cbct", args.cbct_series_uid)
    except ValueError as error:
        return {"case": name, "status": "skipped", "reason": f"缺少 Varian CBCT 序列（{error}）"}
    try:
        ct_series = select_series(case_series, "ct", args.ct_series_uid)
    except ValueError as error:
        return {"case": name, "status": "skipped", "reason": f"缺少计划 CT 序列（{error}）"}
    if not cbct_series.paths or not ct_series.paths:
        return {"case": name, "status": "skipped", "reason": "CT 或 CBCT 序列为空"}

    registered_ct_path: Path | None = None
    if args.gt_source == "registered-ct":
        registered_ct_path = resolve_registered_ct(args, name)
        if registered_ct_path is None:
            return {
                "case": name,
                "status": "skipped",
                "reason": "缺少 --registered-ct-root 或 --registered-ct-mu",
            }
        if not registered_ct_path.is_file():
            return {
                "case": name,
                "status": "skipped",
                "reason": f"配准 CT 不存在（请先跑 registration.py）: {registered_ct_path}",
            }

    log(
        f"计划 CT {len(ct_series.paths)} 层 {ct_series.columns}x{ct_series.rows}，"
        f"CBCT {len(cbct_series.paths)} 层 {cbct_series.columns}x{cbct_series.rows}"
    )
    log("读取 DICOM 体数据 ...")
    cbct_hu, cbct_native_reference = load_hu(cbct_series.paths)
    cbct_reference = cbct_native_reference
    cbct_depth_mm = cbct_reference.GetSize()[2] * cbct_reference.GetSpacing()[2]
    ct_paths = crop_paths(
        ct_series,
        parse_slice_range(args.ct_slices),
        cbct_depth_mm
        if args.ct_crop == "match-cbct-center" and args.ct_slices is None
        else None,
    )
    ct_hu, ct_reference = load_hu(ct_paths)
    ct_mu = hu_to_mu(ct_hu)

    registered_ct_image: sitk.Image | None = None
    if registered_ct_path is not None:
        registered_ct_image = sitk.ReadImage(str(registered_ct_path), sitk.sitkFloat32)
        if not np.allclose(
            registered_ct_image.GetDirection(),
            cbct_native_reference.GetDirection(),
            atol=1e-6,
        ):
            return {"case": name, "status": "failed", "reason": "配准 CT 方向与 CBCT 方向不一致"}
        native_center = np.asarray(
            cbct_native_reference.TransformContinuousIndexToPhysicalPoint(
                tuple((np.asarray(cbct_native_reference.GetSize()) - 1.0) / 2.0)
            )
        )
        registered_center = np.asarray(
            registered_ct_image.TransformContinuousIndexToPhysicalPoint(
                tuple((np.asarray(registered_ct_image.GetSize()) - 1.0) / 2.0)
            )
        )
        if not np.allclose(native_center, registered_center, atol=1e-3):
            return {
                "case": name,
                "status": "failed",
                "reason": "配准 CT 与 CBCT 网格不共享同一物理中心",
            }
        cbct_reference = sitk.Resample(
            cbct_native_reference,
            registered_ct_image,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkLinear,
            -1000.0,
            sitk.sitkFloat32,
        )
        cbct_hu = sitk.GetArrayFromImage(cbct_reference)
    cbct_mu = hu_to_mu(cbct_hu)

    destination.mkdir(parents=True)
    (destination / ".thorax_prepare_output").write_text(
        "Generated by tools.thorax_preprocessing.prepare_thorax\n", encoding="utf-8"
    )
    # Volume writes happen before projection conversion, so a failure in between
    # would otherwise leave a case folder with volumes but no proj.nii.gz.
    try:
        cbct_image = image_with_array(cbct_mu, cbct_reference)
        ct_image = image_with_array(ct_mu, ct_reference)
        sitk.WriteImage(cbct_image, str(destination / "cbct_volume.nii.gz"), useCompression=True)
        sitk.WriteImage(ct_image, str(destination / "ct_volume.nii.gz"), useCompression=True)
        if args.gt_source == "cbct":
            gt_image = cbct_image
        elif args.gt_source == "ct":
            gt_image = ct_image
        else:
            assert registered_ct_image is not None
            gt_image = registered_ct_image
        sitk.WriteImage(gt_image, str(destination / "gt_volume.nii.gz"), useCompression=True)

        geometry = read_scan_geometry(projection_case / "Scan.xml")
        projections, frames, projection_spacing = convert_projections(
            projection_case,
            destination / "proj.nii.gz",
            geometry,
            bin_factor=args.projection_bin,
            mode=args.projection_mode,
            max_line_integral=args.max_line_integral,
            detector_offset_u=args.detector_offset_u_mm,
            detector_offset_v=args.detector_offset_v_mm,
            output_views=args.output_views,
            output_resolution=(args.projection_resolution, args.projection_resolution)
            if args.projection_resolution > 0
            else None,
        )
        params: dict[str, object] = {
            "obj_index": name,
            "start": 0.0,
            "end": 360.0,
            "angle_per_view": 360.0 / len(frames),
            "N_views": len(frames),
            "sad": geometry.sad,
            "sid": geometry.sid,
            "proj_resolution": [int(projections.shape[2]), int(projections.shape[1])],
            "proj_spacing": list(projection_spacing),
            "proj_phy": [
                projections.shape[2] * projection_spacing[0],
                projections.shape[1] * projection_spacing[1],
            ],
            "detector_offset_mm": [
                geometry.imager_lateral
                if args.detector_offset_u_mm is None
                else args.detector_offset_u_mm,
                geometry.imager_longitudinal
                if args.detector_offset_v_mm is None
                else args.detector_offset_v_mm,
            ],
            "gt_source": args.gt_source,
            "ct_slice_range": [
                ct_series.paths.index(ct_paths[0]),
                ct_series.paths.index(ct_paths[-1]) + 1,
            ],
            "ct_frame_of_reference_uid": ct_series.frame_of_reference_uid,
            "cbct_frame_of_reference_uid": cbct_series.frame_of_reference_uid,
            "frames": frames,
        }
        params.update(centered_geometry(gt_image))
        write_transforms(destination / "transforms.json", params)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    elapsed = time.perf_counter() - started
    gt_size = [int(value) for value in gt_image.GetSize()]
    gt_spacing = [float(value) for value in gt_image.GetSpacing()]
    log(
        f"完成 {elapsed:.1f}s  GT {gt_size} @ {gt_spacing} mm  "
        f"投影 {int(projections.shape[2])}x{int(projections.shape[1])} @ "
        f"{float(projection_spacing[0]):.3f}x{float(projection_spacing[1]):.3f} mm  "
        f"{len(frames)} 帧"
    )
    return {
        "case": name,
        "status": "ok",
        "output": str(destination),
        "gt_source": args.gt_source,
        "gt_size": gt_size,
        "gt_spacing": gt_spacing,
        "gt_divisible_by_4": all(value % 4 == 0 for value in gt_size),
        "projection_size": [int(projections.shape[2]), int(projections.shape[1])],
        "projection_spacing": [float(value) for value in projection_spacing],
        "views": len(frames),
        "air_frames": len(air_frames),
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
        default=Path("dataset/thorax/syn_data"),
        help="Parent folder holding one output sub-folder per patient",
    )
    parser.add_argument("--ct-series-uid", default=None)
    parser.add_argument("--cbct-series-uid", default=None)
    parser.add_argument("--ct-slices", default=None, help="Explicit sorted CT slice range START:END")
    parser.add_argument(
        "--ct-crop",
        choices=("match-cbct-center", "none"),
        default="match-cbct-center",
        help="Default CT crop matches the CBCT physical z coverage and is centered",
    )
    parser.add_argument("--gt-source", choices=("cbct", "ct", "registered-ct"), default="cbct")
    parser.add_argument(
        "--registered-ct-root",
        type=Path,
        default=None,
        help=(
            "Batch form of --gt-source registered-ct: parent folder holding one "
            "<patient>/registered_ct_mu.nii.gz per patient, as written by registration.py"
        ),
    )
    parser.add_argument(
        "--registered-ct-mu",
        type=Path,
        default=None,
        help="Single CT already registered to the CBCT grid and converted to mu",
    )
    parser.add_argument("--projection-bin", type=int, default=1)
    parser.add_argument(
        "--projection-resolution",
        type=int,
        default=256,
        help="Square detector resolution after physical-space interpolation; 0 keeps binned shape",
    )
    parser.add_argument("--projection-mode", choices=("log", "raw"), default="log")
    parser.add_argument("--output-views", type=int, default=360)
    parser.add_argument("--max-line-integral", type=float, default=20.0)
    parser.add_argument("--detector-offset-u-mm", type=float, default=None)
    parser.add_argument("--detector-offset-v-mm", type=float, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N patient folders (0 = all); handy for a smoke test",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.projection_bin < 1:
        parser.error("--projection-bin must be >= 1")
    if args.projection_resolution < 0:
        parser.error("--projection-resolution must be >= 0")
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.gt_source == "registered-ct":
        if resolve_registered_ct(args, "placeholder") is None:
            parser.error(
                "--gt-source registered-ct requires --registered-ct-root or --registered-ct-mu"
            )
    elif args.registered_ct_root is not None or args.registered_ct_mu is not None:
        print(
            "[WARN] --registered-ct-root/--registered-ct-mu are ignored unless "
            "--gt-source registered-ct",
            flush=True,
        )

    image_root = args.root / "image"
    all_case_dirs = discover_case_dirs(image_root)
    if not all_case_dirs:
        print(f"[SKIP] No patient folders with DICOM found below {image_root}", flush=True)
        return
    discovered = len(all_case_dirs)
    case_dirs = all_case_dirs[: args.limit] if args.limit else all_case_dirs

    args.output.mkdir(parents=True, exist_ok=True)
    batch_started = time.perf_counter()
    results: list[dict[str, object]] = []
    for index, case_dir in enumerate(case_dirs, 1):
        name = case_dir.name
        print(f"\n[{index}/{len(case_dirs)}] {name}", flush=True)

        def log(message: str, _name: str = name) -> None:
            print(f"    {_name}: {message}", flush=True)

        try:
            result = convert_case(case_dir, args, log)
        except Exception as error:  # noqa: BLE001 - isolate one patient from the batch
            result = {
                "case": name,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
        if result["status"] == "ok":
            print(
                f"    [OK] GT {result['gt_size']} @ {result['gt_spacing']} mm，"
                f"投影 {result['projection_size']}，{result['views']} 帧，{result['elapsed_s']}s",
                flush=True,
            )
        elif result["status"] == "failed":
            print(f"    [FAIL] {result['reason']}", flush=True)
        else:
            print(f"    [SKIP] {result['reason']}", flush=True)
        results.append(result)

    succeeded = [item for item in results if item["status"] == "ok"]
    skipped = [item for item in results if item["status"] == "skipped"]
    existing = [item for item in results if item["status"] == "existing"]
    failed = [item for item in results if item["status"] == "failed"]
    image_names = {path.name for path in all_case_dirs}
    projection_names = set(projection_folder_names(args.root))
    missing_image = [name for name in sorted(projection_names) if name not in image_names]
    missing_projection = [name for name in sorted(image_names) if name not in projection_names]
    not_divisible = [item["case"] for item in succeeded if not item["gt_divisible_by_4"]]
    elapsed_minutes = (time.perf_counter() - batch_started) / 60.0

    summary = {
        "image_root": str(image_root),
        "output_root": str(args.output),
        "gt_source": args.gt_source,
        "registered_ct_root": None
        if args.registered_ct_root is None
        else str(args.registered_ct_root),
        "projection_bin": args.projection_bin,
        "projection_resolution": args.projection_resolution,
        "projection_mode": args.projection_mode,
        "output_views": args.output_views,
        "discovered_cases": discovered,
        "total_cases": len(case_dirs),
        "succeeded": len(succeeded),
        "skipped_missing_data": len(skipped),
        "skipped_existing_output": len(existing),
        "failed": len(failed),
        "gt_not_divisible_by_4": not_divisible,
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
        f"批处理结束：共 {len(case_dirs)} 例，成功 {len(succeeded)}，"
        f"缺数据跳过 {len(skipped)}，已有输出跳过 {len(existing)}，失败 {len(failed)}；"
        f"总用时 {elapsed_minutes:.1f} 分钟",
        flush=True,
    )
    if skipped:
        print("\n缺数据的文件夹（未生成训练数据）：", flush=True)
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
    if not_divisible:
        print(
            "\nGT 体积不能被 4 整除（训练前请改用 --gt-source registered-ct）：",
            flush=True,
        )
        for name in not_divisible:
            print(f"  - {name}", flush=True)
    if missing_image:
        print("\nprojection 有目录但 image 缺文件夹：", flush=True)
        for name in missing_image:
            print(f"  - {name}", flush=True)
    if missing_projection:
        print("\nimage 有文件夹但 projection 缺失（无法生成投影）：", flush=True)
        for name in missing_projection:
            print(f"  - {name}", flush=True)
    if not missing_image and not missing_projection:
        print("\nprojection 与 image 的病人文件夹一一对应，无缺失。", flush=True)

    if succeeded:
        converted_names = [item["case"] for item in succeeded]
        split = make_starter_split(converted_names)
        split_path = args.output / "thorax_split.json"
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        print(f"\n已写 starter split 草稿: {split_path}", flush=True)
        print(
            "注意：这是按本次转换成功病例重新生成的草稿。若 data/dataset_split/thorax_split.json "
            "已有经过确认的划分，请勿直接覆盖；只需把本次被跳过的病例从中移除。",
            flush=True,
        )
    print(f"\n完整汇总 JSON: {args.output / 'batch_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
