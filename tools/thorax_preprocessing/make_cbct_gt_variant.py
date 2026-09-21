"""Build a CBCT-as-GT variant of a prepared thorax dataset.

The projections in ``syn_data/<case>/proj.nii.gz`` were reconstructed from that
patient's Varian CBCT acquisition, so the CBCT itself is the ground truth that is
*natively paired* with the projections.  The default ``gt_volume.nii.gz`` written
by ``prepare_thorax.py --gt-source registered-ct`` instead uses the *planning
pCT*, which comes from a different scan and therefore carries registration error.

This tool produces a side-by-side variant that keeps the projections and the
volume geometry untouched and swaps only the label volume:

```text
<output>/<case>/gt_volume.nii.gz = hu_to_mu(registration/<case>/fixed_cbct_hu.nii.gz)
```

``fixed_cbct_hu.nii.gz`` is the CBCT resampled by ``registration.py`` onto the same
isotropic training grid as ``registered_ct_mu.nii.gz`` (e.g. 248x248x120 @ 2 mm),
so ``transforms.json`` needs no geometry change at all -- only its ``gt_source``
field is rewritten to ``cbct-fixed``.  Every case is verified against the grid
declared in ``transforms.json`` before it is written, so a label volume can never
land on a grid the projections were not generated for.

The output tree links the large shared files instead of copying them, so a 155-case
variant costs roughly the size of the new label volumes only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Callable

import numpy as np
import SimpleITK as sitk

from .dicom_io import hu_to_mu, image_with_array

GT_SOURCE = "cbct-fixed"
SHARED_FILES = ("proj.nii.gz", "cbct_volume.nii.gz", "ct_volume.nii.gz")
MARKER = ".thorax_cbct_gt_variant"


def link_shared(source: Path, destination: Path, mode: str) -> None:
    """Materialise a shared file, preferring links so the variant stays small."""
    if mode == "symlink":
        # A relative target keeps the output tree movable.
        destination.symlink_to(os.path.relpath(source, destination.parent))
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def grid_mismatch(
    label: sitk.Image,
    reference: sitk.Image | None,
    params: dict,
    reference_name: str,
) -> str | None:
    """Check the label sits on the voxel grid the projections were built for.

    ``transforms.json`` declares a *centered* volume grid recomputed from size and
    spacing (``-size*spacing/2``), which is the ray-tracing convention used by
    ``angle_to_vec`` -- it is deliberately not the origin stored inside the NIfTI
    files.  Only size and spacing feed that declaration, so those are compared
    against ``transforms.json``, while the full grid (including the stored origin)
    is compared against the volume ``prepare_thorax.py`` used as GT.
    """
    for name, actual, declared in (
        ("volume_resolution", [int(v) for v in label.GetSize()],
         [int(v) for v in params.get("volume_resolution", [])]),
        ("volume_spacing", [float(v) for v in label.GetSpacing()],
         [float(v) for v in params.get("volume_spacing", [])]),
    ):
        if len(declared) != len(actual) or not np.allclose(actual, declared, atol=1e-3):
            return f"{name} 与 transforms.json 不一致: fixed_cbct_hu={actual} vs {declared}"
    if reference is not None:
        if [int(v) for v in label.GetSize()] != [int(v) for v in reference.GetSize()]:
            return (f"尺寸与 {reference_name} 不一致: "
                    f"{list(label.GetSize())} vs {list(reference.GetSize())}")
        if not np.allclose(label.GetSpacing(), reference.GetSpacing(), atol=1e-4):
            return (f"spacing 与 {reference_name} 不一致: "
                    f"{list(label.GetSpacing())} vs {list(reference.GetSpacing())}")
        if not np.allclose(label.GetOrigin(), reference.GetOrigin(), atol=1e-3):
            return (f"origin 与 {reference_name} 不一致: "
                    f"{list(label.GetOrigin())} vs {list(reference.GetOrigin())}")
    return None


def case_names(args: argparse.Namespace) -> list[str]:
    if args.cases:
        return sorted(args.cases)
    with Path(args.split_file).open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    names: list[str] = []
    for stage in ("train", "val", "test", "visual"):
        for name in split.get(stage, []):
            if name not in names:
                names.append(name)
    return names


def convert_case(
    name: str,
    args: argparse.Namespace,
    log: Callable[[str], None],
) -> dict[str, object]:
    """Build one CBCT-as-GT case; report problems as a record instead of raising."""
    started = time.perf_counter()
    source_case = args.source / name
    registration_case = args.registration_root / name
    destination = args.output / name

    if not (source_case / "proj.nii.gz").is_file():
        return {"case": name, "status": "skipped",
                "reason": f"源数据缺少 proj.nii.gz: {source_case}"}
    transforms_path = source_case / "transforms.json"
    if not transforms_path.is_file():
        return {"case": name, "status": "skipped",
                "reason": f"源数据缺少 transforms.json: {source_case}"}
    cbct_hu_path = registration_case / "fixed_cbct_hu.nii.gz"
    if not cbct_hu_path.is_file():
        return {"case": name, "status": "skipped",
                "reason": f"缺少配准输出的 fixed_cbct_hu.nii.gz: {cbct_hu_path}"}

    if destination.resolve().parent != args.output.resolve():
        return {"case": name, "status": "failed", "reason": f"不安全的输出路径: {destination}"}
    if destination.exists():
        if not args.overwrite:
            return {"case": name, "status": "existing",
                    "reason": f"输出已存在: {destination}（加 --overwrite 可覆盖）"}
        marker = destination / MARKER
        known = (destination / "gt_volume.nii.gz").exists() and (
            destination / "transforms.json"
        ).exists()
        if not marker.exists() and not known:
            return {"case": name, "status": "failed",
                    "reason": f"拒绝覆盖无法识别的目录: {destination}"}
        shutil.rmtree(destination)

    with transforms_path.open("r", encoding="utf-8") as handle:
        params = json.load(handle)
    label_hu = sitk.ReadImage(str(cbct_hu_path), sitk.sitkFloat32)
    reference_path = registration_case / "registered_ct_mu.nii.gz"
    reference = (
        sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
        if reference_path.is_file()
        else None
    )
    mismatch = grid_mismatch(label_hu, reference, params, reference_path.name)
    if mismatch:
        return {"case": name, "status": "failed", "reason": f"标签网格校验失败: {mismatch}"}

    destination.mkdir(parents=True)
    (destination / MARKER).write_text(
        "Generated by tools.thorax_preprocessing.make_cbct_gt_variant\n"
        f"gt_source={GT_SOURCE}\n"
        f"label={cbct_hu_path}\n",
        encoding="utf-8",
    )
    for filename in SHARED_FILES:
        shared = source_case / filename
        if shared.is_file():
            link_shared(shared, destination / filename, args.link_mode)

    label_mu = image_with_array(hu_to_mu(sitk.GetArrayFromImage(label_hu)), label_hu)
    sitk.WriteImage(label_mu, str(destination / "gt_volume.nii.gz"), useCompression=True)

    params["gt_source"] = GT_SOURCE
    params["gt_source_detail"] = (
        "CBCT resampled onto the training grid by registration.py (fixed image); "
        "natively paired with proj.nii.gz"
    )
    params["registered_pct_reference"] = str(
        (registration_case / "registered_ct_mu.nii.gz").resolve()
    )
    (destination / "transforms.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    mu = sitk.GetArrayFromImage(label_mu)
    elapsed = time.perf_counter() - started
    log(f"完成 {elapsed:.1f}s  GT {[int(v) for v in label_mu.GetSize()]} @ "
        f"{[round(float(v), 3) for v in label_mu.GetSpacing()]} mm  "
        f"μ[{mu.min():.5f}, {mu.max():.5f}]")
    return {
        "case": name,
        "status": "ok",
        "output": str(destination),
        "gt_source": GT_SOURCE,
        "label_source": str(cbct_hu_path),
        "gt_size": [int(v) for v in label_mu.GetSize()],
        "gt_spacing": [round(float(v), 4) for v in label_mu.GetSpacing()],
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "mu_above_thorax_clamp": int((mu > 0.09).sum()),
        "gt_divisible_by_4": all(int(v) % 4 == 0 for v in label_mu.GetSize()),
        "link_mode": args.link_mode,
        "elapsed_s": round(elapsed, 2),
    }


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("dataset/thorax/syn_data"),
                        help="Prepared dataset whose projections are reused")
    parser.add_argument("--registration-root", type=Path,
                        default=Path("dataset/thorax/registration/current"),
                        help="registration.py output holding <case>/fixed_cbct_hu.nii.gz")
    parser.add_argument("--output", type=Path,
                        default=Path("dataset/thorax/syn_data_cbct_gt"),
                        help="Parent folder holding one output sub-folder per patient")
    parser.add_argument("--split-file", type=Path,
                        default=Path("data/dataset_split/thorax_split.json"))
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Explicit case list; defaults to every case in --split-file")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"),
                        default="symlink",
                        help="How to materialise proj.nii.gz and the auxiliary volumes")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N cases (0 = all)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be >= 0")
    names = case_names(args)
    if not names:
        print("[SKIP] --split-file/--cases 没有给出任何病例", flush=True)
        return
    if args.limit:
        names = names[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    batch_started = time.perf_counter()
    results: list[dict[str, object]] = []

    for index, name in enumerate(names, 1):
        print(f"\n[{index}/{len(names)}] {name}", flush=True)

        def log(message: str, _name: str = name) -> None:
            print(f"    {_name}: {message}", flush=True)

        try:
            result = convert_case(name, args, log)
        except Exception as error:  # noqa: BLE001 - isolate one case from the batch
            result = {"case": name, "status": "failed",
                      "reason": f"{type(error).__name__}: {error}"}
        if result["status"] == "ok":
            print(f"    [OK] 标签 {result['label_source']}", flush=True)
        elif result["status"] == "failed":
            print(f"    [FAIL] {result['reason']}", flush=True)
        else:
            print(f"    [SKIP] {result['reason']}", flush=True)
        results.append(result)

    succeeded = [item for item in results if item["status"] == "ok"]
    skipped = [item for item in results if item["status"] == "skipped"]
    existing = [item for item in results if item["status"] == "existing"]
    failed = [item for item in results if item["status"] == "failed"]
    above_clamp = [item["case"] for item in succeeded if item["mu_above_thorax_clamp"]]
    clamps = sorted({item["mu_max"] for item in succeeded})
    elapsed_minutes = (time.perf_counter() - batch_started) / 60.0

    summary = {
        "source_root": str(args.source),
        "registration_root": str(args.registration_root),
        "output_root": str(args.output),
        "split_file": str(args.split_file),
        "gt_source": GT_SOURCE,
        "link_mode": args.link_mode,
        "total_cases": len(names),
        "succeeded": len(succeeded),
        "skipped_missing_data": len(skipped),
        "skipped_existing_output": len(existing),
        "failed": len(failed),
        "mu_above_thorax_clamp": above_clamp,
        "mu_max_over_cases": clamps,
        "skipped_detail": {item["case"]: item["reason"] for item in skipped},
        "failed_detail": {item["case"]: item["reason"] for item in failed},
        "elapsed_minutes": round(elapsed_minutes, 2),
        "cases": results,
    }
    # A re-run that only skips existing outputs must not clobber the provenance
    # record (per-case grid check, mu range, size) written by the productive run.
    produced_anything = bool(succeeded) or bool(failed)
    summary_path = args.output / "batch_summary.json"
    if produced_anything or not summary_path.exists():
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n完整汇总 JSON: {summary_path}", flush=True)
    else:
        print(
            f"\n本次没有生成任何新数据（成功 0，失败 0），保留原有汇总: {summary_path}",
            flush=True,
        )
    print("\n" + "=" * 72, flush=True)
    print(f"批处理结束：共 {len(names)} 例，成功 {len(succeeded)}，"
          f"缺数据跳过 {len(skipped)}，已有输出跳过 {len(existing)}，失败 {len(failed)}；"
          f"总用时 {elapsed_minutes:.1f} 分钟", flush=True)
    if skipped:
        print("\n缺数据的病例：", flush=True)
        for item in skipped:
            print(f"  - {item['case']}: {item['reason']}", flush=True)
    if failed:
        print("\n处理失败的病例：", flush=True)
        for item in failed:
            print(f"  - {item['case']}: {item['reason']}", flush=True)
    if above_clamp:
        print(f"\n有 {len(above_clamp)} 例的 μ 超过 thorax 的 clamp_max=0.09，"
              f"训练时会被截断：{above_clamp[:5]}", flush=True)
    if succeeded:
        print(f"\n各病例 μ 上限的取值个数: {len(clamps)}（如 {clamps[:3]} ... {clamps[-1]}）", flush=True)


if __name__ == "__main__":
    main()
