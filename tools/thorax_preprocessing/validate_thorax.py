"""Validate processed thorax cases against ``data/Dataset.py`` expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/thorax/syn_data"))
    args = parser.parse_args()
    failures: list[str] = []
    cases = sorted(path for path in args.data.iterdir() if path.is_dir())
    for case in cases:
        required = [case / "gt_volume.nii.gz", case / "proj.nii.gz", case / "transforms.json"]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            failures.append(f"{case.name}: missing {missing}")
            continue
        params = json.loads((case / "transforms.json").read_text(encoding="utf-8"))
        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "gt_volume.nii.gz")))
        projections = sitk.GetArrayFromImage(sitk.ReadImage(str(case / "proj.nii.gz")))
        if projections.ndim != 3 or volume.ndim != 3:
            failures.append(f"{case.name}: volume/projection arrays must both be 3-D")
        if len(params.get("frames", [])) != projections.shape[0]:
            failures.append(f"{case.name}: frame count differs from projection count")
        if list(params.get("proj_resolution", [])) != [projections.shape[2], projections.shape[1]]:
            failures.append(f"{case.name}: proj_resolution differs from NIfTI shape")
        if list(params.get("volume_resolution", [])) != [volume.shape[2], volume.shape[1], volume.shape[0]]:
            failures.append(f"{case.name}: volume_resolution differs from NIfTI shape")
        if any(size % 4 for size in (volume.shape[2], volume.shape[1], volume.shape[0])):
            failures.append(f"{case.name}: volume dimensions must be divisible by decoder scale 4")
        volume_spacing = np.asarray(params.get("volume_spacing", []), dtype=float)
        if volume_spacing.shape != (3,) or not np.allclose(volume_spacing, volume_spacing[0], atol=1e-6):
            failures.append(f"{case.name}: training volume spacing is not isotropic")
        projection_spacing = np.asarray(params.get("proj_spacing", []), dtype=float)
        if projection_spacing.shape != (2,) or not np.allclose(
            projection_spacing, projection_spacing[0], atol=1e-6
        ):
            failures.append(f"{case.name}: detector spacing is not isotropic")
        frames = params.get("frames", [])
        if frames:
            vec = np.asarray(frames[0].get("vec", []), dtype=float)
            if vec.shape != (12,) or not np.allclose(
                [np.linalg.norm(vec[6:9]), np.linalg.norm(vec[9:12])],
                projection_spacing,
                atol=1e-5,
            ):
                failures.append(f"{case.name}: frame u/v vectors disagree with proj_spacing")
        cbct_path = case / "cbct_volume.nii.gz"
        if cbct_path.exists():
            cbct_image = sitk.ReadImage(str(cbct_path))
            gt_image = sitk.ReadImage(str(case / "gt_volume.nii.gz"))
            same_grid = (
                cbct_image.GetSize() == gt_image.GetSize()
                and np.allclose(cbct_image.GetSpacing(), gt_image.GetSpacing(), atol=1e-6)
                and np.allclose(cbct_image.GetOrigin(), gt_image.GetOrigin(), atol=1e-5)
                and np.allclose(cbct_image.GetDirection(), gt_image.GetDirection(), atol=1e-6)
            )
            if not same_grid:
                failures.append(f"{case.name}: CBCT and GT grids differ")
        if not np.isfinite(volume).all() or not np.isfinite(projections).all():
            failures.append(f"{case.name}: NaN/Inf detected")
        if volume.min() < 0 or projections.min() < 0:
            failures.append(f"{case.name}: negative attenuation values detected")
        print(
            f"{case.name}: volume={volume.shape} range=({volume.min():.5g},{volume.max():.5g}), "
            f"projections={projections.shape} range=({projections.min():.5g},{projections.max():.5g})"
        )
    if not cases:
        failures.append(f"No case directories in {args.data}")
    if failures:
        raise SystemExit("Validation failed:\n  " + "\n  ".join(failures))
    print(f"Validation passed for {len(cases)} case(s).")


if __name__ == "__main__":
    main()
