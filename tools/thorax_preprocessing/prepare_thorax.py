"""Convert mixed DICOM and Varian XIM thorax data into the training layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

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


def usable_projection_cases(root: Path) -> list[Path]:
    """Return non-empty projection cases and report incomplete folders as skipped."""
    projection_root = root / "projection"
    if not projection_root.is_dir():
        print(f"[SKIP] Projection root does not exist: {projection_root}")
        return []

    usable: list[Path] = []
    for case in sorted(path for path in projection_root.iterdir() if path.is_dir()):
        if not any(path.is_file() for path in case.rglob("*")):
            print(f"[SKIP] Empty projection case folder: {case}")
            continue
        scan_xml = case / "Scan.xml"
        if not scan_xml.is_file():
            print(f"[SKIP] Projection case has no Scan.xml: {case}")
            continue
        try:
            acquisition = find_acquisition(case)
        except ValueError as error:
            print(f"[SKIP] {case}: {error}")
            continue
        if not any(acquisition.glob("Proj_*.xim")):
            print(f"[SKIP] Projection case has no Proj_*.xim frames: {case}")
            continue
        usable.append(case)
    return usable


def select_required_series(
    all_series: list, label: str, uid: str | None, image_root: Path
):
    """Select a required DICOM series, returning None for an empty/missing modality."""
    try:
        series = select_series(all_series, label, uid)
    except ValueError as error:
        print(f"[SKIP] Missing or empty {label.upper()} data below {image_root}: {error}")
        return None
    if not series.paths:
        print(f"[SKIP] Empty {label.upper()} DICOM series below {image_root}")
        return None
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/thorax"))
    parser.add_argument("--output", type=Path, default=Path("dataset/thorax/syn_data"))
    parser.add_argument("--case-name", default=None, help="Output case name; defaults to projection folder name")
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
        "--registered-ct-mu",
        type=Path,
        default=None,
        help="CT already registered/resampled to the CBCT grid and converted to mu",
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.projection_bin < 1:
        parser.error("--projection-bin must be >= 1")
    explicit_range = parse_slice_range(args.ct_slices)
    projection_cases = usable_projection_cases(args.root)
    if not projection_cases:
        print(f"[SKIP] No usable projection cases below {args.root / 'projection'}")
        return
    if args.case_name and len(projection_cases) != 1:
        raise ValueError("--case-name is only valid when exactly one projection case exists")

    image_root = args.root / "image"
    all_series = discover_series(image_root)
    if not all_series:
        print(f"[SKIP] No CT DICOM slices found below {image_root}")
        return
    cbct_series = select_required_series(all_series, "cbct", args.cbct_series_uid, image_root)
    ct_series = select_required_series(all_series, "ct", args.ct_series_uid, image_root)
    if cbct_series is None or ct_series is None:
        return
    cbct_hu, cbct_native_reference = load_hu(cbct_series.paths)
    cbct_reference = cbct_native_reference
    cbct_depth_mm = cbct_reference.GetSize()[2] * cbct_reference.GetSpacing()[2]
    ct_paths = crop_paths(
        ct_series,
        explicit_range,
        cbct_depth_mm if args.ct_crop == "match-cbct-center" and explicit_range is None else None,
    )
    ct_hu, ct_reference = load_hu(ct_paths)
    ct_mu = hu_to_mu(ct_hu)
    registered_ct_image: sitk.Image | None = None
    if args.gt_source == "registered-ct":
        if args.registered_ct_mu is None:
            parser.error("--gt-source registered-ct requires --registered-ct-mu")
        registered_ct_image = sitk.ReadImage(str(args.registered_ct_mu), sitk.sitkFloat32)
        if not np.allclose(
            registered_ct_image.GetDirection(), cbct_native_reference.GetDirection(), atol=1e-6
        ):
            raise ValueError("Registered CT direction differs from the CBCT direction")
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
            raise ValueError("Registered CT and CBCT grids do not share the same physical center")
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

    args.output.mkdir(parents=True, exist_ok=True)
    converted_names: list[str] = []
    for projection_case in projection_cases:
        name = args.case_name or projection_case.name
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Case name must be one path component, got {name!r}")
        destination = args.output / name
        if destination.resolve().parent != args.output.resolve():
            raise ValueError(f"Unsafe output case path: {destination}")
        if destination.exists():
            if not args.overwrite:
                raise FileExistsError(f"Output exists: {destination}; pass --overwrite to replace it")
            shutil.rmtree(destination)
        destination.mkdir(parents=True)

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
                geometry.imager_lateral if args.detector_offset_u_mm is None else args.detector_offset_u_mm,
                geometry.imager_longitudinal if args.detector_offset_v_mm is None else args.detector_offset_v_mm,
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
        converted_names.append(name)
        print(f"Converted {name} -> {destination}")

    split = make_starter_split(converted_names)
    split_path = args.output / "thorax_split.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Wrote starter split: {split_path}")


if __name__ == "__main__":
    main()
