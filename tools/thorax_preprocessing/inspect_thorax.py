"""Inspect thorax DICOM series and Varian XIM metadata without converting data."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dicom_io import discover_series
from .projection_io import find_acquisition, find_air_frames, frame_angle, read_scan_geometry
from .xim_io import read_xim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset/thorax"))
    args = parser.parse_args()

    print("DICOM series:")
    series = discover_series(args.root / "image")
    for item in series:
        print(
            f"  {item.label.upper():4s} slices={len(item.paths):3d} size={item.columns}x{item.rows} "
            f"spacing=({item.pixel_spacing[1]:.6g}, {item.pixel_spacing[0]:.6g}, {item.slice_spacing:.6g}) "
            f"manufacturer={item.manufacturer!r}\n"
            f"       series_uid={item.uid}\n"
            f"       frame_of_reference_uid={item.frame_of_reference_uid}"
        )

    projection_root = args.root / "projection"
    if not projection_root.is_dir():
        print(f"[SKIP] Projection root does not exist: {projection_root}")
        return
    for case in sorted(path for path in projection_root.iterdir() if path.is_dir()):
        if not any(path.is_file() for path in case.rglob("*")):
            print(f"[SKIP] Empty projection case folder: {case}")
            continue
        if not (case / "Scan.xml").is_file():
            print(f"[SKIP] Projection case has no Scan.xml: {case}")
            continue
        try:
            acquisition = find_acquisition(case)
        except ValueError as error:
            print(f"[SKIP] {case}: {error}")
            continue
        projections = sorted(acquisition.glob("Proj_*.xim"))
        if not projections:
            print(f"[SKIP] Projection case has no Proj_*.xim frames: {case}")
            continue
        geometry = read_scan_geometry(case / "Scan.xml")
        air = find_air_frames(case)
        sample = read_xim(projections[0], read_pixels=False) if projections else None
        angle = frame_angle(sample.properties) if sample else None
        display_angle = None if angle is None else angle % 360.0
        print(
            f"Projection case {case.name}: projections={len(projections)}, air={len(air)}, "
            f"detector={geometry.detector_size}, spacing={geometry.detector_spacing}, "
            f"SAD/SID={geometry.sad}/{geometry.sid}, first_XIM_source_angle={display_angle}"
        )
        if sample:
            print("  XIM properties:", ", ".join(sorted(sample.properties)))


if __name__ == "__main__":
    main()
