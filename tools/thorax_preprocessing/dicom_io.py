"""DICOM discovery, classification, loading and CT cropping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
import SimpleITK as sitk


@dataclass(frozen=True)
class DicomSeries:
    uid: str
    modality: str
    manufacturer: str
    description: str
    frame_of_reference_uid: str
    paths: tuple[Path, ...]
    rows: int
    columns: int
    pixel_spacing: tuple[float, float]
    slice_spacing: float

    @property
    def label(self) -> str:
        text = f"{self.manufacturer} {self.description}".lower()
        return "cbct" if "varian" in text or "cbct" in text or "cone" in text else "ct"


def _header(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def discover_series(root: str | Path) -> list[DicomSeries]:
    groups: dict[str, list[tuple[Path, object]]] = {}
    for path in sorted(Path(root).rglob("*.dcm")):
        ds = _header(path)
        if str(getattr(ds, "Modality", "")).upper() != "CT":
            continue
        uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if uid:
            groups.setdefault(uid, []).append((path, ds))

    series: list[DicomSeries] = []
    for uid, entries in groups.items():
        first = entries[0][1]
        orientation = np.asarray(getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
        normal = np.cross(orientation[:3], orientation[3:])

        def location(item: tuple[Path, object]) -> float:
            position = np.asarray(getattr(item[1], "ImagePositionPatient", [0, 0, 0]), float)
            return float(np.dot(position, normal))

        entries.sort(key=location)
        locations = np.asarray([location(item) for item in entries])
        diffs = np.diff(locations)
        spacing = float(np.median(np.abs(diffs))) if len(diffs) else float(
            getattr(first, "SliceThickness", 1.0)
        )
        pixel_spacing = tuple(float(x) for x in getattr(first, "PixelSpacing", [1, 1]))
        series.append(
            DicomSeries(
                uid=uid,
                modality=str(getattr(first, "Modality", "")),
                manufacturer=str(getattr(first, "Manufacturer", "")),
                description=str(getattr(first, "SeriesDescription", "")),
                frame_of_reference_uid=str(getattr(first, "FrameOfReferenceUID", "")),
                paths=tuple(path for path, _ in entries),
                rows=int(getattr(first, "Rows", 0)),
                columns=int(getattr(first, "Columns", 0)),
                pixel_spacing=pixel_spacing,
                slice_spacing=spacing,
            )
        )
    return sorted(series, key=lambda item: (item.label, -len(item.paths)))


def select_series(
    series: Iterable[DicomSeries], label: str, uid: str | None = None
) -> DicomSeries:
    candidates = list(series)
    if uid:
        matches = [item for item in candidates if item.uid == uid]
    else:
        matches = [item for item in candidates if item.label == label]
    if not matches:
        available = ", ".join(f"{x.label}:{x.uid}" for x in candidates)
        raise ValueError(f"Could not find {label} DICOM series. Available: {available}")
    return max(matches, key=lambda item: len(item.paths))


def crop_paths(
    series: DicomSeries,
    slice_range: tuple[int, int] | None,
    target_depth_mm: float | None,
) -> tuple[Path, ...]:
    paths = series.paths
    if slice_range is not None:
        start, end = slice_range
        if start < 0 or end > len(paths) or start >= end:
            raise ValueError(f"Invalid CT slice range {start}:{end}; series has {len(paths)} slices")
        return paths[start:end]
    if target_depth_mm is None:
        return paths
    count = min(len(paths), max(1, int(round(target_depth_mm / series.slice_spacing))))
    start = (len(paths) - count) // 2
    return paths[start : start + count]


def load_hu(paths: Iterable[Path]) -> tuple[np.ndarray, sitk.Image]:
    datasets = [pydicom.dcmread(str(path), force=True) for path in paths]
    if not datasets:
        raise ValueError("No DICOM slices selected")
    arrays = []
    for ds in datasets:
        pixels = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(pixels * slope + intercept)
    volume = np.stack(arrays)

    first = datasets[0]
    orientation = np.asarray(getattr(first, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), float)
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    normal = np.cross(row_direction, column_direction)
    positions = [np.asarray(getattr(ds, "ImagePositionPatient", [0, 0, i]), float) for i, ds in enumerate(datasets)]
    projected = np.asarray([float(np.dot(pos, normal)) for pos in positions])
    z_spacing = float(np.median(np.abs(np.diff(projected)))) if len(projected) > 1 else float(
        getattr(first, "SliceThickness", 1.0)
    )
    row_spacing, column_spacing = (float(x) for x in getattr(first, "PixelSpacing", [1, 1]))
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing((column_spacing, row_spacing, z_spacing))
    image.SetOrigin(tuple(float(x) for x in positions[0]))
    direction = np.column_stack((row_direction, column_direction, normal))
    image.SetDirection(tuple(float(x) for x in direction.reshape(-1)))
    return volume, image


def hu_to_mu(
    volume_hu: np.ndarray,
    mu_water: float = 0.022,
    hu_min: float = -1000.0,
    hu_max: float = 3000.0,
) -> np.ndarray:
    """Clip non-physical/outlier CT values before converting HU to attenuation."""
    clipped_hu = np.clip(volume_hu, hu_min, hu_max)
    return ((clipped_hu / 1000.0 + 1.0) * mu_water).astype(np.float32)


def image_with_array(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.CopyInformation(reference)
    return image
