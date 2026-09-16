"""Varian projection correction and scanner-geometry conversion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .xim_io import read_xim


@dataclass(frozen=True)
class ScanGeometry:
    sad: float
    sid: float
    detector_spacing: tuple[float, float]
    detector_size: tuple[int, int]
    imager_lateral: float
    imager_longitudinal: float
    source_angle_offset: float
    start_angle: float
    stop_angle: float


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first_float(root: ET.Element, name: str, default: float | None = None) -> float:
    for element in root.iter():
        if _local_name(element.tag) == name and element.text:
            try:
                return float(element.text)
            except ValueError:
                continue
    if default is None:
        raise ValueError(f"Missing numeric {name} in Scan.xml")
    return default


def read_scan_geometry(scan_xml: str | Path) -> ScanGeometry:
    root = ET.parse(scan_xml).getroot()
    return ScanGeometry(
        sad=_first_float(root, "SAD"),
        sid=_first_float(root, "SID"),
        detector_spacing=(_first_float(root, "ImagerResX"), _first_float(root, "ImagerResY")),
        detector_size=(int(_first_float(root, "ImagerSizeX")), int(_first_float(root, "ImagerSizeY"))),
        imager_lateral=_first_float(root, "ImagerLat", 0.0),
        imager_longitudinal=_first_float(root, "ImagerLng", 0.0),
        source_angle_offset=_first_float(root, "SourceAngleOffset", 0.0),
        start_angle=_first_float(root, "StartAngle", 0.0),
        stop_angle=_first_float(root, "StopAngle", 360.0),
    )


def find_acquisition(case_root: str | Path) -> Path:
    roots = [path for path in Path(case_root).glob("Acquisitions/*") if path.is_dir()]
    if not roots:
        raise ValueError(f"No Acquisitions/<id> directory below {case_root}")
    return max(roots, key=lambda path: len(list(path.glob("Proj_*.xim"))))


def find_air_frames(case_root: str | Path) -> list[Path]:
    candidates = list(Path(case_root).glob("Calibrations/AIR-Bowtie-*/Current/FilterBowtie*.xim"))
    if not candidates:
        candidates = list(Path(case_root).glob("Calibrations/AIR-*/**/*.xim"))
    return sorted(candidates)


def _block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image.astype(np.float32, copy=False)
    height = image.shape[0] // factor * factor
    width = image.shape[1] // factor * factor
    cropped = image[:height, :width]
    return cropped.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def _resample_detector(
    image: np.ndarray,
    input_spacing: tuple[float, float],
    output_resolution: tuple[int, int],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Resample a detector image onto a centered grid with near-isotropic pixels."""
    output_width, output_height = output_resolution
    input_height, input_width = image.shape
    output_spacing = (
        input_spacing[0] * input_width / output_width,
        input_spacing[1] * input_height / output_height,
    )
    source = sitk.GetImageFromArray(np.asarray(image, dtype=np.float32))
    source.SetSpacing(input_spacing)
    input_center = np.array(
        source.TransformContinuousIndexToPhysicalPoint(
            ((input_width - 1.0) / 2.0, (input_height - 1.0) / 2.0)
        )
    )
    output_half_extent = np.array(
        [
            (output_width - 1.0) * output_spacing[0] / 2.0,
            (output_height - 1.0) * output_spacing[1] / 2.0,
        ]
    )
    result = sitk.Resample(
        source,
        [output_width, output_height],
        sitk.Transform(2, sitk.sitkIdentity),
        sitk.sitkLinear,
        tuple(input_center - output_half_extent),
        output_spacing,
        (1.0, 0.0, 0.0, 1.0),
        0.0,
        sitk.sitkFloat32,
    )
    return sitk.GetArrayFromImage(result), output_spacing


def build_air_maps(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("No air/bowtie calibration XIM frames found")
    angles: list[float] = []
    maps: list[np.ndarray] = []
    for path in tqdm(paths, desc="Air calibration"):
        xim = read_xim(path)
        pixels = xim.pixels
        assert pixels is not None
        chamber = float(xim.properties.get("KVNormChamber", 0.0))
        normalized = pixels.astype(np.float64) / chamber if chamber > 0 else pixels.astype(np.float64)
        angle = frame_angle(xim.properties)
        angles.append(0.0 if angle is None else angle % 360.0)
        maps.append(normalized.astype(np.float32))
    return np.asarray(angles), np.stack(maps)


ANGLE_PROPERTY_NAMES = (
    "KVSourceRtn",
    "GantryRtn",
    "GantryAngle",
    "GantryRtnExt",
    "SourceRtn",
)


def frame_angle(properties: dict[str, object]) -> float | None:
    lowered = {key.lower(): value for key, value in properties.items()}
    for name in ANGLE_PROPERTY_NAMES:
        value = lowered.get(name.lower())
        if isinstance(value, (int, float, np.number)):
            return float(value)
    for key, value in properties.items():
        if "gantry" in key.lower() and "angle" in key.lower() and isinstance(value, (int, float, np.number)):
            return float(value)
    return None


def angle_to_vec(
    angle_degrees: float,
    geometry: ScanGeometry,
    spacing: tuple[float, float],
    detector_offset_u: float,
    detector_offset_v: float,
) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    source = np.array([geometry.sad * np.cos(angle), geometry.sad * np.sin(angle), 0.0])
    detector = np.array(
        [-(geometry.sid - geometry.sad) * np.cos(angle), -(geometry.sid - geometry.sad) * np.sin(angle), 0.0]
    )
    u_unit = np.array([-np.sin(angle), np.cos(angle), 0.0])
    v_unit = np.array([0.0, 0.0, -1.0])
    detector = detector + detector_offset_u * u_unit + detector_offset_v * v_unit
    return np.concatenate((source, detector, spacing[0] * u_unit, spacing[1] * v_unit))


def convert_projections(
    case_root: str | Path,
    output_path: str | Path,
    geometry: ScanGeometry,
    *,
    bin_factor: int = 2,
    mode: str = "log",
    max_line_integral: float = 20.0,
    detector_offset_u: float | None = None,
    detector_offset_v: float | None = None,
    output_views: int = 360,
    output_resolution: tuple[int, int] | None = (256, 256),
) -> tuple[np.ndarray, list[dict[str, object]], tuple[float, float]]:
    acquisition = find_acquisition(case_root)
    all_paths = sorted(acquisition.glob("Proj_*.xim"))
    if not all_paths:
        raise ValueError(f"No Proj_*.xim files in {acquisition}")
    air_calibration = build_air_maps(find_air_frames(case_root)) if mode == "log" else None
    metadata: list[tuple[Path, dict[str, object], float]] = []
    fallback_angles = np.linspace(geometry.start_angle, geometry.stop_angle, len(all_paths), endpoint=False)
    for index, path in enumerate(all_paths):
        properties = read_xim(path, read_pixels=False).properties
        chamber = float(properties.get("KVNormChamber", 1.0))
        if chamber <= 0:
            continue
        angle = frame_angle(properties)
        if angle is None:
            # SourceAngleOffset is defined as gantry angle minus source angle.
            angle = float(fallback_angles[index] - geometry.source_angle_offset)
        metadata.append((path, properties, angle % 360.0))
    if output_views < 1 or output_views > len(metadata):
        raise ValueError(f"output_views must be in [1, {len(metadata)}]")
    targets = np.linspace(0.0, 360.0, output_views, endpoint=False)
    measured = np.asarray([item[2] for item in metadata])
    selected_indices = [int(np.argmin(np.abs((measured - target + 180.0) % 360.0 - 180.0))) for target in targets]
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("Uniform angular selection chose duplicate XIM frames; request fewer --output-views")
    selected = [(targets[i], metadata[index]) for i, index in enumerate(selected_indices)]

    first = read_xim(selected[0][1][0], read_pixels=False)
    binned_h = first.height // bin_factor
    binned_w = first.width // bin_factor
    out_w, out_h = output_resolution if output_resolution is not None else (binned_w, binned_h)
    stack = np.empty((len(selected), out_h, out_w), dtype=np.float32)
    records: list[tuple[float, float, Path, np.ndarray, dict[str, object]]] = []

    for target, (path, properties, source_angle) in tqdm(selected, desc="Patient projections"):
        xim = read_xim(path)
        assert xim.pixels is not None
        if mode == "log":
            assert air_calibration is not None
            air_angles, air_maps = air_calibration
            angular_distance = np.abs((air_angles - source_angle + 180.0) % 360.0 - 180.0)
            air = air_maps[int(np.argmin(angular_distance))]
            chamber = float(xim.properties.get("KVNormChamber", 0.0))
            if chamber <= 0:
                raise ValueError(f"Invalid KVNormChamber in {path}")
            patient = xim.pixels.astype(np.float64) / chamber
            ratio = np.clip(patient, np.finfo(np.float64).eps, None) / np.clip(
                air, np.finfo(np.float64).eps, None
            )
            projection = np.clip(-np.log(ratio), 0.0, max_line_integral)
        elif mode == "raw":
            projection = np.clip(xim.pixels, 0, None).astype(np.float32)
        else:
            raise ValueError(f"Unknown projection mode: {mode}")
        projection = _block_mean(projection, bin_factor)
        records.append((target, source_angle, path, projection, properties))
    binned_spacing = (
        geometry.detector_spacing[0] * bin_factor,
        geometry.detector_spacing[1] * bin_factor,
    )
    if output_resolution is None:
        spacing = binned_spacing
    else:
        _, spacing = _resample_detector(records[0][3], binned_spacing, output_resolution)
    offset_u = geometry.imager_lateral if detector_offset_u is None else detector_offset_u
    offset_v = geometry.imager_longitudinal if detector_offset_v is None else detector_offset_v
    frames: list[dict[str, object]] = []
    for index, (target, angle, path, projection, properties) in enumerate(records):
        if output_resolution is not None:
            projection, _ = _resample_detector(projection, binned_spacing, output_resolution)
        stack[index] = projection
        per_frame_u = detector_offset_u
        per_frame_v = detector_offset_v
        if per_frame_u is None and isinstance(properties.get("KVDetectorLat"), (int, float, np.number)):
            per_frame_u = float(properties["KVDetectorLat"]) * 10.0  # XIM stores cm; XML stores mm.
        if per_frame_v is None and isinstance(properties.get("KVDetectorLng"), (int, float, np.number)):
            per_frame_v = float(properties["KVDetectorLng"]) * 10.0
        if per_frame_u is None:
            per_frame_u = offset_u
        if per_frame_v is None:
            per_frame_v = offset_v
        frames.append(
            {
                "file": f"{index:04d}",
                "source_file": path.name,
                "nominal_angle_degrees": target,
                "angle_degrees": angle,
                "vec": angle_to_vec(angle, geometry, spacing, per_frame_u, per_frame_v).tolist(),
            }
        )

    image = sitk.GetImageFromArray(stack)
    sitk.WriteImage(image, str(output_path), useCompression=True)
    return stack, frames, spacing


def write_transforms(path: str | Path, parameters: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(parameters, stream, indent=2, ensure_ascii=False)
