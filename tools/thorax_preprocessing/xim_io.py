"""Small, dependency-free reader for Varian ``.xim`` detector images.

The implementation follows the HND decoder shipped in the open-source Varian
XimReader in ``dataset/thorax/ximreader``.  Unlike that Python-2-era module,
this reader works with current Python/NumPy and does not require docutils or
matplotlib.  It also exposes the per-frame property dictionary needed for
projection geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, BinaryIO

import numpy as np


@dataclass(frozen=True)
class XimImage:
    path: Path
    width: int
    height: int
    bits_per_pixel: int
    bytes_per_pixel: int
    compressed: bool
    pixels: np.ndarray | None
    properties: dict[str, Any]


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of XIM file (wanted {size} bytes)")
    return value


def _unpack(stream: BinaryIO, fmt: str) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(stream, size))


def _decode_hnd(
    stream: BinaryIO, width: int, height: int, bytes_per_pixel: int, lut: np.ndarray
) -> np.ndarray:
    """Vectorized HND decompression.

    HND stores the first row and the first value of row two verbatim.  Every
    later value is represented by a 2-D second difference.  Variable-width
    residuals are gathered with NumPy; the recurrence is then expressed as
    two cumulative sums, avoiding a Python loop over every detector pixel.
    """

    if bytes_per_pixel != 4:
        raise ValueError(f"Only 4-byte Varian detector pixels are supported, got {bytes_per_pixel}")

    initial_count = width + 1
    initial = np.frombuffer(_read_exact(stream, initial_count * 4), dtype="<i4").astype(
        np.int64
    )
    residual_count = width * height - initial_count
    codes = ((lut[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1)
    codes = codes[:residual_count]
    if np.any(codes == 3):
        raise ValueError("Invalid HND LUT code 3")
    sizes = np.choose(codes, (1, 2, 4)).astype(np.int64)
    offsets = np.empty(residual_count, dtype=np.int64)
    offsets[0] = 0
    if residual_count > 1:
        offsets[1:] = np.cumsum(sizes[:-1])
    packed = np.frombuffer(_read_exact(stream, int(sizes.sum())), dtype=np.uint8)

    residuals = np.empty(residual_count, dtype=np.int64)
    mask1 = sizes == 1
    residuals[mask1] = packed[offsets[mask1]].view(np.int8).astype(np.int64)
    mask2 = sizes == 2
    off2 = offsets[mask2]
    value2 = packed[off2].astype(np.uint16) | (packed[off2 + 1].astype(np.uint16) << 8)
    residuals[mask2] = value2.view(np.int16).astype(np.int64)
    mask4 = sizes == 4
    off4 = offsets[mask4]
    value4 = (
        packed[off4].astype(np.uint32)
        | (packed[off4 + 1].astype(np.uint32) << 8)
        | (packed[off4 + 2].astype(np.uint32) << 16)
        | (packed[off4 + 3].astype(np.uint32) << 24)
    )
    residuals[mask4] = value4.view(np.int32).astype(np.int64)

    # Let delta[i] = image[i] - image[i-1].  The HND recurrence becomes
    # delta[i] = residual[i] + delta[i-width], which is a column-wise cumsum.
    delta = np.zeros((height, width), dtype=np.int64)
    initial_delta = np.diff(initial)
    delta[0, 1:] = initial_delta[: width - 1]
    delta[1, 0] = initial_delta[width - 1]
    residual_grid = np.zeros((height, width), dtype=np.int64)
    residual_grid.reshape(-1)[initial_count:] = residuals
    delta[1:, 1:] = delta[0, 1:] + np.cumsum(residual_grid[1:, 1:], axis=0)
    if height > 2:
        delta[2:, 0] = delta[1, 0] + np.cumsum(residual_grid[2:, 0], axis=0)
    flat = np.empty(width * height, dtype=np.int64)
    flat[0] = initial[0]
    flat[1:] = initial[0] + np.cumsum(delta.reshape(-1)[1:])
    return flat.astype(np.int32).reshape(height, width)


def _read_properties(stream: BinaryIO) -> dict[str, Any]:
    raw_count = stream.read(4)
    if not raw_count:
        return {}
    if len(raw_count) != 4:
        raise ValueError("Truncated XIM property count")
    count = struct.unpack("<i", raw_count)[0]
    result: dict[str, Any] = {}
    previous_name = "<none>"
    for index in range(count):
        name_len = _unpack(stream, "<i")[0]
        if not 0 <= name_len <= 1_000_000:
            raise ValueError(
                f"Invalid XIM property-name length {name_len} at property {index}/{count} "
                f"after {previous_name!r}, file offset {stream.tell() - 4}"
            )
        name = _read_exact(stream, name_len).decode("utf-8", errors="replace").rstrip("\x00")
        value_type = _unpack(stream, "<i")[0]
        if value_type == 0:
            value: Any = _unpack(stream, "<i")[0]
        elif value_type == 1:
            value = _unpack(stream, "<d")[0]
        elif value_type == 2:
            length = _unpack(stream, "<i")[0]
            value = _read_exact(stream, length).decode("utf-8", errors="replace").rstrip("\x00")
        elif value_type in (4, 5):
            # XIM stores the array payload size in bytes, not the item count.
            byte_length = _unpack(stream, "<i")[0]
            dtype = "<f8" if value_type == 4 else "<i4"
            item_size = 8 if value_type == 4 else 4
            if byte_length < 0 or byte_length % item_size:
                raise ValueError(f"Invalid XIM array byte length: {byte_length}")
            value = np.frombuffer(_read_exact(stream, byte_length), dtype=dtype).copy()
        else:
            raise ValueError(f"Unsupported XIM property type {value_type} for {name!r}")
        result[name] = value
        previous_name = name
    return result


def read_xim(
    path: str | Path, *, read_pixels: bool = True, read_properties: bool = True
) -> XimImage:
    path = Path(path)
    with path.open("rb") as stream:
        identifier = _read_exact(stream, 8)
        if b"VMS.XI" not in identifier:
            raise ValueError(f"Not a Varian XIM file: {path}")
        _version, width, height, bits, bytes_per_pixel, compression = _unpack(stream, "<6i")
        pixels: np.ndarray | None = None
        if compression:
            lut_size = _unpack(stream, "<i")[0]
            lut = np.frombuffer(_read_exact(stream, lut_size), dtype=np.uint8)
            compressed_size = _unpack(stream, "<i")[0]
            pixel_start = stream.tell()
            if read_pixels:
                pixels = _decode_hnd(stream, width, height, bytes_per_pixel, lut)
                consumed = stream.tell() - pixel_start
                if consumed != compressed_size:
                    raise ValueError(
                        f"HND byte-count mismatch in {path}: decoded {consumed}, header says {compressed_size}"
                    )
            else:
                stream.seek(compressed_size, 1)
            _uncompressed_size = _unpack(stream, "<i")[0]
        else:
            byte_count = _unpack(stream, "<i")[0]
            if read_pixels:
                pixels = np.frombuffer(_read_exact(stream, byte_count), dtype="<i4").copy()
                pixels = pixels.reshape(height, width)
            else:
                stream.seek(byte_count, 1)

        histogram_bins = _unpack(stream, "<i")[0]
        if histogram_bins < 0:
            raise ValueError(f"Invalid XIM histogram size: {histogram_bins}")
        stream.seek(histogram_bins * 4, 1)
        properties = _read_properties(stream) if read_properties else {}

    return XimImage(
        path=path,
        width=width,
        height=height,
        bits_per_pixel=bits,
        bytes_per_pixel=bytes_per_pixel,
        compressed=bool(compression),
        pixels=pixels,
        properties=properties,
    )
