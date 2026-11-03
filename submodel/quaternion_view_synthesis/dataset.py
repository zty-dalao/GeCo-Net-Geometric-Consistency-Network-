"""Dental projection dataset for quaternion view synthesis.

Each subject provides ``proj.nii.gz`` with ``N`` projections (360 by default,
1 degree apart) plus ``transforms.json`` with the matching 12-vectors.  Because
decoding a 360 x 512 x 512 stack is expensive, the (optionally downsampled)
stack is cached as a ``.npy`` file and memory mapped afterwards.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_EPS = 1e-8


def _downsample_stack(stack, image_size):
    """Area-downsample ``[N, H, W]`` to ``[N, image_size, image_size]``."""
    if stack.shape[-1] == image_size and stack.shape[-2] == image_size:
        return stack
    tensor = torch.from_numpy(stack).unsqueeze(1)  # [N,1,H,W]
    chunks = []
    for start in range(0, tensor.shape[0], 32):
        chunk = tensor[start:start + 32].float()
        chunks.append(
            F.interpolate(chunk, size=(image_size, image_size), mode="area")
        )
    return torch.cat(chunks, dim=0).squeeze(1).numpy()


class DentalProjectionDataset(Dataset):
    """Return ``{'subject', 'proj', 'vecs', 'meta'}`` for one subject."""

    def __init__(self, data_root, split_file, split="train", image_size=128,
                 cache_dir=None, limit=None):
        self.data_root = Path(data_root)
        self.image_size = int(image_size)
        with Path(split_file).open("r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        requested = list(split_data[split])
        self.subjects = [
            subject for subject in requested
            if (self.data_root / subject / "proj.nii.gz").is_file()
            and (self.data_root / subject / "transforms.json").is_file()
        ]
        missing = sorted(set(requested) - set(self.subjects))
        if missing:
            warnings.warn(
                f"Skipping {len(missing)} {split} subjects without projections: "
                f"{', '.join(missing[:5])} ...",
                stacklevel=2,
            )
        if limit is not None:
            self.subjects = self.subjects[:limit]
        if not self.subjects:
            raise RuntimeError(f"No usable subjects for split '{split}'.")

        self.cache_dir = Path(cache_dir) if cache_dir else self.data_root.parent / "_nvs_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_cache = {}

    def __len__(self):
        return len(self.subjects)

    def _prepare(self, subject):
        cache_path = self.cache_dir / f"{subject}_{self.image_size}.npy"
        if not cache_path.is_file():
            volume_path = self.data_root / subject / "proj.nii.gz"
            stack = sitk.GetArrayFromImage(sitk.ReadImage(str(volume_path)))
            stack = np.asarray(stack, dtype=np.float32)
            stack = _downsample_stack(stack, self.image_size)
            np.save(cache_path, stack)
        return np.load(cache_path, mmap_mode="r")

    def meta(self, subject):
        if subject not in self._meta_cache:
            with (self.data_root / subject / "transforms.json").open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            vecs = np.asarray([frame["vec"] for frame in raw["frames"]], dtype=np.float32)
            original = raw["proj_resolution"]
            factor = float(original[0]) / float(self.image_size)
            vecs = vecs.copy()
            # u_vector / v_vector fold the pixel spacing, so downsampling the
            # detector means scaling both by the same factor.
            vecs[:, 6:12] *= factor
            self._meta_cache[subject] = {
                "vecs": vecs,
                "factor": factor,
                "angle_per_view": float(raw.get("angle_per_view", 360.0 / len(vecs))),
                "proj_resolution": [self.image_size, self.image_size],
                "sad": float(raw.get("sad", 500.0)),
                "sid": float(raw.get("sid", 700.0)),
                "volume_origin": raw["volume_origin"],
                "volume_phy": raw["volume_phy"],
                "volume_resolution": raw["volume_resolution"],
                "proj_spacing": raw["proj_spacing"],
            }
        return self._meta_cache[subject]

    def __getitem__(self, index):
        subject = self.subjects[index]
        meta = self.meta(subject)
        stack = self._prepare(subject)
        projections = torch.from_numpy(np.array(stack, dtype=np.float32))
        return {
            "subject": subject,
            "proj": projections,  # [N, h, w]
            "vecs": torch.from_numpy(meta["vecs"]),
            "meta": meta,
        }


def load_volume_xyz(volume_path, size=None):
    """Load ``gt_volume.nii.gz`` as ``[1, 1, X, Y, Z]`` (optionally resampled)."""
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(volume_path))).astype(np.float32)
    volume = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 1, 0)))[None, None]
    if size is not None:
        volume = F.interpolate(volume, size=tuple(size), mode="trilinear", align_corners=False)
    return volume


def select_input_indices(num_views, n_views):
    """Evenly spaced input views, e.g. 20 views -> every 18 degrees."""
    angles = np.arange(0.0, 360.0, 360.0 / n_views)
    indices = np.round(angles / (360.0 / num_views)).astype(int) % num_views
    return np.unique(indices)


def select_target_indices(num_views, stride=10, exclude=None, tolerance=1):
    """Candidate target views, avoiding the measured input views."""
    candidates = np.arange(0, num_views, stride)
    if exclude is not None:
        exclude = np.asarray(exclude)
        keep = np.ones(candidates.shape, dtype=bool)
        for value in exclude:
            keep &= np.abs((candidates - value + num_views // 2) % num_views - num_views // 2) > tolerance
        candidates = candidates[keep]
    return candidates
