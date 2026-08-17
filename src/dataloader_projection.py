"""Projection dataloader for the paired ``thorax_fast`` dataset (angle info only).

Loads the preprocessed projection pickles from::

    {root}/processed/projections/{case}.pickle

Each pickle contains::

    projs:     ndarray, uint8,  shape (K, 320, 1280)   # [frame, V, U]
    projs_max: float                                  # inverse-normalization scale
    angles:    ndarray, float32, shape (K,)           # radians, sorted, [-pi, pi]

Only the projection / angle information is used here (no CT/CBCT volumes,
no masks).  Samples are built so that a model can interpolate a "missing"
target view from ``num_inputs`` surrounding source views:

    src      : (num_inputs, 1, H, W)  -- source projections (float, raw range;
                                        z-score normalization is done in the model)
    tgt      : (1, H, W)              -- held-out target projection (ground truth,
                                        same raw range)
    angular  : (num_inputs,)          -- signed circular distances  target - src
                                        (radians, in [-pi, pi])

Cases are taken from ``meta_info.json`` splits: ``train`` / ``eval`` / ``test``.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = r"E:\workspace\LightningRecon\data\thorax_fast"


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    """Signed circular difference wrapped into [-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class ProjectionInterpolationDataset(Dataset):
    """Samples (source views -> target view) pairs from projection pickles.

    Parameters
    ----------
    root_dir:
        Dataset root, i.e. the directory containing ``meta_info.json`` and
        ``processed/projections``.
    split:
        One of ``train`` / ``eval`` / ``test`` (keys in ``meta_info.json``).
        Falls back to *all* pickles if the key is absent.
    num_inputs:
        Number of source projections used to predict the target.
    source_mode:
        ``"nearest"``  - the ``num_inputs`` closest views by circular angle;
        ``"symmetric"``- half from each side of the target (recommended for
                        interpolation), with one extra nearest view if odd.
    cache_cases:
        Number of cases kept in memory (each ~200 MB at full resolution).
    resize:
        Optional ``(H, W)`` to resize each projection to (bilinear).
    """

    def __init__(
        self,
        root_dir: str = DEFAULT_ROOT,
        split: str = "train",
        num_inputs: int = 2,
        source_mode: str = "symmetric",
        cache_cases: int = 4,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.proj_dir = os.path.join(root_dir, "processed", "projections")
        self.num_inputs = num_inputs
        self.source_mode = source_mode
        self.cache_cases = cache_cases
        self.resize = resize

        if not os.path.isdir(self.proj_dir):
            raise FileNotFoundError(f"projections dir not found: {self.proj_dir}")

        # --- split (case list) --------------------------------------------
        meta_path = os.path.join(root_dir, "meta_info.json")
        cases: Optional[list] = None
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if split in meta and isinstance(meta[split], list):
                cases = meta[split]
        if cases is None:
            cases = sorted(
                name[: -len(".pickle")]
                for name in os.listdir(self.proj_dir)
                if name.endswith(".pickle")
            )
        self.cases = list(cases)

        # --- lazy case cache ----------------------------------------------
        self._cache: "OrderedDict[str, dict]" = OrderedDict()

        # Assume a uniform view count (verified on first load, then clamped
        # per case for safety in ``__getitem__``).
        self.views_per_case = self._load_case(self.cases[0])["projs"].shape[0]

    # ------------------------------------------------------------------ cache
    def _load_case(self, case: str) -> dict:
        if case in self._cache:
            self._cache.move_to_end(case)
            return self._cache[case]

        path = os.path.join(self.proj_dir, f"{case}.pickle")
        with open(path, "rb") as f:
            data = pickle.load(f)

        projs = data["projs"].astype(np.float32)          # (K, 320, 1280), raw range
        angles = data["angles"].astype(np.float32)         # (K,)

        item = {"projs": projs, "angles": angles}
        self._cache[case] = item
        while len(self._cache) > self.cache_cases:
            self._cache.popitem(last=False)
        return item

    def load_case(self, case: str) -> dict:
        """Return (and cache) the raw ``projs`` / ``angles`` of one case."""
        return self._load_case(case)

    # ------------------------------------------------------------------ size
    def __len__(self) -> int:
        return len(self.cases) * self.views_per_case

    # ------------------------------------------------------------- source sel
    def _select_source_idx(self, angles: np.ndarray, target_idx: int, target_angle: float) -> np.ndarray:
        s = _wrap_pi(angles - target_angle).astype(np.float64)
        # Exclude the target itself and its 2*pi duplicate (360-deg seam).
        s[np.abs(s) < 1e-4] = np.inf

        if self.source_mode == "nearest":
            return np.argsort(np.abs(s))[: self.num_inputs]

        # symmetric: half from each side, closest first
        half = self.num_inputs // 2
        neg = np.where(s < 0)[0]
        pos = np.where(s > 0)[0]
        neg = neg[np.argsort(np.abs(s[neg]))]
        pos = pos[np.argsort(np.abs(s[pos]))]

        chosen = list(neg[:half]) + list(pos[:half])
        if self.num_inputs % 2 == 1:
            order = np.argsort(np.abs(s))
            for i in order:
                if int(i) not in chosen:
                    chosen.append(int(i))
                    break
        return np.asarray(chosen[: self.num_inputs], dtype=np.int64)

    # -------------------------------------------------------------- getitem
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, float]]:
        case_idx = idx // self.views_per_case
        view_idx = idx % self.views_per_case
        case = self.cases[case_idx]

        item = self._load_case(case)
        projs = item["projs"]                       # (K, H, W) float32
        angles = item["angles"]                     # (K,) float32

        k = int(projs.shape[0])
        view_idx = min(view_idx, k - 1)
        target_angle = float(angles[view_idx])

        src_idx = self._select_source_idx(angles, view_idx, target_angle)
        src_idx = src_idx[src_idx != view_idx]      # belt-and-braces
        src_idx = src_idx[: self.num_inputs]

        src = projs[src_idx]                        # (N, H, W)
        tgt = projs[view_idx]                       # (H, W)
        src_angles = angles[src_idx]                # (N,)
        angular = _wrap_pi(target_angle - src_angles).astype(np.float32)  # (N,)

        if self.resize is not None:
            src = self._resize_np(src, self.resize)
            tgt = self._resize_np(tgt[None], self.resize)[0]

        src_t = torch.from_numpy(np.ascontiguousarray(src)).unsqueeze(1)  # (N,1,H,W)
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt)).unsqueeze(0)  # (1,H,W)

        return {
            "src": src_t,
            "tgt": tgt_t,
            "angular": torch.from_numpy(angular),                     # (N,)
            "src_angles": torch.from_numpy(src_angles.astype(np.float32)),
            "target_angle": target_angle,
            "case": case,
        }

    @staticmethod
    def _resize_np(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        t = torch.from_numpy(img).unsqueeze(0)  # (1, N, H, W)
        t = torch.nn.functional.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return t.squeeze(0).numpy()


def sample_rotor_targets(
    num_views: int,
    num_inputs: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[list, list]:
    """Sample one rotor of ``num_inputs`` equidistant views over 360 deg plus
    one target per angular interval between consecutive rotor views.

    Works in view-index space (angles are uniformly spaced), which naturally
    handles the +/-pi seam.  Returns ``(src_idx, tgt_idx)``:

        src_idx: sorted list of rotor view indices (length ``num_inputs``)
        tgt_idx: one target index inside each interval (length ``num_inputs``)
    """
    if rng is None:
        rng = np.random.default_rng()

    step = num_views / num_inputs
    offset = float(rng.uniform(0.0, num_views))
    ideal = (offset + np.arange(num_inputs) * step) % num_views
    src_idx = sorted({int(round(float(v))) % num_views for v in ideal})

    n = len(src_idx)
    tgt_idx: list = []
    for i in range(n):
        lo = src_idx[i]
        hi = src_idx[(i + 1) % n]
        if i == n - 1:
            hi += num_views  # wrap-around interval
        cand = [j % num_views for j in range(lo + 1, hi)
                if (j % num_views) not in src_idx]
        if cand:
            tgt_idx.append(int(rng.choice(cand)))
    return src_idx, tgt_idx


def compute_zscore_stats(
    dataset: "ProjectionInterpolationDataset",
    max_cases: Optional[int] = None,
) -> Tuple[float, float]:
    """Return ``(mean, std)`` of projection values over (a subset of) cases.

    These statistics feed the input z-score normalization of the interpolation
    model.  ``max_cases`` limits how many cases are scanned (one-time cost).
    """
    cases = dataset.cases if max_cases is None else dataset.cases[:max_cases]
    mean_sum = 0.0
    meansq_sum = 0.0
    for case in cases:
        projs = dataset.load_case(case)["projs"]
        mean_sum += float(np.mean(projs, dtype=np.float64))
        meansq_sum += float(np.mean(projs * projs, dtype=np.float64))
    mean = mean_sum / len(cases)
    meansq = meansq_sum / len(cases)
    var = max(meansq - mean * mean, 0.0)
    return float(mean), float(math.sqrt(var))


def build_projection_dataloader(
    root_dir: str = DEFAULT_ROOT,
    split: str = "train",
    batch_size: int = 1,
    num_inputs: int = 2,
    source_mode: str = "symmetric",
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """Convenience wrapper returning a :class:`DataLoader` for projection data."""
    ds = ProjectionInterpolationDataset(
        root_dir=root_dir, split=split, num_inputs=num_inputs,
        source_mode=source_mode, **kwargs,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def sample_to_model_inputs(
    sample: Dict[str, torch.Tensor],
    use_cos_sin: bool = True,
) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
    """Convert one dataset sample into interpolation-model inputs.

    Returns ``(projections, angular)`` where ``projections`` is a list of
    ``(1, 1, H, W)`` tensors (batch = 1) and ``angular`` is ``(1, angular_dim)``.
    If ``use_cos_sin`` the N angular distances are encoded as
    ``[cos(d)-1, sin(d)]`` -> dimension 2N.
    """
    src = sample["src"]              # (N, 1, H, W)
    angular = sample["angular"]      # (N,)

    proj_list = [src[i].unsqueeze(0) for i in range(src.shape[0])]  # (1,1,H,W)

    if use_cos_sin:
        angular = torch.cat([torch.cos(angular) - 1.0, torch.sin(angular)], dim=-1)  # (2N,)
    return proj_list, angular.unsqueeze(0)  # (1, angular_dim)


if __name__ == "__main__":
    print(f"root: {DEFAULT_ROOT}")

    ds = ProjectionInterpolationDataset(DEFAULT_ROOT, split="train", num_inputs=6)
    print(f"train cases: {len(ds.cases)} | views/case: {ds.views_per_case} | samples: {len(ds)}")

    sample = ds[0]
    print("\n--- sample fields ---")
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:>12}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:>12}: {v}")

    # how to feed the interpolation model (batch = 1)
    proj_list, angular = sample_to_model_inputs(sample, use_cos_sin=True)
    print(f"\nmodel inputs: {len(proj_list)} projections of {tuple(proj_list[0].shape)}")
    print(f"angular (encoded): {tuple(angular.shape)}")

    # quick sanity: source views must differ from the target angle
    d = _wrap_pi(np.asarray([float(sample['target_angle'])]) - sample['src_angles'].numpy())
    print("angular distances to target (deg):", np.round(np.degrees(d), 2))
