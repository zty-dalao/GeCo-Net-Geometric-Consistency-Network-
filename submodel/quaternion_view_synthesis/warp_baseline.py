"""S0 ablation: the pure-geometry (no learning) warp baseline.

    python -m submodel.quaternion_view_synthesis.warp_baseline --splits train,val,test

The synthesiser is ``b = warp(a) + g * residual``.  ``warp`` is the plane-induced
homography built purely from the 12-vector geometry, with **zero parameters**.
This script measures how far that geometric path alone gets, which is the number
any learned contribution has to beat.

Three variants are reported because they behave very differently:

    nearest    warp the single measured view closest in angle to the target
    mean20     uniform average of the 20 warped predictions   (what naive fusion gives)
    median20   pixel-wise median of the 20 warped predictions

The average collapses because the homography maps the **isocenter plane** exactly;
anything off that plane is displaced by an amount growing with |Delta theta|, so
averaging 20 copies blurs the image instead of denoising it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import geometry
from .dataset import DentalProjectionDataset, select_input_indices, select_target_indices
from .metrics import psnr, ssim


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pure-geometry warp baseline")
    parser.add_argument("--data_root", default="dataset/dental/syn_data")
    parser.add_argument("--split_file", default="data/dataset_split/dental_split.json")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--cache_dir", default="dataset/dental/_nvs_cache")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=4, help="subjects per split")
    parser.add_argument("--n_views", type=int, default=20)
    parser.add_argument("--target_stride", type=int, default=10)
    parser.add_argument("--eval_targets", type=int, default=4)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def _circular_distance(a, b, num):
    return np.abs((a - b + num // 2) % num - num // 2)


@torch.no_grad()
def warp_sources(projections, vecs, target, inputs, device):
    height, width = projections.shape[-2:]
    warps = []
    for source in inputs:
        homography = geometry.homography_source_to_target(
            vecs[source:source + 1], vecs[target:target + 1], height, width
        )
        warps.append(
            geometry.warp_with_homography(
                projections[source].view(1, 1, height, width), homography
            )
        )
    return torch.cat(warps, dim=0)


def evaluate_split(dataset, args, device):
    totals = {name: {"psnr": [], "ssim": []} for name in ("nearest", "mean20", "median20")}
    for index in range(min(args.limit, len(dataset))):
        sample = dataset[index]
        projections = sample["proj"].to(device)
        vecs = sample["vecs"].to(device)
        height, width = projections.shape[-2:]
        num_frames = projections.shape[0]
        inputs = select_input_indices(num_frames, args.n_views)
        candidates = select_target_indices(num_frames, args.target_stride, exclude=inputs)
        if candidates.size == 0:
            continue
        picks = np.linspace(0, candidates.size - 1, min(args.eval_targets, candidates.size))
        targets = np.unique(candidates[np.round(picks).astype(int)])

        for target in targets:
            truth = projections[target].view(1, 1, height, width)
            warps = warp_sources(projections, vecs, target, inputs, device)
            nearest = inputs[np.argmin(_circular_distance(inputs, target, num_frames))]
            homography = geometry.homography_source_to_target(
                vecs[nearest:nearest + 1], vecs[target:target + 1], height, width
            )
            nearest_warp = geometry.warp_with_homography(
                projections[nearest].view(1, 1, height, width), homography
            )
            predictions = {
                "nearest": nearest_warp,
                "mean20": warps.mean(dim=0, keepdim=True),
                "median20": warps.median(dim=0).values.view(1, 1, height, width),
            }
            for name, prediction in predictions.items():
                totals[name]["psnr"].append(psnr(prediction, truth))
                totals[name]["ssim"].append(ssim(prediction, truth))

    return {
        name: {
            "psnr": float(np.mean(values["psnr"])) if values["psnr"] else float("nan"),
            "ssim": float(np.mean(values["ssim"])) if values["ssim"] else float("nan"),
            "n": len(values["psnr"]),
        }
        for name, values in totals.items()
    }


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    report = {}
    for split in [name.strip() for name in args.splits.split(",") if name.strip()]:
        dataset = DentalProjectionDataset(
            data_root=args.data_root,
            split_file=args.split_file,
            split=split,
            image_size=args.image_size,
            cache_dir=args.cache_dir,
        )
        metrics = evaluate_split(dataset, args, device)
        report[split] = metrics
        print(f"--- {split} ({min(args.limit, len(dataset))} subjects) ---")
        for name, value in metrics.items():
            print(f"  {name:9s} psnr={value['psnr']:6.2f} dB   ssim={value['ssim']:.4f}  (n={value['n']})")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[warp-baseline] -> {args.output_json}")
    return report


if __name__ == "__main__":
    main()
