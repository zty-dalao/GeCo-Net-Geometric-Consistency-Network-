"""Analyze reconstruction errors in air-, tissue-, and bone-like GT regions.

Example (decomposes the PSNR convention currently used by this project):

    python analyze_region_metrics.py \
        --prediction evaluate/visuals/EXP/.../volume/volume_predict.nii.gz \
        --ground-truth evaluate/visuals/EXP/.../volume/volume_gt.nii.gz

The three masks are defined by normalized ground truth: [0, 1/3),
[1/3, 2/3), and [2/3, 1].  PSNR is not additive, so the script reports each
region's share of global MSE and the hypothetical global PSNR gain if that
region's error alone were completely removed.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def minmax(array: np.ndarray) -> np.ndarray:
    minimum = float(array.min())
    value_range = float(array.max()) - minimum
    if value_range <= 0.0:
        raise ValueError("Cannot min-max normalize a constant volume")
    return (array - minimum) / value_range


def normalize_volumes(prediction, ground_truth, mode, value_min, value_max):
    if mode == "project":
        # Exactly matches data_norm(pred) and data_norm(gt) in evaluate.py.
        return minmax(prediction), minmax(ground_truth)
    if mode == "shared_gt":
        minimum = float(ground_truth.min())
        maximum = float(ground_truth.max())
    elif mode == "fixed":
        if value_min is None or value_max is None:
            raise ValueError("--normalization fixed requires --value-min and --value-max")
        minimum, maximum = float(value_min), float(value_max)
    else:  # none: input volumes are already normalized to [0, 1]
        return prediction, ground_truth

    if maximum <= minimum:
        raise ValueError("Normalization maximum must be greater than minimum")
    prediction = np.clip((prediction - minimum) / (maximum - minimum), 0.0, 1.0)
    ground_truth = np.clip((ground_truth - minimum) / (maximum - minimum), 0.0, 1.0)
    return prediction, ground_truth


def psnr_from_mse(mse: float) -> float:
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def analyze(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    thresholds=(1.0 / 3.0, 2.0 / 3.0),
) -> dict:
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Prediction and GT shapes differ: {prediction.shape} versus {ground_truth.shape}"
        )

    absolute_error = np.abs(prediction - ground_truth)
    squared_error = (prediction - ground_truth) ** 2
    total_voxels = ground_truth.size
    global_l1 = float(absolute_error.mean())
    global_mse = float(squared_error.mean())
    result = {
        "global": {
            "voxel_count": int(total_voxels),
            "l1": global_l1,
            "mse_l2": global_mse,
            "rmse": math.sqrt(global_mse),
            "psnr_db": psnr_from_mse(global_mse),
        },
        "regions": {},
    }

    low, high = float(thresholds[0]), float(thresholds[1])
    if not 0.0 < low < high < 1.0:
        raise ValueError("Thresholds must satisfy 0 < LOW < HIGH < 1")
    regions = (
        ("air", 0.0, low),
        ("tissue", low, high),
        ("bone", high, 1.0),
    )
    for index, (name, lower, upper) in enumerate(regions):
        # Include 1.0 in the last interval to cover every normalized GT voxel.
        mask = ((ground_truth >= lower) & (ground_truth < upper)) if index < 2 else ground_truth >= lower
        count = int(mask.sum())
        if count == 0:
            result["regions"][name] = {
                "normalized_gt_range": [lower, upper],
                "voxel_count": 0,
                "voxel_fraction": 0.0,
                "l1": math.nan,
                "mse_l2": math.nan,
                "rmse": math.nan,
                "region_psnr_db": math.nan,
                "global_mse_contribution": 0.0,
                "global_mse_share": 0.0,
                "oracle_psnr_gain_db": 0.0,
            }
            continue

        region_l1 = float(absolute_error[mask].mean())
        region_mse = float(squared_error[mask].mean())
        mse_contribution = float(squared_error[mask].sum() / total_voxels)
        mse_share = 0.0 if global_mse == 0.0 else mse_contribution / global_mse
        remaining_mse = max(global_mse - mse_contribution, 0.0)
        if global_mse == 0.0:
            oracle_gain = 0.0
        elif remaining_mse == 0.0:
            oracle_gain = math.inf
        else:
            oracle_gain = 10.0 * math.log10(global_mse / remaining_mse)

        result["regions"][name] = {
            "normalized_gt_range": [lower, upper],
            "voxel_count": count,
            "voxel_fraction": count / total_voxels,
            "l1": region_l1,
            "mse_l2": region_mse,
            "rmse": math.sqrt(region_mse),
            "region_psnr_db": psnr_from_mse(region_mse),
            "global_mse_contribution": mse_contribution,
            "global_mse_share": mse_share,
            "oracle_psnr_gain_db": oracle_gain,
        }

    contribution_sum = sum(
        region["global_mse_contribution"] for region in result["regions"].values()
    )
    result["mse_contribution_sum"] = contribution_sum
    return result


def format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    return f"{value:.8g}"


def print_result(result: dict, normalization: str) -> None:
    global_metrics = result["global"]
    print(f"Normalization: {normalization}")
    print(
        "Global: "
        f"L1={format_number(global_metrics['l1'])}, "
        f"MSE(L2)={format_number(global_metrics['mse_l2'])}, "
        f"RMSE={format_number(global_metrics['rmse'])}, "
        f"PSNR={format_number(global_metrics['psnr_db'])} dB"
    )
    print()
    print("region   GT range       voxels%      L1          MSE(L2)     MSE share%   oracle gain")
    for name, metrics in result["regions"].items():
        lower, upper = metrics["normalized_gt_range"]
        print(
            f"{name:<8} [{lower:.3f},{upper:.3f}] "
            f"{100.0 * metrics['voxel_fraction']:>9.3f}  "
            f"{format_number(metrics['l1']):>11} "
            f"{format_number(metrics['mse_l2']):>11} "
            f"{100.0 * metrics['global_mse_share']:>11.3f} "
            f"{format_number(metrics['oracle_psnr_gain_db']):>10} dB"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", "-p", required=True, type=Path, help="Predicted .nii/.nii.gz")
    parser.add_argument("--ground-truth", "-g", required=True, type=Path, help="Ground-truth .nii/.nii.gz")
    parser.add_argument(
        "--normalization",
        choices=("project", "shared_gt", "fixed", "none"),
        default="project",
        help=(
            "project: separately min-max each volume (matches current evaluate.py); "
            "shared_gt: normalize both with GT min/max; fixed: use --value-min/max; "
            "none: inputs are already [0,1]"
        ),
    )
    parser.add_argument("--value-min", type=float, default=None, help="Fixed physical/HU minimum")
    parser.add_argument("--value-max", type=float, default=None, help="Fixed physical/HU maximum")
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional lower clipping bound applied before normalization",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Optional upper clipping bound applied before normalization",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        default=(1.0 / 3.0, 2.0 / 3.0),
        metavar=("LOW", "HIGH"),
        help="Two thresholds in normalized GT; default: equal thirds",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional path for full JSON output")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise SystemExit(
            "SimpleITK is required to read NIfTI files. Install the project's requirements first."
        ) from error
    prediction_image = sitk.ReadImage(str(args.prediction))
    ground_truth_image = sitk.ReadImage(str(args.ground_truth))
    prediction = sitk.GetArrayFromImage(prediction_image).astype(np.float64)
    ground_truth = sitk.GetArrayFromImage(ground_truth_image).astype(np.float64)
    if (args.clip_min is None) != (args.clip_max is None):
        raise SystemExit("Use --clip-min and --clip-max together")
    if args.clip_min is not None:
        if args.clip_max <= args.clip_min:
            raise SystemExit("--clip-max must be greater than --clip-min")
        prediction = np.clip(prediction, args.clip_min, args.clip_max)
        ground_truth = np.clip(ground_truth, args.clip_min, args.clip_max)
    prediction, ground_truth = normalize_volumes(
        prediction,
        ground_truth,
        args.normalization,
        args.value_min,
        args.value_max,
    )
    result = analyze(prediction, ground_truth, thresholds=args.thresholds)
    result["normalization"] = args.normalization
    result["prediction"] = str(args.prediction)
    result["ground_truth"] = str(args.ground_truth)
    print_result(result, args.normalization)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
        print(f"\nJSON saved to: {args.json}")


if __name__ == "__main__":
    main()
