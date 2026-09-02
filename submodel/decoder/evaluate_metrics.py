"""Evaluate a pretrained decoder on every validation/test dental case.

The reported PSNR uses one fixed physical range, rather than separately
min-max normalizing prediction and GT.  Region masks are defined by GT HU:
air < --air-upper-hu, tissue in between, and bone >= --bone-lower-hu.
"""

import argparse
import json
import math
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submodel.decoder.model import DecoderPretrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Pretraining checkpoint, usually ckpt_best_val.pt")
    parser.add_argument("--data-root", default="dataset/dental/syn_data")
    parser.add_argument("--split-file", default="data/dataset_split/dental_split.json")
    parser.add_argument("--conf", default="conf/train.conf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=("val", "test"))
    parser.add_argument("--air-upper-hu", type=float, default=-500.0)
    parser.add_argument("--bone-lower-hu", type=float, default=300.0)
    parser.add_argument("--output-dir", default=None, help="Default: submodel/decoder/metrics/<checkpoint-stem>")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_from_repo(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def make_autocast_context(enabled: bool):
    if not enabled:
        return nullcontext
    if hasattr(torch, "autocast"):
        return lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    return lambda: torch.cuda.amp.autocast(dtype=torch.float16)


def mu_to_hu(volume: torch.Tensor) -> torch.Tensor:
    return (volume / 0.022 - 1.0) * 1000.0


def _safe_psnr(mse: float) -> float:
    return math.inf if mse == 0.0 else -10.0 * math.log10(mse)


def metrics_for_case(
    prediction_mu: torch.Tensor,
    target_mu: torch.Tensor,
    clamp_min: float,
    clamp_max: float,
    air_upper_hu: float,
    bone_lower_hu: float,
) -> dict[str, float]:
    """Return fixed-range global and GT-mask regional reconstruction metrics."""
    prediction_mu = torch.clamp(prediction_mu.float(), clamp_min, clamp_max)
    target_mu = torch.clamp(target_mu.float(), clamp_min, clamp_max)
    normalized_prediction = (prediction_mu - clamp_min) / (clamp_max - clamp_min)
    normalized_target = (target_mu - clamp_min) / (clamp_max - clamp_min)
    hu_prediction = mu_to_hu(prediction_mu)
    hu_target = mu_to_hu(target_mu)

    absolute_error = torch.abs(normalized_prediction - normalized_target)
    squared_error = (normalized_prediction - normalized_target).square()
    absolute_error_hu = torch.abs(hu_prediction - hu_target)
    squared_error_hu = (hu_prediction - hu_target).square()
    global_mse = float(squared_error.mean().item())
    global_mae_hu = float(absolute_error_hu.mean().item())
    total_voxels = target_mu.numel()
    result = {
        "global_voxel_count": float(total_voxels),
        "global_l1_normalized": float(absolute_error.mean().item()),
        "global_mae_hu": global_mae_hu,
        "global_mse_normalized": global_mse,
        "global_rmse_hu": float(torch.sqrt(squared_error_hu.mean()).item()),
        "global_psnr_db": _safe_psnr(global_mse),
    }
    masks = {
        "air": hu_target < air_upper_hu,
        "tissue": (hu_target >= air_upper_hu) & (hu_target < bone_lower_hu),
        "bone": hu_target >= bone_lower_hu,
    }
    for name, mask in masks.items():
        count = int(mask.sum().item())
        prefix = f"{name}_"
        result[prefix + "voxel_count"] = float(count)
        result[prefix + "voxel_fraction"] = count / total_voxels
        if count == 0:
            for key in (
                "l1_normalized", "mae_hu", "mse_normalized", "rmse_hu",
                "region_psnr_db", "global_mse_contribution", "mse_share",
                "oracle_psnr_gain_db",
            ):
                result[prefix + key] = math.nan
            continue
        regional_mse = float(squared_error[mask].mean().item())
        contribution = float(squared_error[mask].sum().item() / total_voxels)
        remaining_mse = max(global_mse - contribution, 0.0)
        if global_mse == 0.0:
            share = 0.0
            oracle_gain = 0.0
        else:
            share = contribution / global_mse
            oracle_gain = (
                math.inf if remaining_mse == 0.0
                else 10.0 * math.log10(global_mse / remaining_mse)
            )
        result.update({
            prefix + "l1_normalized": float(absolute_error[mask].mean().item()),
            prefix + "mae_hu": float(absolute_error_hu[mask].mean().item()),
            prefix + "mse_normalized": regional_mse,
            prefix + "rmse_hu": float(torch.sqrt(squared_error_hu[mask].mean()).item()),
            prefix + "region_psnr_db": _safe_psnr(regional_mse),
            prefix + "global_mse_contribution": contribution,
            prefix + "mse_share": share,
            prefix + "oracle_psnr_gain_db": oracle_gain,
        })
    return result


def summarise(case_metrics: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = sorted({
        key for record in case_metrics for key, value in record.items()
        if key != "subject" and isinstance(value, (float, int))
    })
    mean, variance, std = {}, {}, {}
    for key in numeric_keys:
        values = np.asarray([float(record[key]) for record in case_metrics], dtype=np.float64)
        values = values[np.isfinite(values)]
        mean[key] = float(values.mean()) if values.size else math.nan
        variance[key] = float(values.var(ddof=0)) if values.size else math.nan
        std[key] = float(values.std(ddof=0)) if values.size else math.nan
    return {"cases": len(case_metrics), "mean": mean, "variance": variance, "std": std}


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def print_split_summary(split: str, summary: dict[str, object]) -> None:
    mean = summary["mean"]
    variance = summary["variance"]
    print(
        f"{split}: cases={summary['cases']}, "
        f"PSNR={mean['global_psnr_db']:.4f}±{summary['std']['global_psnr_db']:.4f} dB, "
        f"MAE={mean['global_mae_hu']:.2f} HU"
    )
    for region in ("air", "tissue", "bone"):
        print(
            f"  {region}: MSE share={100 * mean[region + '_mse_share']:.2f}% "
            f"(variance={variance[region + '_mse_share']:.6g}), "
            f"MAE={mean[region + '_mae_hu']:.2f} HU, "
            f"oracle gain={mean[region + '_oracle_psnr_gain_db']:.3f} dB"
        )


def main():
    args = parse_args()
    # Keep numerical helper functions importable in environments that do not
    # have SimpleITK, while requiring it only for actual dataset evaluation.
    from submodel.decoder.dataset import DentalVolumeDataset
    from pyhocon import ConfigFactory

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.bone_lower_hu <= args.air_upper_hu:
        raise ValueError("--bone-lower-hu must be greater than --air-upper-hu")
    checkpoint_path = resolve_from_repo(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    conf = ConfigFactory.parse_file(str(resolve_from_repo(args.conf)))
    data_root = resolve_from_repo(args.data_root)
    split_file = resolve_from_repo(args.split_file)
    clamp_min = conf.get_float("data.dental.clamp_min")
    clamp_max = conf.get_float("data.dental.clamp_max")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir = (
        resolve_from_repo(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parent / "metrics" / checkpoint_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DecoderPretrainer(conf["model.SRGAN.generator"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'model' state: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    autocast_context = make_autocast_context(device.type == "cuda" and not args.no_amp)

    run_info = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch"),
        "data_root": str(data_root),
        "splits": list(args.splits),
        "fixed_mu_range": [clamp_min, clamp_max],
        "fixed_hu_range": [
            float((clamp_min / 0.022 - 1.0) * 1000.0),
            float((clamp_max / 0.022 - 1.0) * 1000.0),
        ],
        "air_upper_hu": args.air_upper_hu,
        "bone_lower_hu": args.bone_lower_hu,
    }
    all_summaries = {}
    with torch.no_grad():
        for split in args.splits:
            dataset = DentalVolumeDataset(
                data_root=str(data_root), split_file=str(split_file), split=split,
                clamp_min=clamp_min, clamp_max=clamp_max,
            )
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=device.type == "cuda",
            )
            records: list[dict[str, object]] = []
            for batch in loader:
                volume = batch["volume"].to(device, non_blocking=True)
                with autocast_context():
                    prediction, _, _ = model(volume)
                for index, subject in enumerate(batch["subject"]):
                    record: dict[str, object] = {"subject": str(subject)}
                    record.update(metrics_for_case(
                        prediction[index], volume[index], clamp_min, clamp_max,
                        args.air_upper_hu, args.bone_lower_hu,
                    ))
                    records.append(record)
            summary = summarise(records)
            all_summaries[split] = summary
            write_jsonl(output_dir / f"{split}_per_case.jsonl", records)
            with (output_dir / f"{split}_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
            print_split_summary(split, summary)

    run_info["summaries"] = all_summaries
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(f"Per-case and split summary metrics saved to: {output_dir}")


if __name__ == "__main__":
    main()
