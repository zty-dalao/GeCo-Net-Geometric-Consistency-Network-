import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submodel.decoder.dataset import DentalVolumeDataset  # noqa: E402
from submodel.decoder.loss import gradient1_loss_3d  # noqa: E402
from submodel.decoder.model import DecoderPretrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pretrain the original 3D decoder with degraded dental pCT volumes."
    )
    parser.add_argument("--data-root", default="dataset/dental/syn_data")
    parser.add_argument("--split-file", default="data/dataset_split/dental_split.json")
    parser.add_argument("--conf", default="conf/train.conf")
    parser.add_argument("--run-name", default="dental_pretrain")
    parser.add_argument("--output-root", default=str(base_dir))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gd1-lambda", type=float, default=None)
    parser.add_argument("--mse-lambda-3d", type=float, default=None)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many training seconds; 0 disables the limit.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_from_repo(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def parameter_report(model: DecoderPretrainer) -> dict[str, float | int]:
    decoder_parameters = sum(p.numel() for p in model.decoder.parameters())
    stem_parameters = sum(p.numel() for p in model.feature_stem.parameters())
    return {
        "decoder_parameters": decoder_parameters,
        "stem_parameters": stem_parameters,
        "total_parameters": decoder_parameters + stem_parameters,
        "decoder_parameter_gib_fp32": decoder_parameters * 4 / 1024**3,
    }


def save_checkpoint(
    path: Path,
    model: DecoderPretrainer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "elapsed_seconds": elapsed_seconds,
            "model": model.state_dict(),
            # This key can be loaded directly into models.model.decoder.
            "decoder": model.decoder.state_dict(),
            "feature_stem": model.feature_stem.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}.")

    conf_path = resolve_from_repo(args.conf)
    data_root = resolve_from_repo(args.data_root)
    split_file = resolve_from_repo(args.split_file)
    conf = ConfigFactory.parse_file(str(conf_path))

    clamp_min = conf.get_float("data.dental.clamp_min")
    clamp_max = conf.get_float("data.dental.clamp_max")
    lr = args.lr if args.lr is not None else conf.get_float("lr_sche.init_lr")
    mse_lambda_3d = (
        args.mse_lambda_3d
        if args.mse_lambda_3d is not None
        else conf.get_float("train.G_loss.mse_lambda_3d")
    )
    gd1_lambda = (
        args.gd1_lambda
        if args.gd1_lambda is not None
        else conf.get_float("train.G_loss.gd1_lambda")
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_root = Path(args.output_root)
    log_dir = output_root / "logs" / args.run_name
    checkpoint_dir = output_root / "checkpoints" / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "train.jsonl"
    summary_path = log_dir / "summary.json"

    dataset = DentalVolumeDataset(
        data_root=str(data_root),
        split_file=str(split_file),
        split="train",
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = DecoderPretrainer(conf["model.SRGAN.generator"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = torch.nn.L1Loss(reduction="mean")
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    autocast_context = (
        (lambda: torch.amp.autocast("cuda", dtype=torch.float16))
        if amp_enabled
        else nullcontext
    )

    report = parameter_report(model)
    run_config = {
        **vars(args),
        **report,
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root),
        "split_file": str(split_file),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "lr": lr,
        "mse_lambda_3d": mse_lambda_3d,
        "gd1_lambda": gd1_lambda,
        "amp_enabled": amp_enabled,
        "created_at": datetime.now().isoformat(),
    }
    with (log_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.monotonic()
    step = 0
    stop_requested = False
    last_epoch = 0
    last_metrics: dict[str, float | int | str] = {}

    for epoch in range(args.epochs):
        last_epoch = epoch
        for batch in loader:
            volume = batch["volume"].to(device, non_blocking=True)
            subjects = list(batch["subject"])

            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                prediction, low_resolution, latent = model(volume)
                # Same 3D L1 and first-gradient losses as the main project,
                # vectorized over the complete [B, C, X, Y, Z] batch.
                loss_3d = l1_loss(prediction, volume) * mse_lambda_3d
                loss_gd1 = gradient1_loss_3d(volume, prediction, l1_loss) * gd1_lambda
                loss = loss_3d + loss_gd1

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            elapsed = time.monotonic() - start_time
            if device.type == "cuda":
                peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
            else:
                peak_allocated_gib = 0.0
                peak_reserved_gib = 0.0

            last_metrics = {
                "timestamp": datetime.now().isoformat(),
                "epoch": epoch,
                "step": step,
                "subjects": subjects,
                "elapsed_seconds": elapsed,
                "loss": float(loss.detach()),
                "loss_3d": float(loss_3d.detach()),
                "loss_gd1": float(loss_gd1.detach()),
                "input_shape": list(volume.shape),
                "low_resolution_shape": list(low_resolution.shape),
                "latent_shape": list(latent.shape),
                "prediction_shape": list(prediction.shape),
                "peak_allocated_gib": peak_allocated_gib,
                "peak_reserved_gib": peak_reserved_gib,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_metrics, ensure_ascii=False) + os.linesep)
            print(json.dumps(last_metrics, ensure_ascii=False), flush=True)

            if args.max_seconds > 0 and elapsed >= args.max_seconds:
                stop_requested = True
                break

        elapsed = time.monotonic() - start_time
        save_checkpoint(
            checkpoint_dir / "ckpt_latest.pt",
            model,
            optimizer,
            scaler,
            epoch,
            step,
            elapsed,
        )
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"ckpt_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                step,
                elapsed,
            )
        if stop_requested:
            break

    elapsed = time.monotonic() - start_time
    summary = {
        "completed": True,
        "stopped_by_time_limit": stop_requested,
        "epoch": last_epoch,
        "steps": step,
        "elapsed_seconds": elapsed,
        "latest_checkpoint": str(checkpoint_dir / "ckpt_latest.pt"),
        **report,
        "last_metrics": last_metrics,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
