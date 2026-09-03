import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submodel.decoder.dataset import DentalVolumeDataset  # noqa: E402
from submodel.decoder.loss import (  # noqa: E402
    bone_gt_mask_l1,
    gradient1_loss_3d,
    soft_tissue_gt_mask_l1,
)
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Checkpoint to continue from. When supplied, --epochs means the "
            "number of additional epochs, not the total epoch count."
        ),
    )
    parser.add_argument("--gd1-lambda", type=float, default=None)
    parser.add_argument("--mse-lambda-3d", type=float, default=None)
    parser.add_argument(
        "--bone-lambda",
        type=float,
        default=0.0,
        help="Weight of the GT-defined bone-region normalized L1 loss; 0 preserves legacy training.",
    )
    parser.add_argument(
        "--bone-lower-hu",
        type=float,
        default=300.0,
        help="GT HU threshold: voxels at or above this value are bone for --bone-lambda.",
    )
    parser.add_argument(
        "--soft-mask-lambda",
        type=float,
        default=0.0,
        help="Weight of the GT-defined soft-tissue-mask L1 loss; 0 preserves legacy training.",
    )
    parser.add_argument("--soft-window-low", type=float, default=-160.0)
    parser.add_argument("--soft-window-high", type=float, default=240.0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many training seconds; 0 disables the limit.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Use at most this many training subjects; intended for smoke tests.",
    )
    parser.add_argument(
        "--eval-limit", type=int, default=None,
        help="Use at most this many subjects from each validation/test split.",
    )
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=10)
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
    scaler: Any,
    epoch: int,
    step: int,
    elapsed_seconds: float,
    best_val_loss: float,
    loss_config: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "elapsed_seconds": elapsed_seconds,
            "best_val_loss": best_val_loss,
            "loss_config": loss_config,
            "model": model.state_dict(),
            # This key loads directly into models.model.decoder.
            "decoder": model.decoder.state_dict(),
            "feature_stem": model.feature_stem.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def append_jsonl(path: Path, metrics: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, ensure_ascii=False) + os.linesep)


def make_loader(
    data_root: Path,
    split_file: Path,
    split: str,
    clamp_min: float,
    clamp_max: float,
    limit: int | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = DentalVolumeDataset(
        data_root=str(data_root),
        split_file=str(split_file),
        split=split,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        limit=limit,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def calculate_losses(
    model: DecoderPretrainer,
    volume: torch.Tensor,
    l1_loss: torch.nn.Module,
    mse_lambda_3d: float,
    gd1_lambda: float,
    bone_lambda: float,
    bone_lower_hu: float,
    soft_mask_lambda: float,
    soft_window_low: float,
    soft_window_high: float,
    clamp_min: float,
    clamp_max: float,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
]:
    prediction, low_resolution, latent = model(volume)
    loss_3d = l1_loss(prediction, volume) * mse_lambda_3d
    loss_gd1 = gradient1_loss_3d(volume, prediction, l1_loss) * gd1_lambda
    zero = loss_3d.new_zeros(())

    # Masks come only from GT. Crucially, prediction is not clamped before the
    # error is measured, so a prediction outside either region still receives
    # a gradient that pulls it back toward the GT HU value.
    if bone_lambda > 0:
        bone_raw = bone_gt_mask_l1(
            prediction, volume, bone_lower_hu, clamp_min, clamp_max,
        )
        loss_bone = bone_raw * bone_lambda
    else:
        bone_raw = zero
        loss_bone = zero

    if soft_mask_lambda > 0:
        soft_raw = soft_tissue_gt_mask_l1(
            prediction, volume, soft_window_low, soft_window_high,
        )
        loss_soft_mask = soft_raw * soft_mask_lambda
    else:
        soft_raw = zero
        loss_soft_mask = zero

    loss_total = loss_3d + loss_gd1 + loss_bone + loss_soft_mask
    return (
        loss_total, loss_3d, loss_gd1, loss_bone, loss_soft_mask,
        bone_raw, soft_raw, prediction, low_resolution, latent,
    )


@torch.no_grad()
def evaluate(
    split: str,
    model: DecoderPretrainer,
    loader: DataLoader,
    device: torch.device,
    autocast_context: Callable,
    l1_loss: torch.nn.Module,
    mse_lambda_3d: float,
    gd1_lambda: float,
    bone_lambda: float,
    bone_lower_hu: float,
    soft_mask_lambda: float,
    soft_window_low: float,
    soft_window_high: float,
    clamp_min: float,
    clamp_max: float,
) -> dict[str, float | int | str]:
    model.eval()
    sums = {
        "loss": 0.0, "loss_3d": 0.0, "loss_gd1": 0.0,
        "loss_bone": 0.0, "loss_soft_mask": 0.0,
        "bone_raw": 0.0, "soft_mask_raw": 0.0,
    }
    sample_count = 0
    for batch in loader:
        volume = batch["volume"].to(device, non_blocking=True)
        with autocast_context():
            loss, loss_3d, loss_gd1, loss_bone, loss_soft_mask, bone_raw, soft_raw, _, _, _ = calculate_losses(
                model, volume, l1_loss, mse_lambda_3d, gd1_lambda,
                bone_lambda, bone_lower_hu, soft_mask_lambda,
                soft_window_low, soft_window_high, clamp_min, clamp_max,
            )
        current_batch_size = volume.shape[0]
        sample_count += current_batch_size
        sums["loss"] += float(loss) * current_batch_size
        sums["loss_3d"] += float(loss_3d) * current_batch_size
        sums["loss_gd1"] += float(loss_gd1) * current_batch_size
        sums["loss_bone"] += float(loss_bone) * current_batch_size
        sums["loss_soft_mask"] += float(loss_soft_mask) * current_batch_size
        sums["bone_raw"] += float(bone_raw) * current_batch_size
        sums["soft_mask_raw"] += float(soft_raw) * current_batch_size

    model.train()
    if sample_count == 0:
        raise RuntimeError(f"The {split} loader produced no samples.")
    return {
        "split": split,
        "samples": sample_count,
        "loss": sums["loss"] / sample_count,
        "loss_3d": sums["loss_3d"] / sample_count,
        "loss_gd1": sums["loss_gd1"] / sample_count,
        "loss_bone": sums["loss_bone"] / sample_count,
        "loss_soft_mask": sums["loss_soft_mask"] / sample_count,
        "bone_raw": sums["bone_raw"] / sample_count,
        "soft_mask_raw": sums["soft_mask_raw"] / sample_count,
    }


def write_epoch_tensorboard(
    writer: SummaryWriter,
    split: str,
    metrics: dict[str, float | int | str],
    epoch_number: int,
) -> None:
    writer.add_scalar(f"epoch/{split}_l1_voxel", metrics["loss_3d"], epoch_number)
    writer.add_scalar(f"epoch/{split}_gradient1", metrics["loss_gd1"], epoch_number)
    writer.add_scalar(f"epoch/{split}_bone_gt_mask", metrics["loss_bone"], epoch_number)
    writer.add_scalar(f"epoch/{split}_bone_gt_mask_raw", metrics["bone_raw"], epoch_number)
    writer.add_scalar(f"epoch/{split}_soft_mask", metrics["loss_soft_mask"], epoch_number)
    writer.add_scalar(f"epoch/{split}_soft_mask_raw", metrics["soft_mask_raw"], epoch_number)
    writer.add_scalar(f"epoch/{split}_total", metrics["loss"], epoch_number)


def make_grad_scaler(enabled: bool):
    """Use the current AMP API when available, with a PyTorch 2.1 fallback."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_autocast_context(enabled: bool) -> Callable:
    if not enabled:
        return nullcontext
    if hasattr(torch, "autocast"):
        return lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    return lambda: torch.cuda.amp.autocast(dtype=torch.float16)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}.")
    if args.val_every < 1 or args.test_every < 1 or args.save_every < 1:
        raise ValueError("val-every, test-every and save-every must all be positive.")
    if args.bone_lambda < 0 or args.soft_mask_lambda < 0:
        raise ValueError("--bone-lambda and --soft-mask-lambda must be non-negative.")
    if args.soft_window_high <= args.soft_window_low:
        raise ValueError("--soft-window-high must be greater than --soft-window-low.")

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
    loss_config = {
        "mse_lambda_3d": float(mse_lambda_3d),
        "gd1_lambda": float(gd1_lambda),
        "bone_lambda": float(args.bone_lambda),
        "bone_lower_hu": float(args.bone_lower_hu),
        "soft_mask_lambda": float(args.soft_mask_lambda),
        "soft_window_low": float(args.soft_window_low),
        "soft_window_high": float(args.soft_window_high),
    }

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_root = Path(args.output_root)
    log_dir = output_root / "logs" / args.run_name
    checkpoint_dir = output_root / "checkpoints" / args.run_name
    tensorboard_dir = log_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    train_metrics_path = log_dir / "train.jsonl"
    epoch_metrics_path = log_dir / "epoch.jsonl"
    summary_path = log_dir / "summary.json"

    common_loader_args = {
        "data_root": data_root,
        "split_file": split_file,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = make_loader(split="train", limit=args.limit, **common_loader_args)
    val_loader = make_loader(split="val", limit=args.eval_limit, **common_loader_args)
    test_loader = make_loader(split="test", limit=args.eval_limit, **common_loader_args)

    model = DecoderPretrainer(conf["model.SRGAN.generator"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = torch.nn.L1Loss(reduction="mean")
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = make_grad_scaler(amp_enabled)
    autocast_context = make_autocast_context(amp_enabled)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    start_epoch = 0
    step = 0
    best_val_loss = float("inf")
    previous_elapsed_seconds = 0.0
    resume_path = None
    if args.resume is not None:
        resume_path = resolve_from_repo(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if "model" not in checkpoint:
            raise KeyError(f"Resume checkpoint has no 'model' state: {resume_path}")
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        # A user-specified --lr deliberately overrides the stored Adam LR.
        if args.lr is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = lr
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        step = int(checkpoint.get("step", 0))
        saved_loss_config = checkpoint.get("loss_config")
        if saved_loss_config is not None and saved_loss_config == loss_config:
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        elif saved_loss_config is not None or args.bone_lambda > 0 or args.soft_mask_lambda > 0:
            # An old checkpoint's validation objective does not include the
            # newly enabled regional losses, hence it is not comparable.
            best_val_loss = float("inf")
            print(
                "Loss configuration changed; resetting best validation loss for the new objective.",
                flush=True,
            )
        else:
            best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        previous_elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        print(
            f"Resumed {resume_path} after epoch {start_epoch}; "
            f"running {args.epochs} additional epochs at lr={optimizer.param_groups[0]['lr']}",
            flush=True,
        )

    report = parameter_report(model)
    run_config = {
        **vars(args),
        **report,
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root),
        "split_file": str(split_file),
        "tensorboard_dir": str(tensorboard_dir),
        "train_subjects": len(train_loader.dataset),
        "val_subjects": len(val_loader.dataset),
        "test_subjects": len(test_loader.dataset),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
        "lr": lr,
        "loss_config": loss_config,
        "amp_enabled": amp_enabled,
        "resume_path": None if resume_path is None else str(resume_path),
        "start_epoch": start_epoch,
        "additional_epochs": args.epochs,
        "created_at": datetime.now().isoformat(),
    }
    with (log_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)
    writer.add_text("run/config", json.dumps(run_config, ensure_ascii=False, indent=2), 0)
    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.monotonic()
    stop_requested = False
    last_epoch = start_epoch - 1
    last_metrics: dict[str, object] = {}
    end_epoch = start_epoch + args.epochs

    try:
        for epoch in range(start_epoch, end_epoch):
            last_epoch = epoch
            epoch_number = epoch + 1
            train_sums = {
                "loss": 0.0, "loss_3d": 0.0, "loss_gd1": 0.0,
                "loss_bone": 0.0, "loss_soft_mask": 0.0,
                "bone_raw": 0.0, "soft_mask_raw": 0.0,
            }
            train_samples = 0

            for batch in train_loader:
                volume = batch["volume"].to(device, non_blocking=True)
                subjects = list(batch["subject"])
                optimizer.zero_grad(set_to_none=True)
                with autocast_context():
                    (
                        loss, loss_3d, loss_gd1, loss_bone, loss_soft_mask,
                        bone_raw, soft_raw, prediction, low_resolution, latent,
                    ) = calculate_losses(
                        model, volume, l1_loss, mse_lambda_3d, gd1_lambda,
                        args.bone_lambda, args.bone_lower_hu, args.soft_mask_lambda,
                        args.soft_window_low, args.soft_window_high, clamp_min, clamp_max,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                step += 1
                current_batch_size = volume.shape[0]
                train_samples += current_batch_size
                train_sums["loss"] += float(loss.detach()) * current_batch_size
                train_sums["loss_3d"] += float(loss_3d.detach()) * current_batch_size
                train_sums["loss_gd1"] += float(loss_gd1.detach()) * current_batch_size
                train_sums["loss_bone"] += float(loss_bone.detach()) * current_batch_size
                train_sums["loss_soft_mask"] += float(loss_soft_mask.detach()) * current_batch_size
                train_sums["bone_raw"] += float(bone_raw.detach()) * current_batch_size
                train_sums["soft_mask_raw"] += float(soft_raw.detach()) * current_batch_size
                elapsed = previous_elapsed_seconds + time.monotonic() - start_time
                if device.type == "cuda":
                    peak_allocated_gib = torch.cuda.max_memory_allocated(device) / 1024**3
                    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 1024**3
                else:
                    peak_allocated_gib = 0.0
                    peak_reserved_gib = 0.0

                last_metrics = {
                    "timestamp": datetime.now().isoformat(), "epoch": epoch, "step": step,
                    "subjects": subjects, "elapsed_seconds": elapsed,
                    "loss": float(loss.detach()), "loss_3d": float(loss_3d.detach()),
                    "loss_gd1": float(loss_gd1.detach()), "input_shape": list(volume.shape),
                    "loss_bone_gt_mask": float(loss_bone.detach()),
                    "bone_gt_mask_raw": float(bone_raw.detach()),
                    "loss_soft_mask": float(loss_soft_mask.detach()),
                    "soft_mask_raw": float(soft_raw.detach()),
                    "low_resolution_shape": list(low_resolution.shape),
                    "latent_shape": list(latent.shape), "prediction_shape": list(prediction.shape),
                    "peak_allocated_gib": peak_allocated_gib,
                    "peak_reserved_gib": peak_reserved_gib,
                }
                append_jsonl(train_metrics_path, last_metrics)
                print(json.dumps(last_metrics, ensure_ascii=False), flush=True)
                writer.add_scalar("train_step/l1_voxel", loss_3d, step)
                writer.add_scalar("train_step/gradient1", loss_gd1, step)
                writer.add_scalar("train_step/bone_gt_mask", loss_bone, step)
                writer.add_scalar("train_step/bone_gt_mask_raw", bone_raw, step)
                writer.add_scalar("train_step/soft_mask", loss_soft_mask, step)
                writer.add_scalar("train_step/soft_mask_raw", soft_raw, step)
                writer.add_scalar("train_step/total", loss, step)
                writer.add_scalar("train_step/learning_rate", optimizer.param_groups[0]["lr"], step)
                if device.type == "cuda":
                    writer.add_scalar("memory/peak_allocated_gib", peak_allocated_gib, step)
                    writer.add_scalar("memory/peak_reserved_gib", peak_reserved_gib, step)
                if args.max_seconds > 0 and elapsed >= args.max_seconds:
                    stop_requested = True
                    break

            if train_samples == 0:
                raise RuntimeError("The train loader produced no samples.")
            train_epoch_metrics = {
                "split": "train", "samples": train_samples,
                "loss": train_sums["loss"] / train_samples,
                "loss_3d": train_sums["loss_3d"] / train_samples,
                "loss_gd1": train_sums["loss_gd1"] / train_samples,
                "loss_bone": train_sums["loss_bone"] / train_samples,
                "loss_soft_mask": train_sums["loss_soft_mask"] / train_samples,
                "bone_raw": train_sums["bone_raw"] / train_samples,
                "soft_mask_raw": train_sums["soft_mask_raw"] / train_samples,
            }
            epoch_record: dict[str, object] = {
                "timestamp": datetime.now().isoformat(), "epoch": epoch,
                "train": train_epoch_metrics,
            }
            write_epoch_tensorboard(writer, "train", train_epoch_metrics, epoch_number)

            # Timed smoke tests stop promptly without adding lengthy eval passes.
            if not stop_requested and epoch_number % args.val_every == 0:
                val_metrics = evaluate(
                    "val", model, val_loader, device, autocast_context,
                    l1_loss, mse_lambda_3d, gd1_lambda,
                    args.bone_lambda, args.bone_lower_hu, args.soft_mask_lambda,
                    args.soft_window_low, args.soft_window_high, clamp_min, clamp_max,
                )
                epoch_record["val"] = val_metrics
                write_epoch_tensorboard(writer, "val", val_metrics, epoch_number)
                if float(val_metrics["loss"]) < best_val_loss:
                    best_val_loss = float(val_metrics["loss"])
                    save_checkpoint(
                        checkpoint_dir / "ckpt_best_val.pt", model, optimizer, scaler,
                        epoch, step, previous_elapsed_seconds + time.monotonic() - start_time,
                        best_val_loss, loss_config,
                    )

            should_test = (
                not stop_requested
                and (epoch_number % args.test_every == 0 or epoch == end_epoch - 1)
            )
            if should_test:
                test_metrics = evaluate(
                    "test", model, test_loader, device, autocast_context,
                    l1_loss, mse_lambda_3d, gd1_lambda,
                    args.bone_lambda, args.bone_lower_hu, args.soft_mask_lambda,
                    args.soft_window_low, args.soft_window_high, clamp_min, clamp_max,
                )
                epoch_record["test"] = test_metrics
                write_epoch_tensorboard(writer, "test", test_metrics, epoch_number)

            append_jsonl(epoch_metrics_path, epoch_record)
            print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
            writer.flush()
            elapsed = previous_elapsed_seconds + time.monotonic() - start_time
            save_checkpoint(
                checkpoint_dir / "ckpt_latest.pt", model, optimizer, scaler,
                epoch, step, elapsed, best_val_loss, loss_config,
            )
            if epoch_number % args.save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"ckpt_epoch_{epoch_number:04d}.pt",
                    model, optimizer, scaler, epoch, step, elapsed, best_val_loss, loss_config,
                )
            if stop_requested:
                break
    finally:
        writer.flush()
        writer.close()

    elapsed = previous_elapsed_seconds + time.monotonic() - start_time
    summary = {
        "completed": True, "stopped_by_time_limit": stop_requested,
        "epoch": last_epoch, "steps": step, "elapsed_seconds": elapsed,
        "latest_checkpoint": str(checkpoint_dir / "ckpt_latest.pt"),
        "best_val_checkpoint": str(checkpoint_dir / "ckpt_best_val.pt"),
        "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
        "tensorboard_dir": str(tensorboard_dir), **report, "last_metrics": last_metrics,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
