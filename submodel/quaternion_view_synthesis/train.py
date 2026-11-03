"""Train the quaternion view synthesiser on the dental projections.

    python -m submodel.quaternion_view_synthesis.train --smoke

One optimisation step uses one subject, ``n_views`` measured views and a few
random target angles.  For every target angle the network sees the 20 views as
a batch, and the 20 predictions are fused with the quaternion softmax.

PSNR / SSIM / L1 are measured in the **projection domain** on the train, val
and test splits every ``--eval_every`` epochs and written to TensorBoard:

    tensorboard --logdir submodel/quaternion_view_synthesis/checkpoints/<name>/tensorboard
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .dataset import (
    DentalProjectionDataset,
    select_input_indices,
    select_target_indices,
)
from .metrics import psnr, ssim
from .losses import elastic_net, parameter_elastic_net
from .model import QuaternionSoftmaxEnsemble, build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quaternion view synthesis training")
    parser.add_argument("--data_root", default="dataset/dental/syn_data")
    parser.add_argument("--split_file", default="data/dataset_split/dental_split.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--cache_dir", default="dataset/dental/_nvs_cache")
    parser.add_argument("--out_dir", default="submodel/quaternion_view_synthesis/checkpoints")
    parser.add_argument("--name", default="qvs_dental")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_views", type=int, default=20)
    parser.add_argument("--target_stride", type=int, default=10)
    parser.add_argument("--n_targets", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--angle_scale", type=float, default=4.0)
    parser.add_argument("--lambda_per_view", type=float, default=0.1)

    # Elastic Net regularisation
    parser.add_argument("--lambda_elastic_net", type=float, default=1e-2,
                        help="Elastic Net weight on the predicted residual (0 disables)")
    parser.add_argument("--elastic_l1_ratio", type=float, default=0.5,
                        help="1 = pure L1 (LASSO), 0 = pure L2 (ridge)")
    parser.add_argument("--lambda_elastic_weight", type=float, default=0.0,
                        help="optional Elastic Net on the parameters (weight decay)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1)

    # metric / TensorBoard options
    parser.add_argument("--eval_every", type=int, default=5,
                        help="score train/val/test every N epochs (0 disables)")
    parser.add_argument("--eval_limit", type=int, default=4,
                        help="subjects per split used for the metric estimate")
    parser.add_argument("--eval_targets", type=int, default=4,
                        help="fixed target angles per subject used for the metric estimate")
    parser.add_argument("--eval_splits", default="train,val,test",
                        help="comma separated splits scored every --eval_every epochs")
    parser.add_argument("--tensorboard_dir", default=None,
                        help="defaults to <out_dir>/<name>/tensorboard")
    parser.add_argument("--no_tensorboard", action="store_true")

    # checkpointing
    parser.add_argument("--save_every", type=int, default=5,
                        help="write ckpt_history/ckpt_XXXX.pt every N epochs (0 disables); "
                             "ckpt_latest.pt is always updated")
    parser.add_argument("--resume", default=None,
                        help="checkpoint to continue from (model + ensemble + optimizer + epoch)")

    parser.add_argument("--smoke", action="store_true", help="tiny run: 1 epoch, 2 subjects")
    args = parser.parse_args(argv)

    if args.smoke:
        args.limit = args.limit or 2
        args.epochs = 1
        args.n_targets = 1
        args.image_size = min(args.image_size, 128)
        args.eval_limit = min(args.eval_limit, 2)
        args.eval_targets = min(args.eval_targets, 2)
        # keep the smoke fast *and* observable: score every epoch, snapshot every epoch
        args.eval_every = 1
        args.save_every = 1
    return args


def optimizer_to(optimizer, device):
    """Move the Adam moment buffers to ``device`` after ``load_state_dict``."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_eval_sets(args):
    """One dataset per scored split; the training split is reused, not reloaded."""
    splits = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    eval_sets = {}
    for split in splits:
        if split == args.split:
            eval_sets[split] = DentalProjectionDataset(
                data_root=args.data_root,
                split_file=args.split_file,
                split=args.split,
                image_size=args.image_size,
                cache_dir=args.cache_dir,
                limit=args.limit,
            )
            continue
        try:
            eval_sets[split] = DentalProjectionDataset(
                data_root=args.data_root,
                split_file=args.split_file,
                split=split,
                image_size=args.image_size,
                cache_dir=args.cache_dir,
            )
        except RuntimeError as error:
            print(f"[qvs] skipping split '{split}': {error}")
    return eval_sets


@torch.no_grad()
def evaluate_split(model, ensemble, dataset, args, device):
    """Projection-domain PSNR / SSIM / L1 on a deterministic subset of a split.

    The subjects and target angles are fixed (``--eval_limit`` closest subjects,
    ``--eval_targets`` evenly spaced target angles), so the curves are directly
    comparable across epochs.
    """
    model.eval()
    ensemble.eval()
    totals = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0}
    count = 0

    for index in range(min(args.eval_limit, len(dataset))):
        sample = dataset[index]
        projections = sample["proj"].to(device)
        vecs = sample["vecs"].to(device)
        num_frames = projections.shape[0]
        inputs = select_input_indices(num_frames, args.n_views)
        candidates = select_target_indices(num_frames, args.target_stride, exclude=inputs)
        if candidates.size == 0:
            continue
        picks = np.linspace(0, candidates.size - 1, min(args.eval_targets, candidates.size))
        targets = np.unique(candidates[np.round(picks).astype(int)])

        source_views = projections[inputs].unsqueeze(1)
        vec_s = vecs[inputs]
        for target in targets:
            output = model(source_views, vec_s, vecs[target].expand_as(vec_s))
            fused = ensemble(
                output["out"].unsqueeze(1),
                output["q_rel"].unsqueeze(1),
                output["confidence"].unsqueeze(1),
            )
            truth = projections[target].view(1, 1, *projections.shape[-2:])
            totals["psnr"] += psnr(fused, truth)
            totals["ssim"] += ssim(fused, truth)
            totals["l1"] += F.l1_loss(fused, truth).item()
            count += 1

    model.train()
    ensemble.train()
    return {name: value / max(count, 1) for name, value in totals.items()}, count


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    dataset = DentalProjectionDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split=args.split,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )
    model = build_model(args.base_channels, args.levels).to(device)
    ensemble = QuaternionSoftmaxEnsemble(args.angle_scale).to(device)
    parameters = list(model.parameters()) + list(ensemble.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    start_epoch = 0
    steps_total = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        ensemble.load_state_dict(state["ensemble"])
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
            optimizer_to(optimizer, device)
        else:
            print("[qvs] resume: checkpoint has no optimizer state, Adam moments restart")
        start_epoch = int(state.get("epoch", -1)) + 1
        steps_total = int(state.get("steps_total", 0))
        if state.get("rng_state") is not None:
            torch.set_rng_state(state["rng_state"].cpu())
        if state.get("numpy_rng_state") is not None:
            np.random.set_state(state["numpy_rng_state"])
        recorded_name = state.get("args", {}).get("name")
        if recorded_name and recorded_name != args.name:
            print(f"[qvs] resume: checkpoint was written by name='{recorded_name}', "
                  f"but --name='{args.name}' -> a new output directory is used")
        print(f"[qvs] resumed from {args.resume} at epoch {start_epoch} "
              f"(steps_total={steps_total})")

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    num_trainable = sum(p.numel() for p in parameters)
    print(f"[qvs] subjects={len(dataset)} parameters={num_trainable/1e6:.3f}M device={device}")

    eval_sets = build_eval_sets(args) if args.eval_every > 0 else {}
    tb_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else out_dir / "tensorboard"
    writer = None if args.no_tensorboard else SummaryWriter(log_dir=str(tb_dir))
    if writer is not None:
        print(f"[qvs] tensorboard -> {tb_dir}")
        print(f"[qvs] view with: tensorboard --logdir {tb_dir}")
    if eval_sets:
        print(f"[qvs] scoring splits {list(eval_sets)} every {args.eval_every} epoch(s), "
              f"{args.eval_limit} subject(s) x {args.eval_targets} target(s) each")

    def record_metrics(epoch):
        if writer is not None:
            writer.add_scalar("train/kappa", float(ensemble.kappa.abs()), epoch)
            writer.add_scalar("train/lambda", float(ensemble.lam), epoch)
        for split, split_dataset in eval_sets.items():
            metrics, count = evaluate_split(model, ensemble, split_dataset, args, device)
            print(f"[qvs] epoch {epoch:3d} {split:5s} " + " ".join(
                f"{name}={value:.4f}" for name, value in metrics.items()
            ) + f" (n={count})")
            if writer is not None:
                for name, value in metrics.items():
                    writer.add_scalar(f"{split}/{name}", value, epoch)

    if eval_sets and start_epoch == 0:
        record_metrics(0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ensemble.train()
        epoch_start = time.time()
        running = 0.0
        steps = 0

        for subject_index in range(len(dataset)):
            sample = dataset[subject_index]
            projections = sample["proj"].to(device)
            vecs = sample["vecs"].to(device)
            num_frames = projections.shape[0]
            inputs = select_input_indices(num_frames, args.n_views)
            candidates = select_target_indices(num_frames, args.target_stride, exclude=inputs)
            picks = np.random.choice(candidates, size=min(args.n_targets, candidates.size), replace=False)

            for target in picks:
                source_views = projections[inputs].unsqueeze(1)          # [V,1,h,w]
                vec_s = vecs[inputs]                                    # [V,12]
                vec_t = vecs[target].expand_as(vec_s)                   # [V,12]
                ground_truth = projections[target].view(1, 1, *projections.shape[-2:])

                output = model(source_views, vec_s, vec_t)
                predictions = output["out"].unsqueeze(1)                # [V,1,1,h,w]
                fused, weights = ensemble(
                    predictions,
                    output["q_rel"].unsqueeze(1),
                    output["confidence"].unsqueeze(1),
                    return_weights=True,
                )

                loss_fused = F.l1_loss(fused, ground_truth)
                loss_view = F.l1_loss(
                    output["out"], ground_truth.expand_as(output["out"])
                )
                loss_elastic = elastic_net(output["residual"], args.elastic_l1_ratio)
                loss = loss_fused + args.lambda_per_view * loss_view
                loss = loss + args.lambda_elastic_net * loss_elastic
                if args.lambda_elastic_weight > 0.0:
                    loss = loss + args.lambda_elastic_weight * parameter_elastic_net(
                        parameters, args.elastic_l1_ratio
                    )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()

                running += float(loss_fused)
                steps += 1
                steps_total += 1

                if writer is not None:
                    writer.add_scalar("train/loss_fused", float(loss_fused), steps_total)
                    writer.add_scalar("train/loss_view", float(loss_view), steps_total)
                    writer.add_scalar("train/loss_elastic", float(loss_elastic), steps_total)
                    writer.add_scalar("train/loss_total", float(loss), steps_total)
                    writer.add_scalar(
                        "train/residual_mean", float(output["residual"].mean()), steps_total
                    )

                if steps % max(args.log_every, 1) == 0:
                    angle = torch.acos(output["q_rel"][:, 0].abs().clamp(max=1.0)) * 2 * 180 / np.pi
                    order = torch.argsort(angle)
                    print(
                        f"[qvs] epoch {epoch} step {steps} subject={sample['subject']} "
                        f"target={int(target)} loss={float(loss_fused):.4f} "
                        f"elastic={float(loss_elastic):.5f} "
                        f"residual_mean={float(output['residual'].mean()):.4f} "
                        f"max_weight_angle={float(angle[torch.argmax(weights[:, 0])]):.1f}deg "
                        f"nearest_angle={float(angle[order[0]]):.1f}deg"
                    )

        mean_loss = running / max(steps, 1)
        step_index = epoch + 1
        checkpoint = {
            "model": model.state_dict(),
            "ensemble": ensemble.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "steps_total": steps_total,
            "loss": mean_loss,
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        torch.save(checkpoint, out_dir / "ckpt_latest.pt")
        if args.save_every and step_index % args.save_every == 0:
            history_dir = out_dir / "ckpt_history"
            history_dir.mkdir(parents=True, exist_ok=True)
            snapshot = history_dir / f"ckpt_{step_index:04d}.pt"
            torch.save(checkpoint, snapshot)
            print(f"[qvs] saved snapshot {snapshot}")
        print(
            f"[qvs] epoch {epoch} done mean_fused_l1={mean_loss:.5f} "
            f"elapsed={time.time() - epoch_start:.1f}s ckpt={out_dir / 'ckpt_latest.pt'}"
        )

        if eval_sets and (epoch + 1) % max(args.eval_every, 1) == 0:
            record_metrics(epoch + 1)
        if writer is not None:
            writer.add_scalar("train/mean_fused_l1_per_epoch", mean_loss, epoch + 1)
            writer.flush()

    if writer is not None:
        writer.close()
        print(f"[qvs] tensorboard events written to {tb_dir}")


if __name__ == "__main__":
    main()
