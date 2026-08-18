"""Training script for the projection interpolation model (real data, Fig. 3).

Sampling strategy (per case, per epoch):

    * pick ``rotors_per_case`` rotors; each rotor is ``num_inputs`` equidistant
      views over 360 deg;
    * for each rotor, pick one target view inside each angular interval between
      consecutive rotor views (=> ``num_inputs`` targets per rotor);
    * total = ``rotors_per_case * num_inputs`` steps per case per epoch.

Loss: ``1 - SSIM + alpha * L_VGG`` (VGG19 perceptual loss); the view-selection
consistency loss is optional (--consistency).

Logging: TensorBoard.  Configuration is read from ``config.json``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from fig3 import InterpolationModel
from loss import InterpolationLoss, VGGPerceptualLoss, consistency_loss
from dataloader_projection import (
    ProjectionInterpolationDataset,
    _wrap_pi,
    compute_zscore_stats,
    sample_rotor_targets,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_root(cfg: dict, data_name: str) -> str:
    """Pick the first existing data root for ``data_name``.

    ``cfg["data_roots"]`` maps a data name to a list of candidate paths
    (``~`` is expanded).  Falls back to ``cfg["data_root"]`` if present.
    """
    candidates = []
    roots = cfg.get("data_roots", {})
    if isinstance(roots, dict) and data_name in roots:
        val = roots[data_name]
        candidates = [val] if isinstance(val, str) else list(val)
    if not candidates and cfg.get("data_root"):
        candidates = [cfg["data_root"]]

    for c in candidates:
        c = os.path.expanduser(str(c))
        if os.path.isdir(c):
            return c
    return os.path.expanduser(str(candidates[0])) if candidates else ""


def _gib(nbytes: int) -> float:
    return nbytes / (1024 ** 3)


def nearest_source_idx(angles: np.ndarray, target_idx: int, num_inputs: int) -> np.ndarray:
    """Nearest ``num_inputs`` views around ``target_idx`` (a local view set)."""
    d = np.abs(_wrap_pi(angles - angles[target_idx]))
    d[target_idx] = np.inf
    return np.argsort(d)[:num_inputs]


def rotate_view_indices(src_idx, num_views: int, delta: float) -> np.ndarray:
    """Rotate a set of (rotor) view indices by ``delta`` views, mod ``num_views``.

    Used for the rotation-consistency branch: the SAME equidistant 360-deg
    rotor is rotated by a small angular offset ``delta`` so that both branches
    keep the identical geometric structure (``num_inputs`` views spanning
    360 deg) but with a different phase, while predicting the same target
    angle.  Returns the rotated indices sorted (ascending).
    """
    return sorted({int(v) % num_views for v in (np.asarray(src_idx, dtype=np.int64) + delta)})


def find_latest_checkpoint(ckpt_dir: str):
    """Return the path of the newest ``interpolation_epoch*.pt`` in ``ckpt_dir``."""
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith("interpolation_epoch") and f.endswith(".pt")]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split("epoch")[1].split(".")[0]))
    return os.path.join(ckpt_dir, files[-1])


def run_eval(model, ds_eval, criterion, device, use_amp, use_cos_sin,
             num_inputs, rotors_per_case, use_consistency, val_range,
             window_size, lambda_consistency, rng) -> dict:
    """Evaluate the model on the eval set and return per-epoch average losses.

    Uses the same 6x6-style rotor sampling as training (deterministic via the
    caller-supplied ``rng``), runs under ``torch.no_grad()`` and averages all
    loss components over the whole eval epoch.  Returns a dict with ``total``,
    ``ssim``, ``vgg``, (``consistency`` if enabled) and ``n`` (samples seen).
    """
    model.eval()
    total_acc = ssim_acc = vgg_acc = cons_acc = 0.0
    n = 0
    for case in ds_eval.cases:
        data = ds_eval.load_case(case)
        projs = data["projs"]                      # (K, H, W) float32, raw
        angles = data["angles"]                    # (K,)
        k = int(projs.shape[0])
        for _ in range(rotors_per_case):
            src_idx, tgt_idx = sample_rotor_targets(k, num_inputs, rng)
            src = projs[src_idx]
            src_angles = angles[src_idx]
            src_list = [
                torch.from_numpy(np.ascontiguousarray(src[i]))
                .unsqueeze(0).unsqueeze(0).to(device)
                for i in range(len(src_idx))
            ]
            for t in tgt_idx:
                with torch.no_grad():
                    target_angle = float(angles[t])
                    angular = _wrap_pi(target_angle - src_angles).astype(np.float32)
                    if use_cos_sin:
                        angular = np.concatenate([np.cos(angular) - 1.0, np.sin(angular)])
                    ang = torch.from_numpy(angular).unsqueeze(0).to(device)

                    target = torch.from_numpy(np.ascontiguousarray(projs[t])) \
                        .unsqueeze(0).unsqueeze(0).to(device)

                    pred_b = None
                    if use_consistency:
                        # rotation consistency: same rotor, small phase offset;
                        # re-sample delta until branch B excludes the target.
                        delta = rng.uniform(1.0, max(k / (2.0 * num_inputs), 2.0))
                        src_b_idx = rotate_view_indices(src_idx, k, delta)
                        attempt = 0
                        while int(t) in src_b_idx and attempt < 20:
                            delta = rng.uniform(1.0, max(k / (2.0 * num_inputs), 2.0))
                            src_b_idx = rotate_view_indices(src_idx, k, delta)
                            attempt += 1
                        src_b = projs[src_b_idx]
                        src_b_angles = angles[src_b_idx]
                        src_b_list = [
                            torch.from_numpy(np.ascontiguousarray(src_b[i]))
                            .unsqueeze(0).unsqueeze(0).to(device)
                            for i in range(len(src_b_idx))
                        ]
                        angular_b = _wrap_pi(target_angle - src_b_angles).astype(np.float32)
                        if use_cos_sin:
                            angular_b = np.concatenate(
                                [np.cos(angular_b) - 1.0, np.sin(angular_b)])
                        ang_b = torch.from_numpy(angular_b).unsqueeze(0).to(device)

                    with torch.autocast(device_type="cuda", dtype=torch.float16,
                                        enabled=use_amp):
                        pred_a = model(src_list, ang)
                        if use_consistency:
                            pred_b = model(src_b_list, ang_b)

                    l_ssim, l_vgg, sup_loss = criterion.compute(
                        pred_a.float(), target.float())
                    loss = sup_loss
                    if use_consistency:
                        l_cons = consistency_loss(pred_a.float(), pred_b.float(),
                                                  val_range=val_range,
                                                  window_size=window_size)
                        loss = loss + lambda_consistency * l_cons

                total_acc += float(loss.item())
                ssim_acc += float(l_ssim.item())
                vgg_acc += float(l_vgg.item())
                if use_consistency:
                    cons_acc += float(l_cons.item())
                n += 1

    model.train()
    res = {"total": total_acc / max(n, 1),
           "ssim": ssim_acc / max(n, 1),
           "vgg": vgg_acc / max(n, 1),
           "n": n}
    if use_consistency:
        res["consistency"] = cons_acc / max(n, 1)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the projection interpolation model (Fig. 3).")
    parser.add_argument("--config",
                        default=os.path.join(PROJECT_ROOT, "config", "config.json"))
    parser.add_argument("--version", default="v0",
                        help="run version tag embedded in the log dir name")
    parser.add_argument("--final_view", type=int, default=None,
                        help="number of input projections (overrides config model.in_channels)")
    parser.add_argument("--data_name", default=None,
                        help="dataset/organ name (thorax, head, ...); overrides config data_name")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="enable mixed precision")
    parser.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--alpha", type=float, default=None,
                        help="VGG perceptual loss weight (default from config loss.alpha)")
    parser.add_argument("--consistency", action="store_true",
                        help="enable the view-selection consistency loss (optional)")
    parser.add_argument("--lambda_consistency", type=float, default=None,
                        help="consistency loss weight (default from config loss.lambda_consistency)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--clip_grad", type=float, default=None,
                        help="gradient norm clipping threshold (0 disables; "
                             "default from config training.clip_grad_norm)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the latest checkpoint in the run's checkpoints dir")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="stop after N optimizer steps (smoke test)")
    parser.add_argument("--max_cases", type=int, default=None,
                        help="limit cases per epoch (smoke test)")
    parser.add_argument("--min_views", type=int, default=450,
                        help="drop cases with fewer than N projections "
                             "(skips missing/sparse-view cases; default from config training.min_views)")
    parser.add_argument("--eval_split", default=None,
                        help="split to evaluate on each epoch (default config eval_split / 'eval'; "
                             "use empty string to disable eval)")
    parser.add_argument("--test_split", default=None,
                        help="split to evaluate on each epoch as test (default config test_split / 'test'; "
                             "use empty string to disable test)")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="run eval every N epochs (default config logging.eval_every_epochs / 1)")
    parser.add_argument("--eval_rotors", type=int, default=None,
                        help="rotors per case during eval (default = rotors_per_case)")
    parser.add_argument("--eval_max_cases", type=int, default=None,
                        help="limit eval cases (smoke test)")
    parser.add_argument("--zscore_max_cases", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log_root", default=os.path.join(PROJECT_ROOT, "logs"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_name = args.data_name or cfg.get("data_name", "thorax")
    root = resolve_data_root(cfg, data_name)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    loss_cfg = cfg.get("loss", {})
    log_cfg = cfg.get("logging", {})

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = bool(train_cfg.get("amp", False))
    if args.amp:
        use_amp = True
    if args.no_amp:
        use_amp = False
    if use_amp and device != "cuda":
        use_amp = False

    num_inputs = args.final_view if args.final_view is not None else int(model_cfg["in_channels"])
    use_cos_sin = bool(train_cfg.get("use_cos_sin", True))
    angular_dim = 2 * num_inputs if use_cos_sin else num_inputs

    seed = int(train_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---------------- run dirs (before resume so we can locate checkpoints) ----------------
    run_dir = os.path.join(args.log_root, f"{args.version}_{data_name}_fv{num_inputs}")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    min_views = (args.min_views if args.min_views is not None
                 else train_cfg.get("min_views"))
    ds = ProjectionInterpolationDataset(root, split=cfg.get("split", "train"),
                                        num_inputs=num_inputs, min_views=min_views)

    # ---------------- val / test datasets (optional, same min_views) ------
    eval_split = args.eval_split if args.eval_split is not None \
        else cfg.get("eval_split", "eval")
    test_split = args.test_split if args.test_split is not None \
        else cfg.get("test_split", "test")
    ds_eval = None
    if eval_split:
        ds_eval = ProjectionInterpolationDataset(
            root, split=eval_split, num_inputs=num_inputs, min_views=min_views)
        if args.eval_max_cases is not None:
            ds_eval.cases = ds_eval.cases[: args.eval_max_cases]
    ds_test = None
    if test_split:
        ds_test = ProjectionInterpolationDataset(
            root, split=test_split, num_inputs=num_inputs, min_views=min_views)
        if args.eval_max_cases is not None:
            ds_test.cases = ds_test.cases[: args.eval_max_cases]

    eval_every = (args.eval_every if args.eval_every is not None
                  else int(log_cfg.get("eval_every_epochs", 1)))
    eval_rotors = (args.eval_rotors if args.eval_rotors is not None
                   else int(train_cfg.get("rotors_per_case", 6)))
    eval_rng = np.random.default_rng(seed + 1000)   # deterministic val sampling
    test_rng = np.random.default_rng(seed + 2000)   # deterministic test sampling

    # ---------------- resume or fresh z-score stats ----------------
    start_epoch = 1
    global_step = 0
    resume_ckpt = None
    if args.resume:
        ckpt_path = find_latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"no checkpoint found to resume in {ckpt_dir}")
        resume_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        mean = float(resume_ckpt["zscore_mean"])
        std = float(resume_ckpt["zscore_std"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        print(f"[resume] {os.path.basename(ckpt_path)} | start_epoch={start_epoch} "
              f"global_step={global_step} | z-score mean={mean:.4f} std={std:.4f}")
    else:
        zmax = args.zscore_max_cases if args.zscore_max_cases is not None else train_cfg.get("zscore_max_cases", 16)
        mean, std = compute_zscore_stats(ds, max_cases=zmax)
        print(f"[z-score] mean={mean:.4f} std={std:.4f}")

    # ---------------- model ----------------
    model = InterpolationModel(
        in_channels=num_inputs,
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        num_down=int(model_cfg.get("num_down", 4)),
        num_regions=int(model_cfg.get("num_regions", 4)),
        angular_dim=angular_dim,
        zscore_mean=mean,
        zscore_std=std,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scaler is not None and resume_ckpt.get("scaler") is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])

    # ---------------- loss -------------
    val_range = float(loss_cfg.get("val_range", 255.0))
    window_size = int(loss_cfg.get("ssim_window", 11))
    alpha = args.alpha if args.alpha is not None else float(loss_cfg.get("alpha", 1e-2))
    lambda_consistency = (args.lambda_consistency if args.lambda_consistency is not None
                          else float(loss_cfg.get("lambda_consistency", 1.0)))
    use_consistency = args.consistency
    clip_grad = (args.clip_grad if args.clip_grad is not None
                 else float(train_cfg.get("clip_grad_norm", 0.0)))

    vgg = VGGPerceptualLoss(
        feature_layers=int(loss_cfg.get("vgg_feature_layers", 9)),
        input_range=val_range,
        weights=loss_cfg.get("vgg_weights", "IMAGENET1K_V1"),
    ).to(device)
    criterion = InterpolationLoss(alpha=alpha, val_range=val_range,
                                  window_size=window_size, vgg=vgg)

    # ---------------- logging ----------------
    writer = SummaryWriter(run_dir) if SummaryWriter is not None else None
    print(f"[run] version={args.version} | data={data_name} | final_view={num_inputs} "
          f"| amp={use_amp} | consistency={use_consistency} "
          f"| clip_grad={clip_grad} | resume={resume_ckpt is not None} | log_dir={run_dir}")

    log_every = int(log_cfg.get("log_every_steps", 50))
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 300))
    rotors_per_case = int(train_cfg.get("rotors_per_case", 6))

    cases = ds.cases if args.max_cases is None else ds.cases[: args.max_cases]

    print(f"data_root={root}")
    print(f"cases={len(cases)} views/case={ds.views_per_case} "
          f"steps/case={rotors_per_case * num_inputs} device={device}")
    if ds_eval is not None:
        print(f"[val] split={eval_split} cases={len(ds_eval.cases)} "
              f"every={eval_every} epoch(s) rotors/case={eval_rotors} "
              f"steps/val_epoch={eval_rotors * num_inputs * len(ds_eval.cases)}")
    if ds_test is not None:
        print(f"[test] split={test_split} cases={len(ds_test.cases)} "
              f"every={eval_every} epoch(s) rotors/case={eval_rotors} "
              f"steps/test_epoch={eval_rotors * num_inputs * len(ds_test.cases)}")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    stop = False
    for epoch in range(start_epoch, epochs + 1):
        if stop:
            break
        model.train()
        order = rng.permutation(len(cases))
        epoch_loss = 0.0
        epoch_n = 0
        sum_ssim = 0.0
        sum_vgg = 0.0
        sum_cons = 0.0

        for case_i in order:
            if stop:
                break
            case = cases[int(case_i)]
            data = ds.load_case(case)
            projs = data["projs"]            # (K, H, W) float32, raw range
            angles = data["angles"]          # (K,)
            k = int(projs.shape[0])

            for _ in range(rotors_per_case):
                if stop:
                    break
                src_idx, tgt_idx = sample_rotor_targets(k, num_inputs, rng)
                src = projs[src_idx]                     # (N, H, W)
                src_angles = angles[src_idx]             # (N,)
                src_list = [
                    torch.from_numpy(np.ascontiguousarray(src[i]))
                    .unsqueeze(0).unsqueeze(0).to(device)
                    for i in range(len(src_idx))
                ]

                for t in tgt_idx:
                    target_angle = float(angles[t])
                    angular = _wrap_pi(target_angle - src_angles).astype(np.float32)
                    if use_cos_sin:
                        angular = np.concatenate([np.cos(angular) - 1.0, np.sin(angular)])
                    ang = torch.from_numpy(angular).unsqueeze(0).to(device)  # (1, angular_dim)

                    target = torch.from_numpy(np.ascontiguousarray(projs[t])) \
                        .unsqueeze(0).unsqueeze(0).to(device)                 # (1, 1, H, W)

                    # second branch (rotation consistency): the SAME rotor
                    # (num_inputs equidistant views over 360 deg) rotated by a
                    # small angular offset, predicting the SAME target angle.
                    pred_b = None
                    if use_consistency:
                        # re-sample delta until branch B does NOT contain the
                        # target view itself (avoids trivial consistency where
                        # branch B directly "sees" the answer).
                        delta = rng.uniform(1.0, max(k / (2.0 * num_inputs), 2.0))
                        src_b_idx = rotate_view_indices(src_idx, k, delta)
                        attempt = 0
                        while int(t) in src_b_idx and attempt < 20:
                            delta = rng.uniform(1.0, max(k / (2.0 * num_inputs), 2.0))
                            src_b_idx = rotate_view_indices(src_idx, k, delta)
                            attempt += 1
                        src_b = projs[src_b_idx]
                        src_b_angles = angles[src_b_idx]
                        src_b_list = [
                            torch.from_numpy(np.ascontiguousarray(src_b[i]))
                            .unsqueeze(0).unsqueeze(0).to(device)
                            for i in range(len(src_b_idx))
                        ]
                        angular_b = _wrap_pi(target_angle - src_b_angles).astype(np.float32)
                        if use_cos_sin:
                            angular_b = np.concatenate([np.cos(angular_b) - 1.0, np.sin(angular_b)])
                        ang_b = torch.from_numpy(angular_b).unsqueeze(0).to(device)

                    with torch.autocast(device_type="cuda", dtype=torch.float16,
                                        enabled=use_amp):
                        pred_a = model(src_list, ang)                        # (1, 1, H, W)
                        if use_consistency:
                            pred_b = model(src_b_list, ang_b)

                    # SSIM / VGG / consistency in fp32 (0..255 squares overflow fp16).
                    l_ssim, l_vgg, sup_loss = criterion.compute(pred_a.float(), target.float())
                    loss = sup_loss
                    if use_consistency:
                        l_cons = consistency_loss(pred_a.float(), pred_b.float(),
                                                  val_range=val_range, window_size=window_size)
                        loss = loss + lambda_consistency * l_cons

                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if clip_grad > 0.0:
                            # Unscale before clipping so the norm threshold
                            # applies to the true (unscaled) gradients.
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if clip_grad > 0.0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), clip_grad)
                        optimizer.step()

                    global_step += 1
                    epoch_loss += loss.item()
                    epoch_n += 1
                    sum_ssim += l_ssim.item()
                    sum_vgg += l_vgg.item()
                    if use_consistency:
                        sum_cons += l_cons.item()

                    if global_step % log_every == 0 or global_step == 1:
                        alloc = _gib(torch.cuda.memory_allocated()) if device == "cuda" else 0.0
                        peak = _gib(torch.cuda.max_memory_allocated()) if device == "cuda" else 0.0
                        print(f"epoch {epoch}/{epochs} | step {global_step} | "
                              f"loss {loss.item():.4f} (ssim {l_ssim.item():.4f}, "
                              f"vgg {l_vgg.item():.4f}) | gpu {alloc:.2f}G (peak {peak:.2f}G) | "
                              f"lr {optimizer.param_groups[0]['lr']:.2e}")
                        if writer is not None:
                            writer.add_scalar("train/total", loss.item(), global_step)
                            writer.add_scalar("train/ssim", l_ssim.item(), global_step)
                            writer.add_scalar("train/vgg", l_vgg.item(), global_step)
                            if use_consistency:
                                writer.add_scalar("train/consistency", l_cons.item(), global_step)
                            if device == "cuda":
                                writer.add_scalar("gpu/memory_allocated_GB", alloc, global_step)
                                writer.add_scalar("gpu/max_memory_GB", peak, global_step)

                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop = True
                        break

        avg = epoch_loss / max(epoch_n, 1)
        print(f"== epoch {epoch}/{epochs} | avg loss {avg:.4f} ==")
        if writer is not None:
            writer.add_scalar("train/total_epoch", avg, epoch)
            writer.add_scalar("train/ssim_epoch", sum_ssim / max(epoch_n, 1), epoch)
            writer.add_scalar("train/vgg_epoch", sum_vgg / max(epoch_n, 1), epoch)
            if use_consistency:
                writer.add_scalar("train/consistency_epoch", sum_cons / max(epoch_n, 1), epoch)

        # ---------------- validation (per-epoch average) ----------------
        if ds_eval is not None and (epoch % eval_every == 0 or stop):
            er = run_eval(
                model, ds_eval, criterion, device, use_amp, use_cos_sin,
                num_inputs, eval_rotors, use_consistency, val_range,
                window_size, lambda_consistency, eval_rng)
            cons_str = f", consistency {er.get('consistency', float('nan')):.4f}" \
                if use_consistency else ""
            print(f"== [val] epoch {epoch}/{epochs} | avg loss {er['total']:.4f} "
                  f"(ssim {er['ssim']:.4f}, vgg {er['vgg']:.4f})"
                  f"{cons_str} | n={er['n']} ==")
            if writer is not None:
                writer.add_scalar("val/total_epoch", er["total"], epoch)
                writer.add_scalar("val/ssim_epoch", er["ssim"], epoch)
                writer.add_scalar("val/vgg_epoch", er["vgg"], epoch)
                if use_consistency:
                    writer.add_scalar("val/consistency_epoch",
                                      er["consistency"], epoch)

        # ---------------- test (per-epoch average) -----------------------
        if ds_test is not None and (epoch % eval_every == 0 or stop):
            tr = run_eval(
                model, ds_test, criterion, device, use_amp, use_cos_sin,
                num_inputs, eval_rotors, use_consistency, val_range,
                window_size, lambda_consistency, test_rng)
            cons_str = f", consistency {tr.get('consistency', float('nan')):.4f}" \
                if use_consistency else ""
            print(f"== [test] epoch {epoch}/{epochs} | avg loss {tr['total']:.4f} "
                  f"(ssim {tr['ssim']:.4f}, vgg {tr['vgg']:.4f})"
                  f"{cons_str} | n={tr['n']} ==")
            if writer is not None:
                writer.add_scalar("test/total_epoch", tr["total"], epoch)
                writer.add_scalar("test/ssim_epoch", tr["ssim"], epoch)
                writer.add_scalar("test/vgg_epoch", tr["vgg"], epoch)
                if use_consistency:
                    writer.add_scalar("test/consistency_epoch",
                                      tr["consistency"], epoch)

        if epoch % int(log_cfg.get("checkpoint_every_epochs", 10)) == 0 or stop:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "zscore_mean": mean,
                    "zscore_std": std,
                    "config": cfg,
                    "final_view": num_inputs,
                },
                os.path.join(ckpt_dir, f"interpolation_epoch{epoch}.pt"),
            )

    if device == "cuda":
        print(f"[gpu] peak allocated during training: {_gib(torch.cuda.max_memory_allocated()):.2f} GB")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
