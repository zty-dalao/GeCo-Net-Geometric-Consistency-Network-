"""Inference: 20 measured views -> all remaining views -> weighted FDK -> files.

    python -m submodel.quaternion_view_synthesis.infer --limit 1 --n_synth 340

For every subject of the chosen split:

1. take the 20 measured views (every 18 degrees);
2. synthesise every remaining angle (``--n_synth 0`` means all of them, i.e. 340)
   with the quaternion synthesiser and the quaternion softmax;
3. run weighted FDK over the 20 real + N synthetic views;
4. write the reconstruction as ``.nii.gz`` (plus an FDK-20 baseline) and,
   optionally, a slice figure.

Split choice: ``test`` (20 subjects) is the reporting split.  ``visual``
(2 subjects) is a *subset of test* and only exists for figures - never report
numbers on it.  ``val`` is unused by ``train.py`` and should stay reserved for
model selection.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from . import fdk as fdk_module
from .dataset import DentalProjectionDataset, load_volume_xyz, select_input_indices
from .model import QuaternionSoftmaxEnsemble, build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Quaternion view synthesis inference + FDK")
    parser.add_argument("--ckpt",
                        default="submodel/quaternion_view_synthesis/checkpoints/qvs_dental/ckpt_latest.pt")
    parser.add_argument("--data_root", default="dataset/dental/syn_data")
    parser.add_argument("--split_file", default="data/dataset_split/dental_split.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "visual"])
    parser.add_argument("--cache_dir", default="dataset/dental/_nvs_cache")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n_views", type=int, default=20)
    parser.add_argument("--n_synth", type=int, default=0,
                        help="0 (default) = synthesise every remaining angle")
    parser.add_argument("--synth_weight", type=float, default=0.25,
                        help="FDK weight of a synthetic view relative to a measured one")
    parser.add_argument("--fdk_res", type=int, default=64)
    parser.add_argument("--out_dir", default="submodel/quaternion_view_synthesis/infer")
    parser.add_argument("--run_name", default="qvs_dental")
    parser.add_argument("--save_projections", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--with_dense", action="store_true",
                        help="also reconstruct from the same number of *real* views (ceiling)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


@torch.no_grad()
def synthesize(model, ensemble, projections, vecs, source_indices, target_indices, log=None):
    source_views = projections[source_indices].unsqueeze(1)   # [V,1,h,w]
    vec_s = vecs[source_indices]                              # [V,12]
    outputs = []
    for step, target in enumerate(target_indices):
        vec_t = vecs[target].expand_as(vec_s)
        output = model(source_views, vec_s, vec_t)
        fused = ensemble(
            output["out"].unsqueeze(1),
            output["q_rel"].unsqueeze(1),
            output["confidence"].unsqueeze(1),
        )
        outputs.append(fused.squeeze(1).squeeze(0))
        if log is not None and (step + 1) % 50 == 0:
            log(f"  synthesised {step + 1}/{len(target_indices)}")
    return torch.stack(outputs, dim=0)


def save_volume(path, volume_xyz, meta, resolution):
    """Write an ``[X, Y, Z]`` reconstruction as a NIfTI volume in world space."""
    array = volume_xyz.detach().cpu().numpy().transpose(2, 1, 0)  # -> [Z, Y, X]
    image = sitk.GetImageFromArray(np.ascontiguousarray(array))
    spacing = [float(meta["volume_phy"][axis]) / resolution[axis] for axis in range(3)]
    image.SetSpacing(spacing)
    image.SetOrigin([float(v) for v in meta["volume_origin"]])
    sitk.WriteImage(image, str(path))


def save_slices(path, volumes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(volumes)
    figure, axes = plt.subplots(3, len(names), figsize=(3.2 * len(names), 9.0))
    for column, name in enumerate(names):
        volume = volumes[name]
        planes = (
            volume[:, :, volume.shape[2] // 2],
            volume[:, volume.shape[1] // 2, :],
            volume[volume.shape[0] // 2, :, :],
        )
        for row, plane in enumerate(planes):
            axis = axes[row, column]
            axis.imshow(plane.detach().cpu().numpy(), cmap="gray")
            axis.set_title(name if row == 0 else "")
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    model = build_model(train_args.get("base_channels", 16), train_args.get("levels", 2)).to(device)
    ensemble = QuaternionSoftmaxEnsemble(train_args.get("angle_scale", 4.0)).to(device)
    model.load_state_dict(checkpoint["model"])
    ensemble.load_state_dict(checkpoint["ensemble"])
    model.eval()
    ensemble.eval()

    dataset = DentalProjectionDataset(
        data_root=args.data_root,
        split_file=args.split_file,
        split=args.split,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
        limit=args.limit,
    )

    out_dir = Path(args.out_dir) / f"{args.run_name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = (args.fdk_res, args.fdk_res, args.fdk_res)
    report = {"split": args.split, "n_views": args.n_views, "subjects": {}}

    for subject_index in range(len(dataset)):
        sample = dataset[subject_index]
        subject = sample["subject"]
        projections = sample["proj"].to(device)
        vecs = sample["vecs"].to(device)
        meta = sample["meta"]
        num_frames = projections.shape[0]

        source_indices = select_input_indices(num_frames, args.n_views)
        mask = np.ones(num_frames, dtype=bool)
        mask[source_indices] = False
        target_indices = np.flatnonzero(mask)
        if args.n_synth:
            target_indices = target_indices[: args.n_synth]

        print(f"[qvs-infer] {subject}: {source_indices.size} measured -> "
              f"{target_indices.size} synthesised -> FDK {resolution}")

        synth = synthesize(
            model, ensemble, projections, vecs, source_indices, target_indices,
            log=print,
        )

        real_proj = projections[source_indices]
        real_vecs = vecs[source_indices]
        rec_20 = fdk_module.reconstruct(
            real_proj, real_vecs, meta["volume_origin"], meta["volume_phy"], resolution,
            view_weights=torch.ones(real_proj.shape[0], device=device),
        )
        rec_fused = fdk_module.reconstruct(
            torch.cat((real_proj, synth), dim=0),
            torch.cat((real_vecs, vecs[target_indices]), dim=0),
            meta["volume_origin"], meta["volume_phy"], resolution,
            view_weights=torch.cat((
                torch.ones(real_proj.shape[0], device=device),
                torch.full((synth.shape[0],), args.synth_weight, device=device),
            )),
        )

        save_volume(out_dir / f"{subject}_fdk_fused.nii.gz", rec_fused, meta, resolution)
        save_volume(out_dir / f"{subject}_fdk_20.nii.gz", rec_20, meta, resolution)

        entry = {"n_synth": int(target_indices.size), "fdk_res": list(resolution)}
        truth_path = Path(args.data_root) / subject / "gt_volume.nii.gz"
        if truth_path.is_file():
            truth = load_volume_xyz(truth_path, size=resolution).to(device)[0, 0]
            data_range = float(truth.max() - truth.min())
            for name, reconstruction in (("fdk_20", rec_20), ("fdk_fused", rec_fused)):
                scale = fdk_module.fit_scale(reconstruction, truth)
                entry[name] = {
                    "psnr": fdk_module.psnr(reconstruction * scale, truth, data_range),
                    "rmse": float((reconstruction * scale - truth).pow(2).mean().sqrt()),
                }
            if args.with_dense:
                dense = np.arange(0, num_frames, max(1, num_frames // target_indices.size))
                rec_dense = fdk_module.reconstruct(
                    projections[dense], vecs[dense], meta["volume_origin"], meta["volume_phy"],
                    resolution, view_weights=torch.ones(dense.size, device=device),
                )
                scale = fdk_module.fit_scale(rec_dense, truth)
                entry["fdk_dense"] = {
                    "psnr": fdk_module.psnr(rec_dense * scale, truth, data_range),
                    "rmse": float((rec_dense * scale - truth).pow(2).mean().sqrt()),
                }
        report["subjects"][subject] = entry
        print(f"[qvs-infer] {subject}: " + " | ".join(
            f"{k} psnr={v['psnr']:.2f}" for k, v in entry.items() if isinstance(v, dict)
        ))

        if args.save_projections:
            stack = torch.cat((real_proj, synth), dim=0).detach().cpu().numpy()
            sitk.WriteImage(
                sitk.GetImageFromArray(np.ascontiguousarray(stack)),
                str(out_dir / f"{subject}_synthetic_views.nii.gz"),
            )

        if args.save_png:
            volumes = {"FDK-20": rec_20, "FDK-fused": rec_fused}
            if truth_path.is_file():
                volumes = {"GT": truth, **volumes}
            save_slices(out_dir / f"{subject}_slices.png", volumes)

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[qvs-infer] volumes -> {out_dir}")
    return report


if __name__ == "__main__":
    main()
