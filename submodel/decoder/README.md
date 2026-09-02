# Decoder pretraining

This submodel pretrains the project's original 3D decoder on dental pCT/CT
volumes. It uses a fixed `4x4x4` average-pooling degradation, a small
`1 -> 64 -> 256` feature stem, and the unchanged decoder from
`models/SRGAN.py`:

```text
[B, 1, 256, 256, 256]
  -> fixed average pool x4
[B, 1, 64, 64, 64]
  -> feature stem
[B, 256, 64, 64, 64]
  -> original decoder
[B, 1, 256, 256, 256]
```

The loss matches the main project's 3D supervision:

```text
total loss = mse_lambda_3d * voxel L1
           + gd1_lambda * first-gradient L1
           + bone_lambda * GT-bone-mask normalized L1
           + soft_mask_lambda * GT-soft-mask window-normalized L1
```

Despite its historical config name, `mse_lambda_3d` weights an L1 loss in this
repository. Both terms operate on the complete `[B, 1, X, Y, Z]` batch. The
projection-domain 2D loss is not used because decoder pretraining intentionally
loads only ground-truth volumes.

The two regional terms are disabled by default (`bone_lambda=0` and
`soft_mask_lambda=0`), preserving the previous objective and full checkpoint
compatibility. Their masks are defined **only from GT HU**. Predictions are not
clamped before these terms are calculated, so a predicted voxel outside the
target region still receives a gradient back toward its correct value.

```text
bone mask:       GT HU >= bone_lower_hu                 (default 300 HU)
bone raw loss:   mean_mask(|pred_mu - gt_mu| / (mu_max - mu_min))

soft mask:       soft_window_low <= GT HU < soft_window_high
soft raw loss:   mean_mask(|pred_HU - gt_HU| / window_width)
```

## Training

Install dependencies from the repository root. The requirements select a
PyTorch CUDA 12.8 build so that Blackwell/RTX 5090 is supported:

```bash
python -m pip install -r requirements.txt
```

Verify the environment before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

Run from the repository root. A suggested RTX 5090 command in **Linux Bash**
is shown below. Bash uses a backslash as the final character on each continued
line; do not put spaces after it.

```bash
python -m submodel.decoder.train \
  --run-name dental_batch3 \
  --device cuda \
  --batch-size 3 \
  --epochs 100 \
  --val-every 1 \
  --test-every 10 \
  --save-every 5
```

The equivalent command in **Windows PowerShell** uses backticks:

```powershell
python -m submodel.decoder.train `
  --run-name dental_batch3 `
  --device cuda `
  --batch-size 3 `
  --epochs 100 `
  --val-every 1 `
  --test-every 10 `
  --save-every 5
```

The most portable form is a single line:

```bash
python -m submodel.decoder.train --run-name dental_batch3 --device cuda --batch-size 3 --epochs 100 --val-every 1 --test-every 10 --save-every 5
```

For a small timed smoke test in Bash:

```bash
python -m submodel.decoder.train \
  --run-name dental_smoke_60s \
  --device cuda \
  --batch-size 1 \
  --limit 1 \
  --eval-limit 1 \
  --max-seconds 60
```

A timed run stops after the first completed training step that reaches the time
limit. It skips validation/test at that partial epoch so that a smoke test stops
promptly.

### Continue a finished run

`--resume` restores the complete pretraining model, Adam state, AMP scaler,
step counter, and best validation loss. When it is present, `--epochs` means
the number of **additional** epochs. For example, continue from the best
checkpoint for 50 epochs with a lower learning rate:

```bash
python -m submodel.decoder.train \
  --run-name dental_batch3_region_refine \
  --device cuda \
  --batch-size 3 \
  --resume submodel/decoder/checkpoints/dental_batch3/ckpt_best_val.pt \
  --epochs 50 \
  --lr 2e-5 \
  --bone-lambda 0.05 \
  --bone-lower-hu 300 \
  --soft-mask-lambda 0.01 \
  --soft-window-low -160 \
  --soft-window-high 240 \
  --val-every 1 \
  --test-every 10 \
  --save-every 5
```

Use `ckpt_latest.pt` instead of `ckpt_best_val.pt` only when the intent is to
continue exactly from the final epoch, rather than from the best validation
model. A new `--run-name` is recommended for a changed loss function so that
the original checkpoint and TensorBoard curves remain intact. When a resumed
checkpoint uses a different stored loss configuration, the best validation
loss is reset automatically because old and new total losses are not
comparable.

## Command-line parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `--data-root` | `dataset/dental/syn_data` | Root directory containing one folder per dental subject and its `gt_volume.nii.gz`. |
| `--split-file` | `data/dataset_split/dental_split.json` | JSON file defining the `train`, `val`, and `test` subject lists. |
| `--conf` | `conf/train.conf` | Main project configuration; supplies decoder architecture, dental clamp range, loss weights, and default learning rate. |
| `--run-name` | `dental_pretrain` | Experiment name used as the log and checkpoint subdirectory. Use a new name for a new run. |
| `--output-root` | `submodel/decoder` | Root under which `logs/<run-name>` and `checkpoints/<run-name>` are created. |
| `--device` | `cuda` | PyTorch device, for example `cuda`, `cuda:0`, or `cpu`. |
| `--epochs` | `500` | Maximum number of complete passes over the training split. A practical first run is 100 epochs. |
| `--batch-size` | `1` | Number of full 3D volumes processed together. Batch 3 is intended for the 32 GB GPU; AMP is enabled by default. |
| `--num-workers` | `0` | Number of DataLoader worker processes. On Windows, start with 0; increase only after verifying host RAM and I/O behavior. |
| `--lr` | config value (`1e-4`) | Adam learning rate. If omitted, uses `lr_sche.init_lr` from the config. |
| `--resume` | none | Checkpoint to restore. With this option, `--epochs` is the number of additional epochs. A supplied `--lr` replaces the learning rate stored in the checkpoint. |
| `--mse-lambda-3d` | config value (`1.0`) | Weight of the voxel-wise L1 loss. The name is retained for compatibility with the original project. |
| `--gd1-lambda` | config value (`1.0`) | Weight of the first-order spatial-gradient L1 loss. |
| `--bone-lambda` | `0` | Weight of the GT-defined bone-region loss. `0` exactly reproduces the old objective. Start with `0.05` for refinement. |
| `--bone-lower-hu` | `300` | GT threshold defining the bone mask. Prediction HU is never used to choose this mask. |
| `--soft-mask-lambda` | `0` | Weight of the GT-defined soft-tissue-mask loss. `0` exactly reproduces the old objective. Start with `0.01` for refinement. |
| `--soft-window-low` | `-160` | Lower HU bound of the GT soft-tissue mask. |
| `--soft-window-high` | `240` | Upper HU bound of the GT soft-tissue mask. |
| `--max-seconds` | `0` | Wall-clock training limit in seconds; `0` disables it. Intended for smoke tests, not normal training. |
| `--limit` | all | Maximum number of training subjects to use. Intended for smoke tests. |
| `--eval-limit` | all | Maximum number of subjects used from each of the validation and test splits. |
| `--val-every` | `1` | Run validation every N completed epochs. The lowest validation total loss is saved as `ckpt_best_val.pt`. |
| `--test-every` | `10` | Run test evaluation every N completed epochs and always on the final configured epoch. Validation should select checkpoints; test metrics are for periodic reporting. |
| `--save-every` | `10` | Save a numbered `ckpt_epoch_NNNN.pt` every N epochs. `ckpt_latest.pt` is updated every epoch. |
| `--no-amp` | off | Disable CUDA automatic mixed precision. This substantially increases memory use and is not recommended for full `256^3` volumes. |

## TensorBoard and text logs

TensorBoard events are written to:

```text
submodel/decoder/logs/<run-name>/tensorboard/
```

Launch TensorBoard from the repository root:

```powershell
tensorboard --logdir submodel/decoder/logs/dental_batch3/tensorboard --port 6006
```

The recorded scalars are:

```text
train_step/l1_voxel
train_step/gradient1
train_step/bone_gt_mask
train_step/bone_gt_mask_raw
train_step/soft_mask
train_step/soft_mask_raw
train_step/total
train_step/learning_rate

epoch/train_l1_voxel
epoch/train_gradient1
epoch/train_bone_gt_mask
epoch/train_bone_gt_mask_raw
epoch/train_soft_mask
epoch/train_soft_mask_raw
epoch/train_total
epoch/val_l1_voxel
epoch/val_gradient1
epoch/val_bone_gt_mask
epoch/val_bone_gt_mask_raw
epoch/val_soft_mask
epoch/val_soft_mask_raw
epoch/val_total
epoch/test_l1_voxel
epoch/test_gradient1
epoch/test_bone_gt_mask
epoch/test_bone_gt_mask_raw
epoch/test_soft_mask
epoch/test_soft_mask_raw
epoch/test_total

memory/peak_allocated_gib
memory/peak_reserved_gib
```

Per-step training records remain available in `train.jsonl`. Epoch-level train,
validation, and test averages are written together to `epoch.jsonl`. The
averages are weighted by the actual number of samples, so a shorter final batch
is handled correctly.

## Checkpoints

Checkpoints are written under:

```text
submodel/decoder/checkpoints/<run-name>/
```

Files include:

```text
ckpt_latest.pt       latest completed/partial epoch
ckpt_best_val.pt     lowest validation total loss
ckpt_epoch_NNNN.pt   periodic history checkpoint
```

Every checkpoint contains separate `decoder` and `feature_stem` state dictionaries.
`decoder` loads directly into `models.model.decoder`; the frozen `feature_stem`
can generate the pCT latent target during full-model transfer training:

```python
checkpoint = torch.load(checkpoint_path, map_location=device)
G_render.decoder.load_state_dict(checkpoint["decoder"], strict=True)
prior_stem.load_state_dict(checkpoint["feature_stem"], strict=True)
```

See the root `README.md` section **Transfer the pCT-pretrained decoder into the
full model** for latent alignment, three-stage training, soft-tissue-window loss,
TensorBoard tags, resume commands, and inference.

## Fixed-HU PSNR and regional error analysis

Evaluate every case in both validation and test splits with the independently
pretrained decoder:

```bash
python -m submodel.decoder.evaluate_metrics \
  --checkpoint submodel/decoder/checkpoints/dental_batch3/ckpt_best_val.pt \
  --device cuda \
  --batch-size 3 \
  --save-volumes
```

Metrics use the shared dental physical range `[-1000, 3095] HU`, not separate
prediction/GT min-max normalization. The default GT-defined regions are air
`HU < -500`, tissue `[-500, 300)`, and bone `HU >= 300`. Every case is written
to `submodel/decoder/metrics/<checkpoint-stem>/<split>_per_case.jsonl`; each
split's `*_summary.json` contains the case-wise mean, population variance, and
standard deviation of PSNR, HU errors, voxel fractions, regional MSE share, and
the theoretical PSNR gain if one region's error were removed. `mse_share` is
the region's actual contribution to global MSE and therefore the relevant
quantity for diagnosing the PSNR bottleneck. Region-level oracle gains cannot
be added together because PSNR is logarithmic.

Each per-case record includes `sct_vs_pct_psnr_db`, which is the fixed-range
PSNR between the generated Decoder output and its full-resolution pCT target.
With `--save-volumes`, inference volumes are additionally written as
`volumes/<split>/<subject>/sct_predict_hu.nii.gz` and `pct_gt_hu.nii.gz`, both
in HU and with the source pCT NIfTI metadata. These files are sizeable; omit
`--save-volumes` when only numerical metrics are needed.
