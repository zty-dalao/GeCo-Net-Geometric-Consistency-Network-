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
```

Despite its historical config name, `mse_lambda_3d` weights an L1 loss in this
repository. Both terms operate on the complete `[B, 1, X, Y, Z]` batch. The
projection-domain 2D loss is not used because decoder pretraining intentionally
loads only ground-truth volumes.

## Training

Run from the repository root. A suggested RTX 5090 command is:

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

For a small timed smoke test:

```powershell
python -m submodel.decoder.train `
  --run-name dental_smoke_60s `
  --device cuda `
  --batch-size 1 `
  --limit 1 `
  --eval-limit 1 `
  --max-seconds 60
```

A timed run stops after the first completed training step that reaches the time
limit. It skips validation/test at that partial epoch so that a smoke test stops
promptly.

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
| `--mse-lambda-3d` | config value (`1.0`) | Weight of the voxel-wise L1 loss. The name is retained for compatibility with the original project. |
| `--gd1-lambda` | config value (`1.0`) | Weight of the first-order spatial-gradient L1 loss. |
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
train_step/total
train_step/learning_rate

epoch/train_l1_voxel
epoch/train_gradient1
epoch/train_total
epoch/val_l1_voxel
epoch/val_gradient1
epoch/val_total
epoch/test_l1_voxel
epoch/test_gradient1
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

Every checkpoint contains a top-level `decoder` state dictionary that loads
directly into `models.model.decoder`:

```python
checkpoint = torch.load(checkpoint_path, map_location=device)
G_render.decoder.load_state_dict(checkpoint["decoder"], strict=True)
```
