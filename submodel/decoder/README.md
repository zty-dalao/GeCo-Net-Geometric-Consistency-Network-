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

The loss matches the main project's available 3D supervision: L1 volume loss
plus first-order gradient L1 loss. Both losses operate on the complete
`[B, 1, X, Y, Z]` batch. The projection-domain 2D loss is not used because this
pretraining task intentionally loads only ground-truth volumes.

Run from the repository root:

```powershell
python -m submodel.decoder.train `
  --run-name dental_pretrain `
  --device cuda `
  --batch-size 3
```

For a timed smoke test:

```powershell
python -m submodel.decoder.train `
  --run-name dental_smoke_60s `
  --device cuda `
  --max-seconds 60
```

Logs and checkpoints are written below this directory:

```text
submodel/decoder/logs/<run-name>/
submodel/decoder/checkpoints/<run-name>/
```

Each checkpoint contains a top-level `decoder` state dictionary that can be
loaded directly into `models.model.decoder`.
