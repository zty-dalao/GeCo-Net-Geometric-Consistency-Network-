# Quaternion Geometry

Optional quaternion conditioning for the geometry-aware sparse-view CBCT model.
The default `--quaternion_mode none` leaves the original data path untouched.

Two independent pieces:

1. **View fusion (stage A)** – a relative quaternion between each view and a
   reference view conditions the per-view weights of the existing aggregator.
   Projection geometry, the 2D encoder, latent sizes and the decoder are
   unchanged.
2. **Geometry calibration (stage B)** – a bounded per-view `(Delta q, Delta t)`
   corrects `[source, detector, u, v]` before back-projection, so the sampled
   feature locations follow the corrected geometry.

## Files

| File | Responsibility |
|---|---|
| `pose.py` | 12-vector geometry -> rotation -> quaternion; relative quaternion; quaternion log map; `apply_pose_delta` |
| `calibration.py` | `PoseRefiner`: bounded `Delta q`, `Delta t` and the pose regularizer |
| `fusion.py` | `QuaternionAwareAggregator`: wraps an existing aggregator and biases its view-weight logits |
| `__init__.py` | Public exports and the `METRIC_KEYS` shared with the trainer |

## Architecture

```text
projection views [V, 1, H, W]
        |
        v
shared 2-D encoder -> F0..F4                    (unchanged)
        |
        +-- [source, detector, u, v] geometry    (unchanged)
        |       |
        |       +-- q_rel = q_ref^-1 (x) q_v      pose.py
        |       +-- optional PoseRefiner          calibration.py
        |             Delta q, Delta t -> corrected 12-vector
        |
3-D query point -> detector coordinates -> bilinear samples   (unchanged)
        |
        +-- base aggregator: local / meanmlp / varmlp / ada
        |       |
        |       +-- weight logits + MLP(q_rel)     fusion.py
        |       +-- sum_v w_v * feat_v
        |
C-channel low-resolution latent -> 3-D decoder   (unchanged)
```

`QuaternionAwareAggregator` does not copy aggregator code. It calls the base
aggregator with an extra `view_bias`, which is added to the weight logits only:

```text
w_v = softmax_v( base_weight_fc(feat_v) + MLP(q_v) )
F   = sum_v w_v * feat_v
```

The bias projection is zero-initialized, so a fresh `fusion` run starts from
exactly the baseline aggregator output.

The quaternion is derived from the detector basis, not from a separate pose
head:

```text
u_hat = u / |u|,  v_hat = v / |v|,  n = u_hat x v_hat
R_v   = [u_hat, v_hat, n]
q_v   = Quat(R_v)
```

Translation stays in the original 12-vector; a quaternion alone cannot encode
source/detector positions, SID/SAD, pixel spacing or half-fan offsets.

## Ablation ladder

Every row states what it **adds** to the row above it inside the same stage.
Implement and read the results in the order `A0 -> A2 -> B3 -> C`.

### Stage A – quaternion view fusion (weights only)

| ID | What it adds | Flags |
|---|---|---|
| **A0** | Nothing. Original encoder, original geometry, original aggregator — the project baseline. | `--quaternion_mode none` |
| **A1** | Absolute view quaternion enters the weight logits: `w_v = f(F_v, q_v)`. | `--quaternion_mode fusion --quaternion_embedding absolute_q` |
| **A2** | Replaces the absolute quaternion with the reference-relative one: `w_v = f(F_v, q_rel,v)` with `q_rel,v = q_ref^-1 (x) q_v`. Usually better, because view relations are relative. | `--quaternion_mode fusion --quaternion_embedding relative_q` |
| **A3** | Appends the relative source translation to the pose embedding: `w_v = f(F_v, q_rel,v, t_rel,v)` — the full pose embedding. | A2 plus `--quaternion_use_translation` |

A1–A3 only rewrite the per-view weights; features, geometry, encoder and
decoder stay untouched.

### Stage B – quaternion geometry calibration (pose only)

| ID | What it adds | Flags |
|---|---|---|
| **B0** | Nothing learned. `PoseRefiner` exists but both corrections stay frozen at `Delta q = identity`, `Delta t = 0`. Strict control for B1–B4 (same modules, same parameter count). | `--quaternion_mode calibration --no_pose_refine_rotation --no_pose_refine_translation` |
| **B1** | Learns the per-view rotation correction only: `Delta q_v`, translation frozen. | `--quaternion_mode calibration --no_pose_refine_translation` |
| **B2** | Learns the per-view translation correction only: `Delta t_v`, rotation frozen. | `--quaternion_mode calibration --no_pose_refine_rotation` |
| **B3** | Learns both: `(Delta q_v, Delta t_v)` — the complete calibration. | `--quaternion_mode calibration` |
| **B4** | B3 plus a projection-domain loss on the corrected geometry: `L_projection` is the 2-D forward-projection ray loss and its weight becomes `--mse_lambda_2d + --projection_loss_weight`. It tells whether the correction really improves forward projection instead of only the volume loss. | B3 plus `--projection_loss_weight 0.01` |

`--projection_loss_weight` is rejected unless a `PoseRefiner` is active, i.e.
it requires `calibration` or `joint`.

### Stage C – joint

| ID | What it adds | Flags |
|---|---|---|
| **C** | Stage A on top of stage B: the quaternion fusion weights are computed from the **corrected** geometry, i.e. `A2/A3 + B3` (optionally `+ B4`). Not a third mathematical module. | `--quaternion_mode joint --quaternion_embedding relative_q [--quaternion_use_translation] [--projection_loss_weight ...]` |

## Modes

| `--quaternion_mode` | Ablation row | Behavior |
|---|---|---|
| `none` | A0 / B0 baseline | Original model |
| `fusion` | A1, A2, A3 | Quaternion-conditioned view weights |
| `calibration` | B0, B1, B2, B3, B4 | Bounded per-view geometry correction before projection |
| `joint` | C | Calibration, then quaternion-conditioned weights on the corrected geometry |

Related options:

| Option | Meaning |
|---|---|
| `--quaternion_embedding relative_q` | `q_ref^-1 * q_v` (A2/A3, recommended) |
| `--quaternion_embedding absolute_q` | absolute view quaternion (A1) |
| `--quaternion_use_translation` | append relative source translation to the pose embedding (A3) |
| `--pose_refine_rotation` / `--no_pose_refine_rotation` | learn / freeze the rotation correction (B1 vs B2/B0) |
| `--pose_refine_translation` / `--no_pose_refine_translation` | learn / freeze the translation correction (B2 vs B1/B0) |
| `--pose_refine_max_angle_deg` | rotation bound, default `1.0` |
| `--pose_refine_max_translation` | translation bound in mm, default `1.0` |
| `--pose_delta_max_angle` / `--pose_delta_max_translation` | aliases of the two bounds above |
| `--pose_regularization_weight` | weight of `lambda_q|log dq|^2 + lambda_t|dt|^2` |
| `--projection_loss_weight` | extra 2-D projection-domain weight, only with a refiner (B4) |

Pose corrections are zero-initialized (`Delta q` is the identity), so
`calibration` starts at the nominal geometry.

## Metrics

The usual `psnr_3d_clamp` / `ssim_3d_clamp` are unchanged. These additional
values are averaged per epoch and written for **train / val / test** and for
the training step stream:

| Key | Meaning |
|---|---|
| `relative_pose_angle_deg_mean` | mean relative-rotation angle (degrees) |
| `view_weight_entropy_norm` | normalized entropy of `w_v` in `[0, 1]`; low = committed to few views |
| `view_weight_max_mean` | mean maximum view weight |
| `pose_delta_regularizer` | `mean|rotvec|^2 + mean|dt|^2` |
| `pose_delta_angle_mean` | mean rotation correction (radians) |
| `pose_delta_translation_mean` | mean translation correction (mm) |
| `pose_regularization_loss` | regularizer times `--pose_regularization_weight` |

Metrics a mode does not produce are reported as `0`.

With an active `PoseRefiner`, `mse_loss_2d` is scaled by
`--mse_lambda_2d + --projection_loss_weight` (the B4 term).

TensorBoard streams: `epoch/train_*`, `epoch/val_*`, `epoch/test_*` and the
per-step `step/train_*` entries.

All commands below assume the project's Python environment is active, for
example `conda activate deeplearning` (or run them with
`conda run -n deeplearning python ...`).

## Stage A – quaternion view fusion (A0–A3)

```bash
# A0: baseline (no quaternion, original aggregator)
python train.py --name dental_a0_baseline --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --is_train --epochs 300 --device cuda:0

# A1: + absolute quaternion in the view weights
python train.py --name dental_a1_qabs --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --quaternion_embedding absolute_q \
  --is_train --epochs 300 --device cuda:0

# A2: + relative quaternion instead (q_ref^-1 * q_v)
python train.py --name dental_a2_qrel --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --quaternion_embedding relative_q \
  --is_train --epochs 300 --device cuda:0

# A3: + relative source translation (full pose embedding)
python train.py --name dental_a3_qrel_t --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --quaternion_embedding relative_q \
  --quaternion_use_translation \
  --is_train --epochs 300 --device cuda:0
```

The wrapper also supports `--fusion local|meanmlp|varmlp`; `ada` is the
intended primary comparison.

## Stage B – quaternion geometry calibration (B0–B4) and stage C

```bash
# B0: refiner built but frozen (Delta q = identity, Delta t = 0)
python train.py --name dental_b0_fixed --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration \
  --no_pose_refine_rotation --no_pose_refine_translation \
  --pose_regularization_weight 0.001 \
  --is_train --epochs 300 --device cuda:0

# B1: + learn the rotation correction only
python train.py --name dental_b1_rot --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration --no_pose_refine_translation \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 0.0 \
  --is_train --epochs 300 --device cuda:0

# B2: + learn the translation correction only
python train.py --name dental_b2_trans --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration --no_pose_refine_rotation \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.0 --pose_refine_max_translation 1.0 \
  --is_train --epochs 300 --device cuda:0

# B3: + learn rotation and translation together
python train.py --name dental_b3_rot_t --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 1.0 \
  --is_train --epochs 300 --device cuda:0

# B4: B3 + extra projection-domain weight
python train.py --name dental_b4_rot_t_proj --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 1.0 \
  --projection_loss_weight 0.01 \
  --is_train --epochs 300 --device cuda:0

# C: A2/A3 on top of B3 (optionally + B4)
python train.py --name dental_c_joint --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode joint --quaternion_embedding relative_q \
  --quaternion_use_translation \
  --pose_regularization_weight 0.001 \
  --is_train --epochs 300 --device cuda:0
```

The 2-D ray loss already exists and defaults to `--mse_lambda_2d 0.01`, so
every run has a weak projection term computed with the (corrected) poses.
B4 adds `--projection_loss_weight` on top — total projection weight
`0.01 + 0.01 = 0.02`. For a strict B3-vs-B4 comparison pass
`--mse_lambda_2d 0` to both so the only difference is the projection term.

A frozen correction (`--no_pose_refine_rotation --no_pose_refine_translation`)
leaves both parameters at zero and contributes no regularizer, which is why B0
is a valid control with the same module and parameter count as B1–B4.

Calibration training should be validated on controlled geometric perturbations
of otherwise identical data; clean-data PSNR alone does not show whether a
correction is meaningful.

## Evaluation

Use exactly the same geometry/mode options as training:

```bash
python evaluate.py --name dental_a2_qrel \
  --datadir ./dataset/dental/syn_data --datatype dental --dataname test \
  --train_scale 4 --eval_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --quaternion_mode fusion --quaternion_embedding relative_q \
  --resume_name 299 --device cuda:0
```

For calibration rows (B0–B4) add `--quaternion_mode calibration` with the same
`--pose_refine_max_*` bounds **and** the same
`--pose_refine_rotation/--pose_refine_translation` switches, e.g. B1:

```bash
python evaluate.py --name dental_b1_rot \
  --datadir ./dataset/dental/syn_data --datatype dental --dataname test \
  --train_scale 4 --eval_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --quaternion_mode calibration --no_pose_refine_translation \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 0.0 \
  --resume_name 299 --device cuda:0
```

Per-case and averaged PSNR/SSIM plus the metrics above are written under
`logs/<name>/...`.

## Smoke tests

Module-level check (no data needed):

```bash
python -c "import torch; from types import SimpleNamespace; from models.aggregator import adafusor; from submodel.quaternion_geometry import QuaternionAwareAggregator, PoseRefiner; p=torch.zeros(4,12); p[:,6]=1; p[:,10]=1; a=adafusor(SimpleNamespace(latent_size=256, activation='GELU')); q=QuaternionAwareAggregator(a); print(q(torch.randn(4,256,32), p).shape, PoseRefiner()(p)[0].shape)"
```

Short real-data runs (about one minute each):

```bash
# A2: fusion
python train.py --name smoke_a2_fusion --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 8 --fusion ada \
  --start 0 --end 360 --nviews 4 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --is_train --epochs 1 --device cuda:0

# B3: calibration
python train.py --name smoke_b3_calib --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 8 --fusion ada \
  --start 0 --end 360 --nviews 4 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration --pose_regularization_weight 0.001 \
  --is_train --epochs 1 --device cuda:0
```

## Limitations

- `--multiscale_decoder` has its own per-scale fusion modules and is not wired
  to quaternion fusion; the model raises instead of silently ignoring it.
  `--fullres_multiscale_decoder` uses `_fuse_views` and is supported.
- `--resume` can start from a pre-quaternion checkpoint: the missing
  quaternion weights keep their zero-initialized values, so training resumes
  from exactly the baseline behaviour.
- A rigid per-view correction cannot model respiratory deformation; that needs
  a deformation field.
- Calibration without a projection-domain loss is weakly constrained under
  sparse views; keep the regularization weight non-zero and the bounds small.
- `--projection_loss_weight` (B4) is rejected unless `--quaternion_mode` is
  `calibration` or `joint`; without a `PoseRefiner` there is no corrected
  geometry to score. It is added on top of `--mse_lambda_2d`, which is `0.01`
  by default — set `--mse_lambda_2d 0` if B3 and B4 must differ only by the
  projection term.
- `--no_pose_refine_rotation` / `--no_pose_refine_translation` freeze a branch;
  freezing both (B0) keeps the parameters allocated but never updates them.
  The evaluation run must repeat the same switches, otherwise the checkpoint
  is loaded into a differently configured refiner.
