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

## Motivation and pipeline

### Stage A – which view should be trusted

Sparse-view CBCT gives only `V` projections (20 in the commands below), so the
volume is strongly under-determined and views are far from equally informative.
The base aggregator (`adafusor` and friends) derives the per-view weights from
the features alone:

```text
logits_v = weight_fc([f_v, mean_v f, var_v f]),   w = softmax_v(logits)
```

Nothing in that rule knows how the views relate in space. Stage A injects that
relation: the relative quaternion against the reference view (index 0) is mapped
by a small MLP into an additive bias on the weight logits, so fusion can favour
geometrically close views while still using complementary ones:

```text
w_v = softmax_v(base_logits_v + MLP(q_rel,v)),   F = sum_v w_v * f_v
q_rel,v = q_ref^-1 (x) q_v,   q_v = Quat([u_hat, v_hat, u_hat x v_hat])
```

A1 uses the absolute `q_v`, A2 the relative `q_rel` (preferred, because absolute
poses are tied to the patient/table frame while only relative poses describe the
view relationship), A3 appends the relative source translation. The bias
projection is zero-initialized, so training starts from exactly the baseline
aggregator output. Nothing else changes: features, geometry, encoder and decoder
are untouched.

### Stage B – the nominal geometry is not exact

Feature back-projection is: 3-D point -> ray/plane intersection with that view's
`[source, detector, u, v]` -> detector pixel -> bilinear sample
([ResEncoder.py](../../models/ResEncoder.py)). C-arm and DRR geometry carry a
residual calibration error of roughly 1 degree and 1 mm, and such an error
shifts every sample at once, so the network reads features from the wrong
tissue and the reconstructed volume blurs or shows ghosting.

Stage B learns a small, bounded correction per view:

```text
rotvec_v = max_angle * tanh(a_v)    # axis-angle, bound e.g. 0.5 deg
dt_v     = max_trans * tanh(b_v)    # mm, bound e.g. 1.0
pose'_v  = apply_pose_delta(pose_v, rotvec_v, dt_v)
```

regularized by `lambda_q |rotvec|^2 + lambda_t |dt|^2` and zero-initialized, so
training starts at the nominal geometry. The corrected pose is then used twice —
for feature sampling and for the forward-projection loss — which is why stage B
moves the result more than stage A. In one sentence: A edits the fusion weights,
B edits the geometry, and C is A applied on the corrected geometry.

### Pipeline

```text
input: V projections [V,1,H,W] + 12-D pose per view [V,12]
       pose = [source(3), detector(3), u(3), v(3)]

(1) 2-D encode (ResEncoder.forward)
    every view -> F0..F4, concatenated to 256 channels -> latent_list,
    poses cached alongside

(2) [B] pose correction (_active_poses -> PoseRefiner)
    pose -> (Delta q, Delta t) -> apply_pose_delta -> pose'
    pose_delta_* metrics recorded; pose' stored in _current_poses

(3) feature back-projection (ResEncoder.queryfeature, uses pose')
    low-resolution grid points xyz [N,3] (scale=4)
    per view: ray/plane intersection -> pixel (u,v) -> grid_sample
    -> per-view features [V, 256, N]

(4) [A] view fusion (_fuse_views -> QuaternionAwareAggregator -> adafusor)
    q_v from the detector basis; q_rel = q_ref^-1 (x) q_v (A3 also t_rel)
    w_v = softmax_v(base_logits + MLP(q_rel)),  F = sum_v w_v * f_v -> [256, N]

(5) fused low-resolution latent volume [1, 256, X/4, Y/4, Z/4]

(6) 3-D decoder -> volume_predict

(7) losses (trainer.py)
    mse_loss_3d    volume vs GT volume (main supervision)
    gd1_loss       volume gradient loss
    latent / prior / completion / ssim / mask   optional auxiliaries
    mse_loss_2d    rays cast with pose' are composited into 2-D projections and
                   compared with the measured ones
                   (weight = mse_lambda_2d + projection_loss_weight, stage B4)
    pose_regularization_loss   only in calibration/joint
```

In short: 2-D features -> use the geometry to fetch features at every 3-D point
from every view -> fuse the views with quaternion-biased weights -> decode a
volume -> project back to 2-D to check the geometry. A acts only in step (4);
B rewrites the geometry in step (2) and therefore influences both (3) and (7).

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

## Learning-rate schedule

The LR is **not** constant. Every command above inherits `conf/train.conf`:

```text
lr_sche { init_lr = 0.0001, step_size = 50, gamma = 0.5 }
```

and the trainer recomputes every parameter group at each epoch as

```text
lr = init_lr * gamma ** (epoch // step_size)
```

so it halves every 50 epochs:

| epochs | effective LR |
|---|---|
| 0-49 | 1e-4 |
| 50-99 | 5e-5 |
| 100-149 | 2.5e-5 |
| 150-199 | 1.25e-5 |
| 200-249 | 6.25e-6 |
| 250-299 | 3.125e-6 |

The current value is logged as `G_lr` in `train/logs/<name>/train_ls.txt`.
Overrides: `--init-lr`, `--lr-step-size`, `--lr-gamma`, and
`--lr-decay-restart` (count the decay from the resumed epoch instead of epoch 0).

To train with a **constant** learning rate, either pass `--lr-gamma 1` (since
`gamma ** n == 1`) or the explicit convenience flag `--no_lr_decay`, which sets
the same thing without having to think about the schedule. The two cannot be
combined with a `--lr-gamma` below 1. This decay is pre-existing behaviour of
the main model (`conf/train.conf`), not something introduced by the quaternion
extension.

For commands like A3 above, `--phase_a_encoder_lr_factor 1` and
`--phase_a_aggregator_lr_factor 1` are inert: the four-phase schedule only turns
on with `--pretrained_decoder` plus `--use_adapter` or `--use_prior_completion`.
Without it `_training_stage` returns 0 and every group — encoder early/late,
aggregator (which also holds the quaternion MLP) and the decoder — shares the
same decayed LR; pass `--stage0_decoder_lr_factor` (e.g. `0` or `0.1`) if the
decoder must be frozen or slower. The `pose_refiner` group only exists in
`calibration`/`joint`.

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

## Existing-projection sampling: quick reference

`random_existing` samples projections that already exist in `proj.nii.gz` and
reads the matching geometry from `transforms.json` (`frames[i]["vec"]`). It never
renders input DRRs from the GT volume, so the image and geometry always remain
paired. This is suitable for pre-generated dental data and measured thorax data.

```text
--angle_sampling random_existing
--existing_view_pattern random|uniform
```

- `random`: default; choose `nviews` distinct frame indices without replacement,
  usually producing non-uniform angular gaps.
- `uniform`: choose equal-spaced existing frames with a random phase. For 360
  frames and 20 views, the spacing is 18 frames, but the first index changes.
- This differs from the original `--angle_sampling uniform`, which always starts
  at frame 0. Use the same two options for training and evaluation. If the range
  contains fewer than `nviews` stored frames, the dataset raises an error.

Training example:

```bash
python train.py --name dental_random_existing20 \
  --datadir ./dataset/dental/syn_data --datatype dental \
  --train_scale 4 --fusion ada --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing --existing_view_pattern random \
  --is_train --epochs 300 --device cuda:0
```

Evaluation must use the same sampling configuration:

```bash
python evaluate.py --name dental_random_existing20 \
  --datadir ./dataset/dental/syn_data --datatype dental --dataname test \
  --train_scale 4 --eval_scale 4 --fusion ada --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing --existing_view_pattern random \
  --resume_name 299 --device cuda:0
```

For thorax, `random_existing` is recommended because it preserves the recorded
frame geometry, including the half-fan detector offset.

## Existing-projection sampling / 已有投影采样（详细说明）

### 中文

本项目增加了 `--angle_sampling random_existing`。它从病例目录中已经保存的
`proj.nii.gz` 和对应的 `transforms.json` 中选择投影，并使用同一索引的
`frames[i]["vec"]` 作为几何，不会从 GT 体数据在线重新渲染 DRR。因此它适用于
预生成的 dental/spine 投影，也适用于带有真实几何的 thorax 测量投影。

新增参数为：

```text
--angle_sampling random_existing
--existing_view_pattern random|uniform
```

`--existing_view_pattern random` 是默认值。它在 `[start,end)` 对应的已有 frame
中无放回随机选择 `nviews` 个索引，选出的角度通常不等间隔：

```bash
python train.py \
  --name dental_random_existing20 \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern random \
  --is_train --epochs 300 --device cuda:0
```

例如完整数据有 `0,1,...,359` 度的 360 个 frame 时，一次取样可能得到
`[3, 17, 41, 88, ...]`；下一个 epoch 或下一次读取同一病例时通常会得到另一组
索引。`--nviews 10` 和 `--nviews 5` 的用法完全相同。

`--existing_view_pattern uniform` 保持等间隔，但每次随机改变等间隔序列的相位。
例如 360 个 frame、20 views 时，间隔为 18 个 frame，一次可能得到
`[2,20,38,...,344]`，另一次可能得到 `[11,29,47,...,353]`。这与原来的
`--angle_sampling uniform` 不同：原来的 uniform 每个病例始终使用固定的
`[0,18,36,...,342]` 索引，而这里的 uniform 是随机相位的等间隔抽样。

```bash
python train.py \
  --name dental_random_phase20 \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern uniform \
  --is_train --epochs 300 --device cuda:0
```

评估时必须使用相同的采样模式和参数。例如：

```bash
python evaluate.py \
  --name dental_random_existing20 \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --dataname test \
  --train_scale 4 --eval_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern random \
  --resume_name 299 --device cuda:0
```

对于 thorax，推荐使用 `random_existing`，因为它直接读取每个实测 frame 的
`vec`，保留 half-fan 探测器偏移和真实角度约定：

```bash
python train.py \
  --name thorax_random_existing20 \
  --datadir ./dataset/thorax/syn_data \
  --datatype thorax --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern random \
  --is_train --epochs 300 --device cuda:0
```

这里的 `random_existing` 不读取 GT 体数据来产生输入投影；GT 体数据仍然只用于
训练/评估目标。若 `nviews` 大于指定角度范围内已有 frame 数量，数据集会直接报错。

### English

The project now supports `--angle_sampling random_existing`. This mode samples from
the projections already stored in `proj.nii.gz` and uses the matching
`transforms.json` entry `frames[i]["vec"]`. It does not render a new DRR from the GT
volume. The image and geometry always come from the same frame index, so this mode is
valid for pre-generated dental/spine projections and measured thorax projections.

The new options are:

```text
--angle_sampling random_existing
--existing_view_pattern random|uniform
```

`--existing_view_pattern random` is the default. It samples `nviews` distinct frame
indices without replacement from the `[start,end)` range. The resulting views are
generally non-uniformly spaced:

```bash
python train.py \
  --name dental_random_existing20 \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern random \
  --is_train --epochs 300 --device cuda:0
```

For a 360-frame sequence sampled at one degree, one call may select
`[3, 17, 41, 88, ...]`, while the next call normally selects a different subset.
Use `--nviews 10` or `--nviews 5` to change the number of selected views.

`--existing_view_pattern uniform` keeps equal spacing but randomizes the phase of the
sequence. With 360 frames and 20 views, the spacing is 18 frames; one call may select
`[2,20,38,...,344]` and another may select `[11,29,47,...,353]`. This differs from the
original `--angle_sampling uniform`, which always selects the fixed indices
`[0,18,36,...,342]` for the same case and command-line range.

```bash
python train.py \
  --name dental_random_phase20 \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern uniform \
  --is_train --epochs 300 --device cuda:0
```

Evaluation must use the same sampling mode and pattern as training:

```bash
python evaluate.py \
  --name dental_random_existing20 \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --dataname test \
  --train_scale 4 --eval_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern random \
  --resume_name 299 --device cuda:0
```

For thorax, `random_existing` is recommended because it reads the measured frame
geometry directly and preserves the half-fan detector offset and the recorded angle
convention:

```bash
python train.py \
  --name thorax_random_existing20 \
  --datadir ./dataset/thorax/syn_data \
  --datatype thorax --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling random_existing \
  --existing_view_pattern random \
  --is_train --epochs 300 --device cuda:0
```

In this mode the GT volume is not used to generate the input projections; it remains
the training/evaluation target. The dataset raises an error if fewer than `nviews`
stored frames are available in the requested angular range.

### Sampling combinations at a glance

`--angle_sampling` selects the sampling family. `--existing_view_pattern` is only
used with `random_existing`:

| Configuration | Existing projections | DRR rendered from GT | View pattern |
|---|---:|---:|---|
| `--angle_sampling uniform` | Yes | No | Fixed equal spacing, starts at frame 0 |
| `random_existing + random` | Yes | No | Random, without replacement; usually non-uniform |
| `random_existing + uniform` | Yes | No | Equal spacing with a random phase |
| `--angle_sampling random` | No | Yes | Randomly generated angles |

For example, with 360 one-degree frames and 20 views, `uniform` selects a fixed
sequence such as `[0,18,36,...,342]`; `random_existing` with `random` selects a
different subset such as `[3,17,41,88,...]`; and `random_existing` with `uniform`
keeps an 18-frame gap but changes the starting phase, such as `[2,20,38,...]`.
The latter two modes always keep each selected projection paired with its original
`frames[i]["vec"]` geometry and are recommended for measured thorax data. Use the
same `--angle_sampling`, `--existing_view_pattern`, `--start`, `--end`, and
`--nviews` values during training and evaluation.

### Train/validation/test metrics

Quaternion runs now use `quaternion_reconstruction_metrics` from `metrics.py` for
clamped 3-D PSNR and SSIM in every split. The helper preserves the root project's
protocol: clamp, per-volume min-max normalization, and three-axis 3-D SSIM. The
trainer writes `train_ssim_3d_clamp`, `val_ssim_3d_clamp`, and
`test_ssim_3d_clamp` to metric logs and TensorBoard; standalone `evaluate.py`
uses the same hook.
