# 四元数几何（Quaternion Geometry）

面向几何感知稀疏视角 CBCT 模型的可选四元数条件化。默认的 `--quaternion_mode none`
不会改动原有数据通路。

包含两个相互独立的部分：

1. **视角融合（stage A）** —— 每个视角与参考视角之间的相对四元数，用于对现有聚合器的
   逐视角权重进行条件化。投影几何、2D 编码器、隐变量尺寸与解码器均保持不变。
2. **几何校准（stage B）** —— 一个有界的逐视角 `(Delta q, Delta t)` 在反投影之前修正
   `[source, detector, u, v]`，从而使采样到的特征位置遵循修正后的几何。

## 文件

| 文件 | 职责 |
|---|---|
| `pose.py` | 12 维几何 -> 旋转矩阵 -> 四元数；相对四元数；四元数对数映射；`apply_pose_delta` |
| `calibration.py` | `PoseRefiner`：有界的 `Delta q`、`Delta t` 以及位姿正则项 |
| `fusion.py` | `QuaternionAwareAggregator`：包装已有的聚合器，并对其视角权重 logits 加偏置 |
| `__init__.py` | 公开导出，以及与 trainer 共用的 `METRIC_KEYS` |

## 架构

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

`QuaternionAwareAggregator` 不会复制聚合器的代码。它以额外的 `view_bias` 调用基础
聚合器，该偏置只加到权重 logits 上：

```text
w_v = softmax_v( base_weight_fc(feat_v) + MLP(q_v) )
F   = sum_v w_v * feat_v
```

偏置投影采用零初始化，因此全新的 `fusion` 运行会从与基线聚合器完全相同的输出开始。

四元数由探测器基向量导出，而不是来自单独的位姿预测头：

```text
u_hat = u / |u|,  v_hat = v / |v|,  n = u_hat x v_hat
R_v   = [u_hat, v_hat, n]
q_v   = Quat(R_v)
```

平移仍然保留在原来的 12 维向量中；仅靠四元数无法编码源/探测器位置、SID/SAD、像素间距
或半扇偏移。

## 消融阶梯（A0–A3 / B0–B4 / C）

每一行都写明它相对“同一阶段内上一行”**增加了什么**。建议按 `A0 -> A2 -> B3 -> C`
的顺序实现并阅读结果。

### 阶段 A —— 四元数视角融合（只改权重）

| 编号 | 增加什么 | 参数 |
|---|---|---|
| **A0** | 什么都不加：原始 Encoder、原始几何、原始 Aggregator，即项目 baseline。 | `--quaternion_mode none` |
| **A1** | 把绝对视角四元数送进权重 logits：`w_v = f(F_v, q_v)`。 | `--quaternion_mode fusion --quaternion_embedding absolute_q` |
| **A2** | 把绝对四元数换成参考视角的相对四元数：`w_v = f(F_v, q_rel,v)`，其中 `q_rel,v = q_ref^-1 (x) q_v`。通常比 A1 更合理，因为视角关系本来就是相对的。 | `--quaternion_mode fusion --quaternion_embedding relative_q` |
| **A3** | 再附加相对源平移：`w_v = f(F_v, q_rel,v, t_rel,v)`，即完整的位姿嵌入。 | A2 再加 `--quaternion_use_translation` |

A1–A3 只改写逐视角权重，特征、几何、编码器与解码器保持不变。

### 阶段 B —— 四元数几何校准（只改位姿）

| 编号 | 增加什么 | 参数 |
|---|---|---|
| **B0** | 不学习任何修正：`PoseRefiner` 存在，但 `Delta q = 单位四元数`、`Delta t = 0` 全程冻结。作为 B1–B4 的严格对照组（模块与参数量完全一致）。 | `--quaternion_mode calibration --no_pose_refine_rotation --no_pose_refine_translation` |
| **B1** | 只学习逐视角旋转修正 `Delta q_v`，平移冻结。 | `--quaternion_mode calibration --no_pose_refine_translation` |
| **B2** | 只学习逐视角平移修正 `Delta t_v`，旋转冻结。 | `--quaternion_mode calibration --no_pose_refine_rotation` |
| **B3** | 同时学习 `(Delta q_v, Delta t_v)`，即完整的几何校准。 | `--quaternion_mode calibration` |
| **B4** | 在 B3 之上加入投影域损失 `L_projection`：即 2D 前向投影射线损失，其总权重变为 `--mse_lambda_2d + --projection_loss_weight`。用来检验几何修正是否真的改善了前向投影，而不只是改善了体数据损失。 | B3 再加 `--projection_loss_weight 0.01` |

没有 `PoseRefiner` 时 `--projection_loss_weight` 会被直接拒绝，也就是说它必须配合
`calibration` 或 `joint` 使用。

### 阶段 C —— 联合

| 编号 | 增加什么 | 参数 |
|---|---|---|
| **C** | 在阶段 B 之上叠加阶段 A：四元数融合权重由**修正后**的几何计算，等价于 `A2/A3 + B3`（可选再 `+ B4`）。它不是第三个独立的数学模块。 | `--quaternion_mode joint --quaternion_embedding relative_q [--quaternion_use_translation] [--projection_loss_weight ...]` |

## 模式

| `--quaternion_mode` | 对应消融行 | 行为 |
|---|---|---|
| `none` | A0 / B0 基线 | 原始模型 |
| `fusion` | A1、A2、A3 | 四元数条件化的视角权重 |
| `calibration` | B0、B1、B2、B3、B4 | 投影前进行有界的逐视角几何修正 |
| `joint` | C | 先校准，再在修正后的几何上做四元数条件化的权重计算 |

相关选项：

| 选项 | 含义 |
|---|---|
| `--quaternion_embedding relative_q` | `q_ref^-1 * q_v`（A2/A3，推荐） |
| `--quaternion_embedding absolute_q` | 绝对视角四元数（A1） |
| `--quaternion_use_translation` | 将相对源平移附加到位姿嵌入中（A3） |
| `--pose_refine_rotation` / `--no_pose_refine_rotation` | 学习 / 冻结旋转修正（B1 与 B2/B0 的区别） |
| `--pose_refine_translation` / `--no_pose_refine_translation` | 学习 / 冻结平移修正（B2 与 B1/B0 的区别） |
| `--pose_refine_max_angle_deg` | 旋转上界，默认 `1.0` |
| `--pose_refine_max_translation` | 平移上界（mm），默认 `1.0` |
| `--pose_delta_max_angle` / `--pose_delta_max_translation` | 上述两个上界的别名 |
| `--pose_regularization_weight` | `lambda_q|log dq|^2 + lambda_t|dt|^2` 的权重 |
| `--projection_loss_weight` | 额外的 2D 投影域权重，仅在启用 refiner 时生效（B4） |

位姿修正采用零初始化（`Delta q` 为单位四元数），因此 `calibration` 从标称几何开始。

## 指标

常规的 `psnr_3d_clamp` / `ssim_3d_clamp` 保持不变。以下附加数值会按 epoch 求平均，
并分别写入 **train / val / test** 以及训练步数据流：

| 键 | 含义 |
|---|---|
| `relative_pose_angle_deg_mean` | 相对旋转角的均值（度） |
| `view_weight_entropy_norm` | `w_v` 在 `[0, 1]` 内的归一化熵；越低表示越集中于少数视角 |
| `view_weight_max_mean` | 最大视角权重的均值 |
| `pose_delta_regularizer` | `mean|rotvec|^2 + mean|dt|^2` |
| `pose_delta_angle_mean` | 旋转修正的均值（弧度） |
| `pose_delta_translation_mean` | 平移修正的均值（mm） |
| `pose_regularization_loss` | 正则项乘以 `--pose_regularization_weight` |

某个模式不产生的指标统一记为 `0`。

启用 `PoseRefiner` 时，`mse_loss_2d` 的权重为
`--mse_lambda_2d + --projection_loss_weight`（即 B4 的投影项）。

TensorBoard 数据流：`epoch/train_*`、`epoch/val_*`、`epoch/test_*`，以及逐步的
`step/train_*` 条目。

以下所有命令均假设项目的 Python 环境已激活，例如
`conda activate deeplearning`（也可以用
`conda run -n deeplearning python ...` 运行）。

## 阶段 A —— 四元数视角融合（A0–A3）

```bash
# A0：基线（无四元数，原始聚合器）
python train.py --name dental_a0_baseline --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --is_train --epochs 300 --device cuda:0

# A1：视角权重中加入绝对四元数
python train.py --name dental_a1_qabs --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --quaternion_embedding absolute_q \
  --is_train --epochs 300 --device cuda:0

# A2：改为相对四元数（q_ref^-1 * q_v）
python train.py --name dental_a2_qrel --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --quaternion_embedding relative_q \
  --is_train --epochs 300 --device cuda:0

# A3：再附加相对源平移（完整位姿嵌入）
python train.py --name dental_a3_qrel_t --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --quaternion_embedding relative_q \
  --quaternion_use_translation \
  --is_train --epochs 300 --device cuda:0
```

该包装器同样支持 `--fusion local|meanmlp|varmlp`；`ada` 是预期的主要对比对象。

## 阶段 B —— 四元数几何校准（B0–B4）与阶段 C

```bash
# B0：refiner 存在但全程冻结（Delta q = 单位四元数，Delta t = 0）
python train.py --name dental_b0_fixed --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration \
  --no_pose_refine_rotation --no_pose_refine_translation \
  --pose_regularization_weight 0.001 \
  --is_train --epochs 300 --device cuda:0

# B1：只学习旋转修正
python train.py --name dental_b1_rot --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration --no_pose_refine_translation \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 0.0 \
  --is_train --epochs 300 --device cuda:0

# B2：只学习平移修正
python train.py --name dental_b2_trans --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration --no_pose_refine_rotation \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.0 --pose_refine_max_translation 1.0 \
  --is_train --epochs 300 --device cuda:0

# B3：同时学习旋转与平移
python train.py --name dental_b3_rot_t --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 1.0 \
  --is_train --epochs 300 --device cuda:0

# B4：B3 + 额外的投影域权重
python train.py --name dental_b4_rot_t_proj --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration \
  --pose_regularization_weight 0.001 \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 1.0 \
  --projection_loss_weight 0.01 \
  --is_train --epochs 300 --device cuda:0

# C：在 B3 之上叠加 A2/A3（可选再 + B4）
python train.py --name dental_c_joint --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode joint --quaternion_embedding relative_q \
  --quaternion_use_translation \
  --pose_regularization_weight 0.001 \
  --is_train --epochs 300 --device cuda:0
```

2D 射线损失本来就存在，默认 `--mse_lambda_2d 0.01`，因此所有实验都会带一个用
（修正后）位姿计算的弱投影项。B4 用 `--projection_loss_weight` 在其之上再加一份，
总投影权重为 `0.01 + 0.01 = 0.02`。若要做严格的 B3 vs B4 对照，请给两者都加
`--mse_lambda_2d 0`，使唯一差别就是投影项。

冻结修正（`--no_pose_refine_rotation --no_pose_refine_translation`）会让两个参数始终
为零、正则项也为零，所以 B0 是模块与参数量都相同、但几何不动的有效对照组。

校准训练应在“除几何扰动外其余完全相同”的受控数据上进行验证；仅凭干净数据上的 PSNR
无法说明该修正是否有意义。

## 评估

请使用与训练完全相同的几何/模式选项：

```bash
python evaluate.py --name dental_a2_qrel \
  --datadir ./dataset/dental/syn_data --datatype dental --dataname test \
  --train_scale 4 --eval_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --quaternion_mode fusion --quaternion_embedding relative_q \
  --resume_name 299 --device cuda:0
```

校准行（B0–B4）需加上 `--quaternion_mode calibration`、与训练一致的
`--pose_refine_max_*` 上界，**以及**相同的
`--pose_refine_rotation/--pose_refine_translation` 开关，例如 B1：

```bash
python evaluate.py --name dental_b1_rot \
  --datadir ./dataset/dental/syn_data --datatype dental --dataname test \
  --train_scale 4 --eval_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --quaternion_mode calibration --no_pose_refine_translation \
  --pose_refine_max_angle_deg 0.5 --pose_refine_max_translation 0.0 \
  --resume_name 299 --device cuda:0
```

逐病例与平均的 PSNR/SSIM 以及上述指标会写入 `logs/<name>/...` 下。

## 冒烟测试

模块级检查（无需数据）：

```bash
python -c "import torch; from types import SimpleNamespace; from models.aggregator import adafusor; from submodel.quaternion_geometry import QuaternionAwareAggregator, PoseRefiner; p=torch.zeros(4,12); p[:,6]=1; p[:,10]=1; a=adafusor(SimpleNamespace(latent_size=256, activation='GELU')); q=QuaternionAwareAggregator(a); print(q(torch.randn(4,256,32), p).shape, PoseRefiner()(p)[0].shape)"
```

短时间真实数据运行（每次约一分钟）：

```bash
# A2：融合
python train.py --name smoke_a2_fusion --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 8 --fusion ada \
  --start 0 --end 360 --nviews 4 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode fusion --is_train --epochs 1 --device cuda:0

# B3：校准
python train.py --name smoke_b3_calib --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 8 --fusion ada \
  --start 0 --end 360 --nviews 4 --angle_sampling uniform \
  --phase_a_encoder_lr_factor 1 --phase_a_aggregator_lr_factor 1 \
  --quaternion_mode calibration --pose_regularization_weight 0.001 \
  --is_train --epochs 1 --device cuda:0
```

## 局限

- `--multiscale_decoder` 拥有自己的一套多尺度融合模块，并未接入四元数融合；模型会直接
  报错，而不是静默忽略该选项。`--fullres_multiscale_decoder` 使用 `_fuse_views`，受支持。
- `--resume` 可以从四元数设计之前的 checkpoint 开始：缺失的四元数权重保持其零初始化值，
  因此训练会从与基线完全一致的行为继续。
- 刚性逐视角修正无法建模呼吸形变；那需要形变场。
- 在稀疏视角下，没有投影域损失的校准约束较弱；请保持正则权重非零，并使用较小的上界。
- `--projection_loss_weight`（B4）在不处于 `calibration` 或 `joint` 模式时会被直接拒绝：
  没有 `PoseRefiner` 就没有可用于投影域评估的修正后几何。它是在 `--mse_lambda_2d`
  （默认 `0.01`）之上叠加的；若希望 B3 与 B4 只差投影项，请把两者都设为
  `--mse_lambda_2d 0`。
- `--no_pose_refine_rotation` / `--no_pose_refine_translation` 用于冻结某个分支；两者都冻结
  即 B0，参数仍会被创建但永不更新。评估时必须重复相同的开关，否则 checkpoint 会被加载
  到一个配置不同的 refiner 中。
