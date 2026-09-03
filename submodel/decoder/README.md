# Decoder 先验预训练与评估

本目录用于预训练主项目中原始的 3D Decoder（`models/SRGAN.py`）。训练的目标是让
Decoder 学到 pCT/CT 的三维结构先验，之后可将其 `decoder` 权重和 `feature_stem`
权重接入稀疏视角 CBCT 主模型。

> 注意：这里的“sCT”是 Decoder 从低分辨率 pCT latent 重建出的 CT-like 体积；训练
> GT 是同一病例的完整分辨率 `gt_volume.nii.gz`。因此本目录的评估衡量的是 Decoder
> 先验本身的上限，不等价于稀疏 CBCT 主模型的最终重建指标。

## 1. 网络与数据流

数据集中的 SimpleITK 体积顺序是 ZYX，训练前会转为模型使用的 XYZ：

```text
完整 pCT/CT GT [B, 1, 256, 256, 256]
    │
    ├── 固定平均池化 4×4×4（生成低分辨率 pCT 输入）
    ▼
[B, 1, 64, 64, 64]
    │
    ├── PriorFeatureStem：1 → 64 → 256
    ▼
[B, 256, 64, 64, 64]
    │
    ├── 主项目原始 SRGAN Decoder（保持不变）
    ▼
生成 sCT [B, 1, 256, 256, 256]
    │
    └── 与完整 pCT/CT GT 比较
```

数据根目录默认为：

```text
dataset/dental/syn_data/<病例编号>/gt_volume.nii.gz
```

训练、验证和测试病例划分来自：

```text
data/dataset_split/dental_split.json
```

## 2. 损失函数

总损失为：

\[
L_{\text{total}}=L_{3D}+L_{grad}+L_{bone}+L_{soft-mask}
\]

其中：

```text
L_3D        = mse_lambda_3d × 全体素 L1
L_grad      = gd1_lambda × X/Y/Z 三方向一阶梯度 L1
L_bone      = bone_lambda × 骨区 GT 掩码损失
L_soft-mask = soft_mask_lambda × 软组织 GT 掩码损失
```

虽然历史名称中带有 `mse`，本仓库的 `mse_lambda_3d` 实际权重的是 **L1**，不是 L2/MSE。

### 骨区 GT 掩码损失

骨区由 GT 定义，而不是预测定义：

\[
M_{bone}=\mathbb{1}[HU_{GT}\ge300]
\]

\[
L_{bone,raw}=\operatorname{mean}_{M_{bone}}
\left(\frac{|\hat{\mu}-\mu_{GT}|}{\mu_{max}-\mu_{min}}\right)
\]

预测不会被截断后再参与此损失。因此 GT 为骨骼、预测却落到空气或软组织时，仍会获得把
预测拉回骨区的梯度。

### 软组织 GT 掩码损失

默认软组织训练窗是 `[-160, 240] HU`：

\[
M_{soft}=\mathbb{1}[-160\le HU_{GT}<240]
\]

\[
L_{soft,raw}=\operatorname{mean}_{M_{soft}}
\left(\frac{|\widehat{HU}-HU_{GT}|}{400}\right)
\]

同样，掩码只依赖 GT，预测值不做窗口截断。因此预测即使已经在软组织窗外，仍有有效梯度。

`*_raw` 是未乘权重的区域误差；实际加入总损失的是：

```text
bone_gt_mask = bone_lambda × bone_gt_mask_raw
soft_mask    = soft_mask_lambda × soft_mask_raw
```

默认 `--bone-lambda 0` 与 `--soft-mask-lambda 0`，即完全保留旧版训练目标和行为。

## 3. 初次训练

在项目根目录执行。Linux Bash 中，反斜杠只能放在每行末尾，参数前必须是普通的 `--`：

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

`batch-size=3` 是面向 32 GB GPU 的建议值；AMP 默认开启。若显存不足，先降低 batch size，
不要关闭 AMP。

## 4. 从 checkpoint 接力训练并启用区域损失

下例从旧版最优 Decoder 继续训练 50 个额外 epoch，并创建一个独立实验目录，保留原始结果：

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

带 `--resume` 时，`--epochs 50` 表示**额外训练 50 个 epoch**，不是训练到总 epoch=50。
`--resume` 会恢复模型、`feature_stem`、Adam 优化器状态、AMP scaler、step 和 epoch。

若已训练到 `dental_batch3_region_refine` 的某个 epoch，需要再接力 50 epoch，应使用该实验的
最新 checkpoint：

```bash
python -m submodel.decoder.train \
  --run-name dental_batch3_region_refine \
  --device cuda \
  --batch-size 3 \
  --resume submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_latest.pt \
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

不加 `--resume` 时，模型与 Adam 都会重新随机初始化，即从头训练。若同时使用旧的
`--run-name`，还可能覆盖原有 `ckpt_latest.pt` 和日志。

当恢复的 checkpoint 与当前损失配置不同（例如旧 checkpoint 未使用 bone/soft mask），代码会
自动重置 best validation loss，因为新旧 `total loss` 不可直接比较。

## 5. 训练输出、Checkpoint 与 TensorBoard

训练输出位置：

```text
submodel/decoder/
├── checkpoints/<run-name>/
│   ├── ckpt_latest.pt       # 最近完成 epoch 的 checkpoint
│   ├── ckpt_best_val.pt     # 当前损失配置下验证总损失最低的 checkpoint
│   └── ckpt_epoch_XXXX.pt   # 按 --save-every 保存
└── logs/<run-name>/
    ├── train.jsonl          # 每个训练 step
    ├── epoch.jsonl          # 每个 epoch 的 train/val/test 平均损失
    ├── config.json
    └── tensorboard/
```

启动 TensorBoard：

```bash
tensorboard --logdir submodel/decoder/logs/dental_batch3_region_refine/tensorboard --port 6006
```

主要曲线：

```text
train_step/l1_voxel
train_step/gradient1
train_step/bone_gt_mask
train_step/bone_gt_mask_raw
train_step/soft_mask
train_step/soft_mask_raw
train_step/total
train_step/learning_rate

epoch/{train,val,test}_l1_voxel
epoch/{train,val,test}_gradient1
epoch/{train,val,test}_bone_gt_mask
epoch/{train,val,test}_bone_gt_mask_raw
epoch/{train,val,test}_soft_mask
epoch/{train,val,test}_soft_mask_raw
epoch/{train,val,test}_total
```

## 6. 全验证集与测试集推理、PSNR 与分区贡献评估

使用最佳 checkpoint 对**全部 val 和 test 病例**推理、计算指标，并保存 NIfTI：

```bash
python -m submodel.decoder.evaluate_metrics \
  --checkpoint submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_best_val.pt \
  --device cuda \
  --batch-size 3 \
  --output-dir submodel/decoder/metrics/dental_batch3_region_refine \
  --save-volumes
```

不写 `--save-volumes` 时，仍会计算并保存全部数值指标，但不会保存 NIfTI，可节省大量磁盘空间。
仅评估一个 split 时，例如测试集：

```bash
python -m submodel.decoder.evaluate_metrics \
  --checkpoint submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_best_val.pt \
  --device cuda \
  --batch-size 3 \
  --splits test \
  --output-dir submodel/decoder/metrics/dental_batch3_region_refine_test \
  --save-volumes
```

### 推理 NIfTI 保存位置

启用 `--save-volumes` 后，每个病例都会生成 HU 单位的 NIfTI：

```text
submodel/decoder/metrics/dental_batch3_region_refine/
└── volumes/
    ├── val/<病例编号>/
    │   ├── sct_predict_hu.nii.gz  # Decoder 推理得到的 sCT
    │   └── pct_gt_hu.nii.gz       # 完整分辨率 pCT/CT GT
    └── test/<病例编号>/
        ├── sct_predict_hu.nii.gz
        └── pct_gt_hu.nii.gz
```

保存时会将模型内部 XYZ 张量转换回 NIfTI 所需的 ZYX 顺序，并复制 GT 的 spacing、origin 和
direction，可直接使用 ITK-SNAP 或 3D Slicer 查看。

### 全局 PSNR

评估不会对预测和 GT 分别做 min-max 归一化，而是使用 dental 固定物理范围：

\[
[-1000,3095]\ \mathrm{HU}
\]

计算：

\[
PSNR=-10\log_{10}(MSE_{[0,1]})
\]

每病例字段：

```text
sct_vs_pct_psnr_db  # 生成 sCT 与完整 pCT/CT GT 的固定范围 PSNR
global_psnr_db      # 与上一字段相同，保留为通用名称
sct_vs_pct_ssim_3d # 生成 sCT 与完整 pCT/CT GT 的三维 SSIM
ssim_3d_clamp       # 与上一字段相同，名称与根目录 evaluate.py 对齐
global_mae_hu
global_rmse_hu
global_mse_normalized
```

SSIM 直接复用根目录 `evaluate.py` 的 `data_norm()` 与 `get_ssim_3d()`：预测与 GT 分别做
min-max 归一化后，计算三个正交方向的 SSIM 并取平均。因此
`sct_vs_pct_ssim_3d` 与主项目的 `ssim_3d_clamp` 使用相同协议；它与固定 HU 范围 PSNR
的归一化协议不同，二者不应相互换算。

### 空气、组织、骨骼对 PSNR 的贡献

评估使用 GT HU 定义三区域，默认与训练软组织窗不同：

```text
air:    HU < -500
tissue: -500 <= HU < 300
bone:   HU >= 300
```

每个病例均记录：

```text
air/tissue/bone_mae_hu
air/tissue/bone_mse_normalized
air/tissue/bone_mse_share
air/tissue/bone_oracle_psnr_gain_db
```

其中：

\[
MSE\ share_r=
\frac{\sum_{i\in r}(\hat{x}_i-x_i)^2}
{\sum_i(\hat{x}_i-x_i)^2}
\]

即 `*_mse_share` 是该区域对该病例**全局 MSE、进而对 PSNR 瓶颈**的贡献率。三类区域的
`mse_share` 之和约为 1。

`*_oracle_psnr_gain_db` 的含义是：假设仅消除该区域的全部误差、其他区域不变时，全局 PSNR
理论上能提高多少 dB。各区域的该指标不能相加，因为 PSNR 是对数指标。

### 评估数值文件

上面的评估命令还会生成：

```text
submodel/decoder/metrics/dental_batch3_region_refine/
├── val_per_case.jsonl     # 验证集每个病例一行，含 PSNR 与三区域指标
├── test_per_case.jsonl    # 测试集每个病例一行，含 PSNR 与三区域指标
├── val_summary.json       # 验证集逐病例指标的 mean / variance / std
├── test_summary.json      # 测试集逐病例指标的 mean / variance / std
└── run_info.json          # checkpoint、HU 范围、区域阈值、运行配置
```

因此：

- 整个 val/test 的平均 PSNR：查看 `*_summary.json` 中
  `mean.sct_vs_pct_psnr_db`；
- PSNR 方差：查看 `variance.sct_vs_pct_psnr_db`；
- 整个 val/test 的平均 SSIM：查看 `mean.sct_vs_pct_ssim_3d`；
- SSIM 方差：查看 `variance.sct_vs_pct_ssim_3d`；
- 整个 split 的平均空气/组织/骨骼贡献率：查看
  `mean.air_mse_share`、`mean.tissue_mse_share`、`mean.bone_mse_share`；
- 对应病例间方差：查看同名字段的 `variance`。

这些 mean/variance 是先计算每个病例指标，再在病例维度上统计；它们不是将所有病例体素直接拼接
后计算的单一 pooled 指标。

## 7. 主要命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data-root` | `dataset/dental/syn_data` | dental 病例根目录。 |
| `--split-file` | `data/dataset_split/dental_split.json` | train/val/test 划分。 |
| `--run-name` | `dental_pretrain` | 日志和 checkpoint 子目录名。 |
| `--epochs` | `500` | 未 resume 时的总训练 epoch；resume 时的额外 epoch。 |
| `--batch-size` | `1` | 完整三维体积 batch。32 GB GPU 推荐从 3 开始。 |
| `--lr` | `1e-4` | Adam 学习率；resume 时显式给出会覆盖 checkpoint 的 LR。 |
| `--resume` | 无 | 恢复预训练 checkpoint。 |
| `--mse-lambda-3d` | `1.0` | 全体素 L1 权重。 |
| `--gd1-lambda` | `1.0` | 全局三维一阶梯度 L1 权重。 |
| `--bone-lambda` | `0` | 骨区 GT 掩码损失权重。 |
| `--bone-lower-hu` | `300` | 骨区 GT HU 下界。 |
| `--soft-mask-lambda` | `0` | 软组织 GT 掩码损失权重。 |
| `--soft-window-low/high` | `-160 / 240` | 软组织训练掩码 HU 范围。 |
| `--val-every` | `1` | 每隔多少 completed epoch 验证。 |
| `--test-every` | `10` | 每隔多少 completed epoch 测试。 |
| `--save-every` | `10` | 每隔多少 epoch 保存历史 checkpoint。 |
| `--no-amp` | 关闭 | 禁用 AMP；通常不建议，会显著增加显存。 |

## 8. 接入主模型

预训练 checkpoint 的键保持与旧版兼容：

```python
checkpoint = torch.load(checkpoint_path, map_location=device)
G_render.decoder.load_state_dict(checkpoint["decoder"], strict=True)
prior_stem.load_state_dict(checkpoint["feature_stem"], strict=True)
```

因此无论 checkpoint 是否启用 bone/soft mask loss，`decoder` 与 `feature_stem` 的网络结构和键名都
不变化，均可作为主模型 Decoder 初始化和 latent 教师使用。
