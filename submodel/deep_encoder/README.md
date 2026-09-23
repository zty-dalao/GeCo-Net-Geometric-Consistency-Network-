# 固定低频 + 学习型高频 Prior Encoder

> **阅读顺序**
> 本文档按「单独训练 → 单独评估 → 接入主模型」组织：
> 第 4 节是本子模型自带的训练循环（`submodel.deep_encoder.train`），
> 第 5 节是它的独立评估，
> 第 6 节是把它的 decoder 作为先验接入根目录主模型训练的两种迁移策略（**四阶段**与 **legacy 三阶段**，并列介绍），
> 第 7 节收纳学习率控制、BatchNorm 陷阱、指标口径等运维经验与注意事项。

---

## 1. 设计目的

本目录实现一个可解释的 pCT 先验编码器：固定平均池化负责保留稳定低频基底，
学习型卷积只编码被平均池化去除的块内高频残差。得到的 latent 随后交给主项目原始
SRGAN Decoder 重建 sCT。

本版本用于从 0 开始训练，不能直接 resume 此前浅层 `submodel/decoder` 或旧版
`submodel/deep_encoder` checkpoint。

数据读取、loss、AMP、TensorBoard、验证、测试、checkpoint 与固定范围指标逻辑均与
`submodel/decoder` 保持一致。训练和评估入口复用原循环，仅替换模型类，防止两套
实验的损失和指标实现发生漂移。

## 2. 模型架构

```text
完整 pCT/CT GT
[B,1,256,256,256]
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        │ 低频分支                                      │ 高频残差分支
        │                                              │
        │ AvgPool3d                                    │ AvgPool3d
        │ kernel=4, stride=4                           │ kernel=4, stride=4
        │                                              │
        │ mean                                         │ mean
        │ [B,1,64,64,64]                               │
        │                                              │ nearest 上采样到 256³
        │                                              │
        │                                              │ high_frequency
        │                                              │ = pCT - mean_up
        │                                              │
        │                                              │ Conv3d 1→31
        │                                              │ kernel=4, stride=4
        │                                              │
        │                                              │ detail
        │                                              │ [B,31,64,64,64]
        │                                              │
        └────────────────────── concat ────────────────┘
                                   │
                                   ▼
                      low_feature [B,32,64,64,64]
                                   │
                                   ├── Conv3d 3×3×3：32→64
                                   ├── GELU
                                   ├── Conv3d 3×3×3：64→64
                                   ├── GELU
                                   ├── Conv3d 3×3×3：64→256
                                   └── GELU
                                   ▼
                      prior latent [B,256,64,64,64]
                                   │
                                   ▼
                      主项目原始 SRGAN Decoder
                                   │
                                   ▼
                         sCT [B,1,256,256,256]
```

对应核心计算：

```python
mean = F.avg_pool3d(volume, kernel_size=4, stride=4)
mean_up = F.interpolate(mean, size=volume.shape[-3:], mode="nearest")
high_frequency = volume - mean_up
detail = detail_downsample(high_frequency)  # 1→31, kernel=4, stride=4
low_feature = torch.cat([mean, detail], dim=1)
latent = feature_stem(low_feature)           # 32→64→64→256
```

第一通道始终是固定块均值，另外 31 个通道只能从高频残差中学习局部测量。相比自由的
`Conv3d 1→32`，该结构更明确地区分低频强度与高频细节。

默认参数量：

```text
LearnedPriorEncoder：       610,655
原 SRGAN Decoder：        25,371,473
完整预训练模型：          25,982,128
```

SRGAN Decoder 的内部分组（第 6 节的解冻调度会逐级放开它们）：

```text
in_blk           输入块（Conv3d + BatchNorm3d）                1,770,240 参数   1 层 BN
res_blk_list     6 个残差块（每块 2×Conv3d + 2×BatchNorm3d）  21,242,880 参数  12 层 BN
res_blk_last     残差尾块（Conv3d + BatchNorm3d）              1,770,240 参数   1 层 BN
up_blk_list      上采样块，数量 = log2(scale)；scale=4 时为 2 个
                 up_blk 内部只有 Conv3d + act + interpolate，不含归一化
out_blk          输出卷积（仅 Conv3d，无归一化）
```

## 3. 目录结构

```text
submodel/deep_encoder/
├── model.py              # LearnedPriorEncoder + 原 SRGAN Decoder
├── dataset.py            # 与原 decoder 预训练一致的数据接口
├── loss.py               # 与原 decoder 预训练一致的 loss 接口
├── train.py              # 从 0 训练 / 接力训练入口
├── evaluate_metrics.py   # 固定范围 PSNR、SSIM 及区域指标
├── README.md
├── checkpoints/          # 训练时自动生成
├── logs/                 # 训练时自动生成
└── metrics/              # 评估时自动生成
```

---

## 4. 单独训练

本子模型自带独立训练循环，**不依赖主模型**。它与 `submodel/decoder` 共用同一套循环，
只是把模型类换成 `LearnedPriorEncoder`。入口是 `submodel.deep_encoder.train`。

数据来源只需要一个体素数据根目录下的 `<病例>/gt_volume.nii.gz`——**预训练不使用任何投影**。
所以 `--data-root` 可以指向 `dataset/<数据集>/syn_data*` 的任一 GT 变体，选哪个变体就
决定了先验适配哪种 GT（见 §7.6）。

### 4.1 从 0 训练命令

在项目根目录运行：

```bash
python -m submodel.deep_encoder.train \
  --run-name dental_mean_detail_prior \
  --data-root dataset/dental/syn_data \
  --split-file data/dataset_split/dental_split.json \
  --conf conf/train.conf \
  --device cuda \
  --epochs 200 \
  --batch-size 1 \
  --num-workers 0 \
  --lr 1e-4 \
  --mse-lambda-3d 1.0 \
  --gd1-lambda 1.0 \
  --bone-lambda 0.05 \
  --bone-lower-hu 300 \
  --soft-mask-lambda 0.01 \
  --soft-window-low -160 \
  --soft-window-high 240 \
  --val-every 1 \
  --test-every 10 \
  --save-every 10
```

这是从 0 训练，**不要添加 `--resume`**。

`submodel/deep_encoder/train.py` 只是一层很薄的包装：它把
`--output-root` 强制指向本目录，然后 `runpy` 转发到 `submodel/decoder/train.py`。
所以**参数表与 `submodel/decoder` 完全一致**（见 §4.3），没有任何本模块独有参数。

**换数据集**：把 `--run-name`、`--data-root`、`--split-file` 三者同时换掉。thorax 示例：

```bash
python -m submodel.deep_encoder.train \
  --run-name thorax_deep_decoder_cbct_prior \
  --data-root dataset/thorax/syn_data_cbct_gt_v2 \
  --split-file data/dataset_split/thorax_split.json \
  --conf conf/train.conf \
  --device cuda \
  --epochs 200 \
  --batch-size 1 \
  --num-workers 0 \
  --lr 1e-4 \
  --mse-lambda-3d 1.0 \
  --gd1-lambda 1.0 \
  --bone-lambda 0.05 \
  --bone-lower-hu 300 \
  --soft-mask-lambda 0.01 \
  --soft-window-low -160 \
  --soft-window-high 240 \
  --val-every 1 \
  --test-every 10 \
  --save-every 10
```

⚠️ `--split-file` **必须显式指定**。它的默认值是 `data/dataset_split/dental_split.json`，
在 thorax 数据上忘记改会导致静默跳过绝大多数病例。另外主模型 `train.py` **没有**这个参数，
它固定读 `data/dataset_split/<datatype>_split.json`，两边不要混淆。

### 4.2 断点续训命令

本模块的 `--resume` 与主模型完全不同：它接收的是一个 **checkpoint 路径**（不是 epoch 标签），
并且在给了 `--resume` 之后 **`--epochs` 表示「再训练多少轮」，而不是总轮数**：

```bash
python -m submodel.deep_encoder.train \
  --run-name thorax_deep_decoder_cbct_prior_86 \
  --data-root dataset/thorax/syn_data_cbct_gt_v2 \
  --split-file data/dataset_split/thorax_split.json \
  --conf conf/train.conf \
  --device cuda \
  --resume submodel/deep_encoder/checkpoints/thorax_deep_decoder_cbct_prior/ckpt_best_val.pt \
  --epochs 100 \
  --lr 5e-5 \
  --batch-size 1 \
  --num-workers 0 \
  --mse-lambda-3d 1.0 \
  --gd1-lambda 1.0 \
  --bone-lambda 0.05 \
  --bone-lower-hu 300 \
  --soft-mask-lambda 0.01 \
  --soft-window-low -160 \
  --soft-window-high 240 \
  --val-every 1 \
  --test-every 10 \
  --save-every 10
```

`--resume` 的实际行为（`submodel/decoder/train.py:394`）：

| 项目 | 行为 |
|---|---|
| 路径解析 | 相对路径按**仓库根目录**解析；文件不存在直接 `FileNotFoundError` |
| 恢复内容 | `model`（`strict=True`）、`optimizer`、`scaler` |
| 起始轮次 | `start_epoch = checkpoint["epoch"] + 1`（`ckpt` 保存在第 N 轮**结束后**） |
| `--epochs` | **增量**轮数。与主模型的「新总轮数」语义相反，容易搞混 |
| `--lr` | 显式给出时**覆盖** checkpoint 里存的 Adam 学习率；不给则沿用旧值 |
| `best_val_loss` | 只有 `loss_config` 与 checkpoint 记录的一致时才继承；不一致（或新开了区域损失）会重置为 `inf` 并打印提示，避免新旧目标下的最佳值互相比烂 |

### 4.3 训练参数解析

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--run-name` | `dental_pretrain` | 日志和 checkpoint 目录名，建议显式指定。 |
| `--data-root` | `dataset/dental/syn_data` | 体数据根目录，只读取每例的 `gt_volume.nii.gz`。 |
| `--split-file` | `data/dataset_split/dental_split.json` | train/val/test 病例划分，**换数据集必须改**。 |
| `--conf` | `conf/train.conf` | Decoder 结构、μ 范围及默认 loss 配置。 |
| `--output-root` | 本目录 | 由 wrapper 强制指向 `submodel/deep_encoder`，不必手动传。 |
| `--device` | `cuda` | `cuda`、`cuda:0` 或 `cpu`。 |
| `--epochs` | `500` | **从 0 训练时是总轮数**；一旦给了 `--resume`，变成**增量轮数**。 |
| `--batch-size` | `1` | 256³ 体数据建议保持 1。 |
| `--num-workers` | `0` | DataLoader 进程数。 |
| `--lr` | 配置文件值 | Adam 学习率。给 `--resume` 时该值会覆盖 checkpoint 里存的 LR。 |
| `--resume` | 无 | **checkpoint 路径**（不是 epoch）；见 §4.2。 |
| `--mse-lambda-3d` | 配置文件值 | 基础 3D 重建项权重；旧命名为 MSE，实际使用 L1。 |
| `--gd1-lambda` | 配置文件值 | XYZ 三方向一阶梯度 loss 权重。 |
| `--bone-lambda` | `0` | GT 骨骼区域归一化 L1 权重。 |
| `--bone-lower-hu` | `300` | GT 骨骼 HU 下限。 |
| `--soft-mask-lambda` | `0` | GT 软组织区域归一化 L1 权重。 |
| `--soft-window-low` | `-160` | 软组织 HU 窗下限。 |
| `--soft-window-high` | `240` | 软组织 HU 窗上限。 |
| `--val-every` | `1` | 验证间隔。 |
| `--test-every` | `10` | 测试间隔。 |
| `--save-every` | `10` | 历史 checkpoint 保存间隔。 |
| `--no-amp` | 关闭 | 禁用 CUDA 混合精度。 |
| `--limit` | 无 | 只使用前 N 个训练病例，供 smoke test 使用。 |
| `--eval-limit` | 无 | val/test 各使用前 N 例。 |
| `--max-seconds` | `0` | 限时训练秒数，0 表示关闭。 |

输出位置：

```text
submodel/deep_encoder/logs/<run-name>/
submodel/deep_encoder/checkpoints/<run-name>/
```

主要权重：

```text
ckpt_latest.pt
ckpt_best_val.pt
ckpt_epoch_0010.pt
ckpt_epoch_0020.pt
...
```

---

## 5. 单独评估

### 5.1 评估命令

评估一个已经预训练好的 deep prior checkpoint：

```bash
python -m submodel.deep_encoder.evaluate_metrics \
  --checkpoint submodel/deep_encoder/checkpoints/thorax_deep_decoder_cbct_prior/ckpt_best_val.pt \
  --data-root dataset/thorax/syn_data_cbct_gt \
  --split-file data/dataset_split/thorax_split.json \
  --conf conf/train.conf \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --splits val test \
  --air-upper-hu -500 \
  --bone-lower-hu 300 \
  --output-dir submodel/deep_encoder/metrics/thorax_cbct_prior
```

评估 dental 数据时把前三行换回：

```bash
  --checkpoint submodel/deep_encoder/checkpoints/<dental-run-name>/ckpt_best_val.pt \
  --data-root dataset/dental/syn_data \
  --split-file data/dataset_split/dental_split.json \
```

`--output-dir` 建议**始终显式指定**。不指定时 wrapper 按 `metrics/<checkpoint文件名>`
生成目录，而所有 run 的最佳 checkpoint 都叫 `ckpt_best_val.pt`，因此不同数据集或不同
训练的评估结果会互相覆盖：

```text
submodel/deep_encoder/metrics/ckpt_best_val/     # 被多次评估共用，会互相覆盖
submodel/deep_encoder/metrics/thorax_cbct_prior/ # 本次评估，互不干扰
```

结果包括每病例和数据集汇总的固定范围 PSNR、RMSE、HU MAE、骨骼/软组织区域指标以及
与根目录评估协议兼容的 SSIM。

### 5.2 评估参数解析

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--checkpoint` | 必填 | 要评估的本模型 checkpoint。 |
| `--data-root` | `dataset/dental/syn_data` | 评估数据根目录，需与 checkpoint 的训练数据一致。 |
| `--split-file` | `data/dataset_split/dental_split.json` | 评估病例划分。 |
| `--conf` | `conf/train.conf` | 与训练一致。 |
| `--device` | `cuda` | 计算设备。 |
| `--batch-size` | `1` | 建议保持 1。 |
| `--num-workers` | `0` | DataLoader 进程数。 |
| `--splits` | `val test` | 评估划分。 |
| `--air-upper-hu` | `-500` | 空气区域 HU 上限。 |
| `--bone-lower-hu` | `300` | 骨骼区域 HU 下限。 |
| `--output-dir` | `metrics/<checkpoint文件名>` | 指标和体数据输出目录，建议显式指定以免覆盖。 |
| `--save-volumes` | 关闭 | 保存预测 sCT 和 GT NIfTI（编码为 HU）。 |
| `--no-amp` | 关闭 | 禁用评估 AMP。 |

几点约定：

- `--air-upper-hu` / `--bone-lower-hu` 是 **HU 空间**阈值。脚本先 `mu_to_hu()` 把预测和
  GT 换回 HU 再分区，所以无论 GT 是 pCT 还是 CBCT 都照抄默认值，不需要调整。
- `--save-volumes` 写出的"HU"是 `mu_to_hu(clamp(mu, 0, 0.09009))`。pCT-GT（μ 上限约
  0.088）和 CBCT-GT（μ 上限约 0.061）都没有触发 clamp，所以往返转换是精确的，可当真实 HU
  使用。
- 输出里的 `sCT-vs-pCT` 文案和 `sct_vs_pct_psnr_db` 键名是历史遗留的硬编码
  （`submodel/decoder/evaluate_metrics.py:197`）。评估 CBCT-GT 时它实际表示 `sCT-vs-CBCT`，
  含义按 `--data-root` 所指的 GT 变体理解，字段名不必改。

### 5.3 两个 GT 变体的先验实测对比

同一套划分上分别用 pCT 和 CBCT 作为 GT 预训练两个先验，再用各自的 GT 评估（不可互相对比，
见下）：

| GT 变体 | 划分 | PSNR(dB) | SSIM3D | HU MAE | bone MAE | tissue MAE | air MAE | 归一化 MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pCT（`thorax_deep_decoder`） | val | 40.9432 | 0.9796 | 13.55 | 96.31 | 31.48 | 7.89 | 8.19e-05 |
| pCT | test | 41.4746 | 0.9783 | 13.78 | 79.02 | 28.15 | 8.38 | 7.21e-05 |
| CBCT（`thorax_deep_decoder_cbct_prior`） | val | 40.3304 | 0.9586 | 16.73 | 96.41 | 41.61 | 9.96 | 9.47e-05 |
| CBCT | test | 40.2547 | 0.9413 | 18.14 | 89.35 | 42.69 | 10.32 | 9.74e-05 |

**跨 GT 变体的 PSNR/SSIM 不可直接比较。** 两者的归一化范围都是 `conf/train.conf` 的
`data.dental`（`0 .. 0.09009`），但 CBCT 本身含散射、射束硬化和噪声，作为拟合目标更难，
所以 CBCT 先验的归一化 MSE 反而更高（9.47e-05 vs 8.19e-05），PSNR/SSIM 更低。实测两者的
`bone_mae_hu` 几乎持平（96.41 vs 96.31），差距集中在 `tissue` 和 `air`——与 CBCT 在低衰减
区域噪声更大的特性一致。

要判断"哪个 teacher 更适合主模型"，只能各自和自己 GT 对齐地看绝对误差，或在主模型侧做
A/B（见第 6 节）。

> 这里报告的 PSNR 是**固定区间**口径（`(x - clamp_min)/(clamp_max - clamp_min)`），
> 可信；与主模型 `psnr_3d_clamp` 的自归一化口径不是一回事，见 §7.3。

---

## 6. 接入主模型

主模型训练时，本子模型的角色是**decoder 先验**。指定
`--pretrained_decoder <本子模型的 ckpt_best_val.pt>` 后：

1. 用 checkpoint 里的 `decoder` 权重初始化主模型的 SRGAN Decoder；
2. 用 checkpoint 里的 `feature_stem`（即本子模型的 `LearnedPriorEncoder`）构建一个
   **冻结 teacher**，用来把 GT 体数据编码成 latent，作为 latent 对齐损失的监督目标；
3. 再复制一份冻结的 decoder 作为 `prior_decoder_ref`，用于先验锚定损失（anchor）。

主模型训练时必须指定：

```text
--prior_encoder_type deep
```

此时主模型会将完整 256³ pCT 送入冻结 teacher，由 teacher 内部完成平均值、高频残差、
31 通道细节编码和 latent 生成。不会在主模型外部提前平均池化。

根目录 `evaluate.py` 评估主模型时**不使用 pCT teacher**，所以无需提供
`--prior_encoder_type deep`；只需要按照训练配置重建正确的 Adapter 结构。

接入主模型有两套迁移策略，用 `--transfer_schedule` 选择：

| 策略 | 取值 | 特点 |
|---|---|---|
| 四阶段（默认） | `four_phase` | 解冻顺序**从输出端向输入端**逐级推进，带 latent 退火与先验锚定；推荐用于新实验 |
| legacy 三阶段 | `legacy_three_stage` | 复现早期 `dental_prior_adapter` 的「冻结 → 联合 → 终端精修」序列 |

下面两节按同样的流程分别介绍。

### 6.1 策略 A：四阶段（`--transfer_schedule four_phase`，默认）

四阶段的骨架是「**A 训主干 → B 收窄到深层 → C 逐级解冻 decoder → D 全参数精修**」，
并且用两个随阶段变化的权重（latent 对齐、先验锚定）把 decoder 先验护住。

#### 6.1.1 从 0 训练命令

以下命令的主干（Encoder/Aggregator）**从 0 训练**，decoder 来自本子模型先验：

```bash
python train.py \
  --name thorax_prior_four_phase_cbct_prior \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --require-gt-source cbct-fixed \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 200 \
  --pretrained_decoder submodel/deep_encoder/checkpoints/thorax_deep_decoder_cbct_prior/ckpt_best_val.pt \
  --prior_encoder_type deep \
  --use_adapter \
  --adapter_type cnn \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 0.1 \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --latent_stat_lambda 0.1 \
  --phase_a_epochs 50 \
  --phase_a_encoder_lr_factor 1.0 \
  --phase_a_aggregator_lr_factor 1.0 \
  --phase_b_epochs 50 \
  --phase_c_epochs 80 \
  --decoder_lr_factor 0.1 \
  --decoder_core_lr_factor 0.01 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

这条命令没有 `--pretrained_backbone`，即主干从 0 训练，因此**必须**用
`--phase_a_encoder_lr_factor` / `--phase_a_aggregator_lr_factor` 打开 Phase A 的主干；
若省略这两个参数，Phase A 会冻结随机初始化的主干、只剩零初始化的 Adapter 可训练，
程序会直接报错而不是空跑。若已有旧主模型 checkpoint，可改为传入 `--pretrained_backbone`
并省略这两个参数。

Phase A/B/C 之和为 180，`--epochs 200` 的剩余 20 轮归 Phase D。

**换数据集**：dental 数据把前三行改为

```text
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --pretrained_decoder submodel/deep_encoder/checkpoints/dental_mean_detail_prior/ckpt_best_val.pt \
```

并去掉 `--require-gt-source`（dental 的 `gt_volume` 没有该字段）。用 CBCT 作 GT 时见 §7.6。

**换 Adapter 结构**：Transformer Adapter 把 `--adapter_type cnn` 换成

```text
--adapter_type transformer
--adapter_transformer_pool_size 8
--adapter_transformer_layers 2
--adapter_transformer_heads 4
--adapter_transformer_dropout 0.1
```

#### 6.1.2 四个阶段分别做什么

阶段边界是**绝对 epoch**（`_training_stage(epoch)`），所以下面的区间随 `--phase_*_epochs`
和起点平移。以 §6.1.1 的 `--epochs 200` + `50/50/80` 为例：

```text
epoch:   0 ─────── 50 ─────── 100 ──────────── 180 ── 200
         │ Phase A │ Phase B │      Phase C      │Phase D│
         │  50 轮  │  50 轮  │      80 轮        │ 20 轮 │
                            └┬───┬───┬───┬──────┘
                            lvl1 lvl2 lvl3 lvl4   (每 20 轮解锁一级 decoder)
                            100  120  140  160
```

**Phase A（第 0–49 轮）**

| 组 | LR factor | 相对基础 LR |
|---|---:|---:|
| `encoder_early`（layer1/2） | 1.0 | 1.0× |
| `encoder_late`（layer3/4） | 1.0 | 1.0× |
| `aggregator` | 1.0 | 1.0× |
| `adapter` | `--adapter_lr_factor` | 0.1× |
| **decoder 四组** | 不在 factor 表内 → 0 | **0，完全冻结在先验上** |

- latent 对齐权重：`--latent_lambda`（示例为 0.1）恒定；注意**默认值是 0，即关闭**
- 先验锚定权重：0

目的：decoder 冻结在先验上，先让**投影支路**（Encoder + Aggregator）把输出的 latent
对齐到先验期望的基底。监督目标是冻结 teacher 从 `volume_gt` 算出的 latent，
损失为 `combined = smooth_l1 + latent_cosine_lambda·cosine + latent_stat_lambda·(mean_l1 + std_l1)`。
这个阶段固定随机初始化的主干是不可行的（程序会报错），所以必须用
`--phase_a_*_lr_factor` 打开主干，或改用 `--pretrained_backbone` / `--resume`。

**Phase B（第 50–99 轮）**

| 组 | LR factor | 相对基础 LR |
|---|---:|---:|
| `encoder_early` | 不在 factor 表内 → 0 | **0，冻结** |
| `encoder_late` | `--phase_b_encoder_lr_factor` | 0.2× |
| `aggregator` | `--phase_b_aggregator_lr_factor` | 0.5× |
| `adapter` | `--adapter_lr_factor` | 0.1× |
| **decoder 四组** | 0 | **仍冻结** |

- latent 对齐权重：0.1 → 0.07（线性退火，终点由 `--phase_b_latent_end_factor` 决定）
- 先验锚定权重：0

目的：只微调 Encoder 后半段与 Aggregator，把它们锁进先验的 latent 空间；同时开始
放松对 teacher 的依赖，让 Aggregator 有机会自由学习。

**Phase C（第 100–179 轮）★ 核心阶段**

主干大幅降速，decoder **逐级解冻**，解冻顺序是**从输出端向输入端**：

| 级别 | Phase C 内偏移 | 新解锁的 decoder 模块 | 参数量 | 占 decoder |
|---|---|---|---:|---:|
| lvl1 | 第 0–19 轮 | `out_blk`（输出卷积） | 433 | 0.00% |
| lvl2 | 第 20–39 轮 | `up_blk_list[-1]`（最后一个上采样块） | 34,592 | 0.14% |
| lvl3 | 第 40–59 轮 | `up_blk_list[0]`（前一个上采样块） | 553,088 | 2.18% |
| **lvl4** | 第 60–79 轮 | `in_blk` + `res_blk_last` + `res_blk_list`（残差主体） | **24,783,360** | **97.68%** |

⚠️ 注意 lvl4 一次释放了 decoder 的 **97.68%** 参数，并且**全部 14 层 BatchNorm3d 都在这
一级**。这是一个强烈的状态切换，详见 §7.2。

| 组 | LR factor |
|---|---:|
| `encoder_early` / `encoder_late` | `--phase_c_backbone_lr_factor`（0.1） |
| `aggregator` | `--phase_c_aggregator_lr_factor`（0.3） |
| `adapter` | `--adapter_lr_factor`（0.1） |
| `decoder_out` | `--decoder_lr_factor`（0.1），lvl1 起 |
| `decoder_up_high` | `--decoder_lr_factor`，lvl2 起 |
| `decoder_up_low` | `--decoder_lr_factor`，lvl3 起 |
| `decoder_core` | `--decoder_core_lr_factor`（0.01），lvl4 起 |

- latent 对齐权重：0.07 → `--phase_c_latent_end_factor`（0.1）
- **先验锚定权重：`--prior_anchor_lambda`（0.1）开启**

anchor 是这一阶段的护栏：`prior_decoder_ref` 是预训练 decoder 的冻结副本，anchor 损失
约束训练中的 decoder 输出不要偏离先验输出太远，防止 decoder 在解冻后把先验冲掉。

**Phase D（第 180–199 轮）**

| 组 | LR factor |
|---|---:|
| `encoder_early` / `encoder_late` | `--phase_d_backbone_lr_factor`（0.01） |
| `aggregator` | `--phase_d_aggregator_lr_factor`（0.05） |
| `adapter` | `--adapter_lr_factor`（0.1） |
| `decoder_out` / `decoder_up_low` / `decoder_up_high` | `--decoder_lr_factor`（0.1） |
| `decoder_core` | `--decoder_core_lr_factor`（0.01） |

- latent 对齐权重：`--phase_c_latent_end_factor`（0.1）→ 0
- 先验锚定权重：`--prior_anchor_lambda × --phase_d_anchor_factor` = 0.1 × 0.25 = 0.025

全参数小 LR 精修收敛，latent 对齐完全退出，只靠 3D/2D 监督。

**全程恒定的损失权重**：`mse_lambda_3d`(1.0)、`gd1_lambda`(1.0)、`bone_lambda`(0.05)、
`soft_mask_lambda`(0.01)、`ssim_lambda`(0.01)、`mse_lambda_2d`(0.01，重投影)。
随阶段变化的只有 **latent 权重**和 **anchor 权重**。

**读日志时的两个坑**

1. `train/logs/<name>/train_lr.txt` 里可能出现 `G_lr:0.0`，这是**正常的**。它记录的是
   `param_groups[0]["lr"]`，而 group[0] 是 `encoder_early`——它在 Phase B 的 factor 表里
   不存在，取默认 0.0。按下表对照即可：

   | epoch 区间 | `G_lr` | 含义 |
   |---|---:|---|
   | 0–49 | `0.0001` | Phase A |
   | 50–99 | `0.0` | Phase B（`encoder_early` 被冻结，不是出错） |
   | 100–179 | 逐级下降 | Phase C |
   | 180–199 | 更低 | Phase D |

2. 判断当前处于哪个时期，别靠 loss 大小，看 checkpoint 里存的 `training_stage`
   （1=A, 2=B, 3=C, 4=D），或按上面的 epoch 表对照。

#### 6.1.3 断点续训命令

主模型的续训语义与子模型**完全相反**，先看清三点：

- `--resume` 是**开关**，配 `--resume_name <epoch>`；`--epochs` 给的是**新的总轮数**（不是增量）。
- `--resume_name` 只会在 `train/checkpoints/<--name>/ckpt_history/ckpt_<resume_name>` 下查找，
  **跨实验续训必须先把 ckpt 复制过去**，否则 `FileNotFoundError`。
- `ckpt_N` 保存在第 N 轮**结束后**，所以 `--resume_name 199` 从 epoch 200 开始
  （`ckpt` 里存 `iter = N + 1`）。

**情形一：只是继续训练（保留 checkpoint 里的 decoder）**

```bash
python train.py \
  --name thorax_prior_four_phase_cbct_prior_cont \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --require-gt-source cbct-fixed \
  --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling uniform --is_train \
  --epochs 300 \
  --resume --resume_name 199 \
  --lr-decay-restart \
  --pretrained_decoder submodel/deep_encoder/checkpoints/thorax_deep_decoder_cbct_prior/ckpt_best_val.pt \
  --prior_encoder_type deep \
  --use_adapter --adapter_type cnn --adapter_hidden_channels 64 \
  --adapter_lr_factor 0.1 \
  --latent_lambda 0.1 --latent_cosine_lambda 0.1 --latent_stat_lambda 0.1 \
  --phase_a_epochs 50 --phase_b_epochs 50 --phase_c_epochs 80 \
  --decoder_lr_factor 0.1 --decoder_core_lr_factor 0.01 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01
```

注意两点：

- `--pretrained_decoder` 仍然**必须**传（虽然它的权重不会生效，见下），否则
  `use_four_phase` 会变成 `False`，四阶段调度与 latent 对齐全部失效。
- 由于阶段边界是绝对 epoch，从 epoch 200 续训且不重放阶段表时，**整段都会落在 Phase D**。

**情形二：重置 decoder 回先验，并完整重放四阶段**

如果 checkpoint 里的 decoder 已经被 Phase C 的 lvl4 解冻冲偏（§7.2），想把它找回先验，
需要额外两个开关。原因是 `train.py` 的加载逻辑：

```python
if args.pretrained_decoder is not None and not args.resume:   # ← --resume 时整块跳过
    G_render.decoder.load_state_dict(pretrained["decoder"], strict=True)
```

即 **`--resume` 会让 `--pretrained_decoder` 的权重被静默忽略**，它此时只作为"开启四阶段"
的开关。所以要显式要求重注入：

```bash
# --resume_name 只在 <--name> 自己的目录下查找，跨实验必须先复制
mkdir -p train/checkpoints/thorax_prior_four_phase_v2_prior_reinject/ckpt_history
cp train/checkpoints/thorax_prior_four_phase_v2_cleansplit/ckpt_history/ckpt_199 \
   train/checkpoints/thorax_prior_four_phase_v2_prior_reinject/ckpt_history/

python train.py \
  --name thorax_prior_four_phase_v2_prior_reinject \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --require-gt-source cbct-fixed \
  --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling uniform --is_train \
  --epochs 400 \
  --resume --resume_name 199 \
  --resume_reload_decoder \
  --phase_restart \
  --lr-decay-restart \
  --pretrained_decoder submodel/deep_encoder/checkpoints/thorax_deep_decoder_cbct_prior/ckpt_best_val.pt \
  --prior_encoder_type deep \
  --use_adapter --adapter_type cnn --adapter_hidden_channels 64 \
  --adapter_lr_factor 0.1 \
  --latent_lambda 0.1 --latent_cosine_lambda 0.1 --latent_stat_lambda 0.1 \
  --phase_a_epochs 50 --phase_b_epochs 50 --phase_c_epochs 80 \
  --phase_a_encoder_lr_factor 1.0 --phase_a_aggregator_lr_factor 1.0 \
  --decoder_lr_factor 0.1 --decoder_core_lr_factor 0.01 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01
```

四个参数缺一不可：

| 参数 | 值 | 漏掉的后果 |
|---|---|---|
| `--epochs` | `400` | `ckpt_199` 存的是 `iter = 200`，`range(200, 200)` 为空 → **一轮都不跑**，静默空转 |
| `--resume_reload_decoder` | 必加 | decoder 沿用 checkpoint 里已漂移的权重，先验拿不回来 |
| `--phase_restart` | 必加 | 直接落进 Phase D，A/B/C 全部跳过 |
| `--lr-decay-restart` | 必加 | `decay = 0.5**(200//50) = 0.0625`，再乘 Phase D 的 0.01 → LR ≈ 6e-8，等于冻结 |

上例重放后的绝对区间：Phase A 200–249、B 250–299、C 300–379、D 380–399。
`phase_d_epochs` 会按新终点自动重算；余量不足时程序直接报错并给出所需 `--epochs`。
`--phase_restart` 与 `--lr-decay-restart` 是配套的，只给前者时程序会警告。

续训前先做两项检查：

```bash
# 1) 确认目标实验的训练进程已经结束（正在写入时会覆盖同一份 ckpt_latest）
ps -ef | grep train.py | grep -v grep

# 2) 确认可用的 checkpoint 编号
ls train/checkpoints/<name>/ckpt_history/
```

#### 6.1.4 参数解析

**数据与几何**

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--datadir` | `dataset/dental/syn_data` | 体数据根目录，主模型会读该目录下的投影与 `gt_volume.nii.gz`。 |
| `--datatype` | `dental` | `dental` / `spine` / `thorax` / `Walnuts`；决定 mask 划分与 clamp 范围。 |
| `--require-gt-source` | 无 | 硬校验 `transforms.json` 的 `gt_source`，见 §7.6。 |
| `--train_scale` | `4` | Decoder 上采样倍数，必须与 decoder 预训练时一致（决定 `up_blk_num`）。 |
| `--fusion` | `ada` | 多视角特征融合策略。 |
| `--start` / `--end` | `0` / `360` | 扫描角度范围。 |
| `--nviews` | `20` | 稀疏视角数。 |
| `--angle_sampling` | `uniform` | `uniform` / `random`。thorax 真实投影数据只能用 `uniform`（`random` 分支用不了探测器偏移，程序会拦下）。 |

**先验来源**

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--pretrained_decoder` | 无 | 本子模型的 `ckpt_best_val.pt`。**四阶段必备**（同时决定 `use_four_phase`）。 |
| `--prior_encoder_type` | `shallow` | 必须设为 `deep` 才能匹配本子模型的 `LearnedPriorEncoder`；若 checkpoint 来自 `submodel/decoder`（浅层）则保持默认。 |
| `--pretrained_backbone` | 无 | 旧主模型 checkpoint，用来跳过 Phase A 的主干预热。 |
| `--stage0_decoder_lr_factor` | 无 | 只在不启用四阶段（`stage 0`，即无 adapter 的普通联合训练）时生效，用来减速或冻结 decoder。 |

**Adapter 与 latent**

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--use_adapter` | 关闭 | 开启 `LatentAdapter`，位于 latent 之后、decoder 之前。 |
| `--adapter_type` | `cnn` | `cnn` / `transformer`。 |
| `--adapter_hidden_channels` | `64` | CNN adapter 隐藏通道数。 |
| `--adapter_lr_factor` | `1.0` | adapter 的学习率因子。**为 0 时 adapter 全程冻结，而它的最后一层是零初始化，所以效果恒等于恒等映射**——相当于没有 adapter。想要 adapter 真正学习必须给正值。 |
| `--adapter_use_global_alpha` | 关闭 | 启用可学习的全局缩放。 |
| `--latent_lambda` | `0.0` | latent 对齐基础权重，乘上各阶段的退火因子得到实际权重。**默认 `0.0` 表示完全关闭 latent 对齐**（`_latent_weight` 在 `latent_lambda <= 0` 时直接返回 0），所以四阶段实验必须显式传 `0.1`；`--latent_cosine_lambda` / `--latent_stat_lambda` 两者都只在这个权重 > 0 时才起作用。 |
| `--latent_cosine_lambda` | `0.1` | 余弦项权重。 |
| `--latent_stat_lambda` | `0.1` | 均值/标准差统计项权重。 |

**四阶段调度**

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--phase_a_epochs` | `20` | Phase A 轮数。 |
| `--phase_b_epochs` | `40` | Phase B 轮数。 |
| `--phase_c_epochs` | `80` | Phase C 轮数；解码器 4 级按此四等分。 |
| `--phase_c_hold_epochs` | `0` | Phase C 末尾额外保持轮数（不改变已解锁层级）。 |
| `--phase_a_encoder_lr_factor` | `0.0` | Phase A 的 Encoder LR 因子。主干从 0 训练时必须 > 0。 |
| `--phase_a_aggregator_lr_factor` | `0.0` | 同上，Aggregator。 |
| `--phase_b_encoder_lr_factor` | `0.2` | Phase B 的 `encoder_late` 因子。 |
| `--phase_b_aggregator_lr_factor` | `0.5` | Phase B 的 Aggregator 因子。 |
| `--phase_c_backbone_lr_factor` | `0.1` | Phase C 的主干因子。 |
| `--phase_c_aggregator_lr_factor` | `0.3` | Phase C 的 Aggregator 因子。 |
| `--phase_d_backbone_lr_factor` | `0.01` | Phase D 的主干因子。 |
| `--phase_d_aggregator_lr_factor` | `0.05` | Phase D 的 Aggregator 因子。 |
| `--decoder_lr_factor` | `0.1` | decoder 的 `out`/`up_low`/`up_high` 三组因子。 |
| `--decoder_core_lr_factor` | `0.01` | decoder 残差主体（`in_blk`/`res_blk*`）因子。 |
| `--prior_anchor_lambda` | `0.1` | 先验锚定损失权重，仅 Phase C/D 生效。设为 0 会跳过锚定（并少建一份冻结 decoder）。 |
| `--phase_d_anchor_factor` | `0.25` | Phase D 的锚定权重再乘此因子。 |
| `--phase_b_latent_end_factor` | `0.7` | Phase B 末 latent 权重相对值。 |
| `--phase_c_latent_end_factor` | `0.1` | Phase C 末 latent 权重相对值。建议提到 `0.3`，见 §7.2。 |
| `--freeze_decoder_bn_stats` | 关闭 | 让 decoder 的 BatchNorm 始终用预训练 running 统计。**强烈建议开启**，见 §7.2。 |

**学习率**

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--init-lr` | 配置文件 `lr_sche.init_lr` | 覆盖基础学习率。 |
| `--lr-step-size` | 配置文件值（50） | 覆盖衰减间隔；设得很大即等于恒定学习率。 |
| `--lr-gamma` | 配置文件值（0.5） | 覆盖衰减倍率。 |
| `--lr-decay-restart` | 关闭 | 让衰减从续训 epoch 重新起算；续训时基本都要开。细节见 §7.1。 |

**续训**

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--resume` | 关闭 | 开关，配 `--resume_name` 使用。 |
| `--resume_name` | 无 | epoch 标签（不是路径），在 `<--name>/ckpt_history/` 下查找。 |
| `--resume_reload_decoder` | 关闭 | 恢复后用 `--pretrained_decoder` 覆盖 decoder，并清空 decoder 四个参数组的 Adam 动量。 |
| `--phase_restart` | 关闭 | 把阶段表计时原点锚定到续训 epoch 并完整重放。 |

**损失与开销**

⚠️ 下表中默认值为 `0.0` 的三项**不在 `conf/train.conf` 里**，所以不显式传就等于关闭。
`conf/train.conf` 的 `G_loss` 只有 `mse_lambda_2d=0.01`、`mse_lambda_3d=1`、`gd1_lambda=1`。

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--mse_lambda_3d` | 配置文件值（`1`） | 基础 3D 重建项权重；名为 MSE，实际用 L1。 |
| `--gd1_lambda` | `1.0` | XYZ 三方向一阶梯度损失权重。 |
| `--mse_lambda_2d` | `0.01` | 2D 重投影损失权重。 |
| `--bone_lambda` / `--bone_lower_hu` | **`0.0`** / `300.0` | GT 骨骼区域归一化 L1 及 HU 阈值；默认关闭。 |
| `--soft_mask_lambda` / `--soft_window_low` / `--soft_window_high` | **`0.0`** / `-160.0` / `240.0` | 软组织区域归一化 L1 及 HU 窗；默认关闭。 |
| `--ssim_lambda` | **`0.0`** | SSIM 损失权重；默认关闭。 |
| `--query_chunk_size` | `25000` | 逐点查询的 chunk 大小，显存不够就调小。 |

### 6.2 策略 B：legacy 三阶段（`--transfer_schedule legacy_three_stage`）

这是早期 `dental_prior_adapter` 用的序列，冻结/解冻方向与四阶段**相反**：先只训 Adapter，
再**一次性全解冻 decoder**，最后收回只留输出端。

#### 6.2.1 训练命令

`dental_prior_adapter` 的等价命令（参数名已改用现行的 `--legacy_*`）：

```bash
python train.py \
  --name dental_prior_adapter \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 200 \
  --transfer_schedule legacy_three_stage \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_latest.pt \
  --pretrained_backbone train/checkpoints/dental_prior_transfer_after_refine/ckpt_history/ckpt_299 \
  --use_adapter \
  --adapter_type cnn \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --legacy_stage1_epochs 20 \
  --legacy_stage2_epochs 100 \
  --legacy_stage3_backbone_lr_factor 0.01 \
  --decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

三点注意：

1. `--transfer_schedule` 的默认值是 `four_phase`，复现这套序列**必须显式**写
   `legacy_three_stage`。
2. 因为 `--legacy_stage1_epochs > 0` 会冻结主干，所以**必须**提供
   `--pretrained_backbone` 或 `--resume`，否则程序报错。
3. `--legacy_stage1_epochs + --legacy_stage2_epochs` 必须**小于** `--epochs`，至少留 1 轮给
   Stage 3。上例 20 + 100 = 120 < 200 ✓。

**thorax 实例：在 CBCT 数据上跑策略 B，主干沿用四阶段实验**

```bash
python train.py \
  --name thorax_prior_decoder_with_adapter \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --require-gt-source cbct-fixed \
  --train_scale 4 \
  --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 200 \
  --transfer_schedule legacy_three_stage \
  --pretrained_decoder submodel/deep_encoder/checkpoints/thorax_deep_decoder_cbct_prior/ckpt_best_val.pt \
  --prior_encoder_type deep \
  --pretrained_backbone train/checkpoints/thorax_prior_four_phase_v2_prior_reinject/ckpt_history/ckpt_399 \
  --use_adapter \
  --adapter_type cnn \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --prior_anchor_lambda 0 \
  --freeze_decoder_bn_stats \
  --legacy_stage1_epochs 20 \
  --legacy_stage2_epochs 100 \
  --legacy_stage3_backbone_lr_factor 0.01 \
  --decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

⚠️ **上面的命令刻意不带行内注释。** bash 的 `\` 续行符**必须是该行最后一个字符**，后面跟
`# 注释` 会让命令在此处断开（实测 `exit=127`）；把注释放成独立一行同样会断开。所以多行命令
里不能夹注释，差异只能用下面的表格表达。

**`--pretrained_backbone` 该取哪个 checkpoint 见 §7.4。**

**相对「把上面 dental 命令照抄到 thorax」的 6 处改动**：

```text
（仅示意差异，无续行符，不可直接执行）
  --require-gt-source cbct-fixed              🆕 硬校验 GT 来源
  --prior_encoder_type deep                   🆕 必加，不写直接崩
  --prior_anchor_lambda 0                     🆕 legacy 用不到 anchor，省约 100 MB
  --freeze_decoder_bn_stats                   🆕 防 Stage 2 的 BN 跳变
  --pretrained_decoder .../ckpt_best_val.pt   🆕 原用 ckpt_latest.pt
  --pretrained_backbone .../ckpt_399          🆕 原用 ckpt_340
```

| 参数 | 改动 | 原因 |
|---|---|---|
| <span style="color:#d32f2f">**`--prior_encoder_type deep`**</span> | **必加** | `--pretrained_decoder` 是 deep_encoder 的 checkpoint，其 `feature_stem` 是 `LearnedPriorEncoder`（8 个张量，含 `detail_downsample`）。默认的 `shallow` 会去建 `PriorFeatureStem`（4 个张量），而 `trainer.py` 用 `strict=True` 加载 → **直接 `RuntimeError`**。 |
| <span style="color:#d32f2f">**`--require-gt-source cbct-fixed`**</span> | 建议加 | 硬校验每例 `transforms.json` 的 `gt_source`。`--datadir` 指向 `syn_data_cbct_gt_v2` 时应当开启。 |
| <span style="color:#d32f2f">**`--freeze_decoder_bn_stats`**</span> | 建议加 | Stage 2 在第 20 轮一次性解冻全部 decoder 模块，**14 层 `BatchNorm3d` 全在其中**（机制同 §7.2）。而 legacy **没有 anchor 保护**（stage 6/7/8 的 anchor 权重恒为 0），比四阶段更裸。 |
| <span style="color:#d32f2f">**`--prior_anchor_lambda 0`**</span> | 可选 | legacy 全程 anchor 权重为 0，但 `use_four_phase and prior_anchor_lambda > 0` 仍会构建 `prior_decoder_ref`（25.4M 参数的冻结 decoder 副本 ≈ 100 MB 显存）。设 0 可跳过，无副作用。 |
| <span style="color:#d32f2f">**`--pretrained_decoder` 用 `ckpt_best_val.pt`**</span> | 可选 | `ckpt_latest.pt` 是 epoch 200 的状态，`ckpt_best_val.pt` 是 epoch 190（val 最优）。decoder 预训练用的是可信的固定区间口径，按 best_val 取更稳妥。 |

漏掉 `--prior_encoder_type deep` 的报错可以精确复现（实测）：

```text
RuntimeError: Error(s) in loading state_dict for PriorFeatureStem:
    Missing key(s) in state_dict: "0.weight", "0.bias", "2.weight", "2.bias".
    Unexpected key(s) in state_dict: "detail_downsample.weight", "detail_downsample.bias",
        "feature_stem.0.weight", "feature_stem.0.bias", "feature_stem.2.weight",
        "feature_stem.2.bias", "feature_stem.4.weight", "feature_stem.4.bias".
```

#### 6.2.2 三个阶段分别做什么

```text
epoch:   0 ─── 20 ────────────── 120 ──────────── 200
         │  S1  │       S2        │      S3       │
         │ 20轮 │     100 轮      │     80 轮     │
         │只训adapter│ decoder 全解冻 │ 收回只留输出端 │
```

**Stage 1（第 0–19 轮，内部 `stage 6`）**

| 组 | LR | 状态 |
|---|---:|---|
| `adapter` | `decay × --adapter_lr_factor`（1.0） | **唯一可训练** |
| encoder / aggregator / decoder | 0 | 全部冻结 |

- latent 对齐权重：`--latent_lambda`（0.1）恒定
- 先验锚定权重：0

目的：只让 Adapter 把投影支路的 latent 分布对齐到 decoder 期望的基底，其余全部保持
预训练状态不动。这是"两段式"策略里最关键的安全垫——它让后面的全解冻不会突然冲击分布。

**Stage 2（第 20–119 轮，内部 `stage 7`）**

| 组 | LR factor | 相对基础 LR |
|---|---:|---:|
| `encoder_early` / `encoder_late` | 1.0 | 1.0× |
| `aggregator` | 1.0 | 1.0× |
| `adapter` | `--adapter_lr_factor` | 1.0× |
| `decoder_out` / `decoder_up_low` / `decoder_up_high` | `--decoder_lr_factor` | 0.1× |
| `decoder_core` | `--decoder_lr_factor` | **0.1×**（注意：不是 `decoder_core_lr_factor`） |

- latent 对齐权重：`0.1 × linear(0.5 → 0)`，即在 100 轮内从 0.05 线性退到 0
- 先验锚定权重：**0**（legacy 三阶段不用 anchor）

decoder 的全部模块（含全部 14 层 BatchNorm3d）在**第 20 轮一次性解冻**。

**Stage 3（第 120–199 轮，内部 `stage 8`）**

| 组 | LR factor |
|---|---:|
| `encoder_early` / `encoder_late` | `--legacy_stage3_backbone_lr_factor`（0.01） |
| `aggregator` | 同上（0.01） |
| `adapter` | `--adapter_lr_factor`（1.0） |
| `decoder_out` / `decoder_up_high` | `--decoder_lr_factor`（0.1） |
| `decoder_up_low` / `decoder_core` | **0（重新冻结）** |

- latent 对齐权重：0
- 先验锚定权重：0

decoder **被收回**，只留 `out_blk` 与 `up_blk_list[-1]` 两个模块继续微调；主干降到 0.01
做终端精修。

#### 6.2.3 断点续训命令

续训机制与 §6.1.3 **完全相同**（`--resume` 开关 + `--resume_name` epoch + 复制 ckpt）。
差别在于阶段边界由 `--legacy_stage1_epochs` / `--legacy_stage2_epochs` 决定，而且因为边界
同样是**绝对 epoch**，直接续训也会跳过前两阶段。要让调度重新计时，同样用 `--phase_restart`：

```bash
mkdir -p train/checkpoints/dental_prior_adapter_cont/ckpt_history
cp train/checkpoints/dental_prior_adapter/ckpt_history/ckpt_199 \
   train/checkpoints/dental_prior_adapter_cont/ckpt_history/

python train.py \
  --name dental_prior_adapter_cont \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling uniform --is_train \
  --epochs 400 \
  --resume --resume_name 199 \
  --phase_restart \
  --lr-decay-restart \
  --transfer_schedule legacy_three_stage \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_latest.pt \
  --use_adapter --adapter_type cnn --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --latent_lambda 0.1 --latent_cosine_lambda 0.1 \
  --legacy_stage1_epochs 20 \
  --legacy_stage2_epochs 100 \
  --legacy_stage3_backbone_lr_factor 0.01 \
  --decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01
```

`--epochs 400` = 200（已训）+ 200（重放整张三阶段表：20 + 100 + 80）。Stage 3 的余量由
`legacy_stage1 + legacy_stage2` 与总轮数的差决定，不足时 `train_args.py` 直接
`parser.error` 拒绝启动。

如果只想让 decoder 找回先验而不重放阶段表，把 `--phase_restart` 换成
`--resume_reload_decoder`（可同时使用）。

#### 6.2.4 参数解析：与四阶段的差异

legacy 三阶段独有的参数：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--transfer_schedule` | `four_phase` | 设为 `legacy_three_stage` 才走本节流程。 |
| `--legacy_stage1_epochs` | `20` | Stage 1 轮数（只训 adapter）。> 0 时必须有 `--pretrained_backbone`。 |
| `--legacy_stage2_epochs` | `100` | Stage 2 轮数（decoder 全解冻）。 |
| `--legacy_stage3_backbone_lr_factor` | `0.01` | Stage 3 的主干/Aggregator 因子。 |

`--phase_*`、`--prior_anchor_lambda`、`--phase_d_anchor_factor`、
`--phase_b/c_latent_end_factor`、`--decoder_core_lr_factor`、`--freeze_decoder_bn_stats`
**在这套流程里都不生效**（除 `--phase_restart` 作为调度重启开关仍然有效）。

两者的对照：

| 维度 | 四阶段 | legacy 三阶段 |
|---|---|---|
| decoder 解冻方式 | 从输出端向输入端**逐级**（4 级） | 第 2 阶段**一次性全解冻**，第 3 阶段收回 |
| 第 1 阶段训什么 | 主干（encoder + aggregator）+ adapter | **只有 adapter** |
| 先验锚定（anchor） | Phase C/D 启用（0.1 → 0.025） | **全程关闭** |
| latent 退火 | A 恒定 → B 0.7 → C 0.1 → D 0 | Stage 2 内 0.5 → 0 |
| `decoder_core` 因子 | `--decoder_core_lr_factor`（0.01） | 与其余 decoder 组同用 `--decoder_lr_factor`（0.1） |
| 依赖 `--pretrained_backbone` | 否（可用 `--phase_a_*_lr_factor` 从 0 训） | **是**（Stage 1 冻结主干） |
| 参考实验 | `thorax_prior_four_phase_cbct_prior_v2` | `dental_prior_adapter` |

有趣的是，`dental_prior_adapter` 一次解冻全部 14 层 BN 却毫无波动（20 轮 val PSNR
28.745 → 30.154，120 轮 31.224 → 31.279），而四阶段在 lvl4 会崩。原因不在于"解冻多少"，
而在于它的主干来自 `--pretrained_backbone`、与 decoder 同域，**latent 分布本就匹配**。
详见 §7.2。

---

## 7. 运维经验与注意事项

### 7.1 学习率控制与提高学习率

主模型训练到 200 epoch 后往往还没收敛，但此时学习率已被 `lr_sche` 的衰减压到很低的水平。
以 `thorax_prior_joint_no_adapter` 为例，`train/logs/<实验名>/train_lr.txt` 记录的实测学习率：

```text
Epoch:0    G_lr:0.0001      # 1e-4
Epoch:80   G_lr:5e-05       # 第 50 轮衰减一次
Epoch:120  G_lr:2.5e-05     # 第 100 轮
Epoch:160  G_lr:1.25e-05    # 第 150 轮，此时只有初值的 1/8
Epoch:198  G_lr:1.25e-05
```

也就是 `decay = gamma ** (epoch // step_size) = 0.5 ** (epoch // 50)`。**断点续训时这个衰减
不会重置**，如果只是重新加载 checkpoint，第 200 轮仍会以 1/8 的学习率继续跑。因此有 4 个
学习率开关：`--init-lr`、`--lr-decay-restart`、`--lr-step-size`、`--lr-gamma`（见 §6.1.4）。

三种配置下 `encoder` 参数组的实际学习率（续训起点为 epoch 200，数值由 `_lr_multiplier` 实算）：

| 配置 | epoch 200 | epoch 249 | epoch 250 |
|---|---:|---:|---:|
| 不加 `--lr-decay-restart`（`--init-lr 3e-4`） | 1.875e-05 | 1.875e-05 | 9.375e-06 |
| `--init-lr 1e-4 --lr-decay-restart` | 1.000e-04 | 1.000e-04 | 5.000e-05 |
| `--init-lr 3e-4 --lr-decay-restart` | 3.000e-04 | 3.000e-04 | 1.500e-04 |

对比可知：**不重启衰减时，把 `--init-lr` 提到 3e-4 也只有 1.875e-05**（因为 epoch//50 已经
累计到 4 次衰减）；而重启衰减后 `1e-4` 就能拿到 1e-04，是原来第 199 轮 1.25e-05 的 8 倍。
所以提升学习率的关键是 `--lr-decay-restart`，不是单纯调大 `--init-lr`。

**学习率选多大**

| `--init-lr` | `--lr-step-size` | encoder @200 | encoder @250 | decoder @200 | 相对第 199 轮 |
|---|---:|---:|---:|---:|---:|
| `1e-4` | 50 | 1.00e-04 | 5.00e-05 | 1.00e-05 | 8× |
| `3e-4` | 50 | 3.00e-04 | 1.50e-04 | 3.00e-05 | 24× |
| `1e-3` | 50 | 1.00e-03 | 5.00e-04 | 1.00e-04 | 80× |
| `1e-3` | 25 | 1.00e-03 | 2.50e-04 | 1.00e-04 | 80× |

`1e-3` 可以用，但要注意这套配置里**没有任何梯度裁剪**（`grep clip_grad trainer.py` 无结果）、
AMP 默认开启、batch size 为 1，所以 80× 的跳变有真实的发散风险。建议同时做三件事降风险：

1. 加 `--lr-step-size 25`，让学习率每 25 轮减半而不是每 50 轮；
2. 先把 `--stage0_decoder_lr_factor` 设为 `0`，**冻结 decoder**，让高学习率只作用在随机初始化
   的主干上，避免把先验冲掉；主干稳定后再改回 `0.1` 解冻 decoder 继续（四阶段实验也可用
   `--decoder_lr_factor 0 --decoder_core_lr_factor 0 --freeze_decoder_bn_stats` 达到同样效果）；
3. 用一个**新的实验名**试高学习率，把原 200-epoch 基线留在原地：

```bash
mkdir -p train/checkpoints/thorax_prior_joint_lr1e3/ckpt_history
cp train/checkpoints/thorax_prior_joint_no_adapter/ckpt_history/ckpt_199 \
   train/checkpoints/thorax_prior_joint_lr1e3/ckpt_history/ckpt_199
# 然后把 --name 换成 thorax_prior_joint_lr1e3，其余参数照抄
```

判据：前 10～20 轮如果公平口径 PSNR/SSIM 先掉后升，说明这个跳变可以接受；如果持续下滑，
或 `train_ls.txt` 里总损失突然抬升 2～3 倍，就把 `--init-lr` 折半再试。注意 AMP 的
GradScaler 在梯度出现 inf/NaN 时会**跳过该步**并缩小 scale，所以发散有时表现为"loss 不再
下降"而不是报错，需要主动看曲线而不是等它崩。

⚠️ **不要用 `val_psnr_3d_clamp` 做判据**，它被自归一化污染（§7.3）。改用
`val_ssim_3d_clamp`。

### 7.2 Phase C lvl4 的 BatchNorm 陷阱

Phase C 的第 4 级（lvl4，解冻残差主体）会出现一次**突变**：PSNR 在那一轮陡然下跌。实测
（`thorax_prior_four_phase_v2_prior_reinject`，Phase C 从 epoch 300 开始 → lvl4 落在 ep360）：

| epoch | 层级 | 预测 HU 峰值 | 报告 PSNR | **公平 PSNR** | 报告 SSIM | **公平 SSIM** |
|---|---|---:|---:|---:|---:|---:|
| 340 | lvl3 | 1584 | 23.690 | 23.531 | 0.6782 | 0.6766 |
| 350 | lvl3 | 1535 | 23.556 | **23.510** | 0.6841 | **0.6837** |
| **360** | **lvl4** | **822** ⬇ | **16.542** | **20.605** | **0.6435** | **0.6712** |
| 370 | lvl4 | 1082 | 21.363 | **23.668** | 0.7295 | **0.7368** |
| 380 | lvl4 | 1150 | 21.851 | 23.610 | 0.7314 | 0.7385 |

（case `2026-06-04_072952`，GT 的 HU 峰值 1516。"公平"= 用同一个固定区间
`(x - clamp_min)/(clamp_max - clamp_min)` 归一化双方，口径见 §7.3。）

读法：

- lvl3（340）几乎无退化：公平 PSNR 只差 **−0.02 dB**，公平 SSIM 反而上升。所以"解冻上采样
  就掉点"是错觉。
- **lvl4（360）才是真正的崩**：报告 PSNR 掉 **−7.01 dB**，但公平口径只掉 **−2.90 dB**
  —— 报告值把真实退化**放大了 2.4 倍**。
- **它是瞬态的**：10 轮内公平 PSNR 恢复到 23.668（已超过 340 的 23.531）、公平 SSIM 到 0.7368
  （远超 340 的 0.6766）。但报告 PSNR 永远回不来（380 的 21.851 仍低于 340 的 23.690），
  如果按报告 PSNR 选 checkpoint 会一直选到崩掉的那个。

**机制：这是 BatchNorm 模式切换，不是权重解冻**

| 级别 | 解锁模块 | 含 BN 层数 |
|---|---|---:|
| lvl1 | `out_blk` | **0** |
| lvl2 | `up_blk_list[-1]` | **0** |
| lvl3 | `up_blk_list[0]` | **0** |
| **lvl4** | `in_blk` + `res_blk_last` + `res_blk_list` | **14** |

`up_blk` 与 `out_blk` 的结构里**根本没有归一化层**，所以 lvl1-3 一个 BN 都不含，全部 14 层
BN 都在 lvl4（`in_blk` 1 层、`res_blk_last` 1 层、`res_blk_list` 6 块 × 2 层 = 12 层）。

实测 BN `running_mean` 相对预训练先验的最大偏差：

| epoch | 300–350（lvl1/2/3） | **360（lvl4 首轮）** | 370 | 380 |
|---|---:|---:|---:|---:|
| BN `running_mean` 相对先验的最大偏差 | **0.000e+00** | **8.174** | 6.466 | 6.256 |

lvl1-3 全程**一个字节都没变**（decoder 一直 `eval()`）；lvl4 那一轮跳 **8.17**。而**权重更新
解释不了这个突变**：epoch 360 时 `decoder_core` 的 LR = `1e-4 × 0.125 × 0.01 = 1.25e-7`，
10 轮累计权重移动只在 `1e-5` 量级，不可能让预测 HU 峰值从 1535 塌到 822。

所以 lvl4 不是"多放开一些卷积权重"，而是**一次性把 14 层 BN 从 `eval()` 切到 `train()`**，
从"用先验累积的 running 统计"变成"用当前 batch 统计"。这是**离散跳变**。

**病根是 latent 分布失配，不是"解冻太多"。** 反证：`dental_prior_adapter`（§6.2）在
Stage 2 一次性解冻整个 decoder（含全部 14 层 BN）却毫无波动，因为它有
`--pretrained_backbone`，主干与 decoder 同域，**batch 统计 ≈ running 统计**，切换不产生跳变。
反过来，如果主干是投影支路从 0 训了 200 轮出来的 latent，其分布与 decoder 先验标定的分布
并不一致；latent 对齐权重此时已退火到 0.01，管不住，BN 一放手就把失配暴露成 8.17 的跳变。

**规避手段（尚未 A/B 验证）**

加 `--freeze_decoder_bn_stats`。它的实现是 `_apply_decoder_bn_policy()`，在
`_apply_training_stage` 末尾（也就是所有 `module.train()` 之后）把 14 层 BN 统一打回
`eval()`，于是它们始终用先验的 running 统计；同时注释明确写了**不会改动 BN 的
`requires_grad`，γ/β 照常训练**。这样 lvl4 只剩纯卷积权重的连续变化，理论上不再有 8.17 的
跳变。但这是代码层面的推断，需要自己跑一次 A/B 确认。

配套的两个旋钮（比 BN 更根本，针对 latent 分布失配）：

| 参数 | 默认 | 建议 | 理由 |
|---|---:|---:|---|
| `--phase_c_latent_end_factor` | 0.1 | 0.3 | latent 对齐权重在 Phase C 从 0.7 退火到 0.1，解冻残差主体前已几乎失效，拉不住分布 |
| `--adapter_lr_factor` | 0.1 | 0.3 | adapter 位于 latent 与 decoder 之间，是分布失配的直接可补偿处；参照 `dental_prior_adapter` 用的是 1.0 |

### 7.3 指标口径警告：`data_norm` 的自归一化伪影

`util/util_func.py`：

```python
def data_norm(x):
    # self normalization
    return (x-x.min())/(x.max()-x.min())
```

主模型的 `psnr_3d_clamp` / `ssim_3d_clamp`（`trainer.py`、`evaluate.py`）是
`get_psnr(data_norm(pred_clamp), data_norm(gt))`——**预测和真值各自用自己的 min/max 拉到
[0,1]**，于是给预测强加了一个隐式增溢 `g = R_gt / R_pred`。真值范围固定，但**预测的动态
范围在训练中会剧烈收缩**（这正是网络在变好的证据），`g` 随之偏离 1，PSNR 就被与重建精度
无关的全局增溢主导。

`data_norm` **只用于指标，不进任何网络输入**（`data/`、`models/` 下零引用；decoder 的输入
是 aggregator/adapter 输出的 latent，中间没有任何归一化层）。

| 指标 | 归一化方式 | 可信度 |
|---|---|---|
| 主模型 `psnr_3d_clamp` | `data_norm` ×2（各自 min/max） | **与真实质量可能反向，不要用作选点判据** |
| 主模型 `ssim_3d_clamp` | `data_norm` ×2 | 基本可用（SSIM 对全局增溢免疫） |
| 本子模型的 `global_psnr_db` | **固定区间** | 可信（§5.3 的表格即此口径） |

后果：一个 run 的 `val_psnr_3d_clamp` 可能在 epoch 120 达峰、之后一路下滑，而同期的公平
PSNR 与 SSIM 一直在涨。**判断优劣请用 `val_ssim_3d_clamp`，或把保存下来的
`train/visuals/<name>/<case>/volume/volume_<epoch>.nii.gz` 用固定区间回算 PSNR。**

### 7.4 如何选 `--pretrained_backbone` 的 checkpoint

`--pretrained_backbone` 只取该 checkpoint 的 `encoder.*` 与 `aggregator.*`（例如
`thorax_prior_four_phase_v2_prior_reinject` 的 `ckpt_399` 贡献 224 个张量），其余部分（decoder、
adapter、优化器状态）一律忽略。所以判据是"**主干本身，以及它的 latent 与 decoder 先验的相容性**"，
而不是那个 checkpoint 的报告 PSNR。

下面用 `thorax_prior_four_phase_v2_prior_reinject` 的 2 个 visual 病例做实测（口径见 §7.3）：

| epoch | 阶段 | 固定区间 PSNR | 固定区间 SSIM | latent `smooth_l1_raw` | latent `cosine_raw` |
|---|---|---:|---:|---:|---:|
| 200 | A | 21.178 | 0.5434 | 0.00141 | 0.4301 |
| 280 | B | 24.934 | 0.6961 | 0.00168 | 0.3253 |
| 320 | C-lvl2 | 27.382 | 0.7209 | 0.00155 | 0.3269 |
| 330 | C-lvl2 | 28.086 | 0.7459 | 0.00130 | 0.3223 |
| 340 | C-lvl3 | 28.722 | 0.7642 | 0.00118 | 0.3192 |
| 350 | C-lvl3 | 28.877 | 0.7717 | 0.00108 | 0.3088 |
| 360 | C-lvl4 | 25.322 | 0.7221 | 0.00105 | 0.3100 |
| 370 | C-lvl4 | 28.715 | 0.7914 | 0.00098 | 0.2675 |
| 380 | D | 28.871 | 0.7998 | 0.00093 | 0.2449 |
| 390 | D | 28.815 | **0.8006** ← 最高 | 0.00091 | 0.2415 |
| **399** | **D** | **28.959** ← 最高 | — | — | — |

三项指标都指向**越靠后越好**：

- **固定区间 PSNR 的峰值在 epoch 399**（28.959），比 350 / 380 高 0.06～0.09 dB。
- **固定区间 SSIM 的峰值在 epoch 390**（0.8006），且 370 之后一直在 0.79 以上。
- **latent 对齐的原始值单调改善**（`cosine_raw` 从 epoch 200 的 0.4301 降到 390 的 0.2415），说明
  主干的 latent 与先验教师越来越一致。这里有个反直觉点：latent **权重**在退火，但**实际对齐度**
  一直在变好——退火只是让主干逐步接管，而不是放弃对齐。

**推荐 `ckpt_399`**：收敛态、PSNR 最高、latent 对齐最好。若更看重 SSIM，可用 `ckpt_390`。
**不要用 `ckpt_360`**——那是 lvl4 突变发生的当轮（§7.2）。

另两个理由支持取收敛态而不是 Phase C 中途：

- 340/350 处在 Phase C 中途，主干仍在以 0.1× 移动，是"未收敛"的状态；399 落在 Phase D（主干
  0.01×、latent 权重 0），权重已稳定。
- 拿一个还在移动的状态当新实验起点，会让后续的对照实验多一个不确定变量。

⚠️ **两项诚实保留**

1. 上表的 PSNR / SSIM 来自 `volume_<epoch>.nii.gz`，反映的是 **backbone + adapter + decoder 三者
   联合**的输出，而 `--pretrained_backbone` 只取 encoder/aggregator，所以它们是**代理指标**。更
   直接的证据是 latent 原始对齐值（与 decoder 无关），结论一致。
2. 该表只覆盖 2 个 visual 病例，样本很小。想要更稳的结论，可以用 `evaluate.py --resume_name <epoch>`
   在完整 val/test 上回算，或把 `print.vis_interval` 调小以保存更多病例的体数据。

### 7.5 Checkpoint 结构

本子模型的共享训练循环仍保存以下字段：

```python
checkpoint["model"]
checkpoint["decoder"]
checkpoint["feature_stem"]
checkpoint["optimizer"]
checkpoint["scaler"]
checkpoint["epoch"]
checkpoint["step"]
checkpoint["loss_config"]
checkpoint["best_val_loss"]
```

这里 `checkpoint["feature_stem"]` 虽然沿用旧名称，但保存的是完整 `LearnedPriorEncoder`，
包括：

```text
detail_downsample
feature_stem 32→64→64→256
```

因此学习型高频下采样权重不会遗漏。

主模型侧 checkpoint 保存 `G_render` 全量权重、四个参数组的 `G_optim`、`G_lr_scheduler`、
`G_scaler`、`prior_stem`，以及 `iter`（= 保存时的 `epoch + 1`，续训起点由它决定）和
`training_stage`。

### 7.6 GT 变体：用 CBCT 替代配准 pCT

`proj.nii.gz` 是从该病人的 CBCT 重建出来的，所以 CBCT 才是与投影同源配对的标签；默认的
`gt_volume.nii.gz` 用的是配准后的计划 CT，带配准残差。若要改用 CBCT，**不要让**
`prepare_thorax.py --gt-source cbct` 来写（它产出的是原生网格 512×512×123，z 不能被 4 整除，
无法训练），而应生成 `fixed_cbct_hu.nii.gz` 版本的变体：

```powershell
python -m tools.thorax_preprocessing.make_cbct_gt_variant `
  --output dataset/thorax/syn_data_cbct_gt
```

然后把训练命令的 `--datadir` 改为 `./dataset/thorax/syn_data_cbct_gt`、
`--require-gt-source` 改为 `cbct-fixed`，其余参数一字不动——两组实验仅 GT 不同，严格配对可比。
该变体与配准 pCT 版本共用同一体素网格（248×248×N @2mm），投影用软链接复用，155 例约 1.8 GB。

注意 CBCT 的 μ 上限只有 0.05–0.06（pCT 到 0.088），固定范围 PSNR 会天然偏高，两组指标不可
直接横比；细节与评估口径见 `tools/thorax_preprocessing/README.md` 的「GT 变体」一节。

如果投影几何有变（例如角度约定修正），还有 `make_angle_convention_variant.py` 生成的
`syn_data_cbct_gt_v2`，它只重算 `frames[].vec`，其余共享软链接。同一旧划分上的受控对比：
v1（错误角度约定）val PSNR 20.65 / test 19.50，v2（正确约定）val 24.43 / test 23.03，
SSIM 0.503 → 0.681。换用 v2 时训练侧**零代码改动**，只改 `--datadir`。

### 7.7 重要注意事项

- 必须从 0 训练本 prior encoder 与 decoder。
- 旧 35 dB decoder 可以人工作为初始化来源，但不能视为已适配新 latent；本说明默认完全从 0。
- 浅层 prior checkpoint 与当前 checkpoint 结构不兼容。
- 旧版 4 个 ResidualBlock 的 deep_encoder checkpoint 也不兼容。
- submodel 得到更高 PSNR 并不保证主模型同步提高，仍需比较接入 Adapter 后的 val/test 结果。
- 31 通道 detail 保留的信息远多于单通道平均值，需关注 train/val/test 差距与错误细节生成。
- **四阶段训练在 Phase C 的 lvl4（解冻残差主体）存在 BatchNorm 模式切换引起的突变**：全部
  14 层 BN 都在 lvl4，`--freeze_decoder_bn_stats` 默认关闭时 `running_mean` 会在那一轮跳变 8
  以上。**推荐同时加 `--freeze_decoder_bn_stats`**，并把 `--phase_c_latent_end_factor` 从 0.1
  提到 0.3 左右、`--adapter_lr_factor` 从 0.1 提到 0.3 左右，让 latent 分布失配在解冻前就被
  收紧。机制与实测见 §7.2。
- **不要用 `val_psnr_3d_clamp` / `test_psnr_3d_clamp` 选 checkpoint**：它们经过 `data_norm`
  的各自 min-max 归一化，与真实质量可能反向。改用 `val_ssim_3d_clamp` 或用固定区间回算，
  见 §7.3。
- **`--resume` 在子模型与主模型上的语义不同**：子模型 `--resume <路径>` 且 `--epochs` 是
  **增量轮数**；主模型 `--resume` 是开关 + `--resume_name <epoch>`，且 `--epochs` 是**新总
  轮数**。混用会静默跑错轮数。
- **`--resume` 会让主模型的 `--pretrained_decoder` 权重被静默忽略**（只保留"开启四阶段"的
  作用）。要真正重注入先验，必须加 `--resume_reload_decoder`，见 §6.1.3。
- **不要跨数据集比较 PSNR 曲线。** `train_psnr_3d_clamp` 上 30 dB 左右的曲线来自 synthetic
  dental（投影由同一份 CT 用 DRR 生成），而 thorax 是**真实投影**且 GT 是**配准后的 pCT**
  （含配准残差），二者目标与天花板完全不同。thorax 的 16～17 dB 不能直接判定为"训练失败"。
