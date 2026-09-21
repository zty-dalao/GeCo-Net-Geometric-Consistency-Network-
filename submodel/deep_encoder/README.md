# 固定低频 + 学习型高频 Prior Encoder

## 1. 设计目的

本目录实现一个可解释的pCT先验编码器：固定平均池化负责保留稳定低频基底，
学习型卷积只编码被平均池化去除的块内高频残差。得到的latent随后交给主项目原始
SRGAN Decoder重建sCT。

本版本用于从0开始训练，不能直接resume此前浅层 `submodel/decoder` 或旧版
`submodel/deep_encoder` checkpoint。

数据读取、loss、AMP、TensorBoard、验证、测试、checkpoint与固定范围指标逻辑均与
`submodel/decoder` 保持一致。训练和评估入口复用原循环，仅替换模型类，防止两套
实验的损失和指标实现发生漂移。

## 2. 完整模型架构

```text
完整pCT/CT GT
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
        │                                              │ nearest上采样到256³
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
                      主项目原始SRGAN Decoder
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

第一通道始终是固定块均值，另外31个通道只能从高频残差中学习局部测量。相比自由的
`Conv3d 1→32`，该结构更明确地区分低频强度与高频细节。

默认参数量：

```text
LearnedPriorEncoder：       610,655
原SRGAN Decoder：        25,371,473
完整预训练模型：          25,982,128
```

## 3. 目录结构

```text
submodel/deep_encoder/
├── model.py              # LearnedPriorEncoder + 原SRGAN Decoder
├── dataset.py            # 与原decoder预训练一致的数据接口
├── loss.py               # 与原decoder预训练一致的loss接口
├── train.py              # 从0训练/接力训练入口
├── evaluate_metrics.py   # 固定范围PSNR、SSIM及区域指标
├── README.md
├── checkpoints/          # 训练时自动生成
├── logs/                 # 训练时自动生成
└── metrics/              # 评估时自动生成
```

## 4. 从0训练命令

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

这是从0训练，不要添加 `--resume`。输出位置：

```text
submodel/deep_encoder/logs/dental_mean_detail_prior/
submodel/deep_encoder/checkpoints/dental_mean_detail_prior/
```

主要权重：

```text
ckpt_latest.pt
ckpt_best_val.pt
ckpt_epoch_0010.pt
ckpt_epoch_0020.pt
...
```

## 5. 独立评估命令

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
生成目录，而所有 run 的最佳checkpoint都叫 `ckpt_best_val.pt`，因此不同数据集或不同
训练的评估结果会互相覆盖：

```text
submodel/deep_encoder/metrics/ckpt_best_val/     # 被多次评估共用，会互相覆盖
submodel/deep_encoder/metrics/thorax_cbct_prior/ # 本次评估，互不干扰
```

结果包括每病例和数据集汇总的固定范围PSNR、RMSE、HU MAE、骨骼/软组织区域指标以及
与根目录评估协议兼容的SSIM。

几点约定：

- `--air-upper-hu` / `--bone-lower-hu` 是**HU空间**阈值。脚本先 `mu_to_hu()` 把预测和
  GT换回HU再分区，所以无论GT是pCT还是CBCT都照抄默认值，不需要调整。
- `--save-volumes` 写出的"HU"是 `mu_to_hu(clamp(mu, 0, 0.09009))`。pCT-GT（μ上限约
  0.088）和CBCT-GT（μ上限约0.061）都没有触发clamp，所以往返转换是精确的，可当真实HU
  使用。
- 输出里的 `sCT-vs-pCT` 文案和 `sct_vs_pct_psnr_db` 键名是历史遗留的硬编码
  （`submodel/decoder/evaluate_metrics.py:197`）。评估CBCT-GT时它实际表示 `sCT-vs-CBCT`，
  含义按 `--data-root` 所指的GT变体理解，字段名不必改。

### 5.1 两个GT变体的先验实测对比

同一套划分上分别用pCT和CBCT作为GT预训练两个先验，再用各自的GT评估（不可互相对比，
见下）：

| GT变体 | 划分 | PSNR(dB) | SSIM3D | HU MAE | bone MAE | tissue MAE | air MAE | 归一化MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pCT（`thorax_deep_decoder`） | val | 40.9432 | 0.9796 | 13.55 | 96.31 | 31.48 | 7.89 | 8.19e-05 |
| pCT | test | 41.4746 | 0.9783 | 13.78 | 79.02 | 28.15 | 8.38 | 7.21e-05 |
| CBCT（`thorax_deep_decoder_cbct_prior`） | val | 40.3304 | 0.9586 | 16.73 | 96.41 | 41.61 | 9.96 | 9.47e-05 |
| CBCT | test | 40.2547 | 0.9413 | 18.14 | 89.35 | 42.69 | 10.32 | 9.74e-05 |

**跨GT变体的PSNR/SSIM不可直接比较。** 两者的归一化范围都是 `conf/train.conf` 的
`data.dental`（`0 .. 0.09009`），但CBCT本身含散射、射束硬化和噪声，作为拟合目标更难，
所以CBCT先验的归一化MSE反而更高（9.47e-05 vs 8.19e-05），PSNR/SSIM更低。实测两者的
`bone_mae_hu`几乎持平（96.41 vs 96.31），差距集中在`tissue`和`air`——与CBCT在低衰减
区域噪声更大的特性一致。

要判断"哪个teacher更适合主模型"，只能各自和自己GT对齐地看绝对误差，或在主模型侧做
A/B（见第8节）。

## 6. 参数解析

### 6.1 训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--run-name` | `dental_pretrain` | 日志和checkpoint目录名，建议显式指定。 |
| `--data-root` | `dataset/dental/syn_data` | dental体数据根目录。 |
| `--split-file` | `data/dataset_split/dental_split.json` | train/val/test病例划分。 |
| `--conf` | `conf/train.conf` | Decoder结构、μ范围及默认loss配置。 |
| `--device` | `cuda` | `cuda`、`cuda:0`或`cpu`。 |
| `--epochs` | `500` | 从0训练时的总轮数。 |
| `--batch-size` | `1` | 256³体数据建议保持1。 |
| `--num-workers` | `0` | DataLoader进程数。 |
| `--lr` | 配置文件值 | Adam学习率。 |
| `--mse-lambda-3d` | 配置文件值 | 基础3D重建项权重；旧命名为MSE，实际使用L1。 |
| `--gd1-lambda` | 配置文件值 | XYZ三方向一阶梯度loss权重。 |
| `--bone-lambda` | `0` | GT骨骼区域归一化L1权重。 |
| `--bone-lower-hu` | `300` | GT骨骼HU下限。 |
| `--soft-mask-lambda` | `0` | GT软组织区域归一化L1权重。 |
| `--soft-window-low` | `-160` | 软组织HU窗下限。 |
| `--soft-window-high` | `240` | 软组织HU窗上限。 |
| `--val-every` | `1` | 验证间隔。 |
| `--test-every` | `10` | 测试间隔。 |
| `--save-every` | `10` | 历史checkpoint保存间隔。 |
| `--no-amp` | 关闭 | 禁用CUDA混合精度。 |
| `--limit` | 无 | 只使用前N个训练病例，供smoke test使用。 |
| `--eval-limit` | 无 | val/test各使用前N例。 |
| `--max-seconds` | `0` | 限时训练秒数，0表示关闭。 |

### 6.2 评估参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--checkpoint` | 必填 | 要评估的本模型checkpoint。 |
| `--data-root` | `dataset/dental/syn_data` | 评估数据根目录，需与checkpoint的训练数据一致。 |
| `--split-file` | `data/dataset_split/dental_split.json` | 评估病例划分。 |
| `--splits` | `val test` | 评估划分。 |
| `--air-upper-hu` | `-500` | 空气区域HU上限。 |
| `--bone-lower-hu` | `300` | 骨骼区域HU下限。 |
| `--output-dir` | `metrics/<checkpoint文件名>` | 指标和体数据输出目录，建议显式指定以免覆盖。 |
| `--save-volumes` | 关闭 | 保存预测sCT和GT NIfTI（编码为HU）。 |
| `--no-amp` | 关闭 | 禁用评估AMP。 |

## 7. Checkpoint结构

共享训练循环仍保存以下字段：

```python
checkpoint["model"]
checkpoint["decoder"]
checkpoint["feature_stem"]
checkpoint["optimizer"]
checkpoint["scaler"]
```

这里 `checkpoint["feature_stem"]` 虽然沿用旧名称，但保存的是完整
`LearnedPriorEncoder`，包括：

```text
detail_downsample
feature_stem 32→64→64→256
```

因此学习型高频下采样权重不会遗漏。

## 8. 接入主模型

主模型训练时必须指定：

```text
--prior_encoder_type deep
```

此时主模型会将完整256³ pCT送入冻结teacher，由teacher内部完成平均值、高频残差、
31通道细节编码和latent生成。不会在主模型外部提前平均池化。

使用CNN Adapter的示例：

```bash
python train.py \
  --name dental_mean_detail_prior_adapter \
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
  --pretrained_decoder submodel/deep_encoder/checkpoints/dental_mean_detail_prior/ckpt_best_val.pt \
  --prior_encoder_type deep \
  --use_adapter \
  --adapter_type cnn \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --phase_a_epochs 50 \
  --phase_a_encoder_lr_factor 1.0 \
  --phase_a_aggregator_lr_factor 1.0 \
  --phase_b_epochs 50 \
  --phase_c_epochs 80 \
  --decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

上面这条命令没有 `--pretrained_backbone`，即主模型 Encoder/Aggregator 从 0 训练，因此
必须用 `--phase_a_encoder_lr_factor` / `--phase_a_aggregator_lr_factor` 打开 Phase A 的
主干；若省略这两个参数，Phase A 会冻结随机初始化的主干、只剩零初始化的 Adapter 可训练，
程序会直接报错而不是空跑。若已有旧主模型 checkpoint，可改为传入 `--pretrained_backbone`
并省略这两个参数。Phase A/B/C 之和为 180，`--epochs 200` 的剩余 20 轮归 Phase D。

若使用Transformer Adapter，将 `--adapter_type cnn` 改成：

```text
--adapter_type transformer
--adapter_transformer_pool_size 8
--adapter_transformer_layers 2
--adapter_transformer_heads 4
--adapter_transformer_dropout 0.1
```

根目录 `evaluate.py` 评估主模型时不使用pCT teacher，所以无需提供
`--prior_encoder_type deep`；只需要按照训练配置重建正确的Adapter结构。

**换用 CBCT 作为 GT**：`proj.nii.gz` 是从该病人的 CBCT 重建出来的，所以 CBCT 才是与投影同源配对
的标签；默认的 `gt_volume.nii.gz` 用的是配准后的计划 CT，带配准残差。若要改用 CBCT，不要让
`prepare_thorax.py --gt-source cbct` 来写（它产出的是原生网格 512×512×123，z 不能被 4 整除，无法
训练），而应生成 `fixed_cbct_hu.nii.gz` 版本的变体：

```powershell
python -m tools.thorax_preprocessing.make_cbct_gt_variant `
  --output dataset/thorax/syn_data_cbct_gt
```

然后把训练命令的 `--datadir` 改为 `./dataset/thorax/syn_data_cbct_gt`、`--require-gt-source` 改为
`cbct-fixed`，其余参数一字不动——两组实验仅 GT 不同，严格配对可比。该变体与配准 pCT 版本共用同一
体素网格（248×248×N @2mm），投影用软链接复用，155 例约 1.8 GB。注意 CBCT 的 μ 上限只有
0.05–0.06（pCT 到 0.088），固定范围 PSNR 会天然偏高，两组指标不可直接横比；细节与评估口径见
`tools/thorax_preprocessing/README.md` 的「GT 变体」一节。

### 8.1 在主模型 checkpoint 上断点续训并提高学习率

主模型训练到 200 epoch 后往往还没收敛，但此时学习率已被 `lr_sche` 的衰减压到很低的水平。
以 `thorax_prior_joint_no_adapter` 为例，`train/logs/<实验名>/train_lr.txt` 记录的实测学习率：

```text
Epoch:0    G_lr:0.0001      # 1e-4
Epoch:80   G_lr:5e-05       # 第 50 轮衰减一次
Epoch:120  G_lr:2.5e-05     # 第 100 轮
Epoch:160  G_lr:1.25e-05    # 第 150 轮，此时只有初值的 1/8
Epoch:198  G_lr:1.25e-05
```

也就是 `decay = gamma ** (epoch // step_size) = 0.5 ** (epoch // 50)`。**断点续训时这个衰减不会重置**，
如果只是重新加载 checkpoint，第 200 轮仍会以 1/8 的学习率继续跑。因此新增了 4 个学习率开关：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `--init-lr` | 配置文件 `lr_sche.init_lr` | 覆盖基础学习率。主模型训练此前只能改 `conf/train.conf`，现在可在此直接指定。 |
| `--lr-decay-restart` | 关闭 | **让衰减从续训的那个 epoch 重新起算**，于是续训一开始就回到完整的 `init_lr`，而不是继续用已经衰减过的值。 |
| `--lr-step-size` | 配置文件值（50） | 覆盖衰减间隔；设得很大即等于恒定学习率。 |
| `--lr-gamma` | 配置文件值（0.5） | 覆盖衰减倍率。 |

续训前先做两项检查：

```bash
# 1) 确认目标实验的训练进程已经结束（正在写入时会覆盖同一份 ckpt_latest）
ps -ef | grep train.py | grep -v grep

# 2) 确认可用的 checkpoint 编号
ls train/checkpoints/thorax_prior_joint_no_adapter/ckpt_history/
```

从 `ckpt_199` 续训 100 个 epoch（总轮数改为 300），并把学习率拉回 `1e-4` 重新衰减：

```bash
/autdl-tmp/conda_env/GeoAware/bin/python train.py \
  --name thorax_prior_joint_no_adapter \
  --datadir ./dataset/thorax/syn_data \
  --datatype thorax \
  --require-gt-source registered-ct \
  --train_scale 4 \
  --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 300 \
  --resume \
  --resume_name 199 \
  --init-lr 1e-4 \
  --lr-decay-restart \
  --stage0_decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01
```

想更激进地把学习率提高到 `3e-4`，把 `--init-lr 1e-4` 改成 `--init-lr 3e-4`；如果希望学习率
在续训期间完全不衰减，再补 `--lr-step-size 1000`。三种配置下 `encoder` 参数组的实际学习率
（续训起点为 epoch 200，数值由 `_lr_multiplier` 实算）：

| 配置 | epoch 200 | epoch 249 | epoch 250 |
|---|---:|---:|---:|
| 不加 `--lr-decay-restart`（`--init-lr 3e-4`） | 1.875e-05 | 1.875e-05 | 9.375e-06 |
| `--init-lr 1e-4 --lr-decay-restart` | 1.000e-04 | 1.000e-04 | 5.000e-05 |
| `--init-lr 3e-4 --lr-decay-restart` | 3.000e-04 | 3.000e-04 | 1.500e-04 |

对比可知：**不重启衰减时，把 `--init-lr` 提到 3e-4 也只有 1.875e-05**（因为 epoch//50 已经累计到 4 次
衰减）；而重启衰减后 `1e-4` 就能拿到 1e-04，是原来第 199 轮 1.25e-05 的 8 倍。所以提升学习率的
关键是 `--lr-decay-restart`，不是单纯调大 `--init-lr`。

`decoder` 参数组始终再乘 `--stage0_decoder_lr_factor`（上面命令里是 0.1），即上表数值的 1/10。

**学习率选多大**

续训起点（epoch 200）的实际学习率，以及相对老实验第 199 轮 `1.25e-05` 的倍数：

| `--init-lr` | `--lr-step-size` | encoder @200 | encoder @250 | decoder @200 | 相对第 199 轮 |
|---|---:|---:|---:|---:|---:|
| `1e-4` | 50 | 1.00e-04 | 5.00e-05 | 1.00e-05 | 8× |
| `3e-4` | 50 | 3.00e-04 | 1.50e-04 | 3.00e-05 | 24× |
| `1e-3` | 50 | 1.00e-03 | 5.00e-04 | 1.00e-04 | 80× |
| `1e-3` | 25 | 1.00e-03 | 2.50e-04 | 1.00e-04 | 80× |

`1e-3` 可以用，但要注意这套配置里**没有任何梯度裁剪**（`grep clip_grad trainer.py` 无结果）、
AMP 默认开启、batch size 为 1，所以 80× 的跳变有真实的发散风险。建议同时做三件事降风险：

1. 加 `--lr-step-size 25`，让学习率每 25 轮减半而不是每 50 轮；
2. 先把 `--stage0_decoder_lr_factor` 设为 `0`，**冻结 decoder**，让高学习率只作用在随机初始化的
   主干上，避免把 pCT 先验冲掉；主干稳定后再改回 `0.1` 解冻 decoder 继续；
3. 用一个**新的实验名**试高学习率，把原 200-epoch 基线留在原地：

```bash
mkdir -p train/checkpoints/thorax_prior_joint_lr1e3/ckpt_history
cp train/checkpoints/thorax_prior_joint_no_adapter/ckpt_history/ckpt_199 \
   train/checkpoints/thorax_prior_joint_lr1e3/ckpt_history/ckpt_199
# 然后把 --name 换成 thorax_prior_joint_lr1e3，其余参数照抄
```

判据：前 10～20 轮如果 `epoch/val_psnr_3d_clamp` 先掉后升，说明这个跳变可以接受；如果持续下滑，
或 `train_ls.txt` 里总损失突然抬升 2～3 倍，就把 `--init-lr` 折半再试。注意 AMP 的 GradScaler
在梯度出现 inf/NaN 时会**跳过该步**并缩小 scale，所以发散有时表现为"loss 不再下降"而不是报错，
需要主动看曲线而不是等它崩。

**使用要点**

- `--epochs` 给的是**新的总轮数**，不是增量。`--epochs 300` + `--resume_name 199` 表示
  继续训练 epoch 200～299。另外 `ckpt_N` 保存于第 N 轮**结束之后**，所以 `--resume_name 199`
  从 epoch 200 开始（实测 `begin_epochs = resume_name + 1`）。
- `--name`、`--checkpoints_path`、`--resume_name` 三者共同定位
  `train/checkpoints/<name>/ckpt_history/ckpt_<resume_name>`。该文件不存在时程序会直接
  `FileNotFoundError`，不会再静默从 epoch 0 重跑并覆盖日志。
- 模型结构参数必须与保存时一致（`--fusion`、`--train_scale`、`--nviews` 无关结构，但
  Adapter 相关的 `--use_adapter`/`--adapter_type`/`--adapter_hidden_channels`/
  `--adapter_use_global_alpha` 必须一致），否则严格加载会报 `Checkpoint/model mismatch`。
- **无 Adapter 的实验**（stage 0）：续训时不必传 `--pretrained_decoder`，checkpoint 里已含
  decoder 权重，传了也只会多建一个用不到的冻结 teacher。
- **有 Adapter 的实验**：续训时**必须**继续传 `--pretrained_decoder` 与
  `--prior_encoder_type deep`，否则 `use_four_phase` 会变成 `False`，四阶段调度与 latent 对齐
  全部失效；`--resume` 模式下它不会覆盖 checkpoint 里的 decoder 权重，只用于重建 teacher。
  同时阶段边界按新参数重算，因此可以顺便调整 `--phase_*_epochs`。
- 提高学习率会让 `val/test_psnr_3d_clamp` 先回落再上升，属正常现象；建议 `--val-every` 保持 1，
  并在 `epoch/val_psnr_3d_clamp` 连续 3 次验证不再改善时把 `--init-lr` 折半。

> ⚠️ 不要跨数据集比较 PSNR 曲线。`train_psnr_3d_clamp` 上 30 dB 左右的曲线来自 synthetic dental
> （投影由同一份 CT 用 DRR 生成），而 thorax 是**真实投影**且 GT 是**配准后的 pCT**（含配准残差），
> 二者目标与天花板完全不同。thorax 的 16～17 dB 不能直接判定为"训练失败"。

## 9. 重要注意事项

- 必须从0训练本prior encoder与decoder。
- 旧35 dB decoder可以人工作为初始化来源，但不能视为已适配新latent；本说明默认完全从0。
- 浅层prior checkpoint与当前checkpoint结构不兼容。
- 旧版4个ResidualBlock的deep_encoder checkpoint也不兼容。
- submodel得到更高PSNR并不保证主模型同步提高，仍需比较接入Adapter后的val/test结果。
- 31通道detail保留的信息远多于单通道平均值，需关注train/val/test差距与错误细节生成。
