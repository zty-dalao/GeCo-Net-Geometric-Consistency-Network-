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

评估最佳验证checkpoint：

```bash
python -m submodel.deep_encoder.evaluate_metrics \
  --checkpoint submodel/deep_encoder/checkpoints/dental_mean_detail_prior/ckpt_best_val.pt \
  --data-root dataset/dental/syn_data \
  --split-file data/dataset_split/dental_split.json \
  --conf conf/train.conf \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --splits val test \
  --air-upper-hu -500 \
  --bone-lower-hu 300 \
  --save-volumes
```

若只需要指标、不保存NIfTI，去掉 `--save-volumes`。默认结果保存在：

```text
submodel/deep_encoder/metrics/ckpt_best_val/
```

包括每病例和数据集汇总的固定范围PSNR、RMSE、HU MAE、骨骼/软组织区域指标以及
与根目录评估协议兼容的SSIM。

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
| `--splits` | `val test` | 评估划分。 |
| `--air-upper-hu` | `-500` | 空气区域HU上限。 |
| `--bone-lower-hu` | `300` | 骨骼区域HU下限。 |
| `--output-dir` | 自动 | 指标和体数据输出目录。 |
| `--save-volumes` | 关闭 | 保存预测sCT和pCT GT NIfTI。 |
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
  --stage1_epochs 50 \
  --stage1_backbone_lr_factor 1.0 \
  --stage2_epochs 100 \
  --decoder_lr_factor 0.1 \
  --stage3_backbone_lr_factor 0.01 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

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

## 9. 重要注意事项

- 必须从0训练本prior encoder与decoder。
- 旧35 dB decoder可以人工作为初始化来源，但不能视为已适配新latent；本说明默认完全从0。
- 浅层prior checkpoint与当前checkpoint结构不兼容。
- 旧版4个ResidualBlock的deep_encoder checkpoint也不兼容。
- submodel得到更高PSNR并不保证主模型同步提高，仍需比较接入Adapter后的val/test结果。
- 31通道detail保留的信息远多于单通道平均值，需关注train/val/test差距与错误细节生成。
