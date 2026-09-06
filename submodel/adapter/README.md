# Latent Adapter 使用说明

## 1. 作用

本目录实现一个放在主模型 `Aggregator` 与 3D SRGAN `Decoder` 之间的轻量级
latent 适配器：

```text
稀疏视角投影
    ↓
2D Encoder + 几何查询 + Aggregator
    ↓
z_projection  [B, 256, X/4, Y/4, Z/4]
    ↓
LatentAdapter
    ↓
z_adapted     [B, 256, X/4, Y/4, Z/4]
    ↓
预训练 pCT Decoder
    ↓
sCT
```

Adapter 用于修正投影分支 latent 与 pCT 预训练 decoder 所使用 latent 之间的
通道组合、局部空间编码和数值分布差异。它不能凭空恢复投影中完全缺失的信息，
因此仍需要 Encoder/Aggregator 和重建损失共同学习。

默认不启用 Adapter，原主模型的行为保持不变。

## 2. 网络结构

实现位于 `submodel/adapter/model.py`：

```text
输入 z
  ├──────────────────────────────────────────────┐
  │                                              │
  └→ Conv3d 1×1×1: 256→64                       │
     → GELU                                      │
     → Conv3d 3×3×3: 64→64                      │
     → GELU                                      │
     → Conv3d 1×1×1: 64→256                     │
     ────────────────────────────────────────────┤
                                                 ↓
                                           z + residual
```

当 bottleneck 为64时约有14.3万权重参数。最后一个卷积的权重和偏置采用零初始化，
所以刚插入网络时严格满足 `Adapter(z) = z`，不会在第一次前向传播时给已有 latent
增加随机扰动。

未使用 `BatchNorm3d`，避免 batch size 很小时统计量不稳定以及额外改变 decoder
所依赖的 latent 绝对尺度。

## 3. 新增命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--use_adapter` | 关闭 | 在 Aggregator 与 Decoder 之间启用 Adapter。训练和评估必须保持一致。 |
| `--adapter_hidden_channels` | `64` | Adapter 的 bottleneck 通道数。第一组实验建议保持64。 |
| `--adapter_lr_factor` | `1.0` | Adapter 学习率相对于配置文件 `init_lr` 的倍率。设为0可冻结 Adapter。 |
| `--stage1_backbone_lr_factor` | `1.0` | 第一阶段 Encoder/Aggregator 的学习率倍率。设为0时，第一阶段只训练 Adapter。 |
| `--pretrained_backbone PATH` | 无 | 从旧主模型 checkpoint 中只加载 Encoder 和 Aggregator，不加载旧 Decoder、优化器或训练轮数。 |
| `--pretrained_decoder PATH` | 无 | 加载 submodel 预训练得到的原始 pCT Decoder，同时提供冻结的 `feature_stem` 作为 latent 教师。 |

`--pretrained_backbone` 所指 checkpoint 可以是主模型的 `ckpt_latest`，也可以是
`ckpt_history/ckpt_199` 这类历史 checkpoint。文件内部需要包含 `G_render`。

## 4. 推荐训练方式：旧几何主干 + 原始 prior decoder + 新 Adapter

这是最推荐的初始化方式：

- 保留旧主模型200 epoch得到的 Encoder/Aggregator；
- 不使用联合训练后可能已经漂移的旧 Decoder；
- 重新加载 pCT 预训练得到的原始 Decoder；
- 第一阶段冻结 Encoder、Aggregator 和 Decoder，只训练 Adapter；
- 第二阶段联合训练；第三阶段保持 Adapter 可训练，并仅微调 Decoder 高分辨率末端。

以下命令假定旧主模型的第199轮 checkpoint 和 prior decoder 路径实际存在：

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
  --use_adapter \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --pretrained_backbone train/checkpoints/dental_prior_transfer_after_refine/ckpt_history/ckpt_199 \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_best_val.pt \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --stage1_epochs 20 \
  --stage1_backbone_lr_factor 0 \
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

这里的200表示总训练轮数，而不是在旧模型训练轮数上继续计数。新实验会从 epoch 0
开始记录，但 Encoder/Aggregator 参数来自旧 checkpoint。

训练阶段如下：

| 阶段 | epoch范围 | Encoder/Aggregator | Adapter | Decoder |
|---|---|---|---|---|
| Stage 1 | 0～19 | 冻结 | 训练 | 完全冻结并保持 eval |
| Stage 2 | 20～119 | 正常训练 | 训练 | 以 `0.1×` 学习率联合训练 |
| Stage 3 | 120～199 | 以 `0.01×` 学习率训练 | 训练 | 只训练最后上采样块和输出层 |

Stage 1 的20轮是诊断性阶段，并非必须固定为20。如果验证集 PSNR/SSIM 在10轮左右
已经不再改善，可以提前缩短；如果 Adapter loss 仍在稳定下降，可以适当延长。

## 5. 完全从头训练的对照实验

不传 `--pretrained_backbone` 即可。由于 Encoder/Aggregator 此时是随机初始化，
Stage 1 不能把 backbone 冻结，应使用：

```bash
python train.py \
  --name dental_prior_adapter_from_scratch \
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
  --use_adapter \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_best_val.pt \
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

这个实验应作为对照，不建议替代上一节的推荐实验。

## 6. 在旧主模型上直接续训

代码允许在旧的、尚未包含 Adapter 的主模型 checkpoint 上使用 `--resume` 和
`--use_adapter`。缺失的 Adapter 参数会自动按恒等映射初始化，旧优化器若参数组不兼容
则自动重建。但这种方式会继续使用旧主模型中已经联合训练过的 Decoder，不会重新加载
`--pretrained_decoder`，因此它不能验证“原始35 dB decoder + Adapter”的假设，只适合作为
对照实验。

续训时 `--name`、`--checkpoints_path` 和 `--resume_name` 必须能定位旧 checkpoint，示例：

```bash
python train.py \
  --name dental_prior_transfer_after_refine \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 400 \
  --resume \
  --resume_name 199 \
  --use_adapter \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0
```

这里 `--epochs 400` 是最终总 epoch，若 checkpoint 的 `iter=200`，则继续执行
epoch 200～399。不要同时期待 `--pretrained_decoder` 覆盖旧 Decoder；resume 模式以完整
主模型 checkpoint 为准。

## 7. 评估命令

评估含 Adapter 的 checkpoint 时必须传入和训练时一致的两个结构参数：

```bash
python evaluate.py \
  --name dental_prior_adapter \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --eval_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --resume_name 199 \
  --use_adapter \
  --adapter_hidden_channels 64 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

结果默认写入：

```text
evaluate/logs/dental_prior_adapter/
evaluate/visuals/dental_prior_adapter/
```

可用 `--logs_path`、`--visual_path` 和 `--checkpoints_path` 修改位置。

如果 checkpoint 是使用 Adapter 训练的，却在评估时漏掉 `--use_adapter`，模型结构与
checkpoint 不一致，严格加载会报错；这可以避免在无意中绕过 Adapter 得到错误结果。

## 8. 建议比较的实验

至少保存以下三组结果，并使用完全相同的固定 HU/μ 范围计算 PSNR、SSIM：

1. 原主模型，不使用 Adapter；
2. 旧 Encoder/Aggregator + 原始 prior Decoder + Adapter（推荐实验）；
3. 随机初始化 Encoder/Aggregator + 原始 prior Decoder + Adapter（对照实验）。

若第二组在 Decoder 冻结的 Stage 1 就明显优于第一组，说明主要问题确实包含 latent
接口不兼容。若 Adapter latent loss 下降但 PSNR/SSIM 几乎不变，则更可能是稀疏投影
latent 本身缺少信息，需要进一步加入多尺度投影观测，而不是简单增加 Adapter 参数量。
