# CNN + Transformer Latent Adapter

## 1. 作用与模型结构

本模块放在主模型的 Aggregator 与 pCT 预训练 Decoder 之间，用于把稀疏投影形成的
三维 latent 转换为 Decoder 更容易解释的表示。它随根目录 `train.py` 一起训练，权重
保存在主模型 checkpoint 的 `G_render.adapter` 中，没有独立训练入口。

```text
z_sparse [B,256,64,64,64]
       │
       ├── 局部分支（复用 submodel/adapter/LatentAdapter）
       │     1×1×1 Conv：256→64
       │     GELU
       │     3×3×3 Conv：64→64
       │     GELU
       │
       └── 全局分支
             1×1×1 Conv：256→64 + GELU
             AdaptiveAvgPool3d：8×8×8
             512个token + 可学习位置编码
             2个Transformer Encoder Block（默认4 heads）
             三线性上采样到64×64×64
                    │
                    ▼
             局部特征 + 全局特征
                    ↓
             1×1×1 Conv：64→256
                    ↓
       z_adapted = z_sparse + residual
```

输出投影层采用零初始化，因此刚接入时严格为恒等映射。默认参数量约 259,904。

## 2. 当前唯一训练流程：Phase A/B/C/D

旧三阶段参数已经移除。训练必须同时提供 `--pretrained_backbone`、
`--pretrained_decoder` 和 `--use_adapter`。以总共 200 epoch，A/B/C 分别为
20/40/80 epoch 为例，Phase D 自动使用剩余 60 epoch：

| 阶段 | epoch | Encoder | Aggregator | Adapter | Decoder | latent教师 | prior anchor |
|---|---:|---|---|---|---|---|---|
| A | 0～19 | 冻结 | 冻结 | 训练 | 完全冻结并保持eval | 0.1 | 关闭 |
| B | 20～59 | 只训练layer3/layer4 | 训练 | 训练 | 完全冻结并保持eval | 0.1平滑降至0.07 | 关闭 |
| C1 | 60～79 | 全部训练 | 训练 | 训练 | 只解冻输出层 | 0.07继续下降 | 0.1 |
| C2 | 80～99 | 同上 | 同上 | 同上 | 再解冻最后一个上采样块 | 继续下降 | 0.1 |
| C3 | 100～119 | 同上 | 同上 | 同上 | 再解冻前面的上采样块 | 继续下降 | 0.1 |
| C4 | 120～139 | 同上 | 同上 | 同上 | 最后解冻低分辨率残差主体 | 降至0.01 | 0.1 |
| D | 140～199 | 小学习率 | 小学习率 | 训练 | 全部训练，主体保持更小LR | 0.01平滑降至0 | 0.025 |

Phase B 的 Encoder 后部明确对应 `ResEncoder.model.layer3` 和 `layer4`；conv1、
layer1、layer2 保持冻结且为 eval 模式。Phase C 的四段由程序按
`--phase_c_epochs` 等分，无需再传四个边界。

Decoder 的解冻顺序为：

```text
out_blk
→ 最后一个up_blk（最高分辨率）
→ 更早的up_blk
→ in_blk + res_blk_list + res_blk_last（低分辨率残差主体）
```

## 3. latent监督与prior保持

现在 latent 损失直接比较未标准化的原始张量：

```text
L_latent = L_raw
         + latent_cosine_lambda × L_cos
         + latent_stat_lambda × (L_mean + L_std)
```

- `L_raw`：`smooth_l1(z_adapted, z_teacher)`，保留幅值、均值和方差信息；
- `L_cos`：展平空间维度后的通道方向余弦距离；
- `L_mean`：逐样本逐通道空间均值的 L1；
- `L_std`：逐样本逐通道空间标准差的 L1；
- 标准化后的 Smooth L1 仍记录为监控指标，但不参与总损失。

TensorBoard 对应名称为：

```text
latent_smooth_l1_raw
latent_cosine_raw
latent_mean_l1_raw
latent_std_l1_raw
latent_normalized_smooth_l1_raw
latent_weight
```

Phase C/D 还使用冻结的原始 pCT Decoder 作为参考：

```text
L_anchor = |D_current(z_pCT) - D_frozen(z_pCT)|_1
```

它约束正在解冻的 Decoder 不要遗忘 pCT 恢复能力。参考 Decoder 会额外占用一份
Decoder 参数显存；当前 Decoder 的 anchor 前向使用激活重计算以降低反向传播显存。

## 4. 推荐训练命令

```bash
python train.py \
  --name dental_prior_adapter_transformer \
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
  --pretrained_backbone train/checkpoints/dental_prior_transfer_after_refine/ckpt_history/ckpt_199 \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_best_val.pt \
  --prior_encoder_type shallow \
  --use_adapter \
  --adapter_type transformer \
  --adapter_hidden_channels 64 \
  --adapter_transformer_pool_size 8 \
  --adapter_transformer_layers 2 \
  --adapter_transformer_heads 4 \
  --adapter_transformer_dropout 0.1 \
  --adapter_lr_factor 1.0 \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --latent_stat_lambda 0.1 \
  --phase_a_epochs 20 \
  --phase_b_epochs 40 \
  --phase_c_epochs 80 \
  --phase_b_encoder_lr_factor 0.2 \
  --phase_b_aggregator_lr_factor 0.5 \
  --phase_c_backbone_lr_factor 0.1 \
  --phase_c_aggregator_lr_factor 0.3 \
  --phase_d_backbone_lr_factor 0.01 \
  --phase_d_aggregator_lr_factor 0.05 \
  --decoder_lr_factor 0.1 \
  --decoder_core_lr_factor 0.01 \
  --prior_anchor_lambda 0.1 \
  --phase_d_anchor_factor 0.25 \
  --phase_b_latent_end_factor 0.7 \
  --phase_c_latent_end_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

若 prior checkpoint 来自 `submodel/deep_encoder`，必须成对替换为：

```bash
--pretrained_decoder submodel/deep_encoder/checkpoints/你的实验/ckpt_best_val.pt \
--prior_encoder_type deep
```

不要把 deep checkpoint 与 `shallow` 混用。

## 5. 参数解释

### 5.1 阶段与学习率

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--phase_a_epochs` | 20 | 只训练Adapter的轮数。 |
| `--phase_b_epochs` | 40 | 独立Phase B轮数，只放开Encoder后部和Aggregator。 |
| `--phase_c_epochs` | 80 | Decoder由后向前分四段解冻的总轮数。 |
| `--phase_b_encoder_lr_factor` | 0.2 | Phase B中layer3/layer4相对基础LR。 |
| `--phase_b_aggregator_lr_factor` | 0.5 | Phase B中Aggregator相对基础LR。 |
| `--phase_c_backbone_lr_factor` | 0.1 | Phase C中完整Encoder相对基础LR。 |
| `--phase_c_aggregator_lr_factor` | 0.3 | Phase C中Aggregator相对基础LR。 |
| `--phase_d_backbone_lr_factor` | 0.01 | Phase D中完整Encoder相对基础LR。 |
| `--phase_d_aggregator_lr_factor` | 0.05 | Phase D中Aggregator相对基础LR。 |
| `--adapter_lr_factor` | 1.0 | 所有阶段中Adapter相对基础LR。 |
| `--decoder_lr_factor` | 0.1 | Decoder输出层和上采样块相对基础LR。 |
| `--decoder_core_lr_factor` | 0.01 | Decoder低分辨率残差主体相对基础LR。 |

Phase D 轮数为：

```text
epochs - phase_a_epochs - phase_b_epochs - phase_c_epochs
```

必须至少为 1。配置文件原有的 step/gamma 学习率衰减仍会乘到上述倍率上。

### 5.2 latent与anchor

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--latent_lambda` | 0 | Phase A起始latent总权重；推荐显式设为0.1。 |
| `--latent_cosine_lambda` | 0.1 | latent内部余弦项系数。 |
| `--latent_stat_lambda` | 0.1 | mean与std两项共同的系数。 |
| `--phase_b_latent_end_factor` | 0.7 | Phase B结束时为初始latent权重的0.7倍。 |
| `--phase_c_latent_end_factor` | 0.1 | Phase C结束时为初始latent权重的0.1倍。 |
| `--prior_anchor_lambda` | 0.1 | Phase C的prior保持损失权重。 |
| `--phase_d_anchor_factor` | 0.25 | Phase D anchor相对Phase C的倍率。 |

要求 `0 ≤ phase_c_latent_end_factor ≤ phase_b_latent_end_factor ≤ 1`。
Phase D 会逐 epoch 连续衰减到 0，而不是在边界突然关闭 latent 教师。

### 5.3 Adapter结构

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--use_adapter` | 关闭 | 四阶段训练必须启用。 |
| `--adapter_type` | cnn | 本模块必须设为transformer。 |
| `--adapter_hidden_channels` | 64 | CNN局部分支和token维度。 |
| `--adapter_transformer_pool_size` | 8 | 每轴池化尺寸，8对应512个token。 |
| `--adapter_transformer_layers` | 2 | Transformer Block数量。 |
| `--adapter_transformer_heads` | 4 | 注意力头数，必须整除hidden channels。 |
| `--adapter_transformer_dropout` | 0.1 | 注意力和FFN的dropout。 |

## 6. 继续训练

恢复训练时必须保留原来的模型结构、prior路径和所有阶段边界。例如从 epoch 99
继续跑到总 epoch 200，在上面的完整命令中额外加入：

```bash
--resume \
--resume_name 99
```

并保持 `--epochs 200`。程序会从 checkpoint 中的 `iter=100` 开始，而不是额外训练
200轮。旧三阶段 checkpoint 的优化器参数组与当前实现不同，因此不建议用它直接恢复；
为公平比较，应使用新的实验名从 epoch 0 开始。

## 7. 评估命令

```bash
python evaluate.py \
  --name dental_prior_adapter_transformer \
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
  --adapter_type transformer \
  --adapter_hidden_channels 64 \
  --adapter_transformer_pool_size 8 \
  --adapter_transformer_layers 2 \
  --adapter_transformer_heads 4 \
  --adapter_transformer_dropout 0.1
```

该命令加载：

```text
train/checkpoints/dental_prior_adapter_transformer/ckpt_history/ckpt_199
```

结果保存在：

```text
evaluate/logs/dental_prior_adapter_transformer/
evaluate/visuals/dental_prior_adapter_transformer/
```

评估不构造训练期teacher和anchor参考网络，因此不传prior、latent或阶段参数。若希望
额外报告骨/软组织/局部SSIM loss，可再传对应loss参数；它们不会改变预测结果、PSNR、
`ssim_3d_clamp`或NIfTI输出。

## 8. 观察重点

- Phase A：冻结 Decoder 时，`D_pre(A(z_sparse))` 是否明显优于无Adapter；
- Phase B：raw、cosine、mean、std是否同步改善，而不只是标准化latent指标下降；
- Phase C：每次解冻边界是否发生短暂波动，以及验证/测试指标能否随后恢复并提高；
- Phase C/D：`prior_anchor_raw`若明显上升，应降低Decoder LR或提高anchor；
- Phase D：latent趋近0后PSNR/SSIM是否保持，判断模型是否真正转向投影观测；
- 同时比较train/val/test、骨边缘、软组织和差异热图，不能只看训练PSNR。
