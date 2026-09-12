# Continuous Prior Completion（连续先验残差补全）

## 1. 目标与位置

本模块位于 `Adapter` 与预训练 `Decoder` 之间。它不把稀疏视角 latent 强制复制成
pCT latent，而是根据当前观测 latent 和本次扫描的 12 维投影几何，预测一个连续残差：

```text
投影图像 + poses
      │
      ├─ 2D Encoder → 几何查询 → Aggregator → Adapter
      │                                      ↓
      │                              z_aligned = A
      │                                      │
      └──────── poses = B ──→ ContinuousPriorCompletion f(A,B)
                                             │
                              z_completed = A + f(A,B)
                                             │
                                      Prior Decoder
                                             │
                                            sCT
```

训练时额外使用 pCT/CT GT 经过预训练 Prior Encoder 得到的 `z_prior=C`，只构造教师目标：

```text
delta_target = stopgrad(C - A)
delta_pred   = f(A, B)
L_completion = SmoothL1(delta_pred, delta_target)
```

`C` 从不作为补全网络输入，因此推理时不需要 pCT。几何 `B` 是条件，不与 latent
直接相乘；它经过集合编码器后用 FiLM 调制 3D 特征。

## 2. 模型结构

实现文件为 `submodel/continuous_prior_completion/model.py`。

```text
A: z_aligned [B,256,64,64,64]
 │
 ├─ 1×1 Conv：256→16
 ├─ 几何FiLM：h×(1+gamma)+beta
 ├─ Depthwise 3×3 Conv，dilation=1 → 1×1 Conv，残差连接
 ├─ Depthwise 3×3 Conv，dilation=2 → 1×1 Conv，残差连接
 ├─ Depthwise 3×3 Conv，dilation=4 → 1×1 Conv，残差连接
 │
B: poses [V,12] 或 [B,V,12]
 ├─ 每视角物理几何规范化
 ├─ 共享 MLP
 ├─ view mean + view max 集合池化（与视角排列顺序无关）
 └─ MLP → gamma、beta
                  │
             1×1 Conv：16→256（权重和bias零初始化）
                  │
             delta_pred [B,256,64³]
                  │
             z_completed=A+delta_pred
```

12维 pose 按以下顺序解析：

```text
[source_xyz, detector_center_xyz, detector_u_step_xyz, detector_v_step_xyz]
```

前两项是物理坐标；`u/v step` 同时携带探测器方向和单像素物理间距。编码器由它们
派生相对源位置3维、相对探测器位置3维、u/v单位方向各3维、相对u/v步长各1维，
再加入源角度的1～4阶正余弦特征8维，共22维。
位置先用所有“射线源→探测器中心”的中心射线做最小二乘交点，估计扫描等中心；再去除
全局平移并用平均SID归一化。相比直接使用射线源坐标均值，这在缺失角度、非均匀角度和
有限角扫描下不会把偏置后的轨迹均值误当成旋转中心。

补全器不使用 BatchNorm。最后一层单独零初始化，所以新模块接入时
`delta_pred=0`、`z_completed=A`，不会在第一步破坏原 Adapter/Decoder 接口；并不是
把整个网络全零初始化。

默认配置（`hidden=16`、几何隐藏宽度32、几何输出64）约有 **18,656个参数**，相对主模型
很小。主要显存来自64³特征图而不是参数量，因此默认开启内部 activation checkpoint。

## 3. 已接入主模型的位置

- `models/model.py`：`Aggregator → Adapter → prior_completion → Decoder`；从
  `ResEncoder.poses` 取得同一组几何。
- `train.py`、`evaluate.py`：按命令行参数构建相同结构。
- `trainer.py`：增加 Completion-only 阶段、独立优化器参数组、残差教师损失、
  checkpoint兼容和 TensorBoard 监控。
- `util/train_args.py`、`util/evaluate_args.py`：增加训练与评估参数。

不开启 `--use_prior_completion` 时，原主模型的数据流和 checkpoint 键保持不变。

## 4. 推荐训练阶段

开启本模块后，在原 A/B/C/D 中间插入独立阶段 P：

| 阶段 | 示例epoch | 可训练模块 | Completion是否进入前向 | 目的 |
|---|---:|---|---|---|
| A | 0–19 | Adapter | 否 | 先校准 latent 接口 |
| B | 20–59 | Encoder后部、Aggregator、Adapter | 否 | 让观测端产生 Decoder 可理解的表示 |
| P | 60–99 | 仅 Continuous Prior Completion | 是 | 在固定接口上学习 `C-A` 的统计补偿 |
| C | 100–179 | Completion、Adapter、Encoder、Aggregator、Decoder分层 | 是 | 联合适配并由后向前解冻 Decoder |
| D | 180–299 | 上述模块，小学习率主干 | 是 | pCT latent/残差教师逐渐降到0，转为3D/2D真实观测优化 |

阶段 P 中 Adapter、Encoder、Aggregator 和 Decoder 都处于冻结/eval状态。进入 C/D 后，
补全器继续训练；残差教师目标中的 `A` 与 `C` 都会 `detach`，因此该辅助损失只更新
补全器，重建和重投影损失仍可联合更新允许训练的其他模块。残差教师权重在P阶段保持
`completion_residual_lambda`，C阶段平滑减弱，D阶段继续平滑降到0；因此推理目标最终不会
被强制限制为逐元素复制 pCT latent。

阶段边界由下式确定：

```text
A结束 = phase_a_epochs
B结束 = phase_a_epochs + phase_b_epochs
P结束 = A + B + completion_phase_epochs
C结束 = A + B + P + phase_c_epochs + phase_c_hold_epochs
D轮数 = epochs - 上述所有轮数
```

## 5. 推荐训练命令（CNN Adapter，300 epoch）

以下命令使用已有主模型的 Encoder/Aggregator 作为起点，使用浅层 prior encoder 对应的
pCT Decoder checkpoint。路径应按实际文件修改：

```bash
python train.py \
  --name dental_continuous_prior_completion \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 300 \
  --pretrained_backbone train/checkpoints/dental_prior_transfer_after_refine/ckpt_history/ckpt_199 \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_latest.pt \
  --prior_encoder_type shallow \
  --use_adapter \
  --adapter_type cnn \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --use_prior_completion \
  --completion_hidden_channels 16 \
  --completion_geometry_hidden_channels 32 \
  --completion_geometry_channels 64 \
  --completion_residual_scale 1.0 \
  --completion_phase_epochs 40 \
  --completion_lr_factor 1.0 \
  --completion_residual_lambda 0.1 \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --latent_stat_lambda 0.1 \
  --phase_a_epochs 20 \
  --phase_b_epochs 40 \
  --phase_c_epochs 80 \
  --phase_c_hold_epochs 0 \
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
  --freeze_decoder_bn_stats \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

如果 Decoder 来自 `submodel/deep_encoder`，必须把两项同时改为匹配的 checkpoint 与类型：

```bash
--pretrained_decoder submodel/deep_encoder/checkpoints/<实验名>/ckpt_latest.pt \
--prior_encoder_type deep
```

不能把 shallow checkpoint 和 `--prior_encoder_type deep` 混用。

若要使用 Transformer Adapter，额外替换/增加：

```bash
--adapter_type transformer \
--adapter_transformer_pool_size 8 \
--adapter_transformer_layers 2 \
--adapter_transformer_heads 4 \
--adapter_transformer_dropout 0.1 \
--adapter_use_global_alpha \
--adapter_global_alpha_init 0.0
```

当前实验结果中 CNN Adapter 更稳，建议先以 CNN 版本验证 Continuous Completion 的独立
贡献，再做 Transformer 组合实验。

## 6. 断点续训

同一实验结构从 latest 继续：

```bash
python train.py \
  --name dental_continuous_prior_completion \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --is_train --epochs 300 \
  --resume \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_latest.pt \
  --prior_encoder_type shallow \
  --use_adapter --adapter_type cnn --adapter_hidden_channels 64 \
  --use_prior_completion \
  --completion_hidden_channels 16 \
  --completion_geometry_hidden_channels 32 \
  --completion_geometry_channels 64 \
  --completion_residual_scale 1.0 \
  --completion_phase_epochs 40 \
  --completion_lr_factor 1.0 \
  --completion_residual_lambda 0.1 \
  --latent_lambda 0.1 --latent_cosine_lambda 0.1 --latent_stat_lambda 0.1 \
  --phase_a_epochs 20 --phase_b_epochs 40 --phase_c_epochs 80 \
  --freeze_decoder_bn_stats
```

`--resume` 不带 `--resume_name` 时读取
`train/checkpoints/dental_continuous_prior_completion/ckpt_latest`；指定例如
`--resume_name 110` 时读取 `ckpt_history/ckpt_110`。续训必须保留与原实验相同的模型结构
参数和阶段边界。虽然主模型 checkpoint 已含 Decoder 权重，当前训练器仍需要
`--pretrained_decoder` 来重建冻结的 pCT Prior Encoder/anchor reference；主 Decoder 本身
在 `--resume` 时不会被该参数覆盖。

## 7. 评估命令

评估 `ckpt_history/ckpt_299`：

```bash
python evaluate.py \
  --name dental_continuous_prior_completion \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --eval_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --resume_name 299 \
  --use_adapter \
  --adapter_type cnn \
  --adapter_hidden_channels 64 \
  --use_prior_completion \
  --completion_hidden_channels 16 \
  --completion_geometry_hidden_channels 32 \
  --completion_geometry_channels 64 \
  --completion_residual_scale 1.0 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

评估始终报告 PSNR 和 SSIM。末尾 bone/soft/SSIM-loss 参数只控制附加区域 loss 报告，
不参与推理优化；不需要这些报告时可以删除。结果保存到：

```text
evaluate/logs/dental_continuous_prior_completion/<几何配置>/
evaluate/visuals/dental_continuous_prior_completion/<几何配置>/<病例>/volume/
```

评估命令中的 Adapter 类型、Transformer 参数（若使用）以及 Completion 的结构参数必须
与训练 checkpoint 一致，否则 state_dict 形状不匹配。

若 checkpoint 尚未训练并保存 `prior_completion.*` 权重，评估程序会直接报错，而不会
悄悄使用零残差恒等分支。此时应去掉 `--use_prior_completion`，或改用完成P/C/D阶段训练后
保存的 checkpoint。

## 8. 新参数解释

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--use_prior_completion` | 关闭 | 构建并启用 Continuous Prior Completion；要求同时启用 Adapter。 |
| `--completion_hidden_channels` | 16 | 3D补全主干的瓶颈通道数。越大表达力和显存越高；32GB显存先用16。 |
| `--completion_geometry_hidden_channels` | 32 | 几何编码器中逐视角共享MLP的隐藏宽度。 |
| `--completion_geometry_channels` | 64 | 视角集合池化后的几何条件向量宽度。 |
| `--completion_residual_scale` | 1.0 | `tanh`残差的最大缩放系数；训练和评估必须一致。 |
| `--completion_phase_epochs` | 40 | 插在B与C之间、仅训练补全器的轮数；只在训练中使用。 |
| `--completion_lr_factor` | 1.0 | 补全器学习率相对 `conf` 中基础学习率的倍率。 |
| `--completion_residual_lambda` | 0.1 | P阶段的 `SmoothL1(delta_pred, stopgrad(C-A))` 初始权重；C/D中自动平滑衰减。 |
| `--disable_completion_checkpoint` | 关闭 | 关闭补全器内部 activation checkpoint；速度略快但显存更高。 |

主模型已有参数中，`--pretrained_decoder` 同时提供 Decoder 初值、pCT Prior Encoder
权重和 anchor reference；`--pretrained_backbone` 只加载旧主模型的 Encoder/Aggregator；
`--freeze_decoder_bn_stats` 在 Decoder 解冻后固定 BN running mean/variance，不冻结卷积层，
也不冻结 BN affine weight/bias。

如果 `completion_target_abs_mean` 长期接近或超过 `completion_residual_scale`，说明 `tanh`
可能饱和，可以适度增大 scale；若预测残差明显过强、PSNR反而下降，则优先减小 scale 或
`completion_residual_lambda`。不要只根据训练集残差 loss 调参，应以验证/测试 PSNR、SSIM
和差异图为准。

## 9. TensorBoard与诊断指标

除原有 train/val/test 的 PSNR、SSIM、3D/2D/区域 loss 和 latent 指标外，新增：

| 指标 | 含义 |
|---|---|
| `completion_residual_raw` | 未乘权重的残差 SmoothL1。 |
| `completion_residual_loss` | 加入总 loss 的加权残差损失。 |
| `completion_pred_abs_mean` | 预测补偿量绝对值均值。 |
| `completion_target_abs_mean` | 教师 `C-A` 绝对值均值。 |
| `step/lr_prior_completion` | 补全器当前实际学习率。 |
| `step/completion_residual_weight` | 当前实际残差教师权重；应在D阶段平滑降到0。 |
| `step/training_stage=5` | 当前处于 Completion-only 阶段P。 |

重点判断：P阶段后验证/测试 PSNR、SSIM是否同步改善；预测残差幅度是否逐渐接近目标幅度；
训练集改善而验证/测试下降，说明网络在记忆 pCT 残差或产生不可观测细节，应降低
`completion_hidden_channels`、`completion_residual_lambda` 或缩短 P 阶段。

## 10. 角度条件的必要注意事项

如果所有训练病例始终采用完全相同的 `20个uniform角度`，那么 pose 集合条件对每个样本
都是常量，网络不可能学会“不同缺失角度对应不同补偿规律”；此时它退化为带固定几何
条件的连续残差网络。要验证角度编码价值，训练集必须出现多种角度子集、起始角、缺口
或视角数。

当前 `CBCTDataset` 的 `random` 会在线重新生成 DRR，代价很高。第一轮可以保留
`uniform` 做结构消融，但结论只能证明补全器本身是否有效；正式训练角度条件时，建议
预先生成多套投影/pose组合，或后续增加“从已有全角度 proj 中随机抽取视角”的轻量采样，
并保证投影图像与 pose 使用完全相同的索引。

建议至少比较：

1. 原 CNN Adapter，不启用 Completion；
2. CNN Adapter + Completion，但固定 uniform 角度；
3. CNN Adapter + Completion，训练时变化角度子集；
4. 第3组在测试时使用未见过的角度子集。

这样才能区分提升来自额外参数、连续先验残差，还是来自真正的几何条件泛化。

## 11. 延长A/B/P后的推荐300 epoch训练方案

延长A、B、P阶段的思路合理，但不建议无限延长A和P。A阶段只训练Adapter，训练过久
容易让Adapter独自承担本不应由它负责的信息补全；P阶段只训练Continuous Prior
Completion，训练过久则容易记忆训练集中的 `C-A` 残差。更值得延长的是B阶段，因为
Encoder后部和Aggregator决定了稀疏投影可提取观测信息的上限。

对于总计300 epoch，推荐按下表分配：

| 阶段 | Epoch范围 | 数量 | 作用 |
|---|---:|---:|---|
| A | 0–29 | 30 | 冻结观测端和Decoder，只训练Adapter。 |
| B | 30–99 | 70 | 训练Encoder后部、Aggregator和Adapter，建立稳定的观测表示。 |
| P | 100–149 | 50 | 固定A/B阶段形成的接口，只训练Continuous Prior Completion。 |
| C-progressive | 150–209 | 60 | Decoder由后向前分层解冻。 |
| C-hold | 210–229 | 20 | Decoder已经完全解冻，以相同状态继续稳定训练。 |
| D | 230–299 | 70 | latent/completion教师逐渐降到0，转为依靠真实重建和投影约束。 |

总轮数为：

```text
30 + 70 + 50 + 60 + 20 + 70 = 300
```

与原推荐方案相比：

```text
原方案：A20 + B40 + P40 + C80 + D120
新方案：A30 + B70 + P50 + C-progressive60 + C-hold20 + D70
```

新方案更符合“先把接口训练到平台，再开启补全”的思路，并将D阶段限制为70 epoch，
避免后期自由微调时间过长而进一步破坏prior空间。

### 11.1 阶段边界参数

对应的阶段参数为：

```bash
--epochs 300 \
--phase_a_epochs 30 \
--phase_b_epochs 70 \
--completion_phase_epochs 50 \
--phase_c_epochs 60 \
--phase_c_hold_epochs 20
```

D阶段不需要单独传入epoch数，训练器会自动计算：

```text
D = epochs
    - phase_a_epochs
    - phase_b_epochs
    - completion_phase_epochs
    - phase_c_epochs
    - phase_c_hold_epochs
  = 300 - 30 - 70 - 50 - 60 - 20
  = 70
```

### 11.2 推荐的完整阶段与学习率参数

```bash
--epochs 300 \
--phase_a_epochs 30 \
--phase_b_epochs 70 \
--completion_phase_epochs 50 \
--phase_c_epochs 60 \
--phase_c_hold_epochs 20 \
--adapter_lr_factor 1.0 \
--phase_b_encoder_lr_factor 0.2 \
--phase_b_aggregator_lr_factor 0.5 \
--completion_lr_factor 1.0 \
--phase_c_backbone_lr_factor 0.1 \
--phase_c_aggregator_lr_factor 0.3 \
--phase_d_backbone_lr_factor 0.01 \
--phase_d_aggregator_lr_factor 0.05 \
--decoder_lr_factor 0.1 \
--decoder_core_lr_factor 0.01
```

### 11.3 如何判断各阶段是否已经到达瓶颈

- **A阶段**：观察 `val/test PSNR` 和latent loss，而不能只看train PSNR。连续10个epoch
  的验证/测试PSNR提升不足约 `0.05 dB`，可以认为Adapter接口校准已接近平台。
- **B阶段**：重点观察val/test PSNR、SSIM和 `latent_smooth_l1_raw`。如果train指标继续
  提升，但val/test不再提升，不应继续延长B阶段。
- **P阶段**：同时观察 `completion_residual_raw`、`completion_pred_abs_mean`、
  `completion_target_abs_mean` 和val/test PSNR。如果残差loss持续下降但PSNR、SSIM不再
  提升，说明补全器可能只是在拟合teacher residual，应停止继续延长P阶段。
- **C阶段**：每次Decoder解冻都可能产生短期指标波动，应观察解冻后5～10个epoch能否
  恢复并超过解冻前的最佳验证指标，不应根据切换当轮的下降立即判定失败。
- **D阶段**：70 epoch通常已经足够。D阶段的主要任务是逐渐摆脱latent teacher和
  completion residual teacher，而不是继续长期复制pCT latent。

### 11.4 固定阶段与平台判断的关系

当前训练器仍然按照命令中指定的epoch边界切换阶段，不会自动检测“瓶颈”。因此
`30/70/50/60/20/70` 应理解为一组合理的阶段上限，而不是每个数据集都必须机械训练到
该轮数。

如果某阶段提前稳定，可以停止实验，修改对应的阶段epoch参数后从合适的checkpoint
重新开始或续训。不要只为了凑足预设轮数继续单模块拟合，否则A阶段可能让Adapter承担
过多补全，P阶段也可能开始记忆训练病例的 `C-A` 残差。
