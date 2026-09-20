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
| `--adapter_type` | `cnn` | Adapter类型：`cnn`为原始CNN版本，`transformer`为局部CNN与全局Transformer双分支版本。 |
| `--adapter_hidden_channels` | `64` | Adapter 的 bottleneck 通道数。第一组实验建议保持64。 |
| `--adapter_lr_factor` | `1.0` | Adapter 学习率相对于配置文件 `init_lr` 的倍率。设为0可冻结 Adapter。 |
| `--adapter_transformer_pool_size` | `8` | 全局分支池化后的单轴尺寸；8对应512个token。 |
| `--adapter_transformer_layers` | `2` | Transformer Encoder Block数量。 |
| `--adapter_transformer_heads` | `4` | 多头注意力的head数量，必须整除hidden channels。 |
| `--adapter_transformer_dropout` | `0.1` | Transformer注意力及FFN的dropout。 |
| `--phase_a_encoder_lr_factor` | `0.0` | Phase A 的 Encoder 学习率倍率。默认 0 时 Phase A 只训练 Adapter，这要求 backbone 已有预训练权重；从 0 训练时应设为正值。 |
| `--phase_a_aggregator_lr_factor` | `0.0` | Phase A 的 Aggregator 学习率倍率。默认 0；从 0 训练时应设为正值。 |
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
  --phase_a_epochs 20 \
  --phase_b_epochs 100 \
  --phase_c_epochs 60 \
  --decoder_lr_factor 0.1 \
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

训练阶段如下（`--phase_a_epochs 20 --phase_b_epochs 100 --phase_c_epochs 60`，
`--epochs 200` 的剩余轮次归 Phase D）：

| 阶段 | epoch范围 | Encoder/Aggregator | Adapter | Decoder |
|---|---|---|---|---|
| Phase A | 0～19 | 冻结（`phase_a_encoder_lr_factor`/`phase_a_aggregator_lr_factor` 为 0） | 训练 | 完全冻结并保持 eval |
| Phase B | 20～119 | 仅 `encoder.layer3/layer4` 与 Aggregator，倍率见 `phase_b_encoder_lr_factor`/`phase_b_aggregator_lr_factor` | 训练 | 冻结并保持 eval |
| Phase C | 120～179 | `phase_c_backbone_lr_factor`/`phase_c_aggregator_lr_factor` | 训练 | 由后向前分四段解冻 |
| Phase D | 180～199 | `phase_d_backbone_lr_factor`/`phase_d_aggregator_lr_factor` | 训练 | 全解冻，倍率 `decoder_lr_factor` |

Stage 1 的20轮是诊断性阶段，并非必须固定为20。如果验证集 PSNR/SSIM 在10轮左右
已经不再改善，可以提前缩短；如果 Adapter loss 仍在稳定下降，可以适当延长。

Phase A 默认把 Encoder/Aggregator 冻结，只训练 Adapter。这个默认值成立的前提是
backbone 已有合理的预训练权重（由 `--pretrained_backbone` 提供）。若 backbone 是随机
初始化的，冻结它会让整个阶段只有零初始化的 Adapter 在更新，等于空跑：此时必须要么
提供 `--pretrained_backbone`，要么按第 5 节显式打开 Phase A 的主干学习率。程序会对
“从 0 训练 + Phase A 冻结主干”直接报错，而不是静默浪费该阶段。

## 5. 完全从头训练的对照实验

不传 `--pretrained_backbone` 即可。由于 Encoder/Aggregator 此时是随机初始化，
Phase A 不能把 backbone 冻结，必须用 `--phase_a_encoder_lr_factor` 和
`--phase_a_aggregator_lr_factor` 打开主干：

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

各阶段主干是否可训练：

| 阶段 | Encoder early | Encoder late | Aggregator | Adapter | Decoder |
|---|---:|---:|---:|---:|---:|
| Phase A | `phase_a_encoder_lr_factor` | `phase_a_encoder_lr_factor` | `phase_a_aggregator_lr_factor` | `adapter_lr_factor` | 冻结并保持 eval |
| Phase B | 冻结 | `phase_b_encoder_lr_factor` | `phase_b_aggregator_lr_factor` | `adapter_lr_factor` | 冻结并保持 eval |
| Phase C | `phase_c_backbone_lr_factor` | `phase_c_backbone_lr_factor` | `phase_c_aggregator_lr_factor` | `adapter_lr_factor` | 由后向前分四段解冻 |
| Phase D | `phase_d_backbone_lr_factor` | `phase_d_backbone_lr_factor` | `phase_d_aggregator_lr_factor` | `adapter_lr_factor` | `decoder_lr_factor` 起 |

注意 Phase B 只解冻 `encoder.layer3/layer4`，`encoder.layer1/layer2` 到 Phase C 才会
以 `phase_c_backbone_lr_factor` 重新参与。从 0 训练时如果希望 A 阶段之后主干保持连续
可训练，可以把 `--phase_b_epochs` 设小或设为 0，让训练从 Phase A 直接进入 Phase C。

这个实验应作为对照，不建议替代上一节的推荐实验。

## 5.1 两个可直接对比的命令：有 Adapter / 无 Adapter

下面两条命令**只有 `--use_adapter` 这一个开关不同**，数据、几何、损失权重完全一致，
因此可以直接比较"latent 接口是否需要适配"。

运行前先定位到项目根目录。注意要用装有 SimpleITK 的解释器；本机的 conda base 环境没有
SimpleITK，直接 `python train.py` 会在建数据集时报 `ModuleNotFoundError`：

```bash
cd /root/autodl-tmp/workspace/Geometry-Aware-Attenuation-Learning-for-Sparse-View-CBCT-Reconstruction-main
```

两条命令共用的前提：

- 数据是 `tools/thorax_preprocessing.prepare_thorax` 生成的 `dataset/thorax/syn_data`，
  划分是 `data/dataset_split/thorax_split.json`（125/15/15/2）；
- `--require-gt-source registered-ct` 让 `train.py` 在加载数据前校验每个病例的
  `transforms.json` 记录 `gt_source=registered-ct`，确保 3D 标签是**配准后的 pCT** 而不是 CBCT；
- Decoder 由 `submodel/deep_encoder/checkpoints/thorax_deep_decoder/ckpt_best_val.pt` 初始化。
  `--prior_encoder_type deep` **在两种命令里都必须传**：该 checkpoint 的 `feature_stem` 是
  mean/detail 版（键为 `detail_downsample.*`、`feature_stem.0/2/4.*`），而默认的 shallow
  `PriorFeatureStem` 键是 `0.weight`/`2.weight`，教师权重用 `strict=True` 加载，键名不符会直接报错；
- 都**不传 `--pretrained_backbone`**，即 Encoder/Aggregator 从 0 训练。

### A. 有 Adapter（四阶段 prior transfer）

```bash
/autdl-tmp/conda_env/GeoAware/bin/python train.py \
  --name thorax_prior_adapter_from_scratch \
  --datadir ./dataset/thorax/syn_data \
  --datatype thorax \
  --require-gt-source registered-ct \
  --train_scale 4 \
  --fusion ada \
  --start 0 --end 360 --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 200 \
  --use_adapter \
  --adapter_hidden_channels 64 \
  --adapter_lr_factor 1.0 \
  --pretrained_decoder submodel/deep_encoder/checkpoints/thorax_deep_decoder/ckpt_best_val.pt \
  --prior_encoder_type deep \
  --latent_lambda 0.1 --latent_cosine_lambda 0.1 \
  --phase_a_epochs 50 \
  --phase_a_encoder_lr_factor 1.0 --phase_a_aggregator_lr_factor 1.0 \
  --phase_b_epochs 50 --phase_c_epochs 80 \
  --decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01
```

**这条命令在做什么**

`--use_adapter` 与 `--pretrained_decoder` 同时存在会激活四阶段调度。Adapter 是插在
Aggregator 与 Decoder 之间的零初始化残差 CNN（hidden 64，约 14.3 万参数），插入时严格
等于恒等映射 `Adapter(z)=z`，所以它不会在第一步就给已有的 latent 加噪声。

`--latent_lambda 0.1 --latent_cosine_lambda 0.1` 打开 latent 对齐：冻结的 prior stem 把
**GT 体数据**编码成教师 latent，投影分支的 latent 去拟合它（L1 + cosine + 通道 mean/std）。
这是"让投影分支产出的 latent 落进预训练 Decoder 的 latent 基座"的唯一显式监督。

因为不传 `--pretrained_backbone`（主干随机初始化），**Phase A 必须用
`--phase_a_encoder_lr_factor 1.0 --phase_a_aggregator_lr_factor 1.0` 打开主干**。这两个参数
默认是 0，即"Phase A 只训练 Adapter"——对随机主干那是空跑，程序会直接报错拦住你。

阶段划分（200 epoch）：

| 阶段 | epoch | Encoder | Aggregator | Adapter | Decoder | latent 权重 |
|---|---|---|---|---|---|---|
| A | 0–49 | 1.0× | 1.0× | 1.0× | 冻结 + eval | 0.1（满） |
| B | 50–99 | 仅 layer3/4 0.2× | 0.5× | 1.0× | 冻结 + eval | 0.1 → 0.07 |
| C | 100–179 | 0.1× | 0.3× | 1.0× | 由后向前分四段解冻 | 0.07 → 0.01 |
| D | 180–199 | 0.01× | 0.05× | 1.0× | 全解冻，`decoder_lr_factor` 0.1× | 0.01 → 0 |

基础学习率 `init_lr=1e-4`，每个参数组的实际 LR = `1e-4 × 阶段倍率 × 0.5^(epoch//50)`。
另外 `--prior_anchor_lambda` 默认 0.1，会在 Phase C/D 生效，用一份冻结的 decoder 副本约束
输出不要偏离先验 decoder。

**生效的损失**：`mse_3d 1.0 + gd1 1.0 + latent 0.1(→0) + prior_anchor 0.1(仅 C/D) +
bone 0.05 + soft_mask 0.01 + ssim 0.01 + 重投影 mse_2d 0.01`。

**它验证的假设**：预训练 Decoder 需要的 latent 与投影分支产出的 latent 之间存在接口不兼容，
这个不兼容可以被一个轻量 Adapter 修正。

### B. 无 Adapter（stage 0 联合训练）

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
  --epochs 200 \
  --pretrained_decoder submodel/deep_encoder/checkpoints/thorax_deep_decoder/ckpt_best_val.pt \
  --prior_encoder_type deep \
  --stage0_decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01
```

**这条命令在做什么**

去掉 `--use_adapter` 后 `use_four_phase=False`，`_training_stage()` 对所有 epoch 都返回 0：
**四阶段调度完全不激活，也不需要写任何 `--phase_*` 参数**（传了会被静默忽略，不报错）。
此时：

- 整个模型（encoder / aggregator / decoder）全部可训练；
- 所有参数组共用同一个学习率，即 `1e-4 × 0.5^(epoch//50)`，没有分阶段倍率。

因此有三个**静默行为**必须注意：

| 项 | 行为 | 说明 |
|---|---|---|
| `--latent_lambda` | **静默失效**（实测 `latent_weight=0.0`） | `_latent_weight()` 在 stage 0 走 `else: return 0.0`，latent 对齐被关闭，别传 |
| `--decoder_lr_factor` | **静默失效** | 它只在 stage 3/4/7/8 被引用，stage 0 不读它 |
| `--prior_encoder_type deep` | **仍然必须传** | 见前面的共用前提，教师权重是 `strict=True` 加载的 |

为了让预训练 Decoder 不被随机主干的噪声梯度以满学习率冲掉，用
`--stage0_decoder_lr_factor 0.1` 把它降到 1e-5；改成 `0` 则完全冻结 Decoder
（只训练随机初始化的 Encoder/Aggregator，最"纯粹"的 baseline）。不传该参数时所有组都是
1e-4，即 Decoder 从第 0 轮就以满学习率微调。

**生效的损失**：`mse_3d 1.0 + gd1 1.0 + bone 0.05 + soft_mask 0.01 + ssim 0.01 +
重投影 mse_2d 0.01`——**没有 latent 对齐，也没有 prior anchor**（anchor 只在 Phase C/D 生效）。

**它的角色**：对照组。它回答"不加适配、直接联合训练能到什么水平"，但它**不检验** latent
接口假设：Decoder 的 latent 基座只能靠重建损失间接对齐。

### C. 两者的差异与结果判读

| 维度 | A. 有 Adapter | B. 无 Adapter |
|---|---|---|
| 调度 | 四阶段 A/B/C/D | 单一 stage 0 |
| 主干 LR | 分阶段（1.0→0.2/0.5→0.1→0.01） | 恒 1e-4 |
| Decoder | A/B 冻结，C 渐进解冻，D 0.1× | 由 `--stage0_decoder_lr_factor` 控制（本例 1e-5） |
| latent 对齐 | 0.1，跨阶段衰减到 0 | 无 |
| prior anchor | Phase C/D 为 0.1 | 无 |
| 额外参数 | +14.3 万（Adapter） | 0 |
| 对应开关 | `--use_adapter` 及其相关参数 | 只需删掉 `--use_adapter` 和 `--phase_*`，加 `--stage0_decoder_lr_factor` |

判读方式（两组用同一固定 HU/μ 范围算 PSNR/SSIM）：

- A 明显优于 B → latent 接口不兼容确实是主要瓶颈，Adapter 起了作用；
- A 的 latent loss 稳定下降但 PSNR/SSIM 与 B 基本持平 → 稀疏投影的 latent 本身缺信息，
  继续加大 Adapter 参数量不会有收益，应该去增加多尺度投影观测；
- A 反而差于 B（尤其 Phase A/B 就落后）→ 预训练 Decoder 的 latent 基座不适合这批 thorax
  数据，需要回到 `submodel/deep_encoder` 重新预训练 Decoder，而不是调 Adapter。

> 如果你想要"保留四阶段调度、但 Adapter 不产生任何作用"，可以改用
> `--use_adapter --adapter_lr_factor 0`：CNN Adapter 末层零初始化且
> `adapter_lr_factor=0` 时其参数 `requires_grad=False`，前向恒为恒等映射，数学上等价于
> 没有 Adapter，但阶段划分、分阶段学习率和 latent 对齐全部保留。这比 B 更接近"只去掉
> Adapter 模块、保留先验迁移调度"的对照。

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

评估含 Adapter 的 checkpoint 时必须传入和训练时一致的结构参数：

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

## 9. CNN + Transformer双分支Adapter

Transformer版本实现在：

```text
submodel/adapter_with_transformer/model.py
```

主模型中的接入位置不变：

```text
Encoder → 几何查询 → Aggregator → TransformerLatentAdapter → Decoder
```

### 9.1 模型结构

输入为：

```text
z_sparse [B,256,64,64,64]
```

完整结构为：

```text
z_sparse [B,256,64,64,64]
       │
       ├── 局部分支：直接复用现有 LatentAdapter
       │     1×1×1 Conv：256→64
       │     GELU
       │     3×3×3 Conv：64→64
       │     GELU
       │     得到 local_feature [B,64,64,64,64]
       │
       └── 全局分支
             1×1×1 Conv：256→64
             GELU
             ↓
             AdaptiveAvgPool3d：8×8×8
             ↓
             [B,64,8,8,8]
             ↓ flatten + transpose
             512个token，每个token 64维
             ↓
             加入可学习3D位置序列编码
             ↓
             2个Transformer Encoder Block
             每个Block：4-head attention + 128维FFN
             ↓
             恢复为 [B,64,8,8,8]
             ↓
             三线性插值到 [B,64,64,64,64]
             ↓
             global_feature
                    │
                    ▼
       fused_feature = local_feature + global_feature
                    ↓
       复用现有LatentAdapter最后的1×1×1 Conv：64→256
                    ↓
                 residual
                    ↓
       z_out = z_sparse + residual
```

这里不是复制一套近似的CNN局部分支，而是代码层面直接实例化并复用
`submodel.adapter.LatentAdapter`：

```python
self.local_adapter = LatentAdapter(256, 64)
local_feature = self.local_adapter.encode(z_sparse)
residual = self.local_adapter.project(local_feature + global_feature)
```

因此两种Adapter的局部结构保持一致，方便进行严格消融实验。

默认配置下：

```text
hidden channels：64
pool size：8×8×8
token数量：512
Transformer层数：2
attention heads：4
FFN维度：128
参数量：259,904
```

最后的 `64→256` 卷积仍为零初始化，所以即使Transformer及位置编码为随机初始化，
整个模块初始仍严格满足：

```text
TransformerLatentAdapter(z) = z
```

### 9.2 为什么不在64³空间直接做全局注意力

`64×64×64`共有262,144个token，标准注意力矩阵规模与token数量平方成正比，
无法在常规3D训练显存下使用。本实现先池化到`8×8×8`，只对512个token做全局
注意力；局部细节由未池化的CNN分支保存。

### 9.3 推荐训练命令

建议从相同的旧主模型Encoder/Aggregator和原始prior decoder初始化，以便和CNN
Adapter公平比较：

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
  --use_adapter \
  --adapter_type transformer \
  --adapter_hidden_channels 64 \
  --adapter_transformer_pool_size 8 \
  --adapter_transformer_layers 2 \
  --adapter_transformer_heads 4 \
  --adapter_transformer_dropout 0.1 \
  --adapter_lr_factor 1.0 \
  --pretrained_backbone train/checkpoints/dental_prior_transfer_after_refine/ckpt_history/ckpt_199 \
  --pretrained_decoder submodel/decoder/checkpoints/dental_batch3_region_refine/ckpt_best_val.pt \
  --latent_lambda 0.1 \
  --latent_cosine_lambda 0.1 \
  --phase_a_epochs 20 \
  --phase_b_epochs 100 \
  --phase_c_epochs 60 \
  --decoder_lr_factor 0.1 \
  --query_chunk_size 25000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

路径中的checkpoint名称是示例，必须替换为实际存在的文件。

Phase A 保持 `--phase_a_encoder_lr_factor`/`--phase_a_aggregator_lr_factor` 为默认的
`0` 时：

```text
Encoder/Aggregator：冻结
Transformer Adapter：训练
Decoder：冻结并保持eval
```

其余阶段的训练逻辑与CNN Adapter一致；若从 0 训练，两个 Phase A 倍率都要设为正值。

### 9.4 Transformer Adapter评估命令

评估时必须使用与训练完全相同的结构参数：

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
  --adapter_transformer_dropout 0.1 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01
```

训练和评估时只要pool size、层数、head数或hidden channels任意一个不一致，
checkpoint都可能无法严格加载。原CNN Adapter checkpoint应继续使用：

```text
--use_adapter --adapter_type cnn
```

不能使用 `--resume` 把CNN Adapter checkpoint直接恢复成Transformer Adapter。
如果希望复用CNN实验结果，应使用 `--pretrained_backbone` 只加载其中的Encoder和
Aggregator，再单独加载原始prior decoder。

### 9.5 建议消融顺序

```text
A. 无Adapter
B. CNN Adapter（adapter_type=cnn）
C. CNN + Transformer Adapter（adapter_type=transformer）
```

三组实验应使用同一个pretrained backbone、同一个prior decoder、相同数据划分和
训练阶段。重点比较Train与Val/Test之间的差距。如果Transformer只提高Train PSNR，
却不提高Val/Test PSNR，说明全局分支主要在记忆训练集解剖结构；如果Val/Test的
PSNR、SSIM及骨骼/软组织损失同步改善，才能说明全局上下文有效。
