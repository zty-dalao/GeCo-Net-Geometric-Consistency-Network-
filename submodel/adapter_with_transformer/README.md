# CNN + Transformer Latent Adapter 使用说明

## 1. 功能与接入位置

本模块用于转换稀疏投影分支生成的三维latent，使其更适合由pCT预训练得到的
SRGAN Decoder：

```text
稀疏视角投影
    ↓
2D ResEncoder
    ↓
几何查询 + 多视角Aggregator
    ↓
z_sparse [B,256,X/4,Y/4,Z/4]
    ↓
TransformerLatentAdapter
    ↓
z_adapted [B,256,X/4,Y/4,Z/4]
    ↓
预训练SRGAN Decoder
    ↓
sCT
```

它不是一个单独训练的图像生成子模型，没有独立的 `train.py`。启用后作为主模型的
一部分，由根目录 `train.py` 联合训练；参数会自动保存在主模型checkpoint的
`G_render` 中。

代码文件：

```text
submodel/adapter_with_transformer/model.py
```

## 2. 模型结构

默认输入以256³重建、4倍Decoder上采样为例：

```text
z_sparse [B,256,64,64,64]
       │
       ├── 局部CNN分支
       │     直接复用 submodel/adapter/LatentAdapter
       │     Conv3d 1×1×1：256→64
       │     GELU
       │     Conv3d 3×3×3：64→64
       │     GELU
       │     local_feature [B,64,64,64,64]
       │
       └── 全局Transformer分支
             Conv3d 1×1×1：256→64
             GELU
             AdaptiveAvgPool3d：8×8×8
             ↓
             512个64维token
             ↓
             加入可学习位置编码
             ↓
             2个Transformer Encoder Block
             每层：4-head attention + 128维FFN
             ↓
             恢复为 [B,64,8,8,8]
             ↓
             三线性插值到 [B,64,64,64,64]
             ↓
             global_feature
                    │
                    ▼
       fused = local_feature + global_feature
                    ↓
       复用CNN Adapter输出投影
       Conv3d 1×1×1：64→256
                    ↓
                 residual
                    ↓
       z_adapted = z_sparse + residual
```

默认参数量约为259,904。输出投影卷积采用零初始化，因此刚接入网络时严格满足：

```text
z_adapted = z_sparse
```

Transformer的随机初始化不会在第一次前向传播时破坏已有latent。

全局分支不直接在64³空间做注意力，因为64³等于262,144个token，标准全局注意力
代价过高。池化到8³后只有512个token；未池化的CNN分支负责保留局部空间细节。

## 3. 推荐训练命令

推荐使用以下组合：

```text
旧主模型Encoder/Aggregator权重
+ 原始prior decoder权重
+ 新建且恒等初始化的Transformer Adapter
```

这样可以保留已经学到的投影几何特征，同时避免继续使用可能已经漂移的旧Decoder。

以下示例使用旧的浅层pCT prior decoder：

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

示例checkpoint路径需要替换成实际存在的路径。

这条命令的阶段行为为：

| 阶段 | epoch | Encoder/Aggregator | Transformer Adapter | Decoder | latent教师 |
|---|---:|---|---|---|---|
| Stage 1 | 0～19 | 冻结 | 正常训练 | 完全冻结 | 权重0.1 |
| Stage 2 | 20～119 | 正常训练 | 正常训练 | 0.1倍学习率 | 逐渐衰减 |
| Stage 3 | 120～199 | 0.01倍学习率 | 正常训练 | 只训练最后上采样块和输出层 | 关闭 |

## 4. 不加载旧主干、从零训练

如果不使用 `--pretrained_backbone`，Encoder和Aggregator是随机初始化的，此时不能在
Stage 1冻结它们，应将 `--stage1_backbone_lr_factor` 设置为1：

```bash
python train.py \
  --name dental_prior_adapter_transformer_from_scratch \
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

这里“从零”指主模型Encoder、Aggregator和Transformer Adapter从随机/恒等初始化开始；
Decoder仍加载pCT预训练权重。

## 5. 使用mean/detail深层prior encoder

如果prior checkpoint来自 `submodel/deep_encoder`，必须同时修改以下参数：

```bash
--pretrained_decoder submodel/deep_encoder/checkpoints/dental_mean_detail_prior/ckpt_best_val.pt \
--prior_encoder_type deep
```

其余Transformer Adapter参数不变。不能把deep prior checkpoint与
`--prior_encoder_type shallow`组合，否则冻结teacher的结构和checkpoint不匹配。

## 6. 继续训练命令

继续同一种Transformer Adapter模型时，使用原实验名称和历史epoch：

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
  --epochs 300 \
  --resume \
  --resume_name 199 \
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

根目录训练代码中的 `--epochs` 是最终总epoch。checkpoint 199的 `iter=200`，因此
`--epochs 300`会继续训练epoch 200～299，而不是额外训练300轮。

resume时所有Adapter结构参数必须与checkpoint一致。CNN Adapter checkpoint不能通过
`--resume`直接变成Transformer Adapter；如需复用，只能通过 `--pretrained_backbone`
提取其中的Encoder/Aggregator。

继续训练时应保持原实验的loss权重和数据参数不变，除非有意开始新的损失配置实验。

## 7. 评估命令

评估Transformer Adapter的第199轮历史checkpoint：

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

并默认保存到：

```text
evaluate/logs/dental_prior_adapter_transformer/
evaluate/visuals/dental_prior_adapter_transformer/
```

评估只需要构造主模型，不使用训练期pCT latent教师，因此不需要传
`--pretrained_decoder`、`--pretrained_backbone`、`--prior_encoder_type`、
`--latent_lambda`或训练阶段参数。

如果还需要报告骨骼、软组织和可微局部SSIM loss，可以额外添加：

```bash
--bone_lambda 0.05 \
--bone_lower_hu 300 \
--soft_mask_lambda 0.01 \
--soft_window_low -160 \
--soft_window_high 240 \
--ssim_lambda 0.01
```

这些评估参数只增加指标报告，不改变预测、PSNR、`ssim_3d_clamp`或输出NIfTI。

## 8. 参数解释

### 8.1 Transformer Adapter结构参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--use_adapter` | 关闭 | 启用Aggregator和Decoder之间的Adapter。Transformer checkpoint必须提供。 |
| `--adapter_type` | `cnn` | 必须设为`transformer`才会构造本模块。 |
| `--adapter_hidden_channels` | `64` | CNN局部分支和Transformer token的通道维度。 |
| `--adapter_transformer_pool_size` | `8` | 池化后每个空间轴尺寸；token数等于该值的三次方。 |
| `--adapter_transformer_layers` | `2` | Transformer Encoder Block数量。 |
| `--adapter_transformer_heads` | `4` | 注意力head数量，必须能整除hidden channels。 |
| `--adapter_transformer_dropout` | `0.1` | 注意力与FFN的dropout概率，评估时自动关闭。 |
| `--adapter_lr_factor` | `1.0` | Adapter学习率相对于配置文件基础学习率的倍率。 |

训练和评估时，hidden channels、pool size、layers和heads必须一致，否则checkpoint
可能因参数形状或键不匹配而无法加载。dropout不改变权重形状，但仍建议保持一致。

### 8.2 初始化与prior参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--pretrained_backbone` | 无 | 只加载旧主模型Encoder/Aggregator，不加载其Decoder或Adapter。 |
| `--pretrained_decoder` | 无 | 加载pCT预训练Decoder，并读取其中的feature_stem作为训练期teacher。 |
| `--prior_encoder_type` | `shallow` | `shallow`对应原两层stem；`deep`对应mean/detail LearnedPriorEncoder。 |
| `--latent_lambda` | `0` | 初始latent对齐总权重。大于0时必须提供pretrained decoder。 |
| `--latent_cosine_lambda` | `0.1` | latent对齐内部余弦项相对权重。 |

### 8.3 分阶段训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--stage1_epochs` | `15` | 完整Decoder冻结的初始训练轮数。 |
| `--stage1_backbone_lr_factor` | `1.0` | Stage 1主干学习率倍率；有可靠旧主干时可设0，仅训练Adapter。 |
| `--stage2_epochs` | `65` | Stage 1之后的联合训练轮数。 |
| `--decoder_lr_factor` | `0.1` | Stage 2及Stage 3可训练Decoder部分的学习率倍率。 |
| `--stage3_backbone_lr_factor` | `0.01` | Stage 3 Encoder/Aggregator学习率倍率；设0完全冻结。 |

只有提供 `--pretrained_decoder` 时才启用三阶段迁移训练；否则训练器使用普通联合训练。

### 8.4 数据、几何与显存参数

| 参数 | 说明 |
|---|---|
| `--datadir` | 数据集根目录。 |
| `--datatype` | 数据类型及对应μ截断范围。 |
| `--train_scale` | Decoder上采样倍数，应与prior decoder训练配置一致。 |
| `--fusion` | 多视角Aggregator类型，当前示例使用`ada`。 |
| `--nviews` | 输入稀疏投影视角数量。 |
| `--start`、`--end` | 扫描角范围。 |
| `--angle_sampling` | 视角采样方式，例如`uniform`。 |
| `--query_chunk_size` | 每次几何查询融合的3D点数；减小可降低峰值显存但会变慢。 |
| `--no_amp` | 关闭自动混合精度，会增加显存占用。 |

## 9. 建议的公平对照

至少比较：

```text
A. 无Adapter
B. CNN Adapter：--adapter_type cnn
C. Transformer Adapter：--adapter_type transformer
```

三组应使用相同的pretrained backbone、prior decoder、数据划分、epoch、loss和学习率。
如果Transformer只提高训练集PSNR而验证/测试不提高，说明全局分支主要在记忆训练
解剖；只有验证/测试PSNR、SSIM以及骨骼和软组织误差同步改善，才能说明全局上下文
确实缓解了latent空间不匹配。
