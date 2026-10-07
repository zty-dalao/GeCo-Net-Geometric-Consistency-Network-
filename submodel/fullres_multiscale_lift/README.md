# Full-resolution Multi-scale Lift

该模块实现F0～F4全部在完整三维物理网格上反投影的实验结构。它与
`submodel/multiscale_lift`并列，只有显式传入
`--fullres_multiscale_decoder`时才会启用，不会改变原模型或原多尺度模型。

## 1. 数据流

编码器输出通道固定为：

```text
F0=16, F1=16, F2=32, F3=64, F4=128，总计256通道
```

对完整`xyz_full`中的每个三维点，模块分别从F0～F4采样二维特征，按原编码器
顺序拼成256通道，然后调用主模型原有的`Aggregator`完成多视角融合。融合后的
完整三维体积再按`16/16/32/64/128`切分为X0～X4。

```text
投影 → ResEncoder → F0...F4
                    ↓
       完整网格采样并拼接为256通道
                    ↓
       原Aggregator进行多视角融合
                    ↓
           256通道完整三维体积
                    ↓ split
       X0(16), X1(16), X2(32), X3(64), X4(128)
```

每个Xi先经过独立的拼接式残差块：

```text
branch = GELU(Conv3d(GELU(Conv3d(Xi))))
Ri = GELU(Conv3d(concat(Xi, branch)))
```

最后自底向上累计通道：

```text
H3  = concat(R4, R3)          # 128 + 64 = 192
H2  = concat(H3, R2)          # 192 + 32 = 224
H01 = concat(H2, R1, R0)      # 224 + 16 + 16 = 256

Hfinal = ConcatResidualBlock3D(H01)  # 256 → 512 → 256
output = Conv3d(Hfinal, 256 → 1, kernel_size=1)
```

`output`返回主模型后继续使用原配置的`last_layer_act`，当前配置为GELU。
`output`层是该子模块唯一的`1×1×1 Conv3d`。

## 2. 与原Aggregator的关系

该模块没有另外建立视角融合器。它复用主模型的：

```text
--fusion ada       → 原adafusor
--fusion local     → 原localfusor
--fusion meanmlp   → 原meanfusor
--fusion varmlp    → 原varfusor
--fusion mean/max  → 原无参数聚合方式
```

因为聚合器接收的仍是原来的256通道，所以已有Aggregator结构和预训练主干权重
可以继续使用。加载主干时可传`--pretrained_backbone`，但不能传原SRGAN的
`--pretrained_decoder`，因为本模块Decoder结构完全不同。

## 3. 训练命令

### 3.1 Dental：`syn_data_cbct_gt_v2`

```bash
python train.py \
  --name dental_fullres_multiscale_concatres \
  --datadir ./dataset/dental/syn_data_cbct_gt_v2 \
  --datatype dental \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 300 \
  --fullres_multiscale_decoder \
  --query_chunk_size 4000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

### 3.2 Thorax：`syn_data_cbct_gt_v2`

```bash
python train.py \
  --name thorax_fullres_multiscale_concatres \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 300 \
  --fullres_multiscale_decoder \
  --query_chunk_size 4000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

如果需要关闭反投影和3D残差块的activation checkpoint，可增加：

```bash
--disable_query_checkpoint
```

这会明显增加显存，一般不建议使用。`--query_chunk_size`只控制反投影/聚合阶段
一次处理的三维点数，不会降低完整三维卷积本身的显存。

## 4. 评估命令

下面示例评估第299轮checkpoint：

```bash
python evaluate.py \
  --name thorax_fullres_multiscale_concatres \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --dataname test \
  --train_scale 4 \
  --eval_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --fullres_multiscale_decoder \
  --query_chunk_size 4000 \
  --resume_name 299 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

## 5. 诊断指标

### 5.1 Train / Val / Test 重建质量

训练期间会对train、val、test三个数据集分别计算并写入TensorBoard：

```text
epoch/train_psnr_3d_clamp
epoch/train_ssim_3d_clamp
epoch/val_psnr_3d_clamp
epoch/val_ssim_3d_clamp
epoch/test_psnr_3d_clamp
epoch/test_ssim_3d_clamp
```

train指标每个训练epoch统计一次；val和test指标分别按照`--val-every`与
`--test-every`指定的周期统计。三组指标均使用clamp后的三维预测体，并沿用项目
当前的`psnr_3d_clamp` / `ssim_3d_clamp`计算口径。

三维SSIM的计算开销明显高于PSNR。加入train SSIM后，每个训练样本都会额外执行
该指标计算，因此单个epoch的耗时会有所增加。

### 5.2 Full-resolution结构诊断

训练时以下指标写入TensorBoard的`step/train_*`，评估时写入每病例和汇总日志：

```text
fullres_x0_abs_mean ... fullres_x4_abs_mean
fullres_r0_abs_mean ... fullres_r4_abs_mean
fullres_r0_change_l1 ... fullres_r4_change_l1
fullres_h3_abs_mean
fullres_h2_abs_mean
fullres_h01_abs_mean
fullres_final_abs_mean
fullres_final_change_l1
fullres_output_abs_mean
```

- `x*_abs_mean`：Aggregator输出切分后，各尺度输入特征的平均绝对值。
- `r*_abs_mean`：各尺度拼接式残差块输出的平均绝对值。
- `r*_change_l1`：残差块输出相对输入的平均绝对变化。
- `h*_abs_mean`：逐级拼接特征的平均绝对值。
- `final_change_l1`：最终256通道残差块改变H01的幅度。
- `output_abs_mean`：最后单通道、进入主模型GELU之前的平均绝对值。

为避免指标统计额外占用大量显存，诊断值在空间维每隔8个体素抽样计算；它们只
用于观察激活是否消失、爆炸或被某一层支配，不参与损失。

## 6. 显存警告

该结构严格保留完整分辨率和全部256通道。若网格确实为`256³`：

```text
256 × 256³ × float16 ≈ 8 GiB
256 × 256³ × float32 ≈ 16 GiB
```

这还不包含卷积中间激活、反向梯度、优化器状态和F0～F4二维特征。即使开启AMP、
query chunk和checkpoint，标准单卡仍极有可能OOM。建议先使用较小体积做结构与
梯度冒烟测试，再决定是否需要分块3D卷积、CPU offload或多卡张量并行。
