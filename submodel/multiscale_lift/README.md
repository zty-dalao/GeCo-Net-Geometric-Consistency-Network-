# 多尺度几何反投影解码器

本子模型实现 `idea.md` 中第一版多尺度方案：

```text
F4 → E4 → 上采样并融合E3
             ↓
          上采样并融合E2
             ↓
          上采样到完整分辨率
             ↓
            输出体数据
```

对应代码：

```text
submodel/multiscale_lift/model.py
```

主模型通过参数开启：

```bash
--multiscale_decoder
```

默认模型不受影响；不传该参数时，仍然使用原来的3D SRGAN Decoder。

## 1. 当前实现的结构

Encoder提取五个2D尺度：

```text
F0 [V, 16, H,   W]
F1 [V, 16, H,   W]
F2 [V, 32, H/2, W/2]
F3 [V, 64, H/4, W/4]
F4 [V,128, H/8, W/8]
```

当前第一版使用E2/E3/E4：

```text
F2 → GeometryLift + ViewFusion → E2 [B,32,D/2,H/2,W/2]
F3 → GeometryLift + ViewFusion → E3 [B,64,D/4,H/4,W/4]
F4 → GeometryLift + ViewFusion → E4 [B,128,D/8,H/8,W/8]
```

解码路径为：

```text
E4 [128通道]
  ↓ ConvUpBlock 128→64
  ↓ 与E3拼接或门控相加
  ↓ FusionBlock

  ↓ ConvUpBlock 64→32
  ↓ 与E2拼接或门控相加
  ↓ FusionBlock

  ↓ ConvUpBlock 32→16，直接恢复到完整体尺寸
  ↓ Conv3d 16→1
```

主项目的`train_scale/eval_scale`仍可保持4；它只影响原始Decoder的查询网格。多尺度分支会额外接收完整物理坐标网格，并在`1/2、1/4、1/8`三个尺度上独立查询F2/F3/F4，最后从E2一次上采样到完整体数据尺寸。

## 2. 参数

### 2.1 训练参数

```bash
--multiscale_decoder
```

启用多尺度E2/E3/E4解码器。

```bash
--multiscale_fusion concat
```

在上采样结果和对应E_i之间使用concat：

```text
concat(Up(A_i), E_i)
    ↓
Conv3d + GELU
    ↓
Conv3d + GELU
```

也可以使用：

```bash
--multiscale_fusion gated_add
```

此时先用`1×1×1 Conv3d`对齐通道，再执行：

```text
A_fused = A_up + gate × Project(E_i)
```

```bash
--multiscale_shallow none
```

默认不使用F0/F1，适合E1基线实验。

```bash
--multiscale_shallow 2d_fuse
```

启用F0/F1二维融合分支：

```text
concat(F0,F1)
    ↓
Conv2d 32→C
    ↓ GELU
Conv2d C→C
    ↓ GELU
GeometryLift到完整3D网格
    ↓
高分辨率残差修正
```

该分支显存开销很大，应在E2/E3/E4结构稳定后再使用。

```bash
--multiscale_shallow_channels 16
```

F0/F1二维融合后的通道数。

E4～E6的开关为：

```bash
--multiscale_highres_fusion concat|gated_add
--use_multiscale_supervision
--multiscale_aux_weights 0.2,0.1,0.05
--cross_scale_lambda 0.01
--use_hierarchical_view_weights
--view_weight_delta_lambda 0.0001
--use_uncertainty_gate
```

其中`multiscale_aux_weights`顺序固定为`E2,E3,E4`；`cross_scale_lambda`、
`view_weight_delta_lambda`默认是0，不开启对应正则。`use_uncertainty_gate`
需要同时使用`gated_add`，否则不确定性只会被计算和记录，不会参与融合。

## 3. 重要限制

当前多尺度解码器是独立消融模型，暂不和以下模块同时使用：

```bash
--use_adapter
--use_prior_completion
```

原因是当前Adapter和Continuous Prior Completion都假设输入latent为256通道，而多尺度Decoder的E2接口为32通道。后续如果要组合，需要专门增加E2→256的接口适配层，不能直接复用原Adapter。

多尺度Decoder内部为F2、F3、F4各自建立了独立的`ScaleViewFusion`。主模型中保留的旧`Aggregator`只是为了兼容模型构造和旧checkpoint，不参与多尺度分支的前向，也不会加入多尺度分支的优化器参数组。

当前多尺度模型也不使用原始SRGAN Decoder checkpoint，因为两者Decoder结构不同。可以加载主模型的Encoder/Aggregator权重：

```bash
--pretrained_backbone <main-checkpoint>
```

但不能使用：

```bash
--pretrained_decoder <old-srgan-decoder-checkpoint>
```

## 4. E1训练命令：只使用E2/E3/E4

建议先从头训练，用于验证多尺度3D观测注入是否有效：

```bash
python train.py \
  --name dental_multiscale_e1_e234_concat \
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
  --multiscale_decoder \
  --multiscale_fusion concat \
  --multiscale_shallow none \
  --query_chunk_size 8000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

`query_chunk_size`建议从8000或4000开始，因为该模型需要在多个3D尺度执行GeometryLift。显存足够时再逐步增大。

## 5. E1评估命令

```bash
python evaluate.py \
  --name dental_multiscale_e1_e234_concat \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --eval_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --multiscale_decoder \
  --multiscale_fusion concat \
  --multiscale_shallow none \
  --resume_name 199 \
  --query_chunk_size 8000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

### Thorax对应命令

训练时将数据参数替换为：

```bash
python train.py \
  --name thorax_multiscale_e1_e234_concat \
  --datadir ./dataset/thorax/processed_registered \
  --datatype thorax \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 200 \
  --multiscale_decoder \
  --multiscale_fusion concat \
  --multiscale_shallow none \
  --query_chunk_size 4000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

如需从已有主模型中只加载Encoder/Aggregator，可附加：

```bash
--pretrained_backbone train/checkpoints/<old_name>/ckpt_history/ckpt_<epoch>
```

这不会加载旧SRGAN Decoder，因为新Decoder结构不兼容。

如果你的 thorax 数据根目录实际为：

```text
./dataset/thorax/syn_data_cbct_gt_v2/
├── <样本1>/gt_volume.nii.gz
├── <样本1>/proj.nii.gz
├── <样本1>/transforms.json
└── ...
```

则 E1 命令应写成：

```bash
python train.py \
  --name thorax_multiscale_e1_e234_concat_cbct_gt_v2 \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --train_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --is_train \
  --epochs 200 \
  --multiscale_decoder \
  --multiscale_fusion concat \
  --multiscale_shallow none \
  --query_chunk_size 2000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

这里的`gt_volume.nii.gz`会被当前`CBCTDataset`自动读取为监督标签，不需要额外添加GT参数。`--datadir`不能写到某一个样本目录，必须写到`<样本>`的父目录。

开始完整训练前，建议先确认所有`thorax_split.json`中的样本都存在以下三个文件：

```text
<datadir>/<样本>/gt_volume.nii.gz
<datadir>/<样本>/proj.nii.gz
<datadir>/<样本>/transforms.json
```

Thorax应继续使用`--angle_sampling uniform`，这样会直接读取`transforms.json`中的`frames[].vec`，保留真实投影的半扇几何和角度约定。

对应评估命令（假设使用第199轮）：

```bash
python evaluate.py \
  --name thorax_multiscale_e1_e234_concat_cbct_gt_v2 \
  --datadir ./dataset/thorax/syn_data_cbct_gt_v2 \
  --datatype thorax \
  --train_scale 4 \
  --eval_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --multiscale_decoder \
  --multiscale_fusion concat \
  --multiscale_shallow none \
  --resume_name 199 \
  --query_chunk_size 2000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

## 6. E2：加入F0/F1二维融合

E1验证成功后，可以使用：

```bash
python train.py \
  --name dental_multiscale_e2_f01_fuse \
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
  --multiscale_decoder \
  --multiscale_fusion concat \
  --multiscale_shallow 2d_fuse \
  --multiscale_shallow_channels 16 \
  --query_chunk_size 2000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

这个版本会在完整3D网格上重新查询F0/F1，显存和运行时间明显增加。建议只在E1确认有效后进行。

E2评估沿用训练时的结构开关，例如：

```bash
python evaluate.py \
  --name dental_multiscale_e2_f01_fuse \
  --datadir ./dataset/dental/syn_data \
  --datatype dental \
  --train_scale 4 \
  --eval_scale 4 \
  --fusion ada \
  --start 0 \
  --end 360 \
  --nviews 20 \
  --angle_sampling uniform \
  --multiscale_decoder \
  --multiscale_fusion concat \
  --multiscale_shallow 2d_fuse \
  --multiscale_shallow_channels 16 \
  --resume_name 199 \
  --query_chunk_size 2000 \
  --bone_lambda 0.05 \
  --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 \
  --soft_window_low -160 \
  --soft_window_high 240 \
  --ssim_lambda 0.01 \
  --device cuda:0
```

评估时的结构参数必须与训练checkpoint一致；`--resume_name`按实际保存的epoch替换。

## 7. E3：F0/F1浅层通道消融

E3只改变F0/F1融合后的二维通道数，其他参数保持E2一致：

```bash
--multiscale_shallow_channels 4
--multiscale_shallow_channels 8
--multiscale_shallow_channels 16
--multiscale_shallow_channels 32
```

推荐顺序是`8 → 16 → 32`。例如C8训练命令只需把E2命令中的名称和通道改为：

```bash
--name dental_multiscale_e3_f01_c8 \
--multiscale_shallow 2d_fuse \
--multiscale_shallow_channels 8
```

如果C16与C32的val/test接近，优先选择C16；如果通道增加只提升train而不提升val/test，则说明浅层分支开始记忆纹理或伪影。

## 8. E4：低分辨率与高分辨率融合消融

低分辨率E2/E3/E4的融合由`--multiscale_fusion`控制；F0/F1高分辨率残差由`--multiscale_highres_fusion`控制：

```text
E4-A：--multiscale_fusion concat    --multiscale_highres_fusion concat
E4-B：--multiscale_fusion concat    --multiscale_highres_fusion gated_add
E4-C：--multiscale_fusion gated_add --multiscale_highres_fusion gated_add
```

E4-B是推荐的显存/表达能力折中：低分辨率保留concat信息，高分辨率使用门控残差。高分辨率`gated_add`带有零初始化的`shallow_alpha`，其初始输出不会改变E1/E2主体路径。

例如E4-A：

```bash
--multiscale_shallow 2d_fuse \
--multiscale_fusion concat \
--multiscale_highres_fusion concat
```

## 9. E5：多尺度辅助监督

E5通过E4、E3、E2中间表示各接一个轻量预测Head，使用固定平均池化得到对应尺度的GT，并加入：

```text
L = L_full
  + 0.20 L_E2
  + 0.10 L_E3
  + 0.05 L_E4
```

开启参数：

```bash
--use_multiscale_supervision \
--multiscale_aux_weights 0.2,0.1,0.05
```

如需增加跨尺度预测一致性：

```bash
--cross_scale_lambda 0.01
```

它约束`Down(y_full)≈y_E2`、`Down(y_E2)≈y_E3`和`Down(y_E3)≈y_E4`，不直接约束不同尺度的feature相等。建议先使用0.0，确认辅助L1有效后再尝试0.01～0.05。

E5训练示例：

```bash
python train.py \
  --name dental_multiscale_e5_supervision \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --is_train --epochs 200 \
  --multiscale_decoder \
  --multiscale_shallow none \
  --use_multiscale_supervision \
  --multiscale_aux_weights 0.2,0.1,0.05 \
  --cross_scale_lambda 0.01 \
  --query_chunk_size 6000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01 --device cuda:0
```

评估E5时必须保留`--use_multiscale_supervision`，使辅助Head结构与checkpoint一致；评估本身仍只使用最终输出计算PSNR/SSIM。

## 10. E6：层级视角权重与不确定性门控

E6有两个可以独立开启的部分。

### 10.1 层级视角权重

```bash
--use_hierarchical_view_weights \
--view_weight_delta_lambda 0.0001
```

E4首先得到粗尺度view logits，E3/E2将其三线性上采样后再学习局部修正：

```text
logits_E3 = upsample(logits_E4) + delta_E3
logits_E2 = upsample(logits_E3) + delta_E2
```

`view_weight_delta_lambda`限制修正量过大，建议从`1e-4`开始。

### 10.2 不确定性门控

```bash
--use_uncertainty_gate \
--multiscale_fusion gated_add \
--multiscale_highres_fusion gated_add
```

每个尺度根据跨视角特征计算：

```text
uncertainty = [归一化视角熵, 归一化特征方差]
```

该二维不确定性图作为额外输入提供给E3/E2以及F0/F1高分辨率gate。高不确定区域由gate抑制观测残差，避免单个异常视角强行覆盖主体解码结果。

E6训练示例：

```bash
python train.py \
  --name dental_multiscale_e6_hierarchical_uncertainty \
  --datadir ./dataset/dental/syn_data \
  --datatype dental --train_scale 4 --fusion ada \
  --start 0 --end 360 --nviews 20 --angle_sampling uniform \
  --is_train --epochs 200 \
  --multiscale_decoder \
  --multiscale_fusion gated_add \
  --multiscale_shallow 2d_fuse \
  --multiscale_shallow_channels 16 \
  --multiscale_highres_fusion gated_add \
  --use_hierarchical_view_weights \
  --view_weight_delta_lambda 0.0001 \
  --use_uncertainty_gate \
  --query_chunk_size 4000 \
  --bone_lambda 0.05 --bone_lower_hu 300 \
  --soft_mask_lambda 0.01 --soft_window_low -160 --soft_window_high 240 \
  --ssim_lambda 0.01 --device cuda:0
```

## 11. 资源和结果记录

每个消融实验至少记录：

```text
train/val/test PSNR
train/val/test SSIM
bone区域误差
soft-tissue区域误差
参数量
训练峰值显存
推理峰值显存
每epoch耗时
单病例推理耗时
```

尤其要注意：该模型参数量比原Decoder小很多，但E0/E1和多尺度GeometryLift可能造成更高的激活显存和运行时间。不能只根据参数量判断显存优势。

## 12. 推荐验证顺序

```text
E0 当前单latent基线
  ↓
E1 E2+E3+E4，多尺度Lift，concat
  ↓
E2 F0+F1二维融合后生成E01
  ↓
E3 比较E01通道数4/8/16/32
  ↓
E4 比较concat与gated_add
  ↓
E5 加入多尺度辅助监督
  ↓
E6 加入层级视角权重和uncertainty gate
```

如果E1没有超过E0，不建议继续增加F0/F1、gate或不确定性模块。此时应先检查：

- 物理坐标网格是否一致；
- E2/E3/E4的视角Softmax是否稳定；
- 上采样尺寸是否与volume_resolution严格一致；
- 是否因为多次GeometryLift导致有效梯度过小；
- query chunk是否造成了错误的reshape或视角维度处理。
