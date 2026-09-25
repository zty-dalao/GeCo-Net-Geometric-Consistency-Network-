# 多尺度2D特征反投影与3D渐进解码实验方案

本文档整理一套围绕 `F0～F4` 多尺度2D特征、对应3D空间表示 `E0～E4`、跨尺度融合和多尺度监督的验证路线。

目标不是一次性实现最复杂的网络，而是通过递进消融回答以下问题：

1. 在多个3D尺度重复注入投影观测，是否优于当前“只生成一次低分辨率latent”的结构？
2. F0/F1浅层高分辨率特征是否能稳定提高PSNR、SSIM和边缘质量？
3. 浅层通道压缩是否真的损失有效信息？
4. `concat`的性能收益是否值得其显存成本？
5. 多尺度监督是否能让中间3D空间具有更稳定的物理意义？
6. 不同尺度的视角权重是否需要层级关联、置信度和不确定性控制？

---

## 1. 当前基线模型

当前Encoder保存五个尺度的2D特征图：

```text
F0 [V, 16, 256, 256]
F1 [V, 16, 256, 256]
F2 [V, 32, 128, 128]
F3 [V, 64,  64,  64]
F4 [V,128,  32,  32]
```

其中 `V=20`。当前主模型对同一个低分辨率3D查询网格分别采样F0～F4，并按通道拼接：

```text
16 + 16 + 32 + 64 + 128 = 256
```

随后只执行一次跨视角Aggregator：

```text
F0～F4
  ↓ 在同一个低分辨率3D网格查询
[V,256,N]
  ↓ Ada Aggregator
[B,256,X/4,Y/4,Z/4]
  ↓ SRGAN Decoder
[B,1,X,Y,Z]
```

基线实验名称建议：

```text
mslift_e0_current_baseline
```

基线必须记录：

- train/val/test PSNR；
- train/val/test SSIM；
- 3D L1、梯度loss、bone/soft-tissue loss；
- 参数量；
- AMP训练峰值显存；
- 推理峰值显存；
- 单体数据推理时间；
- 骨骼边缘和软组织差异图。

---

## 2. 新结构的基本定义

### 2.1 2D尺度与3D尺度不是同一个概念

F0为 `256×256` 不代表E0必须是 `256³`。二维探测器尺寸和三维体数据尺寸属于不同坐标空间。

对于规范化的 `256³` 示例，可以使用：

```text
E0/E1：256³
E2：   128³
E3：    64³
E4：    32³
```

对于实际thorax体数据 `[248,248,120]`，建议使用体数据自身的金字塔：

```text
E0/E1：[248,248,120]
E2：   [124,124, 60]
E3：   [ 62, 62, 30]
E4：   [ 31, 31, 15]
```

所有尺度必须由相同的：

- `volume_origin`；
- `volume_phy`；
- 物理世界坐标范围；
- 体素中心定义；

生成查询网格，不能只根据张量shape进行近似缩放。

### 2.2 公共模块：GeometryLift

每一个 `F_i` 通过对应的3D世界坐标网格进行几何查询：

```text
F_i [V,C_i,H_i,W_i]
       │
       ├─ 3D点投影到各视角探测器
       ├─ grid_sample双线性采样
       └─ 得到 [V,C_i,N_i]
```

随后执行跨视角融合：

```text
[V,C_i,N_i]
      ↓ ViewFusion_i
[C_i,N_i]
      ↓ reshape
E_i [B,C_i,D_i,H_i,W_i]
```

第一版可以复用Ada Aggregator的逻辑：

```text
当前视角特征 + view mean + view variance
              ↓
          Linear 3C→C
              ↓
          Linear C→1
              ↓
       Softmax over views
              ↓
          视角加权求和
```

需要为每个尺度独立设置Aggregator，因为各尺度通道数不同。

### 2.3 公共模块：ChannelProject

当两路特征需要相加但通道数或特征基底不一致时，必须先做投影：

```python
E_projected = Conv3d(C_e, C_a, kernel_size=1)(E_i)
```

即：

```text
E_i [B,C_e,D,H,W]
       ↓ 1×1×1 Conv3d
P(E_i) [B,C_a,D,H,W]
```

`1×1×1 Conv3d`不仅改变通道数，也负责学习E_i到Decoder当前特征基底的线性映射。

### 2.4 公共模块：UpFusionBlock

建议保留两种实现以便消融。

#### Concat版本

```text
A_coarse
   ↓ 上采样×2
A_up
   ↓ concat E_i
[A_up,E_i]
   ↓ Conv3d 3×3×3
   ↓ GELU
   ↓ Conv3d 3×3×3
   ↓ GELU
A_fused
```

#### Gated Addition版本

```text
A_coarse
   ↓ 上采样×2
A_up ---------------------------┐
                                │
E_i → 1×1×1 Conv3d → P(E_i)     │
                 │              │
                 └→ Gate ───────┤
                                ↓
A_fused = A_up + gate × P(E_i)
             ↓
       Refinement Block
```

其中：

```python
gate = sigmoid(gate_net(torch.cat([A_up, P(E_i)], dim=1)))
```

为了保证新模块初始时不会破坏原路径，可以：

- 将投影层最后一层零初始化；或
- 增加零初始化可学习标量 `alpha_i`；或
- 将gate输出bias初始化为较小值。

建议默认使用：

```text
A_fused = A_up + alpha_i × gate × P(E_i)
alpha_i初始值 = 0
```

---

## 3. 实验E1：只使用E2、E3、E4

### 3.1 目的

验证“多尺度3D观测注入”本身是否有效，暂时排除F0/F1带来的高分辨率显存和浅层噪声影响。

### 3.2 结构

```text
F4 → GeometryLift/Softmax Fusion
       ↓
E4 [B,128,32³]
       ↓ UpBlock 128→64，×2
A4 [B,64,64³]
       ↓ concat E3 [B,64,64³]
[B,128,64³]
       ↓ FusionBlock 128→64
       ↓ 上采样×2
A3 [B,64,128³]
       ↓ concat E2 [B,32,128³]
[B,96,128³]
       ↓ FusionBlock 96→32或48
       ↓ 上采样×2
A2_full [B,16或32,256³]
       ↓ Conv3d 3×3×3
输出 [B,1,256³]
```

对于thorax，所有3D尺寸替换成对应的 `[31,31,15] → [62,62,30] → [124,124,60] → [248,248,120]`。

### 3.3 第一版融合方式

E1先采用concat，因为E2～E4处于较低分辨率，concat的显存代价仍然可控。

### 3.4 需要回答的问题

- 是否优于当前只生成单个低分辨率latent的模型？
- E3/E2重新注入投影信息后，骨骼边缘是否更稳定？
- 是否减少Decoder解冻后的指标突变？
- 是否降低对大规模256通道Residual Block的依赖？

实验名称建议：

```text
mslift_e1_e234_concat
```

---

## 4. 实验E2：加入F0/F1浅层高频修正

E2建立在E1有效的前提下。F0/F1不直接承担主体重建，而是作为高分辨率残差修正分支。

### 4.1 推荐方法：先在2D域融合，再反投影一次

```text
F0 [V,16,H,W] ─┐
                 ├─ concat → [V,32,H,W]
F1 [V,16,H,W] ─┘
                       ↓
              1×1 Conv2d：32→16
                       ↓ GELU
              3×3 Conv2d：16→16
                       ↓ GELU
                 F01 [V,16,H,W]
                       ↓
           GeometryLift + ViewFusion
                       ↓
             E01 [B,16,X,Y,Z]
```

主体路径产生：

```text
A2_full [B,C,X,Y,Z]
```

浅层修正可以采用：

```text
A2_full ------------------------┐
                               │
E01 → 1×1×1 Conv3d：16→C       │
               ↓               │
             Gate ─────────────┤
                               ↓
A_refined = A2_full + alpha × gate × P(E01)
                               ↓
                  3×3×3 Refinement
                               ↓
                       Conv3d C→1
```

建议：

- `alpha`零初始化；
- 最终输出卷积不加GELU；
- E01默认16通道；
- F0/F1只执行一次高分辨率GeometryLift。

实验名称建议：

```text
mslift_e2_e234_f01c16_gate
```

### 4.2 备选方法一：F0/F1分别反投影后进行尺度门控

```text
F0 → Lift → E0 ─┐
                 ├─ Scale Gate → E_shallow
F1 → Lift → E1 ─┘
```

可令：

```python
[g0, g1] = softmax(scale_gate(...), dim=scale)
E_shallow = g0 * P0(E0) + g1 * P1(E1)
```

优点：

- F0/F1保持独立；
- 能分析不同区域更依赖哪一层。

缺点：

- 需要两次完整分辨率GeometryLift；
- 同时保存E0和E1；
- 显存和运行时间最高。

建议只在方法二证明F0/F1有效后进行。

实验名称建议：

```text
mslift_e2b_e0e1_separate_scale_gate
```

### 4.3 备选方法三：分块高频修正

不完整生成E0/E1，而是对输出体进行滑窗处理：

```text
A2_full切分为3D patch（带halo）
        ↓
仅查询当前patch对应的F0/F1
        ↓
生成局部E01
        ↓
预测局部高频残差
        ↓
重叠区域加权融合
```

优点：显著降低推理和训练峰值显存。

缺点：实现复杂，需要处理patch边界和halo，建议作为后期显存优化实验，而不是第一版。

实验名称建议：

```text
mslift_e2c_f01_tiled_refine
```

---

## 5. 实验E3：浅层融合通道数消融

固定E2的结构，只改变F0/F1融合后的通道数：

```text
concat(F0,F1)：[V,32,H,W]
       ↓
Conv2d：32→C01
       ↓
F01：[V,C01,H,W]
```

测试：

| 实验 | C01 | 目标 |
|---|---:|---|
| E3-C4 | 4 | 极限压缩，测量信息下限 |
| E3-C8 | 8 | 低显存版本 |
| E3-C16 | 16 | 推荐平衡点 |
| E3-C32 | 32 | 不压缩通道总量，作为性能上限 |

实验名称建议：

```text
mslift_e3_f01_c4
mslift_e3_f01_c8
mslift_e3_f01_c16
mslift_e3_f01_c32
```

除了指标，还必须记录：

- E01的均值、标准差；
- E01的跨视角方差；
- gate均值和分布；
- alpha最终值；
- 峰值显存和推理时间。

判断原则：

- 如果C16与C32指标接近，优先C16；
- 如果C32明显更好，说明浅层特征确实不能强压缩；
- 如果C8已经达到C32水平，说明浅层通道冗余较大；
- 如果通道增加只提高train、不提高val/test，说明高分辨率分支可能在记忆纹理或伪影。

---

## 6. 实验E4：Concat与Gated Addition消融

### 6.1 Concat版本

```text
A_up [B,Ca,D,H,W]
E_i  [B,Ce,D,H,W]
        ↓ concat
[B,Ca+Ce,D,H,W]
        ↓ 3×3×3 Conv
A_fused
```

优势：

- 两路信息完整保留；
- 后续卷积可以自由学习组合关系；
- 表达能力强。

风险：

- 高分辨率下显存增长明显；
- 可能学习到对浅层噪声的依赖。

### 6.2 Gated Addition版本

```text
E_i → 1×1×1 Conv3d：Ce→Ca → P(E_i)

gate = sigmoid(GateNet(A_up, P(E_i), uncertainty_i))

A_fused = A_up + alpha_i × gate × P(E_i)
```

优势：

- 不生成`Ca+Ce`的大通道拼接张量；
- 具有清晰的观测残差解释；
- `alpha_i=0`时严格保留原始路径；
- 可以抑制低置信度的浅层特征。

风险：

- `1×1×1`投影可能造成信息压缩；
- 表达能力可能低于concat；
- gate需要避免过早饱和到0或1。

### 6.3 推荐比较方式

低分辨率E2/E3/E4和高分辨率E01应分别比较：

| 实验 | E2/E3/E4 | E01 |
|---|---|---|
| E4-A | concat | concat |
| E4-B | concat | gated addition |
| E4-C | gated addition | gated addition |

最值得优先验证的是E4-B：

```text
低分辨率使用concat保留信息；
完整分辨率使用gated addition控制显存。
```

实验名称建议：

```text
mslift_e4_all_concat
mslift_e4_low_concat_high_gate
mslift_e4_all_gate
```

---

## 7. 实验E5：多尺度辅助监督

拼接和逐级解码本身不等于多尺度监督。真正的多尺度监督需要从中间层输出体数据预测。

### 7.1 辅助输出头

```text
E4/A4 → Head32  → y32
A3    → Head64  → y64
A2    → Head128 → y128
最终层 → HeadFull → yFull
```

每个Head可以保持简单：

```text
Conv3d 3×3×3：C→C/2
GELU
Conv3d 1×1×1：C/2→1
```

辅助Head只用于训练，推理时可以删除，不增加最终推理显存和耗时。

### 7.2 GT金字塔

使用固定平均池化生成目标：

```text
GT_full
  ↓ AvgPool3d
GT_128
  ↓ AvgPool3d
GT_64
  ↓ AvgPool3d
GT_32
```

不要使用最近邻生成辅助GT，否则会放大阶梯边缘。

### 7.3 损失

第一组推荐权重：

```text
L = L_full
  + 0.20 × L_128
  + 0.10 × L_64
  + 0.05 × L_32
```

每个尺度可以包含：

```text
L_scale = L1 + λg × gradient_loss + λs × SSIM_loss
```

建议先只使用多尺度L1，确认训练稳定后再给辅助尺度增加很小的gradient/SSIM权重。

辅助监督权重需要继续消融：

| 设置 | L128 | L64 | L32 |
|---|---:|---:|---:|
| 弱监督 | 0.10 | 0.05 | 0.025 |
| 中等监督 | 0.20 | 0.10 | 0.05 |
| 强监督 | 0.40 | 0.20 | 0.10 |

强监督可能让最终输出过度平滑，因此不建议作为默认值。

### 7.4 预测空间跨尺度一致性

不直接约束不同尺度feature相等，而约束体数据预测在物理空间一致：

```text
L_cross =
    |Down(y_full) - y_128|₁
  + |Down(y_128) - y_64|₁
  + |Down(y_64) - y_32|₁
```

推荐使用很小的权重，例如：

```text
cross_scale_lambda = 0.01～0.05
```

实验名称建议：

```text
mslift_e5_multiscale_l1
mslift_e5_multiscale_l1_ssim_grad
mslift_e5_multiscale_cross_consistency
```

---

## 8. 实验E6：跨尺度层级权重与不确定性

不同尺度提取的特征不同，因此不应强制所有尺度使用相同视角权重。

### 8.1 层级视角logit

令粗尺度权重logit为基础，细尺度学习修正：

```text
logits_i = Up(logits_{i+1}) + delta_logits_i
weights_i = Softmax(logits_i, dim=view)
```

这样：

- 粗尺度提供稳定的全局视角判断；
- 细尺度可以根据局部边缘和纹理改变权重；
- 不要求 `weights_i == weights_{i+1}`。

可以对修正量增加弱正则：

```text
L_weight_delta = Σ λ_i × |delta_logits_i|₁
```

建议 `λ_i` 很小，例如 `1e-4～1e-3`，避免将不同尺度权重强行拉成一致。

### 8.2 不确定性来源

每个尺度可以计算：

1. 跨视角特征方差；
2. Softmax权重熵；
3. 最大权重与第二大权重的差值；
4. 不同视角预测的一致程度。

例如：

```text
variance_i = Var_view(features_i)
entropy_i = -Σ weights_i × log(weights_i)
```

然后将不确定性用于控制E_i的注入强度：

```text
gate_i = GateNet(A_up, P(E_i), variance_i, entropy_i)

A_fused = A_up + alpha_i × gate_i × P(E_i)
```

高置信度区域允许更强的观测修正；低置信度区域更多保留粗尺度结构和Decoder先验。

### 8.3 必须监控的指标

- 每个尺度的Softmax熵；
- 每个尺度的视角权重最大值；
- `delta_logits_i`绝对值；
- gate均值、标准差和直方图；
- 骨骼、软组织、空气区域的gate分布；
- gate与最终误差热力图的相关性。

实验名称建议：

```text
mslift_e6_hierarchical_view_weights
mslift_e6_uncertainty_gate
mslift_e6_hierarchical_uncertainty_gate
```

---

## 9. 推荐消融顺序

### 第一阶段：证明多尺度GeometryLift有效

```text
E0：当前单latent基线
E1：E2+E3+E4，低分辨率concat，无F0/F1
```

只有E1在val/test上稳定优于E0，才继续增加F0/F1。

### 第二阶段：验证浅层特征

```text
E2：F0+F1 → 2D融合 → 16通道E01 → 高分辨率gated residual
```

主要判断：

- PSNR/SSIM是否提升；
- 边缘是否更清晰；
- 是否出现条纹、噪声或伪细节；
- train与val/test差距是否扩大。

### 第三阶段：通道消融

```text
E3：C01 = 4 / 8 / 16 / 32
```

确定F0/F1在性能和显存之间的最佳通道数。

### 第四阶段：融合形式消融

```text
E4-A：全concat
E4-B：低分辨率concat + 高分辨率gated addition
E4-C：全gated addition
```

优先期望E4-B达到最好平衡。

### 第五阶段：多尺度监督

```text
E5：加入辅助Head和多尺度GT监督
```

先加入多尺度L1，再增加小权重SSIM、gradient和cross-scale consistency。

### 第六阶段：层级权重和不确定性

```text
E6：hierarchical view logits
    + uncertainty-aware gate
```

这是结构最复杂的一步，应建立在E1～E5已经证明有效之后。

---

## 10. 公平比较要求

所有消融实验尽量保持以下条件一致：

- 同一数据划分；
- 同一患者和投影视角；
- 同一随机种子；
- 同一Encoder初始化；
- 同一投影归一化方式；
- 同一基础重建loss；
- 同一训练epoch；
- 同一学习率和衰减策略；
- 同一AMP设置；
- 同一评估clamp范围；
- 同一PSNR/SSIM实现。

需要同时报告：

| 类型 | 指标 |
|---|---|
| 重建质量 | train/val/test PSNR、SSIM |
| 区域质量 | bone、soft-tissue、air区域误差 |
| 结构质量 | gradient loss、边缘误差图 |
| 资源 | 参数量、FLOPs、训练峰值显存、推理峰值显存 |
| 效率 | 每epoch时间、单病例推理时间 |
| 稳定性 | 最佳epoch、最终epoch、标准差、阶段切换波动 |

不能只比较最终train PSNR，也不能只比较某个偶然的最高点。应同时报告：

```text
best validation checkpoint
对应test指标
最终checkpoint指标
多病例均值与标准差
```

---

## 11. 建议的配置开关

后续实现时可以设计以下参数，避免为每个消融复制一套代码：

```text
--decoder_type multiscale_lift
--lift_scales 2,3,4
--lift_fusion concat|gated_add
--shallow_fusion none|2d_fuse|separate_gate|tiled
--shallow_channels 4|8|16|32
--highres_fusion concat|gated_add
--use_multiscale_supervision
--aux_loss_weights 0.2,0.1,0.05
--cross_scale_lambda 0.0
--use_hierarchical_view_weights
--view_weight_delta_lambda 0.0
--use_uncertainty_gate
```

结构参与和训练参与应分开控制，避免出现“模块构造了但推理不经过”或者“模块参与推理但没有进入优化器”的情况。

---

## 12. 当前最推荐的第一版

第一版不要同时实现全部功能，建议只做：

```text
F4 → E4
      ↓ Up + concat E3
F3 → E3
      ↓ Up + concat E2
F2 → E2
      ↓ Up到完整分辨率
      ↓ 16或32通道Refinement
      ↓ 1通道输出
```

即实验E1：

```text
E2 + E3 + E4
低分辨率concat
无F0/F1
无gate
无多尺度监督
无层级权重
```

这一步只回答一个问题：

> 多尺度二维特征分别提升到对应三维尺度，并在解码过程中重复注入观测，是否优于当前单一latent接口？

确认E1有效后，再按照 `E2 → E3 → E4 → E5 → E6` 的顺序逐步增加浅层修正、通道消融、门控、多尺度监督和不确定性机制。
