# 主模型结构说明

本文档对应当前项目中的主模型实现：

- `models/ResEncoder.py`
- `models/aggregator.py`
- `models/SRGAN.py`
- `models/model.py`
- `conf/train.conf`

整体流程为：

```text
20幅稀疏投影
    ↓
共享2D Encoder（每个视角独立提取）
    ↓
五个尺度的2D特征图
    ↓
根据投影几何查询每个3D体素对应的2D特征
    ↓
五尺度通道拼接：16+16+32+64+128=256
    ↓
Aggregator沿视角维度进行自适应融合
    ↓
[B,256,X/4,Y/4,Z/4]
    ↓
Adapter（可选）
    ↓
Continuous Prior Completion（可选）
    ↓
3D SRGAN Decoder
    ↓
[B,1,X,Y,Z]
```

## 1. Encoder

### 1.1 总体配置

当前配置位于 `conf/train.conf`：

```ini
dim_in = 1
inplanes = 16
feat_num_list = [16,32,64,128]
layer_num_list = [3,4,6,3]
block = BasicBlock
activation = GELU
normalization = BatchNorm2d
use_first_pool = False
```

它是一个修改后的轻量化 ResNet34：

- 残差块数量为 `3、4、6、3`；
- 通道数为 `16、32、64、128`，而不是标准ResNet34的 `64、128、256、512`；
- 首层为 `3×3、stride=1` 卷积；
- 不使用首层MaxPool；
- 激活函数使用GELU，而不是ReLU；
- 输入中的视角维度作为批次维度，每个视角共享同一个Encoder。

投影输入的形状通常是：

```text
[V,1,H,W]
```

其中 `V=20` 表示20个投影视角。

### 1.2 Encoder各级输出

| 阶段 | 结构 | 输出尺寸 |
|---|---|---|
| Stem/Conv1 | `Conv2d 1→16, 3×3, stride=1` → `BN` → `GELU` | `[V,16,H,W]` |
| Layer1 | 3个BasicBlock，通道16 | `[V,16,H,W]` |
| Layer2 | 4个BasicBlock，通道16→32，首块stride=2 | `[V,32,H/2,W/2]` |
| Layer3 | 6个BasicBlock，通道32→64，首块stride=2 | `[V,64,H/4,W/4]` |
| Layer4 | 3个BasicBlock，通道64→128，首块stride=2 | `[V,128,H/8,W/8]` |

例如输入为 `[20,1,256,256]` 时，Encoder保存的五个特征图为：

```text
F0：[20, 16, 256, 256]    Stem输出
F1：[20, 16, 256, 256]    Layer1输出
F2：[20, 32, 128, 128]    Layer2输出
F3：[20, 64,  64,  64]    Layer3输出
F4：[20,128,  32,  32]    Layer4输出
```

`ResNet`类中虽然构造了`AdaptiveAvgPool2d`和全连接层，但主模型的`ResEncoder.forward()`不会执行它们，因为本项目需要保留空间特征图，而不是进行图像分类。

### 1.3 BasicBlock

每个BasicBlock的主分支为：

```text
输入 x
  │
  ├─ Conv2d 3×3
  ├─ BatchNorm2d
  ├─ GELU
  ├─ Conv2d 3×3
  └─ BatchNorm2d
       │
       + identity
       │
      GELU
```

当通道数或空间尺寸发生变化时，skip分支使用：

```text
1×1 Conv2d，stride=2
    ↓
BatchNorm2d
```

例如Layer2的首个BasicBlock为：

```text
主分支：16→32，空间尺寸减半
Skip：  16→32，空间尺寸减半
```

## 2. 多尺度几何查询

Encoder不会直接把五个二维特征图resize后拼接。对于一个3D体素点，主模型执行如下过程：

```text
给定一个3D世界坐标点 P
    ↓
根据每个视角的12维投影几何参数
将P投影到对应探测器平面
    ↓
得到二维坐标(u,v)
    ↓
分别在F0、F1、F2、F3、F4上进行双线性采样
    ↓
将五个尺度的采样结果按通道拼接
```

每个视角、每个体素得到：

```text
F0采样：16维
F1采样：16维
F2采样：32维
F3采样：64维
F4采样：128维
----------------
合计：  256维
```

因此，N个3D查询点、V个视角对应的结果是：

```text
[V,256,N]
```

这里的采样位置由源点、探测器中心、探测器u方向和v方向共同决定，并通过`grid_sample`完成双线性插值。

## 3. Aggregator

当前命令通常使用：

```bash
--fusion ada
```

因此使用`adafusor`。

### 3.1 输入

对一个3D体素点，20个视角的多尺度特征为：

```text
latent：[V,256,N]
```

其中每个视角对该点有一个256维特征向量。

### 3.2 计算视角统计量

沿视角维度计算：

```python
mean = torch.mean(latent, dim=0)
var = torch.var(latent, dim=0)
```

得到：

```text
view_mean：[256,N]
view_var： [256,N]
```

然后将mean和var复制到每个视角，与该视角的原始特征拼接：

```text
当前视角特征 f_i：256维
所有视角均值 mean：256维
所有视角方差 var：256维
--------------------------------
拼接后：768维
```

### 3.3 视角特征变换和权重计算

```text
Linear：768→256
GELU
    ↓
Linear：256→1
GELU
    ↓
Softmax(dim=view)
```

因此每个3D体素会得到20个视角权重：

```text
w1(p), w2(p), ..., w20(p)
```

并满足：

```text
sum_v wv(p) = 1
```

### 3.4 加权融合

```python
weighted_feat = torch.sum(global_feat * weight, dim=0)
output_feat = self.output_fc(weighted_feat)
```

其中：

```text
output_fc：Linear 256→256 + GELU
```

最终输出为：

```text
[256,N]
```

然后恢复为三维体素空间：

```text
[B,256,X/4,Y/4,Z/4]
```

### 3.5 当前Aggregator的能力边界

当前Ada Aggregator可以学习：

```text
某个体素应该更相信哪个视角
```

但它不是逐通道的视角门控。它当前为每个视角产生一个标量权重，不能直接表达：

```text
第10通道更相信视角1
第80通道更相信视角5
第200通道更相信视角17
```

因此，当前结构是“逐体素、逐视角”的自适应融合，而不是“逐体素、逐视角、逐通道”的融合。

## 4. 3D Decoder

主模型Decoder是`models/SRGAN.py`中的3D SRGAN Generator，配置为：

```ini
scale = 4
inplanes = 256
res_blk_num = 6
channel_reduce_factor = 4
activation = GELU
normalization = BatchNorm3d
```

输入和输出通常为：

```text
输入：[B,256,X/4,Y/4,Z/4]
输出：[B,1,X,Y,Z]
```

### 4.1 Input Block

```text
Conv3d：256→256，3×3×3，stride=1，padding=1
BatchNorm3d(256)
GELU
```

空间尺寸和通道数不变。

### 4.2 六个Residual Block

每个残差块包含：

```text
Conv3d：256→256，3×3×3
BatchNorm3d
GELU
Conv3d：256→256，3×3×3
BatchNorm3d
与块输入相加
GELU
```

6个Residual Block合计：

```text
12个Conv3d
12个BatchNorm3d
6个块内残差连接
```

### 4.3 Residual Last与全局残差

6个Residual Block之后还有：

```text
Conv3d：256→256，3×3×3
BatchNorm3d
```

然后与Input Block的输出进行一次更大的残差相加：

```text
x = residual_body(x) + input_block_output
x = GELU(x)
```

所以Decoder中存在两级残差：

1. 每个Residual Block内部的局部残差；
2. 整个Residual主体相对于Input Block的全局残差。

### 4.4 第一个上采样块

```text
Conv3d：256→64，3×3×3
GELU
三线性插值上采样×2
Conv3d：64→64，3×3×3
GELU
```

尺寸变化：

```text
[B,256,D,H,W]
    ↓
[B,64,2D,2H,2W]
```

### 4.5 第二个上采样块

```text
Conv3d：64→16，3×3×3
GELU
三线性插值上采样×2
Conv3d：16→16，3×3×3
GELU
```

尺寸变化：

```text
[B,64,2D,2H,2W]
    ↓
[B,16,4D,4H,4W]
```

### 4.6 Output Block

```text
Conv3d：16→1，3×3×3
```

输出层没有BatchNorm、GELU、Sigmoid或Tanh：

```text
[B,16,4D,4H,4W]
    ↓
[B,1,4D,4H,4W]
```

## 5. 完整结构图

```text
输入投影：[V,1,H,W]
        │
        ▼
Stem：Conv 1→16 + BN + GELU
        │
        ├── F0：[V,16,H,W]
        ▼
Layer1：3×BasicBlock
        │
        ├── F1：[V,16,H,W]
        ▼
Layer2：4×BasicBlock，首块stride=2
        │
        ├── F2：[V,32,H/2,W/2]
        ▼
Layer3：6×BasicBlock，首块stride=2
        │
        ├── F3：[V,64,H/4,W/4]
        ▼
Layer4：3×BasicBlock，首块stride=2
        │
        └── F4：[V,128,H/8,W/8]

每个3D体素根据几何投影到每个视角：

F0采样16维
F1采样16维
F2采样32维
F3采样64维
F4采样128维
        │
        ▼
多尺度拼接：[V,256,N]
        │
        ▼
Ada Aggregator：
当前特征 + view mean + view variance
        │
        ▼
跨视角加权融合：[B,256,X/4,Y/4,Z/4]
        │
        ▼
Adapter（可选）
        │
        ▼
Continuous Prior Completion（可选）
        │
        ▼
3D Decoder
        │
        ▼
[B,1,X,Y,Z]
```

## 6. 尺寸示例

假设每幅投影为 `256×256`，目标体数据为 `256×256×256`，训练尺度为4：

```text
投影输入：
[20,1,256,256]

Encoder五尺度：
F0 [20, 16,256,256]
F1 [20, 16,256,256]
F2 [20, 32,128,128]
F3 [20, 64, 64, 64]
F4 [20,128, 32, 32]

几何查询后的多视角特征：
[20,256,64³]

Ada Aggregator输出：
[256,64³]

恢复为3D latent：
[1,256,64,64,64]

Decoder Input/Residual主体：
[1,256,64,64,64]

上采样块1：
[1,64,128,128,128]

上采样块2：
[1,16,256,256,256]

最终输出：
[1,1,256,256,256]
```

## 7. 与先验模块的关系

Adapter和Continuous Prior Completion都位于Aggregator输出与Decoder之间：

```text
Aggregator输出
    ↓
Adapter（可选）
    ↓
Continuous Prior Completion（可选）
    ↓
Decoder
```

其中：

- Adapter主要负责把稀疏投影形成的latent接口调整到Decoder能够理解的表示空间；
- Continuous Prior Completion根据Adapter后的latent和投影几何预测残差补偿；
- Decoder负责将256通道低分辨率三维latent恢复为单通道高分辨率体数据。

