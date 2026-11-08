## QHWT: A WAVELET–QUATERNION NEURAL MODULE FOR UNIVERSAL VISUAL BACKBONES

Abstract
----------

In multiscale visual recognition (classification and semantic segmentation), real-world applications demand joint modeling of global semantics and fine, direction-aware details. However, contemporary CNN/Transformer/Mamba backbones still rely on strided sampling and real-valued feature fusion that blur edges, amplify aliasing, and overlook correlations across frequency subbands; prior wavelet-in-network’ designs typically process LL/LH/HL/HH independently rather than as a coupled entity. To this end, we propose QHWT, a plug-and-play wavelet–quaternion module that unifies multiresolution analysis with hypercomplex algebra. At its core are two mechanisms that directly address the gap: (i) a 2D discrete wavelet transform (DWT) replaces ad-hoc downsampling, providing anti-aliased, direction-selective subbands; (ii) a quaternion convolution (QConv) embeds the four subbands into a single quaternion (LL as the real part, LH/HL/HH as the three imaginary parts) and mixes them via the Hamilton product, learning cross-band couplings with phase consistency. The module is backbone-agnostic and can be inserted at downsampling or residual points without redesigning blocks. On ImageNet-1K classification, QHWT consistently improves Top-1 accuracy by 0.14%–1.50% points across diverse CNN/Transformer/Mamba backbones with only 4–8% parameter/FLOP overhead. On Cityscapes segmentation, integrating QHWT into standard frameworks yields 0.39%–1.97% mIoU, while preserving accuracy on auxiliary metrics. [QHWT repository](https://github.com/2510105628/Wavelet-to-Quaternion)

--------------------------------------------------

Our environments and toolkits
-----------

- OS: Ubuntu 20.04.6
- CUDA: 11.8
- Python: 3.9.21
- Toolkit: PyTorch 2.5.1
- GPU: RTX A6000 (1x)


Module
------

We aim to replace ad-hoc strided sampling and unconstrained feature mixing with a module that (i) prefilters before decimation and (ii) couples oriented subbands in a single algebra.


**QHWT (A pytorch-like implementation).** Detail of implementations, can be found in ``QHWT`` in this repository. 


```python
    class QHWT(nn.Module):
    def __init__(self, in_channels, kernel_size=3, wt_levels=2, scale='bn', wt_type='db1'):
        super(QHWT, self).__init__()

        self.wt_levels = wt_levels
        self.wt_filter, self.iwt_filter = wavelet.create_wavelet_filter(wt_type, in_channels // 4, in_channels // 4,
                                                                        torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.wavelet_convs = nn.ModuleList(
            [QuaternionConv(in_channels, in_channels, kernel_size, 1, (kernel_size - 1) // 2, bias=True)
             for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm2d(in_channels)) for _ in range(self.wt_levels)]
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.downSample = nn.Conv2d(in_channels+in_channels//4, out_channels, 3, 2, 1, bias=False)

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        x1 = self.reduce(x)
        curr_x_ll = x1

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)
            curr_x = wavelet.wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])
        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]
        assert len(x_ll_in_levels) == 0
        x = torch.cat([x, next_x_ll], dim=1)
        x = self.downSample(x)
        return x
```

Experiments
----------
### Ablation on ResNet-34 (ImageNet-100)
| Conv  | WT   | WT Level | Params (M) | Top-1 (%) | Δ |
|:-----:|:----:|:--------:|-----------:|----------:|:--:|
| –     | –    | –        | 21.34 | 80.72 | – |
| QConv | Haar | [3,3,3]  | 22.33 | **81.78** | **+1.06** |
| Conv  | Haar | [3,3,3]  | 24.07 | 81.50 | +0.78 |
| DW    | Haar | [3,3,3]  | 21.76 | 80.56 | -0.16 |
| QConv | Haar | [2,2,2]  | 22.14 | 80.28 | -0.44 |
| QConv | Haar | [4,4,4]  | 22.53 | 81.36 | +0.64 |
| QConv | Haar | [4,3,2]  | 22.19 | 81.22 | +0.50 |
| QConv | Db2  | [3,3,3]  | 22.34 | 80.80 | +0.08 |
| QConv | Db3  | [3,3,3]  | 22.36 | 81.28 | +0.56 |
| QConv | Sym2 | [3,3,3]  | 22.34 | 80.80 | +0.08 |
| QConv | Coif1| [3,3,3]  | 22.36 | 80.34 | -0.38 |
### ImageNet-100
| Type | Model | Baseline Params (M) | Baseline FLOPs (G) | Baseline Top-1 (%) |     QHWT Params (M) |     QHWT FLOPs (G) |     QHWT Top-1 (%) | Δ Top-1 (%) |
|:-----|:------|--------------------:|-------------------:|-------------------:|--------------------:|-------------------:|-------------------:|------------:|
| CNN | (2016 CVPR) ResNet-18 [11] | 11.23 | 1.82 | 80.10 |               12.22 |               1.99 |              80.98 | +0.88 |
| CNN | (2016 CVPR) ResNet-34 [11] | 21.34 | 3.67 | 80.72 |               22.33 |               3.84 |              81.78 | +1.06 |
| CNN | (2016 CVPR) ResNet-50 [11] | 23.72 | 4.11 | 81.04 |               26.92 |               4.71 |              81.92 | +0.88 |
| CNN | (2017 CVPR) DenseNet-121 [12] | 7.06 | 2.86 | 82.40 |                7.07 |               3.01 |              83.20 | +0.80 |
| CNN | (2017 CVPR) ResNeXt-50 [13] | 23.18 | 4.26 | 82.88 |               23.20 |               4.40 |              83.36 | +0.48 |
| CNN | (2018 ECCV) ShuffleNetV2 [14] | 2.07 | 0.21 | 80.58 |                2.07 |               0.24 |              81.32 | +0.74 |
| CNN | (2018 CVPR) MobileNetV2 [15] | 2.35 | 0.31 | 81.34 |                2.35 |               0.35 |              81.76 | +0.42 |
| CNN | (2024 CVPR) RepViT M0 6 [16] | 2.20 | 0.39 | 83.56 |                2.27 |               0.43 |              83.90 | +0.34 |
| CNN | (2024 CVPR) RepViT M0 9 [16] | 4.76 | 0.83 | 85.76 |                4.85 |               0.87 |              86.00 | +0.24 |
| CNN | (2024 ECCV) WTConvNeXt-T [17, 18] | 29.69 | 4.39 | 87.78 |               29.70 |               4.43 |              88.28 | +0.50 |
| Transformer | (2022 CVPR) SwinV2-T [19] | 27.66 | 4.52 | 85.98 |               29.40 |               4.85 |              87.10 | +1.12 |
| Transformer | (2022 CVPR) SwinV2-S [19] | 49.04 | 8.80 | 87.18 |               50.78 |               9.12 |              87.68 | +0.50 |
| Transformer | (2023 TPAMI) DualViT-S [20] | 24.20 | 5.38 | 87.66 |               25.50 |               5.58 |              88.24 | +0.58 |
| Transformer | (2022 ECCV) WaveViT-S [21] | 21.84 | 4.36 | 87.86 |               23.14 |               4.48 |              89.36 | +1.50 |
| Transformer | (2021 NeurIPS) Twins-S [22] | 23.60 | 2.81 | 85.34 |               24.38 |               3.09 |              86.22 | +0.88 |
| Transformer | (2022 CVPR) PoolFormer-S24 [23] | 20.93 | 3.41 | 87.52 |               22.28 |               3.61 |              87.84 | +0.32 |
| Mamba | (2025 AAAI) EfficientVMamba-B [24] | 33.77 | 4.56 | 88.20 |               35.51 |               4.62 |              88.90 | +0.70 |
| Mamba | (2024 arXiv) SiMBA-S (EinFFT) [25] | 14.50 | 1.45 | 87.04 |               15.80 |               1.51 |              87.32 | +0.28 |
| Mamba | (2024 arXiv) SiMBA-S (MLP) [25] | 25.78 | 4.18 | 87.48 |               27.09 |               4.30 |              87.62 | +0.14 |
| Mamba | (2025 CVPR) MambaVision T [26] | 31.22 | 4.46 | 88.88 |               32.77 |               4.55 |              89.24 | +0.36 |
> **Note:** BatchSize 256.
> 
### ImageNet-1000
| Model         | Params (M) | FLOPs (G) | Top-1 (%) | Top-5 (%) | Δ Top-1 (%) |
|:--------------|-----------:|----------:|----------:|----------:|------------:|
| ResNet-18     | 11.69 | 1.82 | 70.33 | 89.58 | – |
| + QHWT        | 12.68 | 1.89 | 71.53 | 90.18 | +1.20 |
| ResNet-34     | 21.80 | 3.67 | 73.75 | 91.60 | – |
| + QHWT        | 22.79 | 3.73 | 74.36 | 91.76 | +0.61 |
| ResNet-50     | 25.56 | 4.11 | 76.34 | 93.12 | – |
| + QHWT        | 28.75 | 4.71 | 76.87 | 93.31 | +0.53 |
| Swin-Tiny     | 28.29 | 4.35 | 80.58 | 95.26 | – |
| + QHWT        | 30.03 | 4.42 | 81.41 | 95.61 | +0.83 |
| Mamba-Vision  | 31.79 | 4.46 | 81.78 | 95.91 | – |
| + QHWT        | 33.34 | 4.55 | 82.17 | 95.94 | +0.39 |

> **Note:** BatchSize 256.

### Cityscapes val. (%) at 512×1024
| Model | aAcc | mIoU | mAcc | Δ mIoU |
|:------|-----:|-----:|-----:|-------:|
| Fast-SCNN [27] | 95.00 | 70.63 | 78.45 | – |
| (2019 BMCV) + QHWT | 95.01 | 71.04 | 79.00 | +0.41 |
| UPerNet-ResNet50 [28, 11] | 95.18 | 71.65 | 79.25 | – |
| (2018 ECCV) + QHWT | **95.53** | **73.62** | **81.19** | **+1.97** |
| Mask2Former-ResNet50 [29, 11] | 95.54 | 74.68 | 83.25 | – |
| (2022 CVPR) + QHWT | 95.63 | 75.07 | 83.86 | +0.39 |
| Mask2Former-SwinT [29, 30] | 95.71 | 73.51 | 82.24 | – |
| (2021 ICCV) + QHWT | 95.62 | 74.53 | 82.68 | +1.02 |
> **Note:** We use [mmdetection](https://github.com/open-mmlab/mmdetection) to train.

--------------------------------------------------------------------


