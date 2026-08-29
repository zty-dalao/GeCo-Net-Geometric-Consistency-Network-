import torch
import torch.nn as nn
from pyhocon import ConfigFactory
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.resnet import _log_api_usage_once, conv1x1, conv3x3
from typing import Type, Callable, Union, List, Optional
from models.render import get_pixel00_center

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.norm1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.norm2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.norm3 = norm_layer(planes * self.expansion)
        self.act = act
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm1 = norm_layer(planes)
        self.act = act
        self.conv2 = conv3x3(planes, planes)
        self.norm2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        layers: List[int],
        feats: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dim_in = 3,         # 输入的维度
        inplanes = 64       # 模型的内部维度
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(dim_in, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = norm_layer(self.inplanes)
        self.act = act
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, act, feats[0], layers[0])
        self.layer2 = self._make_layer(block, act, feats[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, act, feats[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, act, feats[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 网络的初始化
        for m in self.modules():                                # 递归地遍历当前网络（ResNet）里的所有层
            if isinstance(m, nn.Conv2d):                        # 对卷积层（Conv2d）应用 Kaiming 初始化
                # The function value between relu and GELU is very similar, here we directly apply "relu" initialization
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # 对归一化层（BN / GN）进行初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 将每个残差块“最后那个 BN 层”的权重强制设为 0。
        # Zero-initialize the last norm in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.norm3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.norm2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        act: Type[Union[nn.ReLU(inplace=True), nn.GELU()]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:        
            # 触发条件：当空间尺寸需要减半（stride != 1）或者输入通道数 self.inplanes 与当前阶段输出通道数 planes * expansion 不匹配时。
            # 作用：构建一个 1×1 卷积 + 归一化层，将“跳跃连接（identity）”的尺寸和通道数调整到与主干分支的输出一致，以便后续相加（残差连接）。
            # block.expansion：定义“瓶颈块（Bottleneck）”输出通道数与内部压缩通道数的比例。
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, act, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    act,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ResEncoder(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.num_layers = conf.num_layers           # 4
        self.use_first_pool = conf.use_first_pool   # flase
        self.latent_size = conf.latent_size         # 256
        self.layer_num_list = conf.layer_num_list   # [3, 4, 6, 3]
        self.feat_num_list = conf.feat_num_list     # [16,32,64,128]
        self.inplanes = conf.inplanes               # 16            表示当前阶段中，残差块内部“瓶颈”的通道数（即中间特征的通道数）。
        self.dim_in = conf.dim_in                   # 1
        self.activation = conf.activation           # GELU
        self.block = conf.block                     # BasicBlock
        self.normalization = conf.normalization     # Batch
        if self.activation == 'ReLU':
            act = nn.ReLU(inplace=True)
        elif self.activation == 'GELU':
            act = nn.GELU()
        if self.block == 'BasicBlock':
            block = BasicBlock
        elif self.block == 'Bottleneck':
            block = Bottleneck
        if self.normalization == 'Batch':
            norm = nn.BatchNorm2d
        elif self.normalization == 'Instance':
            norm = nn.InstanceNorm2d

        self.model = ResNet(block=block, act=act, layers=self.layer_num_list,
                            feats=self.feat_num_list, norm_layer=norm, inplanes=self.inplanes, dim_in=self.dim_in)

    def forward(self, x, poses):
        self.num_view = x.shape[0]
        self.image_shape = torch.tensor([x.shape[-1], x.shape[-2]]).to(x.device)  # [W, H]

        latent_list = []
        # first layer
        x = self.model.conv1(x)
        x = self.model.norm1(x)
        x = self.model.act(x)
        latent_list.append(x)
        if self.num_layers>=1:
            if self.use_first_pool:
                x = self.model.maxpool(x)
            x = self.model.layer1(x)
            latent_list.append(x)
        if self.num_layers>=2:
            x = self.model.layer2(x)
            latent_list.append(x)
        if self.num_layers>=3:
            x = self.model.layer3(x)
            latent_list.append(x)
        if self.num_layers>=4:
            x = self.model.layer4(x)
            latent_list.append(x)

        self.latent_list = latent_list
        self.poses = poses
        
    def queryfeature(self, xyz,):
        # key component of our method: feature back projection
        # xyz in world coords [N, 3]。N个3D 世界坐标点。
        N = xyz.shape[0]
        xyz = torch.repeat_interleave(xyz.unsqueeze(0), self.num_view, dim=0)  # [nviews, N, 3]，广播：为每个视角复制所有点
        
        vecs = torch.repeat_interleave(self.poses.unsqueeze(1), N, 1)  # [nviews, N, 12]，广播：为每个视角复制所有点
        sources, detectors, uvectors, vvectors = vecs[..., :3], vecs[..., 3:6], vecs[..., 6:9], vecs[..., 9:]  # [nviews, N, 3] each，获取点，探测器，探测平面上的2个垂直的分量

        ray_dir = xyz - sources  # [nviews, N, 3]，# 从源指向点 xyz 的射线方向
        normals = torch.cross(uvectors, vvectors, dim=2)  # [nviews, N, 3]，# u×v = 探测器平面法向量

        '''射线-平面求交：把 3D 点投影到探测器平面.
        下4行公式利用了论文中的公式(11)到(17)'''
        t_numer = torch.sum(normals * (detectors - sources), dim=2, keepdim=True)  # [nviews, N, 1]
        t_denom = torch.sum(normals * ray_dir, dim=2, keepdim=True)  
        t = t_numer / (t_denom + 1e-6)                                              # 解出交点参数 t

        uv_proj = sources + t * ray_dir  # [nviews, N, 3],# 交点世界坐标

        W = self.image_shape[0]
        H = self.image_shape[1]

        ''' 世界坐标 → 探测器像素坐标 (u, v)'''
        pixel00_center = get_pixel00_center(detectors, uvectors, vvectors, H, W)  # [nviews, N, 3]

        # a little different from the paper description, [-1, 1] range for F.grid_sample
        offset = uv_proj - pixel00_center                                                       # 落点相对 (0,0) 像素中心的偏移
        u_proj = torch.sum(offset * uvectors, dim=2) / torch.sum(uvectors * uvectors, dim=2) # [nviews, N]  data range [0, W-1]
        v_proj = torch.sum(offset * vvectors, dim=2) / torch.sum(vvectors * vvectors, dim=2) # [nviews, N]  data range [0, H-1]
        '''归一化到 grid_sample 的 [-1, 1] 坐标'''
        u_proj = 2 * (u_proj / W) - 1  # data range [0, W-1] --> [0, 1] --> [-1, 1]
        v_proj = 2 * (v_proj / H) - 1  # data range [0, H-1] --> [0, 1] --> [-1, 1]

        uv = torch.cat([u_proj.unsqueeze(-1), v_proj.unsqueeze(-1)], dim=-1).unsqueeze(2) # [nviews, N, 1, 2]

        '''多尺度特征采样'''
        feature_list = []
        for i in range(len(self.latent_list)):
            latent = self.latent_list[i]
            feature_list.append(F.grid_sample(latent, uv, align_corners=True, mode='bilinear', padding_mode='zeros',))  
        feature = torch.cat(feature_list, dim=1)# 有尺度按通道拼接 → C_total = 16(conv1)+16+32+64+128 = 256，正好等于配置里的 latent_size=256
        return feature[:,:,:,0]                 # 最终返回 [nviews, 256, N]：每个点 n 在视角 v 下，从多尺度特征图采到的 256 维特征向量。
