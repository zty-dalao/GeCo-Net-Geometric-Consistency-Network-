"""Implementation of Fig. 3 from:

    J. Zhang and L. Ren, "Enhance four-dimensional cone-beam computed
    tomography (4D-CBCT) from sparse view acquisitions using a novel deep
    learning model", Biomedical Signal Processing and Control 119 (2026)
    109935.

Fig. 3(a) - Interpolation model
    A U-Net backbone whose encoder convolutions are replaced by the proposed
    ``DRRConv`` and whose decoder upsampling (transposed) convolutions are
    replaced by the proposed ``DRRTransConv``.  Layer normalization is used
    throughout to learn feature representations that are consistent regardless
    of the input distribution.

    Given two adjacent projections (P_alpha1, P_alpha2) and the target
    interpolated angle ``beta``, the model adaptively adjusts its weights
    according to the angular distances (beta - alpha1, beta - alpha2) and
    predicts the interpolated projection P_beta.

Fig. 3(b) - DRRConv / DRRTransConv
    Dynamic Region- and Rotation-aware (transposed) convolution, obtained by
    modifying the Dynamic Region-Aware Convolution (DRConv) [J. Chen et al.,
    CVPR 2021]:

    * A fully-connected (FC) head consumes the angular distances between the
      adjacent projections and emits *regulators* that dynamically scale the
      module weights (the "rotation-aware" part).
    * Region-aware convolution applies a distinct filter to each spatial
      sub-region, the sub-regions being determined by a learnable guided mask
      (the "region-aware" part).
    * Unlike DRConv, the filters are randomly initialized and learned directly
      instead of being generated from the input content.
    * The base operations are the plain ``Conv`` (regular convolution) and
      ``TransConv`` (transposed convolution) "blue squares" of Fig. 3(b);
      ``DRRConv`` / ``DRRTransConv`` wrap those squares with the guided mask
      and the angular-distance regulators described above.

Note
----
The exact topology/channel counts of the original model are only given in
Algorithm 3 of Supplementary Material S3 (not part of the extracted text),
so the values below are reasonable, configurable defaults consistent with
the paper's description (U-Net backbone + layer norm, 512 x 384 projections,
two input projections and one output projection).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _transposed_output_size(
    h_in: int,
    w_in: int,
    stride: int,
    padding: int,
    output_padding: int,
    kernel_size: int,
    dilation: int,
) -> Tuple[int, int]:
    """Output spatial size of ``nn.functional.conv_transpose2d``."""
    h_out = (h_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    w_out = (w_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    return h_out, w_out


class Conv(nn.Module):
    """Plain 2-D convolution - the ``Conv`` blue square of the DRRConv branch.

    Randomly initialized and learned directly (in contrast to DRConv, whose
    filters are generated from the input content).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


class TransConv(nn.Module):
    """Plain transposed convolution - the ``TransConv`` blue square (Fig. 3(b)).

    Randomly initialized and learned directly; used as the core upsampling
    operation inside :class:`DRRTransConv`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )


class DRRConv(nn.Module):
    """Dynamic Region- and Rotation-aware Convolution (Fig. 3(b), top/left).

    Parameters
    ----------
    in_channels, out_channels:
        Standard convolution channel counts.
    kernel_size:
        Square convolution kernel size.
    num_regions:
        Number of spatial sub-regions, each of which receives its own filter.
    angular_dim:
        Dimensionality of the angular descriptor, typically 2:
        ``(beta - alpha1, beta - alpha2)``.
    regulator_hidden:
        Width of the hidden layer of the fully-connected regulator head.
    stride, padding, dilation:
        Passed through to :func:`torch.nn.functional.conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_regions: int = 4,
        angular_dim: int = 2,
        regulator_hidden: int = 32,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        mask_size: int = 7,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_regions = num_regions
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Region-aware convolution: one ``Conv`` blue square per sub-region.
        # The filters are randomly initialized and *learned directly* (in
        # contrast to DRConv, they are NOT generated from the input).
        self.regions = nn.ModuleList(
            [
                Conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
                for _ in range(num_regions)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Learnable guided mask that softly partitions the spatial plane into
        # ``num_regions`` sub-regions (kept at a small reference resolution and
        # interpolated to the feature-map size in the forward pass).
        self.guide_mask = nn.Parameter(torch.randn(1, num_regions, mask_size, mask_size))

        # Rotation-aware regulator: angular distances -> per-region scalars.
        self.regulator = nn.Sequential(
            nn.Linear(angular_dim, regulator_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(regulator_hidden, num_regions),
        )
        self.reg_activation = nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # ``Conv`` squares initialize their own weights; here we only initialize
        # the shared bias and the regulator head.
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)
        for layer in self.regulator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _region_mask(self, h: int, w: int) -> torch.Tensor:
        """Return the soft region-assignment mask of shape ``(1, R, h, w)``."""
        mask = F.interpolate(
            self.guide_mask, size=(h, w), mode="bilinear", align_corners=False
        )
        return F.softmax(mask, dim=1)

    def forward(self, x: torch.Tensor, angular: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input feature map of shape ``(B, in_channels, H, W)``.
        angular:
            Angular descriptor of shape ``(B, angular_dim)``.  If ``None`` the
            regulators are all ones (i.e. the module degenerates to a plain
            region-aware convolution).
        """
        b = x.shape[0]

        if angular is None:
            regulators = torch.ones(b, self.num_regions, device=x.device, dtype=x.dtype)
        else:
            # (B, R) in (0, 1); each scalar adapts one region's filter weights.
            regulators = self.reg_activation(self.regulator(angular))

        mask = self._region_mask(x.shape[-2], x.shape[-1])  # (1, R, H, W)

        out = torch.zeros(
            b, self.out_channels, x.shape[-2], x.shape[-1], device=x.device, dtype=x.dtype
        )
        for r, conv in enumerate(self.regions):
            conv_r = conv(x)
            # Scaling the mask by the regulator is equivalent to scaling the
            # region filter by the regulator (convolution is linear in weights),
            # and it keeps the batch dimension intact.
            region_mask = mask[:, r : r + 1] * regulators[:, r].view(b, 1, 1, 1)
            if region_mask.shape[-2:] != conv_r.shape[-2:]:
                region_mask = F.interpolate(
                    region_mask,
                    size=conv_r.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            out = out + conv_r * region_mask

        return out + self.bias.view(1, -1, 1, 1)


class DRRTransConv(nn.Module):
    """Dynamic Region- and Rotation-aware Transposed Convolution (Fig. 3(b)).

    Transposed-convolution counterpart of :class:`DRRConv`, used for the
    upsampling path of the U-Net decoder.  Its region mask is defined over the
    (upsampled) output spatial grid.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        num_regions: int = 4,
        angular_dim: int = 2,
        regulator_hidden: int = 32,
        stride: int = 2,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        mask_size: int = 7,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_regions = num_regions
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

        # Region-aware transposed convolution: one ``TransConv`` blue square
        # per sub-region, with randomly initialized, directly-learned filters.
        self.regions = nn.ModuleList(
            [
                TransConv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    dilation=dilation,
                    bias=False,
                )
                for _ in range(num_regions)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.guide_mask = nn.Parameter(torch.randn(1, num_regions, mask_size, mask_size))

        self.regulator = nn.Sequential(
            nn.Linear(angular_dim, regulator_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(regulator_hidden, num_regions),
        )
        self.reg_activation = nn.Sigmoid()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # ``TransConv`` squares initialize their own weights; here we only
        # initialize the shared bias and the regulator head.
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.bias, -bound, bound)
        for layer in self.regulator:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def _region_mask(self, h: int, w: int) -> torch.Tensor:
        mask = F.interpolate(
            self.guide_mask, size=(h, w), mode="bilinear", align_corners=False
        )
        return F.softmax(mask, dim=1)

    def _output_size(self, x: torch.Tensor) -> Tuple[int, int]:
        return _transposed_output_size(
            x.shape[-2],
            x.shape[-1],
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )

    def forward(self, x: torch.Tensor, angular: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = x.shape[0]

        if angular is None:
            regulators = torch.ones(b, self.num_regions, device=x.device, dtype=x.dtype)
        else:
            regulators = self.reg_activation(self.regulator(angular))

        h_out, w_out = self._output_size(x)
        mask = self._region_mask(h_out, w_out)  # (1, R, H_out, W_out)

        out = torch.zeros(
            b, self.out_channels, h_out, w_out, device=x.device, dtype=x.dtype
        )
        for r, conv in enumerate(self.regions):
            conv_r = conv(x)
            region_mask = mask[:, r : r + 1] * regulators[:, r].view(b, 1, 1, 1)
            out = out + conv_r * region_mask

        return out + self.bias.view(1, -1, 1, 1)


class _DRRDoubleConv(nn.Module):
    """Two ``DRRConv`` blocks separated by layer normalization + ReLU.

    Layer normalization over (C, H, W) is implemented with ``GroupNorm(1, C)``,
    which is exactly per-sample layer normalization for 2-D feature maps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_regions: int,
        angular_dim: int,
    ) -> None:
        super().__init__()
        self.conv1 = DRRConv(in_channels, out_channels, num_regions=num_regions,
                             angular_dim=angular_dim)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.conv2 = DRRConv(out_channels, out_channels, num_regions=num_regions,
                             angular_dim=angular_dim)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, angular: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x, angular)))
        x = self.act(self.norm2(self.conv2(x, angular)))
        return x


class InterpolationModel(nn.Module):
    """Interpolation model of Fig. 3(a).

    A U-Net that takes a stack of ``in_channels`` adjacent projections
    (concatenated along the channel axis) plus their angular distances
    ``(beta - alpha_i)`` and returns the interpolated projection ``P_beta``.

    Following Fig. 3(a), the input projections are z-score normalized on entry
    (``(x - zscore_mean) / zscore_std``) and the predicted projection is z-score
    denormalized on exit (``y * zscore_std + zscore_mean``).

    Parameters
    ----------
    in_channels:
        Number of stacked input projections (2 by default).  To use N
        projections, pass ``in_channels=N``; this is unrelated to batch size.
    out_channels:
        Number of output projection channels (1 by default).
    base_channels:
        Width of the first encoder level; doubled after every downsampling.
    num_down:
        Number of downsampling stages (bottleneck is at ``base * 2**num_down``).
    num_regions:
        Passed to every ``DRRConv`` / ``DRRTransConv``.
    angular_dim:
        Dimensionality of the angular descriptor.  Defaults to ``in_channels``
        (one angular distance ``beta - alpha_i`` per input projection).
    zscore_mean, zscore_std:
        Statistics of the input z-score normalization.  Defaults ``0.0`` /
        ``1.0`` (identity).  Provide the training-set mean/std so the input is
        standardized and the output is mapped back to the original range.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        num_down: int = 4,
        num_regions: int = 4,
        angular_dim: Optional[int] = None,
        zscore_mean: float = 0.0,
        zscore_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_down = num_down
        self.in_channels = in_channels
        if angular_dim is None:
            angular_dim = in_channels
        self.angular_dim = angular_dim

        # Fig. 3(a): input z-score normalization / output denormalization.
        self.register_buffer("zscore_mean", torch.tensor(float(zscore_mean)))
        self.register_buffer("zscore_std", torch.tensor(float(zscore_std)))

        # --- Encoder -----------------------------------------------------
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        ch = base_channels
        for _ in range(num_down):
            self.encoders.append(
                _DRRDoubleConv(in_ch, ch, num_regions=num_regions, angular_dim=angular_dim)
            )
            in_ch = ch
            ch *= 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck --------------------------------------------------
        self.bottleneck = _DRRDoubleConv(in_ch, in_ch, num_regions=num_regions,
                                         angular_dim=angular_dim)

        # --- Decoder -----------------------------------------------------
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch //= 2  # now the largest decoder channel count
        for _ in range(num_down):
            # Transposed conv halves the channel count while doubling the
            # spatial size; concatenating the skip connection doubles it back.
            self.upsamplers.append(
                DRRTransConv(in_ch, ch, num_regions=num_regions, angular_dim=angular_dim)
            )
            self.decoders.append(
                _DRRDoubleConv(ch * 2, ch, num_regions=num_regions, angular_dim=angular_dim)
            )
            in_ch = ch
            ch //= 2

        self.final = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(
        self,
        projections: Union[torch.Tensor, Sequence[torch.Tensor]],
        angular: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the interpolated projection.

        Parameters
        ----------
        projections:
            Either a single tensor of shape ``(B, in_channels, H, W)`` (already
            stacked along the channel axis) or a sequence of ``in_channels``
            tensors, each of shape ``(B, C, H, W)``.  The number of projections
            is set by ``in_channels``, not by the batch size.
        angular:
            ``(B, angular_dim)`` tensor holding ``(beta - alpha_i)``, one entry
            per input projection.
        """
        if torch.is_tensor(projections):
            x = projections
        else:
            x = torch.cat(list(projections), dim=1)

        # Fig. 3(a): z-score normalize the stacked input projections.
        x = (x - self.zscore_mean) / self.zscore_std

        # Encoder with skip connections.
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x, angular)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, angular)

        for i in range(self.num_down):
            x = self.upsamplers[i](x, angular)
            skip = skips[self.num_down - 1 - i]
            # Guard against odd-sized inputs where upsampling may differ by a
            # pixel from the pooled skip tensor.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x, angular)

        # Fig. 3(a): z-score denormalize back to the original projection range.
        return self.final(x) * self.zscore_std + self.zscore_mean


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test with the paper's projection size (512 x 384).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InterpolationModel().to(device)   # in_channels=2 by default
    print(f"InterpolationModel parameters: {_count_parameters(model):,}")

    b = 2
    p1 = torch.randn(b, 1, 512, 384, device=device)   # P_alpha1
    p2 = torch.randn(b, 1, 512, 384, device=device)   # P_alpha2
    angular = torch.rand(b, 2, device=device)          # (beta - alpha1, beta - alpha2)

    with torch.no_grad():
        p_beta = model([p1, p2], angular)

    print(f"P_beta output shape: {tuple(p_beta.shape)} (expected {(b, 1, 512, 384)})")

    # The number of input projections is configurable (NOT the batch size).
    model6 = InterpolationModel(in_channels=6).to(device)
    projs6 = [torch.randn(b, 1, 512, 384, device=device) for _ in range(6)]
    angular6 = torch.rand(b, 6, device=device)
    with torch.no_grad():
        out6 = model6(projs6, angular6)
    print(f"6-projection output shape: {tuple(out6.shape)} (expected {(b, 1, 512, 384)})")

    # Verify the plain 'Conv' / 'TransConv' blue squares from Fig. 3(b).
    conv = Conv(64, 128, 3, padding=1).to(device)
    print(f"Conv output shape: {tuple(conv(torch.randn(b, 64, 32, 32, device=device)).shape)}")

    tconv = TransConv(128, 64, kernel_size=2, stride=2).to(device)
    print(f"TransConv output shape: {tuple(tconv(torch.randn(b, 128, 16, 16, device=device)).shape)}")

    # Also verify the two building blocks in isolation.
    drr_conv = DRRConv(64, 128).to(device)
    y = drr_conv(torch.randn(b, 64, 32, 32, device=device), angular)
    print(f"DRRConv output shape: {tuple(y.shape)}")

    drr_tconv = DRRTransConv(128, 64).to(device)
    y = drr_tconv(torch.randn(b, 128, 16, 16, device=device), angular)
    print(f"DRRTransConv output shape: {tuple(y.shape)}")
