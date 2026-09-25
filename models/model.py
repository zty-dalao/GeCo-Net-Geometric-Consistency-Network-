import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.ResEncoder import ResEncoder
from models.SRGAN import generator
from models.aggregator import adafusor, localfusor, meanfusor, varfusor
from submodel.adapter import LatentAdapter
from submodel.adapter_with_transformer import TransformerLatentAdapter
from submodel.continuous_prior_completion import ContinuousPriorCompletion
from submodel.multiscale_lift import MultiScaleLiftDecoder

# Main Model
class model(nn.Module):
    def __init__(
        self,
        model_conf=None,
        device=None,
        query_chunk_size=25000,
        use_query_checkpoint=True,
        use_adapter=False,
        adapter_type="cnn",
        adapter_hidden_channels=64,
        adapter_transformer_pool_size=8,
        adapter_transformer_layers=2,
        adapter_transformer_heads=4,
        adapter_transformer_dropout=0.1,
        adapter_use_global_alpha=False,
        adapter_global_alpha_init=0.0,
        use_prior_completion=False,
        completion_hidden_channels=16,
        completion_geometry_hidden_channels=32,
        completion_geometry_channels=64,
        completion_residual_scale=1.0,
        completion_use_checkpoint=True,
        multiscale_decoder=False,
        multiscale_fusion="concat",
        multiscale_shallow="none",
        multiscale_shallow_channels=16,
        multiscale_highres_fusion="gated_add",
        use_multiscale_supervision=False,
        use_hierarchical_view_weights=False,
        use_uncertainty_gate=False,
    ):
        super(model, self).__init__()
        self.device = device
        self.query_chunk_size = int(query_chunk_size)
        self.use_query_checkpoint = bool(use_query_checkpoint)
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        self.encoder_conf = model_conf['encoder']
        self.decoder_conf = model_conf['SRGAN.generator']
        self.last_layer = model_conf['last_layer']
        self.fusion = model_conf['fusion']
        self.encoder = ResEncoder(self.encoder_conf).to(device)
        self.use_multiscale_decoder = bool(multiscale_decoder)
        if self.use_multiscale_decoder:
            if use_adapter or use_prior_completion:
                raise ValueError(
                    "multiscale_decoder currently uses its own E2/E3/E4 latent path; "
                    "disable --use_adapter and --use_prior_completion for this ablation."
                )
            self.decoder = MultiScaleLiftDecoder(
                decoder_scale=int(self.decoder_conf.scale),
                fusion=str(multiscale_fusion),
                use_shallow_2d_fusion=(str(multiscale_shallow) == "2d_fuse"),
                shallow_channels=int(multiscale_shallow_channels),
                highres_fusion=str(multiscale_highres_fusion),
                use_multiscale_supervision=bool(use_multiscale_supervision),
                use_hierarchical_view_weights=bool(use_hierarchical_view_weights),
                use_uncertainty_gate=bool(use_uncertainty_gate),
                use_query_checkpoint=bool(use_query_checkpoint),
                query_chunk_size=int(query_chunk_size),
            ).to(device)
        else:
            self.decoder = generator(self.decoder_conf).to(device)
        self.use_adapter = bool(use_adapter)
        self.adapter_type = str(adapter_type).lower()
        if self.use_adapter:
            if self.adapter_type == "cnn":
                if adapter_use_global_alpha:
                    raise ValueError(
                        "--adapter_use_global_alpha requires --adapter_type transformer"
                    )
                self.adapter = LatentAdapter(
                    channels=int(self.decoder_conf.inplanes),
                    hidden_channels=int(adapter_hidden_channels),
                ).to(device)
            elif self.adapter_type == "transformer":
                self.adapter = TransformerLatentAdapter(
                    channels=int(self.decoder_conf.inplanes),
                    hidden_channels=int(adapter_hidden_channels),
                    pool_size=int(adapter_transformer_pool_size),
                    num_layers=int(adapter_transformer_layers),
                    num_heads=int(adapter_transformer_heads),
                    dropout=float(adapter_transformer_dropout),
                    use_global_alpha=bool(adapter_use_global_alpha),
                    global_alpha_init=float(adapter_global_alpha_init),
                ).to(device)
            else:
                raise ValueError(
                    f"Unsupported adapter_type {adapter_type!r}; use 'cnn' or 'transformer'."
                )
        else:
            self.adapter = nn.Identity()

        self.use_prior_completion = bool(use_prior_completion)
        self.completion_enabled = self.use_prior_completion
        if self.use_prior_completion:
            self.prior_completion = ContinuousPriorCompletion(
                channels=int(self.decoder_conf.inplanes),
                hidden_channels=int(completion_hidden_channels),
                geometry_hidden_channels=int(completion_geometry_hidden_channels),
                geometry_channels=int(completion_geometry_channels),
                residual_scale=float(completion_residual_scale),
                use_checkpoint=bool(completion_use_checkpoint),
            ).to(device)
        else:
            self.prior_completion = nn.Identity()
        self.last_aligned_latent = None
        self.last_completion_residual = None
        self.last_multiscale_aux = None

        self.aggregator_conf = model_conf['aggregator']
        if self.fusion == 'local':
            self.aggregator = localfusor(self.aggregator_conf).to(device)
        if self.fusion == 'meanmlp':
            self.aggregator = meanfusor(self.aggregator_conf).to(device)
        if self.fusion == 'varmlp':
            self.aggregator = varfusor(self.aggregator_conf).to(device)
        if self.fusion == 'ada':
            self.aggregator = adafusor(self.aggregator_conf).to(device)

        if self.last_layer.act == 'ReLU':
            self.last_layer_act = nn.ReLU(inplace=True)
        elif self.last_layer.act == 'GELU':
            self.last_layer_act = nn.GELU()

    def _query_and_fuse_points(self, pnts):
        latent = self.encoder.queryfeature(pnts)
        if self.fusion == 'max':
            return torch.max(latent, dim=0)[0]
        if self.fusion == 'mean':
            return torch.mean(latent, dim=0)
        return self.aggregator(latent)

    def query_volume_latent(self, xyz_world):
        x,y,z = xyz_world.shape[:3]                                 # 下采样后的尺寸 [x,y,z]=[X/4,Y/4,Z/4]
        points = xyz_world.contiguous().reshape(-1,3)               # 所有点坐标
        pnts_split = torch.split(points, self.query_chunk_size)     # 分块控制融合阶段的瞬时显存
        h = []
        for pnts in pnts_split:
            if self.training and torch.is_grad_enabled() and self.use_query_checkpoint:
                # Recompute projection sampling and view fusion during backward
                # instead of retaining their very large per-point activations.
                fused = checkpoint(
                    self._query_and_fuse_points,
                    pnts,
                    use_reentrant=False,
                )
            else:
                fused = self._query_and_fuse_points(pnts)
            h.append(fused)                                        # ② 聚合多视角特征 [C, npts]
            
        h = torch.cat(h,dim=1)
        return h.reshape(1,-1,x,y,z)                                # ③ 低分辨率特征体积 [1, C, X/4, Y/4, Z/4]

    def forward(self, xyz_world, return_latent=False, xyz_full_world=None):
        if self.use_multiscale_decoder:
            outputs, aux = self.decoder(
                self.encoder.latent_list,
                self.encoder.poses,
                self.encoder.image_shape,
                xyz_full=xyz_full_world,
                return_aux=True,
            )
            # The E2-resolution representation is the observation latent used
            # for optional diagnostics. It is not a pCT teacher target in this
            # standalone multiscale ablation.
            self.last_aligned_latent = aux["e2"]
            self.last_multiscale_aux = aux
            outputs = outputs[0, 0, :, :, :].transpose(0, 2)
            outputs = self.last_layer_act(outputs)
            if return_latent:
                return outputs, self.last_aligned_latent
            return outputs
        latent = self.query_volume_latent(xyz_world)
        latent = self.adapter(latent)
        self.last_aligned_latent = latent
        if self.use_prior_completion and self.completion_enabled:
            latent, residual = self.prior_completion(
                latent,
                self.encoder.poses,
                return_residual=True,
            )
            self.last_completion_residual = residual
        else:
            self.last_completion_residual = None
        outputs = self.decoder(latent)[0,0,:,:,:].transpose(0,2)    # align with ITK-SNAP display format。 这里实际上是拿到针对单个点的，从其在不同视角上的对应点的特征向量，这些特征向量是在维度上进行拼接的
                                                                    # ④ self.decoder(outputs)：★ decoder 3D 上采样 [1, 1, X, Y, Z]
                                                                    # [0,0,:,:,:].transpose(0,2)：⑤ 对齐 ITK-SNAP 显示格式
        outputs = self.last_layer_act(outputs)                      # ⑥ GELU/ReLU 激活 → 非负 μ
        if return_latent:
            return outputs, latent
        return outputs                                              # 返回真正的体素空间表示
