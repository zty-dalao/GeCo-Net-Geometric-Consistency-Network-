import os.path
import itertools
import warnings
from contextlib import nullcontext
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.render import *
from util.util_func import *
import datetime
from models.loss import *
from submodel.decoder.loss import bone_gt_mask_l1, soft_tissue_gt_mask_l1, ssim_loss_3d
from submodel.decoder.model import PriorFeatureStem

class trainer():
    def __init__(self, G_render, train_data_loader, val_data_loader, test_data_loader, visual_data_loader, args,
                 conf, device=None):
        self.G_render = G_render
        self.args = args
        self.conf = conf
        self.device = device

        # dataloader
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.visual_data_loader = visual_data_loader

        # interval
        self.vis_interval = conf.get_int('train.print.vis_interval')    # 每隔多少 epoch 做一次可视化
        self.save_interval = conf.get_int('train.print.save_interval')  # 每隔多少 epoch 存一次历史 checkpoint
        self.val_interval = conf.get_int('train.print.val_interval')    # 每隔多少 epoch 验证一次
        self.test_interval = conf.get_int('train.print.test_interval')  # 每隔多少 epoch 测试一次

        # loss lambda
        self.mse_lambda_2d = conf.get_float('train.G_loss.mse_lambda_2d')
        self.mse_lambda_3d = conf.get_float('train.G_loss.mse_lambda_3d')
        self.gd1_lambda = conf.get_float('train.G_loss.gd1_lambda')
        self.latent_lambda = args.latent_lambda
        self.latent_cosine_lambda = args.latent_cosine_lambda
        self.bone_lambda = args.bone_lambda
        self.soft_mask_lambda = args.soft_mask_lambda
        self.ssim_lambda = args.ssim_lambda
        self.bone_lower_hu = args.bone_lower_hu
        self.soft_window_low = args.soft_window_low
        self.soft_window_high = args.soft_window_high
        if self.soft_window_high <= self.soft_window_low:
            raise ValueError("soft_window_high must be greater than soft_window_low")

        # epoch
        self.is_train = args.is_train
        self.num_epochs = args.epochs
        self.resume_name = args.resume_name  # specify the resume epoch
        if not self.is_train:
            self.num_epochs = self.num_epochs + 1

        # A pretrained checkpoint activates staged transfer learning. Without
        # one, the original joint-training behavior is preserved.
        self.use_staged_training = bool(args.pretrained_decoder) and self.is_train
        self.stage1_epochs = args.stage1_epochs
        self.stage2_epochs = args.stage2_epochs
        self.decoder_lr_factor = args.decoder_lr_factor
        self.stage3_backbone_lr_factor = args.stage3_backbone_lr_factor
        if self.stage1_epochs < 0 or self.stage2_epochs < 0:
            raise ValueError("stage1_epochs and stage2_epochs must be non-negative")
        if min(self.latent_lambda, self.bone_lambda, self.soft_mask_lambda, self.ssim_lambda) < 0:
            raise ValueError("latent_lambda, bone_lambda, soft_mask_lambda, and ssim_lambda must be non-negative")
        if self.latent_lambda > 0 and not args.pretrained_decoder:
            raise ValueError("--latent_lambda > 0 requires --pretrained_decoder")
        if self.use_staged_training and self.stage1_epochs + self.stage2_epochs >= self.num_epochs:
            warnings.warn(
                "No epochs remain for stage 3. Increase --epochs or reduce "
                "--stage1_epochs/--stage2_epochs.",
                stacklevel=2,
            )

        # render 
        self.ray_batch_size = conf.get_int('render.ray_batch_size') # 训练时 2D 损失随机采样的射线数
        self.factor = conf.get_float('render.factor')               # 体渲染采样步长 = volume_spacing * factor
        self.chunksize = conf.get_int('render.chunksize')           # composite 中每块并行处理的射线数（显存控制）
        
        # others
        self.expnorm = args.expnorm
        # We highly recommend to use expnorm, which results in similar projection intensity range between (0, 1].
        # It is beneficial for encoder to extract features, which usually leads to better performance and faster convergence.
        # 布尔开关，决定是否把投影从"衰减线积分域"用 exp(-img/divide) 转换到"(0,1] 透射率域"再喂给编码器/算损失；默认开启，且开启时训练效果更好（传 --expnorm 反而关闭，一般不建议）
        # 根据 args.datatype 设置体积截断范围和投影归一化除数：
        if args.datatype == 'dental':
            self.clamp_min = conf.get_float('data.dental.clamp_min')    # 对 GT/预测体素做 μ 值截断，用于损失计算和指标
            self.clamp_max = conf.get_float('data.dental.clamp_max')    # 对 GT/预测体素做 μ 值截断，用于损失计算和指标
            self.divide = 1                                             # 投影归一化时 exp(-img / divide)，spine 因为衰减系数数值不同所以除以 10
        if args.datatype == 'spine':
            self.clamp_min = conf.get_float('data.spine.clamp_min')
            self.clamp_max = conf.get_float('data.spine.clamp_max')
            self.divide = 10
        if args.datatype == 'Walnuts':
            self.clamp_min = conf.get_float('data.Walnuts.clamp_min')
            self.clamp_max = conf.get_float('data.Walnuts.clamp_max')
            self.divide = 1
            
        # logs
        self.logs_path = os.path.join(args.logs_path, args.name)
        os.makedirs(self.logs_path, exist_ok=True)
        self.visual_path = os.path.join(args.visual_path, args.name)
        os.makedirs(self.visual_path, exist_ok=True)
        self.checkpoints_path = os.path.join(args.checkpoints_path, args.name)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        self.tensorboard_path = os.path.join(self.logs_path, "tensorboard")
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)
        self.amp_enabled = (
            str(device).startswith("cuda")
            and torch.cuda.is_available()
            and not getattr(args, "no_amp", False)
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.G_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
            except TypeError:
                self.G_scaler = torch.amp.GradScaler(enabled=self.amp_enabled)
        else:
            self.G_scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

        # Frozen training-only teacher that maps downsampled pCT to the exact
        # latent basis learned together with the pretrained decoder.
        self.prior_stem = None
        if args.pretrained_decoder is not None:
            pretrained = torch.load(args.pretrained_decoder, map_location="cpu")
            stem_state = pretrained.get("feature_stem")
            if stem_state is None and "model" in pretrained:
                stem_state = {
                    key[len("feature_stem."):]: value
                    for key, value in pretrained["model"].items()
                    if key.startswith("feature_stem.")
                }
            if not stem_state:
                if self.latent_lambda > 0:
                    raise KeyError(
                        f"Checkpoint {args.pretrained_decoder!r} has no feature_stem weights."
                    )
            else:
                self.prior_stem = PriorFeatureStem(
                    int(self.G_render.decoder.inplanes)
                ).to(device)
                self.prior_stem.load_state_dict(stem_state, strict=True)
                self.prior_stem.eval()
                for parameter in self.prior_stem.parameters():
                    parameter.requires_grad = False
            del pretrained

        # lr scheduler & optimizer
        init_lr = conf.get_float('lr_sche.init_lr')
        self.init_lr = init_lr
        step_size = conf.get_float('lr_sche.step_size') # 每 50 个 epoch 衰减一次
        gamma = conf.get_float('lr_sche.gamma')         # 每次衰减为原来的 0.5
        self.lr_step_size = step_size
        self.lr_gamma = gamma
        high_resolution_parameters = list(
            itertools.chain(
                self.G_render.decoder.up_blk_list[-1].parameters(),
                self.G_render.decoder.out_blk.parameters(),
            )
        )
        high_resolution_ids = {id(parameter) for parameter in high_resolution_parameters}
        low_resolution_parameters = [
            parameter
            for parameter in self.G_render.decoder.parameters()
            if id(parameter) not in high_resolution_ids
        ]
        aggregator = getattr(self.G_render, "aggregator", None)
        aggregator_parameters = [] if aggregator is None else list(aggregator.parameters())
        backbone_parameters = list(self.G_render.encoder.parameters()) + aggregator_parameters
        self.G_optim = torch.optim.Adam(
            [
                {"params": backbone_parameters, "lr": init_lr, "name": "backbone"},
                {"params": low_resolution_parameters, "lr": init_lr, "name": "decoder_lowres"},
                {"params": high_resolution_parameters, "lr": init_lr, "name": "decoder_highres"},
            ]
        )
        self.G_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.G_optim,
            lr_lambda=[
                lambda epoch: self._lr_multiplier(epoch, "backbone"),
                lambda epoch: self._lr_multiplier(epoch, "decoder_lowres"),
                lambda epoch: self._lr_multiplier(epoch, "decoder_highres"),
            ],
        )

        # loss
        self.mse_loss = torch.nn.L1Loss(reduction='mean')

        # load weights & optimizer & iterator
        self.begin_epochs = 0
        self.global_step = 0
        os.makedirs("%s/ckpt_history" % (self.checkpoints_path,), exist_ok=True)
        self.latest_model_path = "%s/ckpt_latest" % (self.checkpoints_path,)        # 永远覆盖写"最新"权重
        self.history_model_path = "%s/ckpt_history/ckpt_" % (self.checkpoints_path,)# 按 epoch 归档的历史权重
        if args.resume: 
            self.load_ckpt(self.resume_name)    # 断电加载
        self._apply_training_stage(self.begin_epochs)

    def _training_stage(self, epoch):
        if not self.use_staged_training:
            return 0
        if epoch < self.stage1_epochs:
            return 1
        if epoch < self.stage1_epochs + self.stage2_epochs:
            return 2
        return 3

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)

    def _lr_multiplier(self, epoch, group_name):
        decay = self.lr_gamma ** (epoch // max(1, self.lr_step_size))
        stage = self._training_stage(epoch)
        if stage == 0:
            return decay
        if stage == 1:
            return decay if group_name == "backbone" else 0.0
        if stage == 2:
            if group_name == "backbone":
                return decay
            return decay * self.decoder_lr_factor
        if group_name == "backbone":
            return decay * self.stage3_backbone_lr_factor
        if group_name == "decoder_highres":
            return decay * self.decoder_lr_factor
        return 0.0

    def _latent_weight(self, epoch):
        if self.prior_stem is None or self.latent_lambda <= 0:
            return 0.0
        stage = self._training_stage(epoch)
        if stage in (0, 1):
            return self.latent_lambda
        if stage == 3 or self.stage2_epochs <= 0:
            return 0.0
        stage2_index = epoch - self.stage1_epochs
        progress = stage2_index / max(1, self.stage2_epochs - 1)
        return 0.5 * self.latent_lambda * max(0.0, 1.0 - progress)

    @staticmethod
    def _set_trainable(module, trainable):
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    def _apply_training_stage(self, epoch):
        stage = self._training_stage(epoch)
        self.current_stage = stage
        # Recompute LRs explicitly so a checkpoint saved before scheduler.step()
        # resumes at the correct stage boundary.
        if hasattr(self, "G_optim"):
            for group in self.G_optim.param_groups:
                group["lr"] = self.init_lr * self._lr_multiplier(
                    epoch, group.get("name", "backbone")
                )
        self.G_render.train()

        if stage == 0:
            self._set_trainable(self.G_render, True)
        elif stage == 1:
            self._set_trainable(self.G_render.encoder, True)
            if hasattr(self.G_render, "aggregator"):
                self._set_trainable(self.G_render.aggregator, True)
            self._set_trainable(self.G_render.decoder, False)
            # Frozen BatchNorm running statistics must remain fixed as well.
            self.G_render.decoder.eval()
        elif stage == 2:
            self._set_trainable(self.G_render, True)
        else:
            backbone_trainable = self.stage3_backbone_lr_factor > 0
            self._set_trainable(self.G_render.encoder, backbone_trainable)
            if hasattr(self.G_render, "aggregator"):
                self._set_trainable(self.G_render.aggregator, backbone_trainable)
            self._set_trainable(self.G_render.decoder, False)
            self._set_trainable(self.G_render.decoder.up_blk_list[-1], True)
            self._set_trainable(self.G_render.decoder.out_blk, True)
            self.G_render.decoder.in_blk.eval()
            self.G_render.decoder.res_blk_list.eval()
            self.G_render.decoder.res_blk_last.eval()
            for block in self.G_render.decoder.up_blk_list[:-1]:
                block.eval()

        if self.prior_stem is not None:
            self.prior_stem.eval()

    @staticmethod
    def _normalize_latent(latent):
        mean = latent.mean(dim=(2, 3, 4), keepdim=True)
        std = latent.std(dim=(2, 3, 4), keepdim=True, unbiased=False)
        return (latent - mean) / (std + 1e-6)

    @staticmethod
    def _mu_to_hu(volume):
        return (volume / 0.022 - 1.0) * 1000.0

    def _regional_and_ssim_losses(self, volume_predict, volume_gt):
        """Optional full-model losses shared with Decoder pretraining.

        Masks are defined only by ground truth and prediction is never clamped
        before the regional errors are calculated.  Therefore a voxel predicted
        outside the desired HU interval still receives a corrective gradient.
        """
        zero = volume_predict.new_zeros(())
        results = {
            "bone_gt_mask_raw": zero,
            "bone_gt_mask_loss": zero,
            "soft_mask_raw": zero,
            "soft_mask_loss": zero,
            "ssim_loss_raw": zero,
            "ssim_loss": zero,
        }
        if self.bone_lambda > 0:
            raw = bone_gt_mask_l1(
                volume_predict, volume_gt, self.bone_lower_hu,
                self.clamp_min, self.clamp_max,
            )
            results["bone_gt_mask_raw"] = raw
            results["bone_gt_mask_loss"] = raw * self.bone_lambda
        if self.soft_mask_lambda > 0:
            raw = soft_tissue_gt_mask_l1(
                volume_predict, volume_gt,
                self.soft_window_low, self.soft_window_high,
            )
            results["soft_mask_raw"] = raw
            results["soft_mask_loss"] = raw * self.soft_mask_lambda
        if self.ssim_lambda > 0:
            raw = ssim_loss_3d(
                volume_predict, volume_gt, self.clamp_min, self.clamp_max,
            )
            results["ssim_loss_raw"] = raw
            results["ssim_loss"] = raw * self.ssim_lambda
        return results

    def _make_prior_latent(self, volume_gt):
        if self.prior_stem is None:
            return None
        # SimpleITK/main-dataset volume is ZYX; decoder latent is XYZ.
        volume_xyz = volume_gt.permute(2, 1, 0).contiguous()[None, None]
        low_resolution = F.avg_pool3d(
            volume_xyz,
            kernel_size=self.G_render.decoder.scale,
            stride=self.G_render.decoder.scale,
        )
        with torch.no_grad():
            with self._autocast():
                return self.prior_stem(low_resolution)
        
    def save_ckpt(self, epoch):
        data = {
            'iter': epoch + 1,
            'global_step': self.global_step,
            'training_stage': self._training_stage(epoch),
            'G_render': self.G_render.state_dict(),
            'G_optim': self.G_optim.state_dict(),
            'G_lr_scheduler': self.G_lr_scheduler.state_dict(),
            'G_scaler': self.G_scaler.state_dict(),
        }
        if self.prior_stem is not None:
            data['prior_stem'] = self.prior_stem.state_dict()
        torch.save(data, self.latest_model_path)
        if (epoch % self.save_interval == 0) or epoch == self.num_epochs - 1:
            torch.save(data, self.history_model_path + str(epoch))

    def load_ckpt(self, resume_name=None):
        data = None
        if resume_name is None:
            if os.path.exists(self.latest_model_path):
                data = torch.load(self.latest_model_path, map_location=self.device)
        else:
            history_path = self.history_model_path + str(resume_name)
            if os.path.exists(history_path):
                data = torch.load(history_path, map_location=self.device)
        if data is not None:
            if 'G_render' in data: self.G_render.load_state_dict(data['G_render'])
            if 'iter' in data: self.begin_epochs = data['iter']
            if 'global_step' in data: self.global_step = data['global_step']
            if 'prior_stem' in data and self.prior_stem is not None:
                self.prior_stem.load_state_dict(data['prior_stem'], strict=True)
            if 'G_optim' in data:
                try:
                    self.G_optim.load_state_dict(data['G_optim'])
                except ValueError:
                    warnings.warn(
                        "The checkpoint optimizer predates staged parameter groups; "
                        "model weights were restored but optimizer state was restarted.",
                        stacklevel=2,
                    )
            if 'G_lr_scheduler' in data:
                try:
                    self.G_lr_scheduler.load_state_dict(data['G_lr_scheduler'])
                except (KeyError, ValueError):
                    warnings.warn(
                        "The checkpoint uses an older scheduler format; optimizer "
                        "weights were restored and the staged scheduler was rebuilt.",
                        stacklevel=2,
                    )
            if 'G_scaler' in data:
                self.G_scaler.load_state_dict(data['G_scaler'])

    def train_step(self, data, epoch):
        self._apply_training_stage(epoch)
        device = self.device
        # data loading
        src_images = data["images"].to(device=device).squeeze(0)    # 从 DataLoader 返回的 batch dict 中取出投影图像张量。搬到计算设备（GPU cuda）。即 [20, 1, 512, 512]（20 个视角、单通道灰度、512×512）
        if self.expnorm:
            src_images = torch.exp(-src_images/self.divide)         # 学上把投影从衰减线积分域转到透射率/归一化强度域（Beer-Lambert）
        src_poses = data["poses"].to(device=device).squeeze(0)      # data["poses"]：从 batch 取出每个视角的扫描几何向量。形状：[1, N, 12] → [N, 12]，即 [20, 12]。每个视角的 12 维向量 vec 由 4 组三维坐标拼接而成（来自 transforms.json 的 frames[i]['vec']，由 angle2vec 生成）
        
        # basic information
        _, _, H, W = src_images.shape                                                                           # 得到投影高度/宽度。例如 H=W=512
        volume_phy = torch.tensor(data['paras']['volume_phy']).to(device).to(torch.float32)                     # 是体数据在三个维度的物理长度（单位 mm），形状 [3]。
        volume_origin = torch.tensor(data['paras']['volume_origin']).to(device).to(torch.float32)               # 体数据的物理原点（包围盒左下角），形状 [3]。仿真时生成：
        volume_spacing = torch.min(torch.tensor(data['paras']['volume_spacing'])).to(device).to(torch.float32)  # volume_spacing 原始是 [3] 的数组（X/Y/Z 三方向的体素间距），这里用 torch.min 取三个方向的最小值，得到一个标量。这样做的目的是以最密的体素间距为基准来定采样步长，保证任何方向都不会欠采样。
        render_step_size = volume_spacing * self.factor                                                         # self.factor 来自 conf['render.factor']（默认 0.5），即每个体素内采样 2 个点。这个步长决定了 composite（DRR / 2D 投影损失）沿每条射线等距采样的密度
        volume_gt = data['3Dvolume'].to(device=device).squeeze(0).to(torch.float32)                             # 加载 GT 体积
        volume_gt = torch.clamp(volume_gt, self.clamp_min, self.clamp_max)                                      # GT 体积截断，把 μ 值截断到物理合理的范围（按数据集：dental [0, 0.09009]、spine [0, 0.051744]、Walnuts [0, 0.084]）。作用：
        volume_resolution = torch.tensor(data['paras']['volume_resolution']).to(device).to(torch.int64)         # 各维度的体素数（如 [128, 128, 128]），转成 int64（后面 make_coords 用 torch.linspace 生成网格需要整数长度）。这个值决定了重建网格的采样点数
        
        loss_dict = {}
        
        # 2d projection encoding
        with self._autocast():
            self.G_render.encoder(src_images, src_poses)    # model的ResEncoder
            # 3d volume decoding
            volume_predict, projection_latent = predict_3d_volume(
                model=self.G_render,
                volume_resolution=volume_resolution,
                volume_origin=volume_origin,
                volume_phy=volume_phy,
                scale=self.G_render.decoder.scale,
                device=device,
                return_latent=True,
            )

        self.G_optim.zero_grad()
        # 3d loss，体素空间级别的L1 loss
        mse_loss_3d = self.mse_loss(volume_predict, volume_gt) * self.mse_lambda_3d
        loss_dict['mse_loss_3d'] = round(mse_loss_3d.item(), 8)
        G_loss = mse_loss_3d

        # gd loss，梯度损失
        if self.gd1_lambda > 0:
            gd1_loss = gradient1_loss(volume_gt=volume_gt, volume_predict=volume_predict, loss_func=self.mse_loss) * self.gd1_lambda
            loss_dict['gd1_loss'] = round(gd1_loss.item(), 8)
            G_loss += gd1_loss
        else:
            loss_dict['gd1_loss'] = 0.0

        latent_weight = self._latent_weight(epoch)
        if latent_weight > 0:
            prior_latent = self._make_prior_latent(volume_gt)
            if projection_latent.shape != prior_latent.shape:
                raise RuntimeError(
                    "Projection/prior latent shape mismatch: "
                    f"{tuple(projection_latent.shape)} vs {tuple(prior_latent.shape)}. "
                    "Check XYZ/ZYX ordering and decoder scale."
                )
            projection_latent_norm = self._normalize_latent(projection_latent)
            prior_latent_norm = self._normalize_latent(prior_latent)
            latent_l1 = F.smooth_l1_loss(projection_latent_norm, prior_latent_norm)
            latent_cosine = 1.0 - F.cosine_similarity(
                projection_latent_norm.flatten(2),
                prior_latent_norm.flatten(2),
                dim=1,
            ).mean()
            latent_loss = latent_weight * (
                latent_l1 + self.latent_cosine_lambda * latent_cosine
            )
            G_loss += latent_loss
            loss_dict['latent_loss'] = round(latent_loss.item(), 8)
            loss_dict['latent_smooth_l1_raw'] = round(latent_l1.item(), 8)
            loss_dict['latent_cosine_raw'] = round(latent_cosine.item(), 8)
        else:
            loss_dict['latent_loss'] = 0.0
            loss_dict['latent_smooth_l1_raw'] = 0.0
            loss_dict['latent_cosine_raw'] = 0.0

        optional_losses = self._regional_and_ssim_losses(volume_predict, volume_gt)
        for key, value in optional_losses.items():
            loss_dict[key] = round(value.item(), 8)
        G_loss += (
            optional_losses['bone_gt_mask_loss']
            + optional_losses['soft_mask_loss']
            + optional_losses['ssim_loss']
        )

        # 2d ray batch loss
        if self.mse_lambda_2d > 0:
            pix_inds = torch.randint(
                0,
                src_images.shape[0] * H * W,
                (self.ray_batch_size,),
                device=src_images.device,
            )
            images_gt_all = src_images.reshape(-1, 1)
            proj_gt = images_gt_all[pix_inds]
            src_rays = get_rays(src_poses, H, W)
            proj_rays = src_rays.view(-1, src_rays.shape[-1])[pix_inds].to(device=device)
            proj_predict = composite(rays=proj_rays, volume=volume_predict.float(), volume_origin=volume_origin,
                                        volume_phy=volume_phy, render_step_size=render_step_size, 
                                        chunksize=self.chunksize).reshape(proj_gt.shape)
            if self.expnorm:
                proj_predict = torch.exp(-proj_predict/self.divide)
            mse_loss_2d = self.mse_loss(proj_predict, proj_gt) * self.mse_lambda_2d
            loss_dict['mse_loss_2d'] = round(mse_loss_2d.item(), 8)
            G_loss += mse_loss_2d
        else:
            loss_dict['mse_loss_2d'] = 0.0

        loss_dict['G_loss'] = round(G_loss.item(), 8)

        # update model
        self.G_scaler.scale(G_loss).backward()
        self.G_scaler.step(self.G_optim)
        self.G_scaler.update()

        self.writer.add_scalar("step/train_total", G_loss.detach(), self.global_step)
        for key in (
            'mse_loss_3d', 'gd1_loss', 'mse_loss_2d', 'latent_loss',
            'latent_smooth_l1_raw', 'latent_cosine_raw',
            'bone_gt_mask_raw', 'bone_gt_mask_loss',
            'soft_mask_raw', 'soft_mask_loss', 'ssim_loss_raw', 'ssim_loss',
        ):
            self.writer.add_scalar(f"step/train_{key}", loss_dict[key], self.global_step)
        self.writer.add_scalar("step/latent_weight", latent_weight, self.global_step)
        self.writer.add_scalar("step/training_stage", self.current_stage, self.global_step)
        for group in self.G_optim.param_groups:
            self.writer.add_scalar(
                f"step/lr_{group.get('name', 'group')}",
                group['lr'],
                self.global_step,
            )
        self.global_step += 1

        # first set G_render to eval state, calculate the PSNR, and turn it back to train state
        self.G_render.eval()
        with torch.no_grad():
            with self._autocast():
                self.G_render.encoder(src_images, src_poses)
                volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                                   volume_origin=volume_origin, volume_phy=volume_phy,
                                                   scale=self.G_render.decoder.scale, device=device)
            volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
            # 3d ssim calculation is too slow, so we only calculate psnr
            loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
        self.G_render.train()
        return loss_dict

    def test_step(self, data, epoch=0):
        device = self.device
        # data loading
        src_images = data["images"].to(device=device).squeeze(0)
        if self.expnorm:
            src_images = torch.exp(-src_images/self.divide)
        src_poses = data["poses"].to(device=device).squeeze(0)
        obj_index = data["obj_index"][0]
        _, _, H, W = src_images.shape

        # basic information
        volume_phy = torch.tensor(data['paras']['volume_phy']).to(device).to(torch.float32)
        volume_origin = torch.tensor(data['paras']['volume_origin']).to(device).to(torch.float32)
        volume_spacing = torch.min(torch.tensor(data['paras']['volume_spacing'])).to(device).to(torch.float32)
        render_step_size = volume_spacing * self.factor
        volume_gt = data['3Dvolume'].to(device=device).squeeze(0).to(torch.float32)
        volume_gt = torch.clamp(volume_gt, self.clamp_min, self.clamp_max)
        volume_resolution = torch.tensor(data['paras']['volume_resolution']).to(device).to(torch.int64)

        loss_dict = {
            'obj_index': obj_index,
        }

        # 2d projection encoding
        with self._autocast():
            self.G_render.encoder(src_images, src_poses)
            # 3d volume decoding
            volume_predict, projection_latent = predict_3d_volume(
                model=self.G_render,
                volume_resolution=volume_resolution,
                volume_origin=volume_origin,
                volume_phy=volume_phy,
                scale=self.G_render.decoder.scale,
                device=device,
                return_latent=True,
            )
        
        # metrics calculation
        volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
        loss_dict = {'obj_index': obj_index}
        loss_3d = self.mse_loss(volume_predict, volume_gt) * self.mse_lambda_3d
        loss_dict['mse_loss_3d'] = round(loss_3d.item(), 8)
        total_loss = loss_3d

        if self.gd1_lambda > 0:
            gd1_loss = gradient1_loss(
                volume_gt=volume_gt,
                volume_predict=volume_predict,
                loss_func=self.mse_loss,
            ) * self.gd1_lambda
            total_loss += gd1_loss
            loss_dict['gd1_loss'] = round(gd1_loss.item(), 8)
        else:
            loss_dict['gd1_loss'] = 0.0

        latent_weight = self._latent_weight(epoch)
        if latent_weight > 0:
            prior_latent = self._make_prior_latent(volume_gt)
            if projection_latent.shape != prior_latent.shape:
                raise RuntimeError(
                    "Projection/prior latent shape mismatch during evaluation: "
                    f"{tuple(projection_latent.shape)} vs {tuple(prior_latent.shape)}"
                )
            projection_latent_norm = self._normalize_latent(projection_latent)
            prior_latent_norm = self._normalize_latent(prior_latent)
            latent_l1 = F.smooth_l1_loss(projection_latent_norm, prior_latent_norm)
            latent_cosine = 1.0 - F.cosine_similarity(
                projection_latent_norm.flatten(2),
                prior_latent_norm.flatten(2),
                dim=1,
            ).mean()
            latent_loss = latent_weight * (
                latent_l1 + self.latent_cosine_lambda * latent_cosine
            )
            total_loss += latent_loss
            loss_dict['latent_loss'] = round(latent_loss.item(), 8)
            loss_dict['latent_smooth_l1_raw'] = round(latent_l1.item(), 8)
            loss_dict['latent_cosine_raw'] = round(latent_cosine.item(), 8)
        else:
            loss_dict['latent_loss'] = 0.0
            loss_dict['latent_smooth_l1_raw'] = 0.0
            loss_dict['latent_cosine_raw'] = 0.0

        optional_losses = self._regional_and_ssim_losses(volume_predict, volume_gt)
        for key, value in optional_losses.items():
            loss_dict[key] = round(value.item(), 8)
        total_loss += (
            optional_losses['bone_gt_mask_loss']
            + optional_losses['soft_mask_loss']
            + optional_losses['ssim_loss']
        )

        if self.mse_lambda_2d > 0:
            total_pixels = src_images.shape[0] * H * W
            sample_count = min(self.ray_batch_size, total_pixels)
            pix_inds = torch.linspace(
                0,
                total_pixels - 1,
                steps=sample_count,
                device=src_images.device,
            ).long()
            proj_gt = src_images.reshape(-1, 1)[pix_inds]
            src_rays = get_rays(src_poses, H, W)
            proj_rays = src_rays.reshape(-1, src_rays.shape[-1])[pix_inds]
            proj_predict = composite(
                rays=proj_rays,
                volume=volume_predict.float(),
                volume_origin=volume_origin,
                volume_phy=volume_phy,
                render_step_size=render_step_size,
                chunksize=self.chunksize,
            ).reshape(proj_gt.shape)
            if self.expnorm:
                proj_predict = torch.exp(-proj_predict / self.divide)
            loss_2d = self.mse_loss(proj_predict, proj_gt) * self.mse_lambda_2d
            total_loss += loss_2d
            loss_dict['mse_loss_2d'] = round(loss_2d.item(), 8)
        else:
            loss_dict['mse_loss_2d'] = 0.0

        loss_dict['G_loss'] = round(total_loss.item(), 8)
        loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
        loss_dict['ssim_3d_clamp'] = round(get_ssim_3d(data_norm(volume_predict_clamp), data_norm(volume_gt), data_range=1), 8)

        return loss_dict

    def vis_step(self, data, epoch=0, ):
        device = self.device
        # data loading
        src_images = data["images"].to(device=device).squeeze(0)
        if self.expnorm:
            src_images = torch.exp(-src_images/self.divide)
        src_poses = data["poses"].to(device=device).squeeze(0)
        obj_index = data["obj_index"][0]

        # basic information
        volume_phy = torch.tensor(data['paras']['volume_phy']).to(device).to(torch.float32)
        volume_origin = torch.tensor(data['paras']['volume_origin']).to(device).to(torch.float32)
        volume_gt = data['3Dvolume'].to(device=device).squeeze(0).to(torch.float32)
        volume_gt = torch.clamp(volume_gt, self.clamp_min, self.clamp_max)
        volume_resolution = torch.tensor(data['paras']['volume_resolution']).to(device).to(torch.int64)

        loss_dict = {
            'obj_index': obj_index,
        }

        # 2d projection encoding
        with self._autocast():
            self.G_render.encoder(src_images, src_poses)
            # 3d volume decoding
            volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                               volume_origin=volume_origin, volume_phy=volume_phy,
                                               scale=self.G_render.decoder.scale, device=device)

        # alculate metrics with clamped volume for more accurate evaluation
        volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
        loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
        loss_dict['ssim_3d_clamp'] = round(get_ssim_3d(data_norm(volume_predict_clamp), data_norm(volume_gt), data_range=1), 8)

        # save the volume prediction
        os.makedirs(os.path.join(self.visual_path, obj_index + '/volume'), exist_ok=True)
        volume_gt_nii = self.visual_path + '/' + obj_index + '/volume/volume_gt.nii.gz'
        volume_predict_nii = self.visual_path + '/' + obj_index + '/volume/volume_' + str(epoch) + '.nii.gz'
        volume_gt_hu = mu2ct(volume_gt)  # convert mu to ct number
        volume_predict_hu = mu2ct(volume_predict)
        tensor2nii(volume_gt_hu, volume_gt_nii)
        tensor2nii(volume_predict_hu, volume_predict_nii) # record original volume rather than clamped volume for analysis convinience
        return loss_dict

    @staticmethod
    def _tracked_loss_keys():
        return (
            'G_loss',
            'mse_loss_3d',
            'gd1_loss',
            'mse_loss_2d',
            'latent_loss',
            'latent_smooth_l1_raw',
            'latent_cosine_raw',
            'bone_gt_mask_raw',
            'bone_gt_mask_loss',
            'soft_mask_raw',
            'soft_mask_loss',
            'ssim_loss_raw',
            'ssim_loss',
        )

    def _write_epoch_tensorboard(self, split, sums, count, epoch, metrics=None):
        if count <= 0:
            return
        for key in self._tracked_loss_keys():
            self.writer.add_scalar(
                f"epoch/{split}_{key}",
                sums[key] / count,
                epoch,
            )
        if metrics:
            for key, value in metrics.items():
                self.writer.add_scalar(f"epoch/{split}_{key}", value, epoch)
        self.writer.flush()

    def start(self):
        
        if self.is_train:
            for epoch in range(self.begin_epochs, self.num_epochs):
                self._apply_training_stage(epoch)

                # 在每个 epoch 开始时，把"当前时间"和"当前学习率"追加写入日志文件 train/logs/<实验名>/train_lr.txt
                now = datetime.datetime.now()                                   # ① 获取当前时刻
                f_train_lr = open(self.logs_path + '/train_lr.txt', mode='a')   # ② 以"追加"模式打开日志文件
                f_train_lr.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' G_lr:' + str(
                    self.G_optim.param_groups[0]["lr"]) + '\n')                 # ③ 写入一行记录
                f_train_lr.close()                                              # ④ 关闭文件

                # train with the train dataset
                print('Network Training')
                train_batch = 0
                train_psnr_3d_clamp = 0 
                train_loss_sums = {key: 0.0 for key in self._tracked_loss_keys()}
                for train_data in self.train_data_loader:
                    train_losses = self.train_step(train_data, epoch)
                    train_loss_str = fmt_loss_str(train_losses)
                    now = datetime.datetime.now()
                    f_train_ls = open(self.logs_path + '/train_ls.txt', mode='a')
                    f_train_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(
                        train_batch) + train_loss_str
                                    + " G_lr:" + str(self.G_optim.param_groups[0]["lr"])+'\n')
                    f_train_ls.close()
                    print("*** train:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", train_batch,
                        train_loss_str, "G_lr:", str(self.G_optim.param_groups[0]["lr"]),)
                    train_batch = train_batch + 1

                    # batch psnr
                    train_psnr_3d_clamp = train_psnr_3d_clamp + train_losses['psnr_3d_clamp']
                    for key in self._tracked_loss_keys():
                        train_loss_sums[key] += train_losses[key]

                # epoch psnr
                train_psnr_3d_clamp = train_psnr_3d_clamp / train_batch
                now = datetime.datetime.now()
                f_train_psnr = open(self.logs_path + '/train_metric.txt', mode='a')
                f_train_psnr.write(
                    now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' train_psnr_3d_clamp:' + str(train_psnr_3d_clamp) + '\n')
                f_train_psnr.close()
                print("*** train:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'train_psnr_3d_clamp:', str(train_psnr_3d_clamp))
                self._write_epoch_tensorboard(
                    "train",
                    train_loss_sums,
                    train_batch,
                    epoch,
                    {"psnr_3d_clamp": train_psnr_3d_clamp},
                )

                # network saving
                print("saving network & optimizer")
                self.save_ckpt(epoch)
                
                # validate with the val dataset
                if ((epoch % self.val_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    print('Network validating')
                    val_batch = 0
                    val_psnr_3d_clamp = 0
                    val_ssim_3d_clamp = 0
                    val_loss_sums = {key: 0.0 for key in self._tracked_loss_keys()}
                    for val_data in self.val_data_loader:
                        self.G_render.eval()
                        with torch.no_grad():
                            val_losses = self.test_step(val_data, epoch)
                        self.G_render.train()
                        val_loss_str = fmt_loss_str(val_losses)
                        now = datetime.datetime.now()
                        f_val_ls = open(self.logs_path + '/val_ls.txt', mode='a')
                        f_val_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(
                            val_batch) + val_loss_str + '\n')
                        f_val_ls.close()
                        print("*** validate:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", val_batch, val_loss_str,)
                        val_batch = val_batch + 1

                        # batch psnr
                        val_psnr_3d_clamp = val_psnr_3d_clamp + val_losses['psnr_3d_clamp']
                        
                        # batch ssim
                        val_ssim_3d_clamp = val_ssim_3d_clamp + val_losses['ssim_3d_clamp']
                        for key in self._tracked_loss_keys():
                            val_loss_sums[key] += val_losses[key]

                    # epoch psnr
                    val_psnr_3d_clamp = val_psnr_3d_clamp / val_batch
                    # epoch ssim
                    val_ssim_3d_clamp = val_ssim_3d_clamp / val_batch

                    now = datetime.datetime.now()
                    f_val_psnr = open(self.logs_path + '/val_metric.txt', mode='a')
                    f_val_psnr.write(
                        now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' val_psnr_3d_clamp:' + str(val_psnr_3d_clamp) + 
                        ' val_ssim_3d_clamp:' + str(val_ssim_3d_clamp) +  '\n')
                    f_val_psnr.close()
                    print("*** validate:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'val_psnr_3d_clamp:', str(val_psnr_3d_clamp), 
                          'val_ssim_3d_clamp:'+ str(val_ssim_3d_clamp) + '\n') 
                    self._write_epoch_tensorboard(
                        "val",
                        val_loss_sums,
                        val_batch,
                        epoch,
                        {
                            "psnr_3d_clamp": val_psnr_3d_clamp,
                            "ssim_3d_clamp": val_ssim_3d_clamp,
                        },
                    )

                # test with the test dataset
                if ((epoch % self.test_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    print('Network Testing')
                    test_batch = 0
                    test_psnr_3d_clamp = 0
                    test_ssim_3d_clamp = 0
                    test_loss_sums = {key: 0.0 for key in self._tracked_loss_keys()}
                    for test_data in self.test_data_loader:
                        self.G_render.eval()
                        with torch.no_grad():
                            test_losses = self.test_step(test_data, epoch)
                        self.G_render.train()
                        test_loss_str = fmt_loss_str(test_losses)
                        now = datetime.datetime.now()
                        f_test_ls = open(self.logs_path + '/test_ls.txt', mode='a')
                        f_test_ls.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' Batch:' + str(
                            test_batch) + test_loss_str + '\n')
                        f_test_ls.close()
                        print("*** test:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, "Batch:", test_batch, test_loss_str)
                        test_batch = test_batch + 1

                        # batch psnr
                        test_psnr_3d_clamp = test_psnr_3d_clamp + test_losses['psnr_3d_clamp']

                        # batch ssim
                        test_ssim_3d_clamp = test_ssim_3d_clamp + test_losses['ssim_3d_clamp']
                        for key in self._tracked_loss_keys():
                            test_loss_sums[key] += test_losses[key]

                    # epoch psnr
                    test_psnr_3d_clamp = test_psnr_3d_clamp / test_batch
                    # epoch ssim
                    test_ssim_3d_clamp = test_ssim_3d_clamp / test_batch

                    now = datetime.datetime.now()
                    f_test_psnr = open(self.logs_path + '/test_metric.txt', mode='a')
                    f_test_psnr.write(
                        now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' test_psnr_3d_clamp:' + str(test_psnr_3d_clamp) + 
                        ' test_ssim_3d_clamp:'+ str(test_ssim_3d_clamp) + '\n')
                    f_test_psnr.close()
                    print("*** test:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'test_psnr_3d_clamp:', str(test_psnr_3d_clamp), 
                    'test_ssim_3d_clamp:', str(test_ssim_3d_clamp), '\n')
                    self._write_epoch_tensorboard(
                        "test",
                        test_loss_sums,
                        test_batch,
                        epoch,
                        {
                            "psnr_3d_clamp": test_psnr_3d_clamp,
                            "ssim_3d_clamp": test_ssim_3d_clamp,
                        },
                    )

                # lr schedule
                self.G_lr_scheduler.step()

                # visualization with the visual dataset during training when meet the epoch condition
                if ((epoch % self.vis_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    for vis_data in self.visual_data_loader:
                        print("Generating visualization")
                        self.G_render.eval()
                        with torch.no_grad():
                            vis_losses = self.vis_step(vis_data, epoch=epoch, )
                        self.G_render.train()
                        vis_loss_str = fmt_loss_str(vis_losses)
                        now = datetime.datetime.now()
                        f_vis_psnr = open(self.logs_path + '/visual_metric.txt', mode='a')
                        f_vis_psnr.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + vis_loss_str + '\n')
                        f_vis_psnr.close()
                        print("*** visual:", now.strftime('%Y-%m-%d %H:%M:%S'), " Epoch:", epoch, vis_loss_str)
        
        # visualization when not training (must resume some trained net)
        else:
            epoch = self.begin_epochs
            for vis_data in self.visual_data_loader:
                print("Generating visualization")
                self.G_render.eval()
                with torch.no_grad():
                    vis_losses = self.vis_step(vis_data, epoch=epoch)
                self.G_render.train()
                vis_loss_str = fmt_loss_str(vis_losses)
                now = datetime.datetime.now()
                f_vis_psnr = open(self.logs_path + '/visual_metric.txt', mode='a')
                f_vis_psnr.write(now.strftime('%Y-%m-%d %H:%M:%S') + ' visualization:' + vis_loss_str + '\n')
                f_vis_psnr.close()
                print("*** visual:", now.strftime('%Y-%m-%d %H:%M:%S'), vis_loss_str)
        self.writer.close()
