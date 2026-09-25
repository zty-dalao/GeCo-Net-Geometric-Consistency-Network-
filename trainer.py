import os.path
import itertools
import copy
import warnings
from contextlib import nullcontext
import torch.utils.data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from models.render import *
from util.util_func import *
import datetime
from models.loss import *
from submodel.decoder.loss import bone_gt_mask_l1, soft_tissue_gt_mask_l1, ssim_loss_3d
from submodel.decoder.model import PriorFeatureStem
from submodel.deep_encoder.model import LearnedPriorEncoder

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
        self.latent_stat_lambda = args.latent_stat_lambda
        self.use_prior_completion = bool(args.use_prior_completion)
        self.completion_phase_epochs = (
            args.completion_phase_epochs if self.use_prior_completion else 0
        )
        self.completion_lr_factor = args.completion_lr_factor
        self.completion_residual_lambda = args.completion_residual_lambda
        self.freeze_decoder_bn_stats = args.freeze_decoder_bn_stats
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

        # Canonical four-phase transfer schedule. Phase D occupies all epochs
        # remaining after A/B/C.
        self.phase_a_epochs = args.phase_a_epochs
        self.phase_b_epochs = args.phase_b_epochs
        self.phase_c_epochs = args.phase_c_epochs
        self.phase_c_hold_epochs = args.phase_c_hold_epochs
        self.phase_d_epochs = self.num_epochs - sum((
            self.phase_a_epochs, self.phase_b_epochs,
            self.completion_phase_epochs, self.phase_c_epochs,
            self.phase_c_hold_epochs,
        ))
        self.transfer_schedule = args.transfer_schedule
        self.multiscale_decoder = bool(getattr(args, "multiscale_decoder", False))
        self.use_multiscale_supervision = bool(
            getattr(args, "use_multiscale_supervision", False)
        )
        try:
            self.multiscale_aux_weights = tuple(
                float(value.strip())
                for value in getattr(
                    args, "multiscale_aux_weights", "0.2,0.1,0.05"
                ).split(",")
            )
        except ValueError as exc:
            raise ValueError("--multiscale_aux_weights must be comma-separated numbers") from exc
        if len(self.multiscale_aux_weights) != 3:
            raise ValueError("--multiscale_aux_weights requires three values")
        self.cross_scale_lambda = float(getattr(args, "cross_scale_lambda", 0.0))
        self.view_weight_delta_lambda = float(
            getattr(args, "view_weight_delta_lambda", 0.0)
        )
        self.legacy_stage1_epochs = args.legacy_stage1_epochs
        self.legacy_stage2_epochs = args.legacy_stage2_epochs
        self.legacy_stage3_backbone_lr_factor = (
            args.legacy_stage3_backbone_lr_factor
        )
        # Adapter and Completion are independent optional bridges.  Either one
        # can activate prior-transfer training; enabling both composes them.
        self.use_four_phase = bool(
            self.is_train
            and args.pretrained_decoder
            and (args.use_adapter or args.use_prior_completion)
        )
        # Kept so --resume_reload_decoder can re-inject the prior after load_ckpt;
        # train.py skips --pretrained_decoder entirely whenever --resume is set.
        self.pretrained_decoder_path = args.pretrained_decoder
        self.decoder_lr_factor = args.decoder_lr_factor
        self.adapter_lr_factor = args.adapter_lr_factor
        self.stage0_decoder_lr_factor = args.stage0_decoder_lr_factor
        self.phase_a_encoder_lr_factor = args.phase_a_encoder_lr_factor
        self.phase_a_aggregator_lr_factor = args.phase_a_aggregator_lr_factor
        self.phase_b_encoder_lr_factor = args.phase_b_encoder_lr_factor
        self.phase_b_aggregator_lr_factor = args.phase_b_aggregator_lr_factor
        self.phase_c_backbone_lr_factor = args.phase_c_backbone_lr_factor
        self.phase_c_aggregator_lr_factor = args.phase_c_aggregator_lr_factor
        self.phase_d_backbone_lr_factor = args.phase_d_backbone_lr_factor
        self.phase_d_aggregator_lr_factor = args.phase_d_aggregator_lr_factor
        self.decoder_core_lr_factor = args.decoder_core_lr_factor
        self.prior_anchor_lambda = args.prior_anchor_lambda
        self.phase_d_anchor_factor = args.phase_d_anchor_factor
        self.phase_b_latent_end_factor = args.phase_b_latent_end_factor
        self.phase_c_latent_end_factor = args.phase_c_latent_end_factor
        nonnegative = (
            self.latent_lambda, self.latent_cosine_lambda, self.latent_stat_lambda,
            self.bone_lambda, self.soft_mask_lambda, self.ssim_lambda,
            self.adapter_lr_factor, self.decoder_lr_factor,
            self.completion_lr_factor, self.completion_residual_lambda,
            self.phase_a_encoder_lr_factor, self.phase_a_aggregator_lr_factor,
            self.phase_b_encoder_lr_factor, self.phase_b_aggregator_lr_factor,
            self.phase_c_backbone_lr_factor, self.phase_c_aggregator_lr_factor,
            self.phase_d_backbone_lr_factor, self.phase_d_aggregator_lr_factor,
            self.decoder_core_lr_factor, self.prior_anchor_lambda,
            self.phase_d_anchor_factor,
            self.cross_scale_lambda,
            self.view_weight_delta_lambda,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("Loss weights, LR factors, and anchor factors must be non-negative")
        if self.stage0_decoder_lr_factor is not None and self.stage0_decoder_lr_factor < 0:
            raise ValueError("--stage0_decoder_lr_factor must be non-negative")
        if min(
            self.phase_a_epochs,
            self.phase_b_epochs,
            self.completion_phase_epochs,
            self.phase_c_epochs,
            self.phase_c_hold_epochs,
        ) < 0:
            raise ValueError("Phase A/B/Completion/C and Phase-C hold epoch counts must be non-negative")
        if (
            self.use_four_phase
            and self.transfer_schedule == "four_phase"
            and self.phase_d_epochs <= 0
        ):
            raise ValueError("--epochs must leave at least one epoch for Phase D")
        if not (0 <= self.phase_c_latent_end_factor <= self.phase_b_latent_end_factor <= 1):
            raise ValueError("Require 0 <= phase_c_latent_end_factor <= phase_b_latent_end_factor <= 1")
        if self.is_train and args.use_adapter and not args.pretrained_decoder:
            warnings.warn(
                "Adapter is enabled without a pretrained decoder; using ordinary joint "
                "training because four-phase prior transfer is unavailable.",
                stacklevel=2,
            )
        # Phase A freezes Encoder/Aggregator unless the phase-A backbone LR
        # factors turn them on.  Freezing a randomly initialized backbone leaves
        # only the zero-initialized Adapter trainable, so that combination is
        # rejected rather than silently wasting the whole phase.
        self.phase_a_trains_backbone = (
            self.phase_a_encoder_lr_factor > 0
            or self.phase_a_aggregator_lr_factor > 0
        )
        if self.transfer_schedule == "legacy_three_stage":
            first_phase_freezes_backbone = self.legacy_stage1_epochs > 0
        else:
            first_phase_freezes_backbone = (
                self.phase_a_epochs > 0 and not self.phase_a_trains_backbone
            )
        if (
            self.use_four_phase
            and not (args.pretrained_backbone or args.resume)
            and first_phase_freezes_backbone
        ):
            raise ValueError(
                "The first training stage would freeze the randomly initialized "
                "Encoder/Aggregator, leaving only the zero-initialized Adapter "
                "trainable. Pass --pretrained_backbone (or --resume), or train the "
                "backbone from scratch with --phase_a_encoder_lr_factor and "
                "--phase_a_aggregator_lr_factor set to positive values."
            )
        if (
            self.use_four_phase
            and not (args.pretrained_backbone or args.resume)
            and self.phase_a_epochs == 0
        ):
            warnings.warn(
                "Four-phase training from scratch with --phase_a_epochs 0 skips the "
                "Phase-A backbone warm-up, and Phase B only unfreezes encoder "
                "layer3/layer4, so Encoder layer1/layer2 stay frozen and randomly "
                "initialized until Phase C.",
                stacklevel=2,
            )
        if self.use_prior_completion and not self.use_four_phase:
            raise ValueError(
                "Continuous prior completion requires pretrained-decoder four-phase training"
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
        if args.datatype == 'thorax':
            self.clamp_min = conf.get_float('data.thorax.clamp_min')
            self.clamp_max = conf.get_float('data.thorax.clamp_max')
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

        # Frozen training-only teacher that maps pCT to the exact latent basis
        # learned with the pretrained decoder. The shallow teacher consumes a
        # fixed average-pooled volume; the deep mean/detail teacher consumes
        # the full-resolution pCT and performs its own decomposition.
        self.prior_stem = None
        self.prior_decoder_ref = None
        if args.pretrained_decoder is not None:
            pretrained = torch.load(args.pretrained_decoder, map_location="cpu")
            stem_state = pretrained.get("feature_stem")
            if stem_state is None and "model" in pretrained:
                stem_state = {
                    key[len("feature_stem."):]: value
                    for key, value in pretrained["model"].items()
                    if key.startswith("feature_stem.")
                }
            if not stem_state and self.use_four_phase:
                raise KeyError(
                    f"Checkpoint {args.pretrained_decoder!r} has no feature_stem weights "
                    "required by four-phase latent supervision and prior anchor."
                )
            elif stem_state:
                if args.prior_encoder_type == "deep":
                    self.prior_stem = LearnedPriorEncoder(
                        inplanes=int(self.G_render.decoder.inplanes),
                        scale=int(self.G_render.decoder.scale),
                    ).to(device)
                else:
                    self.prior_stem = PriorFeatureStem(
                        int(self.G_render.decoder.inplanes)
                    ).to(device)
                self.prior_stem.load_state_dict(stem_state, strict=True)
                self.prior_stem.eval()
                for parameter in self.prior_stem.parameters():
                    parameter.requires_grad = False
            if self.use_four_phase and self.prior_anchor_lambda > 0:
                decoder_state = pretrained.get("decoder")
                if decoder_state is None:
                    raise KeyError(
                        f"Checkpoint {args.pretrained_decoder!r} has no decoder weights "
                        "required by --prior_anchor_lambda."
                    )
                self.prior_decoder_ref = copy.deepcopy(self.G_render.decoder).to(device)
                self.prior_decoder_ref.load_state_dict(decoder_state, strict=True)
                self.prior_decoder_ref.eval()
                for parameter in self.prior_decoder_ref.parameters():
                    parameter.requires_grad = False
            del pretrained

        # lr scheduler & optimizer
        init_lr = conf.get_float('lr_sche.init_lr')
        step_size = conf.get_float('lr_sche.step_size') # 每 50 个 epoch 衰减一次
        gamma = conf.get_float('lr_sche.gamma')         # 每次衰减为原来的 0.5
        if getattr(args, "init_lr", None) is not None:
            if args.init_lr <= 0:
                raise ValueError("--init-lr must be positive")
            init_lr = float(args.init_lr)
        if getattr(args, "lr_step_size", None) is not None:
            if args.lr_step_size < 1:
                raise ValueError("--lr-step-size must be >= 1")
            step_size = float(args.lr_step_size)
        if getattr(args, "lr_gamma", None) is not None:
            if not 0 < args.lr_gamma <= 1:
                raise ValueError("--lr-gamma must be in (0, 1]")
            gamma = float(args.lr_gamma)
        self.init_lr = init_lr
        self.lr_step_size = step_size
        self.lr_gamma = gamma
        self.lr_decay_restart = bool(getattr(args, "lr_decay_restart", False))
        # Epoch the LR decay is counted from; updated to the resumed epoch below.
        self.lr_decay_origin = 0
        # Epoch the four-phase schedule counts from. It must exist before
        # LambdaLR is built below, because LambdaLR evaluates the lambdas once
        # during construction. Only --phase_restart moves it off 0.
        self.phase_origin = 0
        encoder_late_parameters = []
        encoder_early_parameters = []
        for name, parameter in self.G_render.encoder.named_parameters():
            destination = encoder_late_parameters if name.startswith(
                ("model.layer3.", "model.layer4.")
            ) else encoder_early_parameters
            destination.append(parameter)
        aggregator = getattr(self.G_render, "aggregator", None)
        aggregator_parameters = [] if aggregator is None else list(aggregator.parameters())
        if self.multiscale_decoder:
            optimizer_groups = [
                {"params": encoder_early_parameters, "lr": init_lr, "name": "encoder_early"},
                {"params": encoder_late_parameters, "lr": init_lr, "name": "encoder_late"},
                {"params": list(self.G_render.decoder.parameters()), "lr": init_lr, "name": "decoder_multiscale"},
            ]
        else:
            decoder_core_parameters = list(itertools.chain(
                self.G_render.decoder.in_blk.parameters(),
                self.G_render.decoder.res_blk_list.parameters(),
                self.G_render.decoder.res_blk_last.parameters(),
            ))
            decoder_up_low_parameters = list(itertools.chain.from_iterable(
                block.parameters() for block in self.G_render.decoder.up_blk_list[:-1]
            ))
            decoder_up_high_parameters = list(self.G_render.decoder.up_blk_list[-1].parameters())
            decoder_out_parameters = list(self.G_render.decoder.out_blk.parameters())
            optimizer_groups = [
            {"params": encoder_early_parameters, "lr": init_lr, "name": "encoder_early"},
            {"params": encoder_late_parameters, "lr": init_lr, "name": "encoder_late"},
            {"params": aggregator_parameters, "lr": init_lr, "name": "aggregator"},
            {"params": list(self.G_render.adapter.parameters()), "lr": init_lr, "name": "adapter"},
            {
                "params": list(self.G_render.prior_completion.parameters()),
                "lr": init_lr,
                "name": "prior_completion",
            },
            {"params": decoder_core_parameters, "lr": init_lr, "name": "decoder_core"},
            {"params": decoder_up_low_parameters, "lr": init_lr, "name": "decoder_up_low"},
            {"params": decoder_up_high_parameters, "lr": init_lr, "name": "decoder_up_high"},
            {"params": decoder_out_parameters, "lr": init_lr, "name": "decoder_out"},
            ]
        lr_lambdas = [
            (lambda epoch, name=group["name"]: self._lr_multiplier(epoch, name))
            for group in optimizer_groups
        ]
        self.G_optim = torch.optim.Adam(optimizer_groups)
        self.G_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.G_optim,
            lr_lambda=lr_lambdas,
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
            if getattr(args, "resume_reload_decoder", False):
                self._reload_pretrained_decoder()
        if getattr(args, "phase_restart", False):
            # Replay the whole A/B/C/D schedule from the resumed epoch. Every phase
            # boundary is expressed as an absolute epoch, so re-anchor the timeline
            # and recompute the Phase-D remainder against the new end point.
            self.phase_origin = self.begin_epochs
            phase_budget = sum((
                self.phase_a_epochs, self.phase_b_epochs,
                self.completion_phase_epochs, self.phase_c_epochs,
                self.phase_c_hold_epochs,
            ))
            self.phase_d_epochs = self.num_epochs - self.phase_origin - phase_budget
            if (
                self.use_four_phase
                and self.transfer_schedule == "four_phase"
                and self.phase_d_epochs <= 0
            ):
                raise ValueError(
                    "--phase_restart 之后 Phase D 没有剩余 epoch："
                    f"重放 A/B/C 需要 {phase_budget} 轮，从 epoch {self.phase_origin} 起"
                    f"至少要 --epochs > {self.phase_origin + phase_budget}。"
                )
            if not self.lr_decay_restart:
                warnings.warn(
                    "--phase_restart 重放了阶段表，但没有同时传 --lr-decay-restart，"
                    "LR 仍按从 epoch 0 起的衰减走（会低很多）。通常两者要一起用。",
                    stacklevel=2,
                )
        if self.lr_decay_restart:
            # A resumed run restarts the decay schedule, so it trains at the full
            # base LR instead of continuing from 0.5**(epoch//step_size).
            self.lr_decay_origin = self.begin_epochs
        self._apply_training_stage(self.begin_epochs)

    def _schedule_epoch(self, epoch):
        """Map an absolute epoch onto the A/B/C/D schedule timeline.

        The schedule normally starts at epoch 0. With --phase_restart a resumed run
        replays the full schedule, so the resumed epoch becomes the new origin.
        """
        return epoch - self.phase_origin

    def _training_stage(self, epoch):
        if not self.use_four_phase:
            return 0
        epoch = self._schedule_epoch(epoch)
        if self.transfer_schedule == "legacy_three_stage":
            if epoch < self.legacy_stage1_epochs:
                return 6
            if epoch < self.legacy_stage1_epochs + self.legacy_stage2_epochs:
                return 7
            return 8
        if epoch < self.phase_a_epochs:
            return 1
        if epoch < self.phase_a_epochs + self.phase_b_epochs:
            return 2
        if epoch < (
            self.phase_a_epochs + self.phase_b_epochs + self.completion_phase_epochs
        ):
            return 5
        if epoch < (
            self.phase_a_epochs
            + self.phase_b_epochs
            + self.completion_phase_epochs
            + self.phase_c_epochs
            + self.phase_c_hold_epochs
        ):
            return 3
        return 4

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        return torch.cuda.amp.autocast(dtype=torch.float16)

    def _lr_multiplier(self, epoch, group_name):
        decay_epoch = max(0, epoch - self.lr_decay_origin)
        decay = self.lr_gamma ** (decay_epoch // max(1, self.lr_step_size))
        stage = self._training_stage(epoch)
        if group_name == "adapter":
            if stage == 5:
                return 0.0
            return decay * self.adapter_lr_factor
        if group_name == "prior_completion":
            if not self.use_prior_completion:
                return 0.0
            return decay * self.completion_lr_factor
        if stage == 0:
            if self.stage0_decoder_lr_factor is None:
                return decay
            # Ordinary joint training: everything shares the base LR except the
            # (usually pretrained) decoder, which this factor slows down or freezes.
            factors = {
                "decoder_core": self.stage0_decoder_lr_factor,
                "decoder_up_low": self.stage0_decoder_lr_factor,
                "decoder_up_high": self.stage0_decoder_lr_factor,
                "decoder_out": self.stage0_decoder_lr_factor,
                "decoder_multiscale": self.stage0_decoder_lr_factor,
            }
            return decay * factors.get(group_name, 1.0)
        if stage == 1:
            factors = {
                "encoder_early": self.phase_a_encoder_lr_factor,
                "encoder_late": self.phase_a_encoder_lr_factor,
                "aggregator": self.phase_a_aggregator_lr_factor,
            }
            return decay * factors.get(group_name, 0.0)
        if stage == 2:
            factors = {
                "encoder_late": self.phase_b_encoder_lr_factor,
                "aggregator": self.phase_b_aggregator_lr_factor,
            }
            return decay * factors.get(group_name, 0.0)
        if stage == 3:
            factors = {
                "encoder_early": self.phase_c_backbone_lr_factor,
                "encoder_late": self.phase_c_backbone_lr_factor,
                "aggregator": self.phase_c_aggregator_lr_factor,
                "decoder_out": self.decoder_lr_factor,
            }
            decoder_level = self._phase_c_decoder_level(epoch)
            if decoder_level >= 2:
                factors["decoder_up_high"] = self.decoder_lr_factor
            if decoder_level >= 3:
                factors["decoder_up_low"] = self.decoder_lr_factor
            if decoder_level >= 4:
                factors["decoder_core"] = self.decoder_core_lr_factor
            return decay * factors.get(group_name, 0.0)
        if stage == 4:
            factors = {
                "encoder_early": self.phase_d_backbone_lr_factor,
                "encoder_late": self.phase_d_backbone_lr_factor,
                "aggregator": self.phase_d_aggregator_lr_factor,
                "decoder_out": self.decoder_lr_factor,
                "decoder_up_high": self.decoder_lr_factor,
                "decoder_up_low": self.decoder_lr_factor,
                "decoder_core": self.decoder_core_lr_factor,
            }
            return decay * factors.get(group_name, 0.0)
        if stage == 6:
            return 0.0
        if stage == 7:
            factors = {
                "encoder_early": 1.0,
                "encoder_late": 1.0,
                "aggregator": 1.0,
                "decoder_out": self.decoder_lr_factor,
                "decoder_up_high": self.decoder_lr_factor,
                "decoder_up_low": self.decoder_lr_factor,
                "decoder_core": self.decoder_lr_factor,
            }
            return decay * factors.get(group_name, 0.0)
        if stage == 8:
            factors = {
                "encoder_early": self.legacy_stage3_backbone_lr_factor,
                "encoder_late": self.legacy_stage3_backbone_lr_factor,
                "aggregator": self.legacy_stage3_backbone_lr_factor,
                "decoder_out": self.decoder_lr_factor,
                "decoder_up_high": self.decoder_lr_factor,
            }
            return decay * factors.get(group_name, 0.0)
        return 0.0

    @staticmethod
    def _linear_value(start, end, index, length):
        if length <= 1:
            return end
        progress = min(1.0, max(0.0, index / (length - 1)))
        return start + (end - start) * progress

    def _phase_c_decoder_level(self, epoch):
        """1=output, 2=last up block, 3=earlier up blocks, 4=residual core."""
        if self.phase_c_epochs <= 0:
            return 4
        epoch = self._schedule_epoch(epoch)
        phase_start = self.phase_a_epochs + self.phase_b_epochs
        phase_start += self.completion_phase_epochs
        progress = (epoch - phase_start) / max(1, self.phase_c_epochs)
        return min(4, max(1, int(progress * 4) + 1))

    def _latent_weight(self, epoch):
        if self.prior_stem is None or self.latent_lambda <= 0:
            return 0.0
        stage = self._training_stage(epoch)
        # The index arithmetic below is relative to the schedule origin.
        epoch = self._schedule_epoch(epoch)
        if stage == 6:
            return self.latent_lambda
        if stage == 7:
            index = epoch - self.legacy_stage1_epochs
            factor = self._linear_value(
                0.5, 0.0, index, self.legacy_stage2_epochs,
            )
            return self.latent_lambda * factor
        if stage == 8:
            return 0.0
        if stage == 1:
            return self.latent_lambda
        if stage == 2:
            index = epoch - self.phase_a_epochs
            factor = self._linear_value(
                1.0, self.phase_b_latent_end_factor, index, self.phase_b_epochs,
            )
        elif stage == 3:
            index = (
                epoch - self.phase_a_epochs - self.phase_b_epochs
                - self.completion_phase_epochs
            )
            factor = self._linear_value(
                self.phase_b_latent_end_factor,
                self.phase_c_latent_end_factor,
                index,
                self.phase_c_epochs,
            )
        elif stage == 4:
            index = (
                epoch
                - self.phase_a_epochs
                - self.phase_b_epochs
                - self.completion_phase_epochs
                - self.phase_c_epochs
                - self.phase_c_hold_epochs
            )
            factor = self._linear_value(
                self.phase_c_latent_end_factor, 0.0, index, self.phase_d_epochs,
            )
        else:
            return 0.0
        return self.latent_lambda * factor

    def _anchor_weight(self, epoch):
        if self.prior_decoder_ref is None or self.prior_anchor_lambda <= 0:
            return 0.0
        stage = self._training_stage(epoch)
        if stage == 3:
            return self.prior_anchor_lambda
        if stage == 4:
            return self.prior_anchor_lambda * self.phase_d_anchor_factor
        return 0.0

    def _completion_weight(self, epoch):
        """Decay the pCT residual teacher while keeping completion trainable."""
        if not self.use_prior_completion or self.completion_residual_lambda <= 0:
            return 0.0
        stage = self._training_stage(epoch)
        # The index arithmetic below is relative to the schedule origin.
        epoch = self._schedule_epoch(epoch)
        if stage == 6:
            return self.completion_residual_lambda
        if stage == 7:
            index = epoch - self.legacy_stage1_epochs
            factor = self._linear_value(
                1.0, 0.0, index, self.legacy_stage2_epochs,
            )
            return self.completion_residual_lambda * factor
        if stage == 8:
            return 0.0
        if stage == 0:
            return self.completion_residual_lambda
        if stage in (1, 2):
            return 0.0
        if stage == 5:
            return self.completion_residual_lambda
        if stage == 3:
            index = (
                epoch - self.phase_a_epochs - self.phase_b_epochs
                - self.completion_phase_epochs
            )
            factor = self._linear_value(
                1.0,
                self.phase_c_latent_end_factor,
                index,
                self.phase_c_epochs,
            )
            return self.completion_residual_lambda * factor
        if stage == 4:
            index = (
                epoch
                - self.phase_a_epochs
                - self.phase_b_epochs
                - self.completion_phase_epochs
                - self.phase_c_epochs
                - self.phase_c_hold_epochs
            )
            factor = self._linear_value(
                self.phase_c_latent_end_factor,
                0.0,
                index,
                self.phase_d_epochs,
            )
            return self.completion_residual_lambda * factor
        return 0.0

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
        # Architecture participation is independent of phase trainability.
        # A zero-initialized/frozen Completion is still an exact identity.
        self.G_render.completion_enabled = self.use_prior_completion

        if stage == 0:
            self._set_trainable(self.G_render, True)
            self._set_trainable(self.G_render.adapter, self.adapter_lr_factor > 0)
            if self.stage0_decoder_lr_factor is not None:
                self._set_trainable(
                    self.G_render.decoder, self.stage0_decoder_lr_factor > 0,
                )
                self.G_render.decoder.train(self.stage0_decoder_lr_factor > 0)
            if self.prior_stem is not None:
                self.prior_stem.eval()
            if self.prior_decoder_ref is not None:
                self.prior_decoder_ref.eval()
            self._apply_decoder_bn_policy()
            return

        # Start from an entirely frozen network, then enable only the groups
        # belonging to the current phase. This also prevents stale requires_grad
        # flags when crossing a phase boundary or resuming a checkpoint.
        self._set_trainable(self.G_render, False)
        self._set_trainable(
            self.G_render.adapter,
            self.G_render.use_adapter and self.adapter_lr_factor > 0,
        )
        self.G_render.encoder.eval()
        aggregator = getattr(self.G_render, "aggregator", None)
        if aggregator is not None:
            aggregator.eval()
        self.G_render.decoder.eval()
        self.G_render.adapter.train(
            self.G_render.use_adapter and self.adapter_lr_factor > 0
        )

        if stage == 5:
            self._set_trainable(self.G_render.adapter, False)
            self._set_trainable(
                self.G_render.prior_completion, self.completion_lr_factor > 0,
            )
            self.G_render.adapter.eval()
            self.G_render.prior_completion.train(self.completion_lr_factor > 0)
        elif stage == 2:
            self._set_trainable(
                self.G_render.prior_completion,
                self.use_prior_completion and self.completion_lr_factor > 0,
            )
            self.G_render.prior_completion.train(
                self.use_prior_completion and self.completion_lr_factor > 0
            )
            for layer in (self.G_render.encoder.model.layer3, self.G_render.encoder.model.layer4):
                self._set_trainable(layer, self.phase_b_encoder_lr_factor > 0)
                layer.train(self.phase_b_encoder_lr_factor > 0)
            if aggregator is not None:
                self._set_trainable(aggregator, self.phase_b_aggregator_lr_factor > 0)
                aggregator.train(self.phase_b_aggregator_lr_factor > 0)
        elif stage == 1:
            self._set_trainable(
                self.G_render.prior_completion,
                self.use_prior_completion and self.completion_lr_factor > 0,
            )
            self.G_render.prior_completion.train(
                self.use_prior_completion and self.completion_lr_factor > 0
            )
            # From-scratch runs need the backbone to learn during Phase A; with
            # the default factors of 0 this stays a pure Adapter warm-up.
            self._set_trainable(
                self.G_render.encoder, self.phase_a_encoder_lr_factor > 0,
            )
            self.G_render.encoder.train(self.phase_a_encoder_lr_factor > 0)
            if aggregator is not None:
                self._set_trainable(
                    aggregator, self.phase_a_aggregator_lr_factor > 0,
                )
                aggregator.train(self.phase_a_aggregator_lr_factor > 0)
        elif stage in (3, 4):
            self._set_trainable(
                self.G_render.prior_completion, self.completion_lr_factor > 0,
            )
            self.G_render.prior_completion.train(self.completion_lr_factor > 0)
            backbone_factor = (
                self.phase_c_backbone_lr_factor if stage == 3
                else self.phase_d_backbone_lr_factor
            )
            aggregator_factor = (
                self.phase_c_aggregator_lr_factor if stage == 3
                else self.phase_d_aggregator_lr_factor
            )
            self._set_trainable(self.G_render.encoder, backbone_factor > 0)
            self.G_render.encoder.train(backbone_factor > 0)
            if aggregator is not None:
                self._set_trainable(aggregator, aggregator_factor > 0)
                aggregator.train(aggregator_factor > 0)

            decoder_level = self._phase_c_decoder_level(epoch) if stage == 3 else 4
            decoder_modules = [self.G_render.decoder.out_blk]
            if decoder_level >= 2:
                decoder_modules.append(self.G_render.decoder.up_blk_list[-1])
            if decoder_level >= 3:
                decoder_modules.extend(self.G_render.decoder.up_blk_list[:-1])
            if decoder_level >= 4:
                decoder_modules.extend((
                    self.G_render.decoder.in_blk,
                    self.G_render.decoder.res_blk_list,
                    self.G_render.decoder.res_blk_last,
                ))
            for module in decoder_modules:
                self._set_trainable(module, True)
                module.train()
        elif stage in (6, 7, 8):
            # Legacy three-stage schedule.  Stage 1 trains only the enabled
            # bridge(s); Stage 2 jointly tunes the whole reconstruction path;
            # Stage 3 retains only the last upsampling/output decoder blocks.
            self._set_trainable(
                self.G_render.prior_completion,
                self.use_prior_completion and self.completion_lr_factor > 0,
            )
            self.G_render.prior_completion.train(
                self.use_prior_completion and self.completion_lr_factor > 0
            )

            if stage in (7, 8):
                backbone_factor = (
                    1.0 if stage == 7
                    else self.legacy_stage3_backbone_lr_factor
                )
                self._set_trainable(self.G_render.encoder, backbone_factor > 0)
                self.G_render.encoder.train(backbone_factor > 0)
                if aggregator is not None:
                    self._set_trainable(aggregator, backbone_factor > 0)
                    aggregator.train(backbone_factor > 0)

                decoder_modules = [
                    self.G_render.decoder.out_blk,
                    self.G_render.decoder.up_blk_list[-1],
                ]
                if stage == 7:
                    decoder_modules.extend((
                        self.G_render.decoder.in_blk,
                        self.G_render.decoder.res_blk_list,
                        self.G_render.decoder.res_blk_last,
                    ))
                    decoder_modules.extend(self.G_render.decoder.up_blk_list[:-1])
                for module in decoder_modules:
                    self._set_trainable(module, True)
                    module.train()

        if self.prior_stem is not None:
            self.prior_stem.eval()
        if self.prior_decoder_ref is not None:
            self.prior_decoder_ref.eval()
        self._apply_decoder_bn_policy()

    def _apply_decoder_bn_policy(self):
        """Freeze BN running stats without freezing its learnable affine terms."""
        if not self.freeze_decoder_bn_stats:
            return
        for module in self.G_render.decoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                # eval selects the stored running_mean/running_var. It does not
                # alter requires_grad on BatchNorm weight or bias.
                module.eval()

    @staticmethod
    def _normalize_latent(latent):
        mean = latent.mean(dim=(2, 3, 4), keepdim=True)
        std = latent.std(dim=(2, 3, 4), keepdim=True, unbiased=False)
        return (latent - mean) / (std + 1e-6)

    def _latent_terms(self, projection_latent, prior_latent):
        raw = F.smooth_l1_loss(projection_latent, prior_latent)
        cosine = 1.0 - F.cosine_similarity(
            projection_latent.flatten(2), prior_latent.flatten(2), dim=1,
        ).mean()
        projection_mean = projection_latent.mean(dim=(2, 3, 4))
        prior_mean = prior_latent.mean(dim=(2, 3, 4))
        projection_std = projection_latent.std(dim=(2, 3, 4), unbiased=False)
        prior_std = prior_latent.std(dim=(2, 3, 4), unbiased=False)
        mean_loss = F.l1_loss(projection_mean, prior_mean)
        std_loss = F.l1_loss(projection_std, prior_std)
        normalized = F.smooth_l1_loss(
            self._normalize_latent(projection_latent),
            self._normalize_latent(prior_latent),
        )
        combined = raw + self.latent_cosine_lambda * cosine + self.latent_stat_lambda * (
            mean_loss + std_loss
        )
        return {
            "combined": combined,
            "raw": raw,
            "cosine": cosine,
            "mean": mean_loss,
            "std": std_loss,
            "normalized_monitor": normalized,
        }

    def _decoder_anchor_loss(self, prior_latent):
        if self.prior_decoder_ref is None:
            return prior_latent.new_zeros(())
        decoder = self.G_render.decoder

        def decoder_eval_forward(latent):
            """Run the current decoder without updating BatchNorm statistics.

            Activation checkpointing calls this function once in the original
            forward and again during backward recomputation.  The temporary
            eval state must therefore live *inside* the checkpointed function;
            restoring it outside checkpoint() makes the recomputation use the
            phase's train/eval state and changes BatchNorm saved-tensor metadata.
            """
            training_states = {
                module: module.training for module in decoder.modules()
            }
            decoder.eval()
            try:
                return decoder(latent)
            finally:
                # Direct assignment restores mixed states exactly: in early
                # Phase C only selected decoder children are in train mode.
                for module, state in training_states.items():
                    module.training = state

        with self._autocast():
            if torch.is_grad_enabled():
                current = checkpoint(
                    decoder_eval_forward,
                    prior_latent,
                    use_reentrant=False,
                )
            else:
                current = decoder_eval_forward(prior_latent)
            current = self.G_render.last_layer_act(current)
            with torch.no_grad():
                reference = self.G_render.last_layer_act(
                    self.prior_decoder_ref(prior_latent)
                )
        return F.l1_loss(current, reference)

    def _multiscale_auxiliary_loss(self, volume_predict, volume_gt):
        """E5 prediction-space supervision and E6 view-logit regularization."""
        zero = volume_predict.new_zeros(())
        if not self.multiscale_decoder:
            return zero, {}
        aux = getattr(self.G_render, "last_multiscale_aux", None) or {}
        values = {}
        total = zero

        # The decoder stores predictions as [1, 1, X, Y, Z], whereas the
        # project-level volume tensors use the display orientation [Z, Y, X].
        predictions = {}
        for name in ("pred_e2", "pred_e3", "pred_e4"):
            prediction = aux.get(name)
            if prediction is not None:
                predictions[name] = prediction[0, 0].transpose(0, 2)

        if self.use_multiscale_supervision:
            for name, weight in zip(
                ("pred_e2", "pred_e3", "pred_e4"), self.multiscale_aux_weights
            ):
                prediction = predictions.get(name)
                if prediction is None or weight <= 0:
                    continue
                target = F.adaptive_avg_pool3d(
                    volume_gt[None, None], prediction.shape,
                )[0, 0]
                term = self.mse_loss(prediction, target) * weight
                total = total + term
                values[f"multiscale_{name}_loss"] = term

            # Enforce consistency in physical prediction space, not equality
            # of latent features.  The full-resolution output is the anchor.
            if self.cross_scale_lambda > 0 and "pred_e2" in predictions:
                pred2 = predictions["pred_e2"]
                consistency = self.mse_loss(
                    F.adaptive_avg_pool3d(
                        volume_predict[None, None], pred2.shape,
                    )[0, 0],
                    pred2,
                )
                if "pred_e3" in predictions:
                    pred3 = predictions["pred_e3"]
                    consistency = consistency + self.mse_loss(
                        F.adaptive_avg_pool3d(
                            pred2[None, None], pred3.shape,
                        )[0, 0],
                        pred3,
                    )
                if "pred_e4" in predictions:
                    pred4 = predictions["pred_e4"]
                    source = predictions.get("pred_e3", pred2)
                    consistency = consistency + self.mse_loss(
                        F.adaptive_avg_pool3d(
                            source[None, None], pred4.shape,
                        )[0, 0],
                        pred4,
                    )
                consistency = consistency * self.cross_scale_lambda
                total = total + consistency
                values["cross_scale_loss"] = consistency

        if self.view_weight_delta_lambda > 0:
            delta = aux.get("view_delta_l1")
            if delta is not None:
                delta_term = delta * self.view_weight_delta_lambda
                total = total + delta_term
                values["view_weight_delta_loss"] = delta_term

        for key in (
            "uncertainty_e2_mean", "uncertainty_e3_mean",
            "uncertainty_e4_mean", "uncertainty_e01_mean",
        ):
            if key in aux:
                values[key] = aux[key]
        values["multiscale_aux_loss"] = total
        return total, values

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
        if self.args.prior_encoder_type == "deep":
            # This encoder needs full-resolution pCT to construct both the
            # fixed mean channel and learned high-frequency residual channels.
            with torch.no_grad():
                with self._autocast():
                    return self.prior_stem(volume_xyz)
        low_resolution = F.avg_pool3d(
            volume_xyz,
            kernel_size=self.G_render.decoder.scale,
            stride=self.G_render.decoder.scale,
        )
        with torch.no_grad():
            with self._autocast():
                return self.prior_stem(low_resolution)

    def _latent_and_anchor_losses(self, projection_latent, volume_gt, epoch):
        if not self.use_four_phase:
            zero = projection_latent.new_zeros(())
            return {
                "latent_loss": zero,
                "latent_smooth_l1_raw": zero,
                "latent_cosine_raw": zero,
                "latent_mean_l1_raw": zero,
                "latent_std_l1_raw": zero,
                "latent_normalized_smooth_l1_raw": zero,
                "prior_anchor_raw": zero,
                "prior_anchor_loss": zero,
            }, 0.0, 0.0, None
        prior_latent = self._make_prior_latent(volume_gt)
        if prior_latent is None:
            raise RuntimeError("Four-phase training requires a loaded prior encoder")
        if projection_latent.shape != prior_latent.shape:
            raise RuntimeError(
                "Projection/prior latent shape mismatch: "
                f"{tuple(projection_latent.shape)} vs {tuple(prior_latent.shape)}. "
                "Check XYZ/ZYX ordering, prior encoder type, and decoder scale."
            )
        latent_weight = self._latent_weight(epoch)
        anchor_weight = self._anchor_weight(epoch)
        terms = self._latent_terms(projection_latent, prior_latent)
        latent_loss = terms["combined"] * latent_weight
        anchor_raw = (
            self._decoder_anchor_loss(prior_latent)
            if anchor_weight > 0 else projection_latent.new_zeros(())
        )
        anchor_loss = anchor_raw * anchor_weight
        values = {
            "latent_loss": latent_loss,
            "latent_smooth_l1_raw": terms["raw"],
            "latent_cosine_raw": terms["cosine"],
            "latent_mean_l1_raw": terms["mean"],
            "latent_std_l1_raw": terms["std"],
            "latent_normalized_smooth_l1_raw": terms["normalized_monitor"],
            "prior_anchor_raw": anchor_raw,
            "prior_anchor_loss": anchor_loss,
        }
        return values, latent_weight, anchor_weight, prior_latent

    def _completion_losses(self, prior_latent, epoch):
        """Supervise only the missing residual; never expose pCT to model input."""
        reference = self.G_render.last_aligned_latent
        prediction = self.G_render.last_completion_residual
        zero = reference.new_zeros(())
        values = {
            "completion_residual_raw": zero,
            "completion_residual_loss": zero,
            "completion_pred_abs_mean": zero,
            "completion_target_abs_mean": zero,
        }
        if (
            not self.use_prior_completion
            or not self.G_render.completion_enabled
            or prior_latent is None
        ):
            return values
        if prediction is None:
            raise RuntimeError("Completion is enabled but produced no residual")
        if reference.shape != prior_latent.shape or prediction.shape != reference.shape:
            raise RuntimeError(
                "Completion latent shape mismatch: aligned="
                f"{tuple(reference.shape)}, residual={tuple(prediction.shape)}, "
                f"prior={tuple(prior_latent.shape)}"
            )
        # Detaching both sides keeps this auxiliary target from pulling the
        # observation path. Reconstruction/reprojection losses still jointly
        # fine-tune that path in later phases.
        target = (prior_latent - reference).detach()
        raw = F.smooth_l1_loss(prediction, target)
        values["completion_residual_raw"] = raw
        values["completion_residual_loss"] = raw * self._completion_weight(epoch)
        values["completion_pred_abs_mean"] = prediction.detach().abs().mean()
        values["completion_target_abs_mean"] = target.abs().mean()
        return values
        
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

    def _reload_pretrained_decoder(self):
        """Re-inject the decoder prior on top of a resumed checkpoint.

        train.py skips --pretrained_decoder whenever --resume is set, so a resumed
        run would otherwise keep the decoder the checkpoint carries. Clearing the
        decoder Adam moments is part of the reset: they were estimated for the
        weights being overwritten.
        """
        if not self.pretrained_decoder_path:
            raise ValueError(
                "--resume_reload_decoder 需要同时传 --pretrained_decoder 指向解码器预训练权重"
            )
        pretrained = torch.load(self.pretrained_decoder_path, map_location="cpu")
        if "decoder" not in pretrained:
            raise KeyError(
                f"Decoder pretraining checkpoint {self.pretrained_decoder_path!r} "
                "has no 'decoder' key."
            )
        self.G_render.decoder.load_state_dict(pretrained["decoder"], strict=True)
        del pretrained
        cleared = 0
        for group in self.G_optim.param_groups:
            if not group.get("name", "").startswith("decoder_"):
                continue
            for parameter in group["params"]:
                if self.G_optim.state.pop(parameter, None) is not None:
                    cleared += 1
        print(
            f"Reloaded pretrained decoder from: {self.pretrained_decoder_path} "
            f"(cleared Adam state for {cleared} decoder tensors)"
        )

    def load_ckpt(self, resume_name=None):
        data = None
        if resume_name is None:
            if os.path.exists(self.latest_model_path):
                data = torch.load(self.latest_model_path, map_location=self.device)
        else:
            history_path = self.history_model_path + str(resume_name)
            if not os.path.exists(history_path):
                raise FileNotFoundError(
                    f"--resume_name {resume_name} 对应的 checkpoint 不存在: {history_path}. "
                    "先用 `ls train/checkpoints/<name>/ckpt_history/` 确认可用的 epoch；"
                    "否则会从 epoch 0 重新开始并覆盖已有日志。"
                )
            data = torch.load(history_path, map_location=self.device)
        if data is not None:
            if 'G_render' in data:
                if getattr(self.G_render, "use_adapter", False):
                    incompatible = self.G_render.load_state_dict(
                        data['G_render'], strict=False
                    )
                    allowed_missing_prefixes = ["adapter."]
                    if self.use_prior_completion:
                        allowed_missing_prefixes.append("prior_completion.")
                    invalid_missing = [
                        key for key in incompatible.missing_keys
                        if not key.startswith(tuple(allowed_missing_prefixes))
                    ]
                    if invalid_missing or incompatible.unexpected_keys:
                        raise RuntimeError(
                            "Checkpoint/model mismatch. Missing keys: "
                            f"{invalid_missing}; unexpected keys: "
                            f"{incompatible.unexpected_keys}"
                        )
                    if incompatible.missing_keys:
                        if all(
                            key.startswith("prior_completion.")
                            for key in incompatible.missing_keys
                        ):
                            warnings.warn(
                                "The resumed checkpoint predates ContinuousPriorCompletion; "
                                "completion weights keep their identity-residual initialization.",
                                stacklevel=2,
                            )
                        elif incompatible.missing_keys == ["adapter.global_alpha"]:
                            warnings.warn(
                                "The resumed checkpoint predates Transformer global alpha; "
                                "adapter.global_alpha keeps its configured initial value.",
                                stacklevel=2,
                            )
                        else:
                            warnings.warn(
                                "The resumed checkpoint predates LatentAdapter; adapter "
                                "weights were initialized as an identity mapping.",
                                stacklevel=2,
                            )
                else:
                    self.G_render.load_state_dict(data['G_render'])
            if 'iter' in data: self.begin_epochs = data['iter']
            if 'global_step' in data: self.global_step = data['global_step']
            if 'prior_stem' in data and self.prior_stem is not None:
                self.prior_stem.load_state_dict(data['prior_stem'], strict=True)
            optimizer_restored = False
            if 'G_optim' in data:
                try:
                    self.G_optim.load_state_dict(data['G_optim'])
                    optimizer_restored = True
                except ValueError:
                    warnings.warn(
                        "The checkpoint optimizer does not match the four-phase parameter groups; "
                        "model weights were restored but optimizer state was restarted.",
                        stacklevel=2,
                    )
            if 'G_lr_scheduler' in data and optimizer_restored:
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

        multiscale_aux, multiscale_aux_values = self._multiscale_auxiliary_loss(
            volume_predict, volume_gt,
        )
        G_loss += multiscale_aux
        for key, value in multiscale_aux_values.items():
            loss_dict[key] = round(value.item(), 8)

        # gd loss，梯度损失
        if self.gd1_lambda > 0:
            gd1_loss = gradient1_loss(volume_gt=volume_gt, volume_predict=volume_predict, loss_func=self.mse_loss) * self.gd1_lambda
            loss_dict['gd1_loss'] = round(gd1_loss.item(), 8)
            G_loss += gd1_loss
        else:
            loss_dict['gd1_loss'] = 0.0

        latent_values, latent_weight, anchor_weight, prior_latent = self._latent_and_anchor_losses(
            projection_latent, volume_gt, epoch,
        )
        for key, value in latent_values.items():
            loss_dict[key] = round(value.item(), 8)
        G_loss += latent_values["latent_loss"] + latent_values["prior_anchor_loss"]
        completion_values = self._completion_losses(prior_latent, epoch)
        for key, value in completion_values.items():
            loss_dict[key] = round(value.item(), 8)
        G_loss += completion_values["completion_residual_loss"]

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
            'latent_mean_l1_raw', 'latent_std_l1_raw',
            'latent_normalized_smooth_l1_raw',
            'prior_anchor_raw', 'prior_anchor_loss',
            'completion_residual_raw', 'completion_residual_loss',
            'completion_pred_abs_mean', 'completion_target_abs_mean',
            'bone_gt_mask_raw', 'bone_gt_mask_loss',
            'soft_mask_raw', 'soft_mask_loss', 'ssim_loss_raw', 'ssim_loss',
        ):
            self.writer.add_scalar(f"step/train_{key}", loss_dict[key], self.global_step)
        for key in (
            'multiscale_aux_loss', 'multiscale_pred_e2_loss',
            'multiscale_pred_e3_loss', 'multiscale_pred_e4_loss',
            'cross_scale_loss', 'view_weight_delta_loss',
            'uncertainty_e2_mean', 'uncertainty_e3_mean',
            'uncertainty_e4_mean', 'uncertainty_e01_mean',
        ):
            if key in loss_dict:
                self.writer.add_scalar(
                    f"step/train_{key}", loss_dict[key], self.global_step,
                )
        self.writer.add_scalar("step/latent_weight", latent_weight, self.global_step)
        self.writer.add_scalar("step/prior_anchor_weight", anchor_weight, self.global_step)
        self.writer.add_scalar(
            "step/completion_residual_weight",
            self._completion_weight(epoch),
            self.global_step,
        )
        self.writer.add_scalar("step/training_stage", self.current_stage, self.global_step)
        global_alpha = getattr(self.G_render.adapter, "global_alpha", None)
        if global_alpha is not None:
            self.writer.add_scalar(
                "step/adapter_global_alpha",
                global_alpha.detach(),
                self.global_step,
            )
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
        # Restore the exact mixed train/eval state of the current phase. This
        # also reapplies the optional Decoder-BN running-statistics policy.
        self._apply_training_stage(epoch)
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

        multiscale_aux, multiscale_aux_values = self._multiscale_auxiliary_loss(
            volume_predict, volume_gt,
        )
        total_loss += multiscale_aux
        for key, value in multiscale_aux_values.items():
            loss_dict[key] = round(value.item(), 8)

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

        latent_values, _, _, prior_latent = self._latent_and_anchor_losses(
            projection_latent, volume_gt, epoch,
        )
        for key, value in latent_values.items():
            loss_dict[key] = round(value.item(), 8)
        total_loss += latent_values["latent_loss"] + latent_values["prior_anchor_loss"]
        completion_values = self._completion_losses(prior_latent, epoch)
        for key, value in completion_values.items():
            loss_dict[key] = round(value.item(), 8)
        total_loss += completion_values["completion_residual_loss"]

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
            'latent_mean_l1_raw',
            'latent_std_l1_raw',
            'latent_normalized_smooth_l1_raw',
            'prior_anchor_raw',
            'prior_anchor_loss',
            'completion_residual_raw',
            'completion_residual_loss',
            'completion_pred_abs_mean',
            'completion_target_abs_mean',
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
