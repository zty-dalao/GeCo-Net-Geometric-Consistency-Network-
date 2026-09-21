import os
import argparse
from pyhocon import ConfigFactory
import datetime
def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-B", type=int, default=1, help="Object batch size | right now we only support 1 batch")
    parser.add_argument("--start", type=int, default=0, help="start scanning angle")  # it is recommended to use integer angle
    parser.add_argument("--end", type=int, default=360, help="end scanning angle")
    parser.add_argument("--nviews", "-V", type=int, default=20, help="Number of selected views",)
    parser.add_argument("--angle_sampling", type=str, default="uniform", help="angle sampling strategy | uniform | random")
    parser.add_argument("--expnorm", action="store_false", help="Whether to use exponential projection normalization") 
    parser.add_argument("--train_scale", type=int, default=4, help="set downsampling scale manually during training stage")
    parser.add_argument("--fusion", type=str, default='ada', help="multi-view feature fusing strategy")
    parser.add_argument("--name", "-n", type=str, default='SVCT_train', help="experiment name")
    parser.add_argument("--logs_path", type=str, default="train/logs", help="logs output directory",)
    parser.add_argument("--checkpoints_path",type=str,default="train/checkpoints",help="checkpoints output directory",)
    parser.add_argument("--visual_path",type=str,default="train/visuals",help="visualization output directory",)
    parser.add_argument("--epochs",type=int,default=500,help="number of epochs to train",)
    parser.add_argument("--datadir", "-D", type=str, default='dataset/dental/syn_data', help="Dataset directory")
    parser.add_argument("--conf", "-c", type=str, default='conf/train.conf', help='Config file')
    parser.add_argument("--device", type=str, default='cuda', help='compute device')
    parser.add_argument("--is_train", action="store_true", help="Training or visualization")
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument("--resume_name", type=str, default=None, help='resume which trained net for continue training')
    parser.add_argument(
        "--init-lr",
        type=float,
        default=None,
        help=(
            "Override conf/train.conf lr_sche.init_lr. Use it to raise the LR when "
            "resuming; without it the main trainer has no CLI control over the base LR."
        ),
    )
    parser.add_argument(
        "--lr-step-size",
        type=float,
        default=None,
        help="Override conf/train.conf lr_sche.step_size (epochs between LR halvings).",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=None,
        help="Override conf/train.conf lr_sche.gamma (LR decay factor).",
    )
    parser.add_argument(
        "--lr-decay-restart",
        action="store_true",
        help=(
            "Count the LR decay from the resumed epoch instead of epoch 0, so a resumed "
            "run starts again at the full base LR. Without it, epoch//step_size keeps "
            "decaying across the resume and epoch 199 runs at 1/8 of init_lr."
        ),
    )
    parser.add_argument("--datatype", type=str, default="dental", help="data type dental | spine | thorax | Walnuts")
    parser.add_argument(
        "--require-gt-source",
        choices=("cbct", "ct", "registered-ct", "cbct-fixed"),
        default=None,
        help=(
            "Fail before training unless every case's transforms.json records this "
            "gt_source for the gt_volume.nii.gz label volume. Pass 'registered-ct' to "
            "guarantee the 3-D labels are the registered pCT and not the CBCT."
        ),
    )
    parser.add_argument("--gd1_lambda", type=float, default=1.0, help='weight for gradient loss')
    parser.add_argument("--mse_lambda_2d", type=float, default=0.01, help='weight for projection loss')  
    parser.add_argument(
        "--pretrained_decoder",
        type=str,
        default=None,
        help="Decoder-pretraining checkpoint containing decoder and feature_stem keys",
    )
    parser.add_argument(
        "--prior_encoder_type",
        choices=("shallow", "deep"),
        default="shallow",
        help="Feature-stem architecture stored in the pretrained decoder checkpoint",
    )
    parser.add_argument(
        "--pretrained_backbone",
        type=str,
        default=None,
        help="Optional main-model checkpoint; load only encoder/aggregator weights",
    )
    parser.add_argument(
        "--use_adapter",
        action="store_true",
        help="Insert a residual latent adapter between the aggregator and decoder",
    )
    parser.add_argument(
        "--adapter_hidden_channels",
        type=int,
        default=64,
        help="Bottleneck channels in the latent adapter",
    )
    parser.add_argument(
        "--adapter_type",
        choices=("cnn", "transformer"),
        default="cnn",
        help="Latent adapter architecture",
    )
    parser.add_argument("--adapter_transformer_pool_size", type=int, default=8)
    parser.add_argument("--adapter_transformer_layers", type=int, default=2)
    parser.add_argument("--adapter_transformer_heads", type=int, default=4)
    parser.add_argument("--adapter_transformer_dropout", type=float, default=0.1)
    parser.add_argument(
        "--adapter_use_global_alpha",
        action="store_true",
        help="Gate the Transformer global feature with a learnable scalar alpha",
    )
    parser.add_argument(
        "--adapter_global_alpha_init",
        type=float,
        default=0.0,
        help="Initial value of the optional Transformer global-feature alpha",
    )
    parser.add_argument(
        "--freeze_decoder_bn_stats",
        action="store_true",
        help="Keep Decoder BatchNorm running mean/variance fixed while training",
    )
    parser.add_argument(
        "--transfer_schedule",
        choices=("four_phase", "legacy_three_stage"),
        default="four_phase",
        help=(
            "Prior-transfer schedule. legacy_three_stage reproduces the old "
            "adapter experiment's freeze/joint/terminal-refinement sequence."
        ),
    )
    parser.add_argument("--legacy_stage1_epochs", type=int, default=20)
    parser.add_argument("--legacy_stage2_epochs", type=int, default=100)
    parser.add_argument(
        "--legacy_stage3_backbone_lr_factor",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--adapter_lr_factor",
        type=float,
        default=1.0,
        help="Adapter learning-rate multiplier relative to the base learning rate",
    )
    parser.add_argument(
        "--stage0_decoder_lr_factor",
        type=float,
        default=None,
        help=(
            "Optional decoder LR multiplier for ordinary joint training (the schedule that "
            "runs when neither --use_adapter nor --use_prior_completion is set). Without it "
            "every group uses the base LR, so a pretrained decoder is fine-tuned at full LR "
            "from epoch 0. Set 0 to freeze the decoder, or e.g. 0.1 to fine-tune it slowly."
        ),
    )
    parser.add_argument(
        "--use_prior_completion",
        action="store_true",
        help="Enable geometry-conditioned continuous latent residual completion",
    )
    parser.add_argument("--completion_hidden_channels", type=int, default=16)
    parser.add_argument("--completion_geometry_hidden_channels", type=int, default=32)
    parser.add_argument("--completion_geometry_channels", type=int, default=64)
    parser.add_argument("--completion_residual_scale", type=float, default=1.0)
    parser.add_argument(
        "--completion_phase_epochs",
        type=int,
        default=40,
        help="Completion-only epochs inserted between Phase B and Phase C",
    )
    parser.add_argument(
        "--completion_lr_factor",
        type=float,
        default=1.0,
        help="Completion learning-rate multiplier relative to the base learning rate",
    )
    parser.add_argument(
        "--completion_residual_lambda",
        type=float,
        default=0.1,
        help="Weight for SmoothL1(delta_pred, stopgrad(z_prior-z_aligned))",
    )
    parser.add_argument(
        "--disable_completion_checkpoint",
        action="store_true",
        help="Disable activation recomputation inside completion residual blocks",
    )
    parser.add_argument(
        "--latent_lambda",
        type=float,
        default=0.0,
        help="Initial weight of raw/cosine/statistical latent alignment; requires --pretrained_decoder",
    )
    parser.add_argument(
        "--latent_cosine_lambda",
        type=float,
        default=0.1,
        help="Cosine term inside the latent alignment loss",
    )
    parser.add_argument(
        "--latent_stat_lambda",
        type=float,
        default=0.1,
        help="Weight of per-channel latent mean/std alignment terms",
    )
    parser.add_argument("--phase_a_epochs", type=int, default=20)
    parser.add_argument("--phase_b_epochs", type=int, default=40)
    parser.add_argument("--phase_c_epochs", type=int, default=80)
    parser.add_argument(
        "--phase_c_hold_epochs",
        type=int,
        default=0,
        help="Extra Phase-C epochs after progressive decoder unfreezing; decoder remains fully unfrozen",
    )
    parser.add_argument(
        "--phase_a_encoder_lr_factor",
        type=float,
        default=0.0,
        help=(
            "Phase-A encoder LR multiplier. Default 0 keeps Phase A adapter-only, "
            "which requires --pretrained_backbone; set >0 to train the randomly "
            "initialized Encoder from scratch in Phase A."
        ),
    )
    parser.add_argument(
        "--phase_a_aggregator_lr_factor",
        type=float,
        default=0.0,
        help=(
            "Phase-A Aggregator LR multiplier. See --phase_a_encoder_lr_factor. "
            "Default 0 keeps Phase A adapter-only."
        ),
    )
    parser.add_argument("--phase_b_encoder_lr_factor", type=float, default=0.2)
    parser.add_argument("--phase_b_aggregator_lr_factor", type=float, default=0.5)
    parser.add_argument("--phase_c_backbone_lr_factor", type=float, default=0.1)
    parser.add_argument("--phase_c_aggregator_lr_factor", type=float, default=0.3)
    parser.add_argument("--phase_d_backbone_lr_factor", type=float, default=0.01)
    parser.add_argument("--phase_d_aggregator_lr_factor", type=float, default=0.05)
    parser.add_argument("--decoder_core_lr_factor", type=float, default=0.01)
    parser.add_argument(
        "--prior_anchor_lambda",
        type=float,
        default=0.1,
        help="Phase-C prior-decoder output preservation weight",
    )
    parser.add_argument(
        "--phase_d_anchor_factor",
        type=float,
        default=0.25,
        help="Phase-D anchor weight relative to prior_anchor_lambda",
    )
    parser.add_argument(
        "--phase_b_latent_end_factor",
        type=float,
        default=0.7,
        help="Latent-weight factor reached at the end of phase B",
    )
    parser.add_argument(
        "--phase_c_latent_end_factor",
        type=float,
        default=0.1,
        help="Latent-weight factor reached at the end of phase C; phase D decays it to zero",
    )
    parser.add_argument(
        "--decoder_lr_factor",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for progressively unfrozen decoder output/upsampling blocks",
    )
    parser.add_argument(
        "--bone_lambda",
        type=float,
        default=0.0,
        help="Weight of the GT-defined bone-region normalized L1 loss; 0 disables it",
    )
    parser.add_argument(
        "--bone_lower_hu",
        type=float,
        default=300.0,
        help="GT HU lower bound for the bone mask",
    )
    parser.add_argument(
        "--soft_mask_lambda",
        type=float,
        default=0.0,
        help="Weight of the GT-defined soft-tissue-mask normalized L1 loss; 0 disables it",
    )
    parser.add_argument("--soft_window_low", type=float, default=-160.0, help="Soft-tissue HU window lower bound")
    parser.add_argument("--soft_window_high", type=float, default=240.0, help="Soft-tissue HU window upper bound")
    parser.add_argument(
        "--ssim_lambda",
        type=float,
        default=0.0,
        help="Weight of differentiable fixed-range local 3-D (1-SSIM) loss; 0 disables it",
    )
    parser.add_argument(
        "--query_chunk_size",
        type=int,
        default=25000,
        help="Number of 3D points processed per backprojection/view-fusion chunk",
    )
    parser.add_argument(
        "--disable_query_checkpoint",
        action="store_true",
        help="Disable activation recomputation for backprojection/view fusion (uses more GPU memory)",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision (uses substantially more GPU memory)",
    )

    args = parser.parse_args()

    if args.use_prior_completion and not args.pretrained_decoder:
        parser.error(
            "--use_prior_completion requires --pretrained_decoder to construct "
            "the pCT latent teacher (also when resuming)"
        )
    if min(
        args.completion_hidden_channels,
        args.completion_geometry_hidden_channels,
        args.completion_geometry_channels,
    ) <= 0:
        parser.error("completion channel counts must be positive")
    if args.completion_phase_epochs < 0:
        parser.error("--completion_phase_epochs must be non-negative")
    if min(args.legacy_stage1_epochs, args.legacy_stage2_epochs) < 0:
        parser.error("legacy three-stage epoch counts must be non-negative")
    if (
        args.transfer_schedule == "legacy_three_stage"
        and args.legacy_stage1_epochs + args.legacy_stage2_epochs >= args.epochs
    ):
        parser.error(
            "legacy_stage1_epochs + legacy_stage2_epochs must leave at least "
            "one epoch for legacy Stage 3"
        )
    if args.legacy_stage3_backbone_lr_factor < 0:
        parser.error("--legacy_stage3_backbone_lr_factor must be non-negative")
    if min(
        args.completion_lr_factor,
        args.completion_residual_lambda,
        args.completion_residual_scale,
    ) < 0:
        parser.error("completion LR/loss factors must be non-negative")

    conf = ConfigFactory.parse_file(args.conf)
    if args.train_scale!=0:
        conf.put("model.SRGAN.generator.scale", args.train_scale)   # 如果命令行指定了 --train_scale（且不为0），则覆盖配置文件里生成器（Generator）的上采样倍数
    conf.put("model.fusion", args.fusion)
    conf.put("train.G_loss.gd1_lambda", args.gd1_lambda)
    conf.put("train.G_loss.mse_lambda_2d", args.mse_lambda_2d)

    now = datetime.datetime.now()
    exp_state_list = [now.strftime('%Y-%m-%d %H:%M:%S'), '\n'
                     'Exp name: ' , args.name , '\n' ,
                     'Training or not: ' , "yes" if args.is_train else "no" , '\n' ,
                     'Resume: ' , "yes" if args.resume else "no" , '\n' ,
                     'resume name: ', str(args.resume_name), '\n',
                     'config file: ' , args.conf , '\n' ,
                     'Dataset: ' , args.datadir , '\n' ,
                     'datatype: ', args.datatype, '\n' ,
                     'require_gt_source: ', str(args.require_gt_source), '\n' ,
                     'init_lr: ', str(args.init_lr), '\n' ,
                     'lr_step_size: ', str(args.lr_step_size), '\n' ,
                     'lr_gamma: ', str(args.lr_gamma), '\n' ,
                     'lr_decay_restart: ', "yes" if args.lr_decay_restart else "no", '\n' ,
                     'stage0_decoder_lr_factor: ', str(args.stage0_decoder_lr_factor), '\n' ,
                     'start scanning angle: ', str(args.start), '\n',
                     'end scanning angle: ', str(args.end), '\n',
                     'input views: ' , str(args.nviews) , '\n',
                     'angle sampling: ', args.angle_sampling, '\n',
                     'expnorm: ', "yes" if args.expnorm else "no", '\n',
                     'ray_batch_size: ', str(conf['render.ray_batch_size']), '\n',
                     'factor: ', str(conf['render.factor']), '\n' ,
                     'scale: ' , str(conf['model.SRGAN.generator.scale']) , '\n',
                     'fusion: ', str(conf['model.fusion']), '\n',
                     'inplanes: ' , str(conf['model.SRGAN.generator.inplanes']) , '\n' ,
                     'mse_lambda_2d: ', str(conf['train.G_loss.mse_lambda_2d']), '\n',
                     'mse_lambda_3d: ', str(conf['train.G_loss.mse_lambda_3d']), '\n',
                     'gd1_lambda: ', str(conf['train.G_loss.gd1_lambda']), '\n']

    exp_state_list.extend([
        'pretrained_decoder: ', str(args.pretrained_decoder), '\n',
        'prior_encoder_type: ', str(args.prior_encoder_type), '\n',
        'pretrained_backbone: ', str(args.pretrained_backbone), '\n',
        'use_adapter: ', "yes" if args.use_adapter else "no", '\n',
        'adapter_hidden_channels: ', str(args.adapter_hidden_channels), '\n',
        'adapter_type: ', str(args.adapter_type), '\n',
        'adapter_transformer_pool_size: ', str(args.adapter_transformer_pool_size), '\n',
        'adapter_transformer_layers: ', str(args.adapter_transformer_layers), '\n',
        'adapter_transformer_heads: ', str(args.adapter_transformer_heads), '\n',
        'adapter_transformer_dropout: ', str(args.adapter_transformer_dropout), '\n',
        'adapter_use_global_alpha: ', "yes" if args.adapter_use_global_alpha else "no", '\n',
        'adapter_global_alpha_init: ', str(args.adapter_global_alpha_init), '\n',
        'freeze_decoder_bn_stats: ', "yes" if args.freeze_decoder_bn_stats else "no", '\n',
        'transfer_schedule: ', str(args.transfer_schedule), '\n',
        'legacy_stage1_epochs: ', str(args.legacy_stage1_epochs), '\n',
        'legacy_stage2_epochs: ', str(args.legacy_stage2_epochs), '\n',
        'legacy_stage3_backbone_lr_factor: ',
        str(args.legacy_stage3_backbone_lr_factor), '\n',
        'adapter_lr_factor: ', str(args.adapter_lr_factor), '\n',
        'use_prior_completion: ', "yes" if args.use_prior_completion else "no", '\n',
        'completion_hidden_channels: ', str(args.completion_hidden_channels), '\n',
        'completion_geometry_hidden_channels: ', str(args.completion_geometry_hidden_channels), '\n',
        'completion_geometry_channels: ', str(args.completion_geometry_channels), '\n',
        'completion_residual_scale: ', str(args.completion_residual_scale), '\n',
        'completion_phase_epochs: ', str(args.completion_phase_epochs), '\n',
        'completion_lr_factor: ', str(args.completion_lr_factor), '\n',
        'completion_residual_lambda: ', str(args.completion_residual_lambda), '\n',
        'completion_checkpoint: ', "no" if args.disable_completion_checkpoint else "yes", '\n',
        'latent_lambda: ', str(args.latent_lambda), '\n',
        'latent_cosine_lambda: ', str(args.latent_cosine_lambda), '\n',
        'latent_stat_lambda: ', str(args.latent_stat_lambda), '\n',
        'phase_epochs[A,B,Completion,C-progressive,C-hold,D]: [', str(args.phase_a_epochs), ', ',
        str(args.phase_b_epochs), ', ',
        str(args.completion_phase_epochs if args.use_prior_completion else 0), ', ',
        str(args.phase_c_epochs), ', ',
        str(args.phase_c_hold_epochs), ', ',
        str(max(0, args.epochs - args.phase_a_epochs - args.phase_b_epochs
                - (args.completion_phase_epochs if args.use_prior_completion else 0)
                - args.phase_c_epochs - args.phase_c_hold_epochs)), ']\n',
        'phase_a_encoder_lr_factor: ', str(args.phase_a_encoder_lr_factor), '\n',
        'phase_a_aggregator_lr_factor: ', str(args.phase_a_aggregator_lr_factor), '\n',
        'phase_b_encoder_lr_factor: ', str(args.phase_b_encoder_lr_factor), '\n',
        'phase_b_aggregator_lr_factor: ', str(args.phase_b_aggregator_lr_factor), '\n',
        'phase_c_backbone_lr_factor: ', str(args.phase_c_backbone_lr_factor), '\n',
        'phase_c_aggregator_lr_factor: ', str(args.phase_c_aggregator_lr_factor), '\n',
        'phase_d_backbone_lr_factor: ', str(args.phase_d_backbone_lr_factor), '\n',
        'phase_d_aggregator_lr_factor: ', str(args.phase_d_aggregator_lr_factor), '\n',
        'decoder_core_lr_factor: ', str(args.decoder_core_lr_factor), '\n',
        'prior_anchor_lambda: ', str(args.prior_anchor_lambda), '\n',
        'phase_d_anchor_factor: ', str(args.phase_d_anchor_factor), '\n',
        'phase_b_latent_end_factor: ', str(args.phase_b_latent_end_factor), '\n',
        'phase_c_latent_end_factor: ', str(args.phase_c_latent_end_factor), '\n',
        'decoder_lr_factor: ', str(args.decoder_lr_factor), '\n',
        'bone_lambda: ', str(args.bone_lambda), '\n',
        'bone_lower_hu: ', str(args.bone_lower_hu), '\n',
        'soft_mask_lambda: ', str(args.soft_mask_lambda), '\n',
        'soft_window: [', str(args.soft_window_low), ', ', str(args.soft_window_high), '] HU\n',
        'ssim_lambda: ', str(args.ssim_lambda), '\n',
        'query_chunk_size: ', str(args.query_chunk_size), '\n',
        'query_checkpoint: ', "no" if args.disable_query_checkpoint else "yes", '\n',
        'amp: ', "no" if args.no_amp else "yes", '\n',
    ])

    exp_state = ''.join(exp_state_list) # 拼接并打印到终端（''.join(exp_state_list)）
    print(exp_state)
    logs_path = os.path.join(args.logs_path, args.name)
    os.makedirs(logs_path, exist_ok=True)   # 创建日志文件夹
    f_exp = open(logs_path + '/exp_state.txt', mode='a')    # 最终文件会存放在 ./logs/你的实验名称/exp_state.txt。并且以追加的方式
    f_exp.write(exp_state)
    f_exp.close()

    return args,conf    # conf 对象此时存储的已经是被 args 覆盖之后的最终生效参数
