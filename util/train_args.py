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
    parser.add_argument("--datatype", type=str, default="dental", help="data type dental | spine | Walnuts")
    parser.add_argument("--gd1_lambda", type=float, default=1.0, help='weight for gradient loss')
    parser.add_argument("--mse_lambda_2d", type=float, default=0.01, help='weight for projection loss')  
    parser.add_argument(
        "--pretrained_decoder",
        type=str,
        default=None,
        help="Decoder-pretraining checkpoint containing decoder and feature_stem keys",
    )
    parser.add_argument(
        "--latent_lambda",
        type=float,
        default=0.0,
        help="Stage-1 weight of normalized latent alignment loss; requires --pretrained_decoder",
    )
    parser.add_argument(
        "--latent_cosine_lambda",
        type=float,
        default=0.1,
        help="Cosine term inside the latent alignment loss",
    )
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=15,
        help="Number of initial epochs with the full decoder frozen",
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=65,
        help="Number of joint-training epochs after stage 1; remaining epochs are stage 3",
    )
    parser.add_argument(
        "--decoder_lr_factor",
        type=float,
        default=0.1,
        help="Decoder learning-rate multiplier during stages 2 and 3",
    )
    parser.add_argument(
        "--stage3_backbone_lr_factor",
        type=float,
        default=0.01,
        help="Encoder/aggregator learning-rate multiplier in stage 3; set 0 to freeze them",
    )
    parser.add_argument(
        "--soft_lambda",
        type=float,
        default=0.0,
        help="Weight of the additional normalized soft-tissue-window L1 loss",
    )
    parser.add_argument("--soft_window_low", type=float, default=-160.0, help="Soft-tissue HU window lower bound")
    parser.add_argument("--soft_window_high", type=float, default=240.0, help="Soft-tissue HU window upper bound")
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
        'latent_lambda: ', str(args.latent_lambda), '\n',
        'latent_cosine_lambda: ', str(args.latent_cosine_lambda), '\n',
        'stage1_epochs: ', str(args.stage1_epochs), '\n',
        'stage2_epochs: ', str(args.stage2_epochs), '\n',
        'decoder_lr_factor: ', str(args.decoder_lr_factor), '\n',
        'stage3_backbone_lr_factor: ', str(args.stage3_backbone_lr_factor), '\n',
        'soft_lambda: ', str(args.soft_lambda), '\n',
        'soft_window: [', str(args.soft_window_low), ', ', str(args.soft_window_high), '] HU\n',
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
