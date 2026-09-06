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
    parser.add_argument("--name", "-n", type=str, default='SVCT_eval', help="experiment name")
    parser.add_argument("--logs_path", type=str, default="evaluate/logs", help="logs output directory",)
    parser.add_argument("--checkpoints_path",type=str,default="train/checkpoints",help="checkpoints resume directory",)
    parser.add_argument("--train_scale", type=int, default=4, help="set downsampling scale manually during training stage")
    parser.add_argument("--eval_scale", type=int, default=-1, help="set downsampling scale manually during evaluation stage")
    parser.add_argument("--fusion", type=str, default='ada', help="multi-view feature fusing strategy")
    parser.add_argument("--visual_path", type=str, default="evaluate/visuals", help="visualization output directory", )
    parser.add_argument("--conf", "-c", type=str, default='conf/evaluate.conf', help='Config file')
    parser.add_argument("--datadir", "-D", type=str, default='dataset/dental/syn_data', help="Dataset directory")
    parser.add_argument("--device", type=str, default='cuda:0', help='compute device')
    parser.add_argument("--resume_name", type=str, default=None, help='resume which trained net for evaluate')
    parser.add_argument(
        "--use_adapter", action="store_true",
        help="Build the model with the latent adapter used during training",
    )
    parser.add_argument(
        "--adapter_hidden_channels", type=int, default=64,
        help="Bottleneck channels in the latent adapter",
    )
    parser.add_argument("--dataname", type=str, default='test', help="evaluate dataname") 
    parser.add_argument("--datatype", type=str, default="dental", help="data type dental | spine | Walnuts")
    parser.add_argument(
        "--bone_lambda", type=float, default=0.0,
        help="Report weighted GT-defined bone-mask L1 (0 disables this loss report)",
    )
    parser.add_argument("--bone_lower_hu", type=float, default=300.0, help="GT HU lower bound for bone mask")
    parser.add_argument(
        "--soft_mask_lambda", type=float, default=0.0,
        help="Report weighted GT-defined soft-tissue-mask L1 (0 disables this loss report)",
    )
    parser.add_argument("--soft_window_low", type=float, default=-160.0, help="Soft-tissue HU window lower bound")
    parser.add_argument("--soft_window_high", type=float, default=240.0, help="Soft-tissue HU window upper bound")
    parser.add_argument(
        "--ssim_lambda", type=float, default=0.0,
        help="Report weighted differentiable local 3-D (1-SSIM) loss (0 disables this loss report)",
    )
    parser.add_argument(
        "--query_chunk_size",
        type=int,
        default=25000,
        help="Number of 3D points processed per backprojection/view-fusion chunk",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision during evaluation",
    )

    args = parser.parse_args()

    if min(args.bone_lambda, args.soft_mask_lambda, args.ssim_lambda) < 0:
        parser.error("bone_lambda, soft_mask_lambda, and ssim_lambda must be non-negative")
    if args.soft_window_high <= args.soft_window_low:
        parser.error("soft_window_high must be greater than soft_window_low")

    conf = ConfigFactory.parse_file(args.conf)
    if args.train_scale!=0:
        conf.put("model.SRGAN.generator.scale", args.train_scale)
    if args.eval_scale==-1:
        args.eval_scale = args.train_scale  # if not set, use the same scale as training
    conf.put("model.fusion", args.fusion)
    
    now = datetime.datetime.now()
    exp_state_list = [now.strftime('%Y-%m-%d %H:%M:%S'), '\n'
                     'Exp name: ' , args.name , '\n' ,
                     'resume name: ', str(args.resume_name), '\n',
                     'dataname: ', args.dataname, '\n', 
                     'config file: ' , args.conf , '\n' ,
                     'Dataset: ' , args.datadir , '\n' ,
                     'datatype: ', args.datatype, '\n' ,
                     'start scanning angle: ', str(args.start), '\n',
                     'end scanning angle: ', str(args.end), '\n',
                     'input views: ' , str(args.nviews) , '\n',
                     'angle sampling: ', args.angle_sampling, '\n',
                     'expnorm: ', "yes" if args.expnorm else "no", '\n',
                     'scale: ' , str(conf['model.SRGAN.generator.scale']) , '\n',
                     'inplanes: ' , str(conf['model.SRGAN.generator.inplanes']) , '\n' ,
                     'fusion strategy: ' , conf['model.fusion'] , '\n',]

    exp_state_list.extend([
        'bone_lambda: ', str(args.bone_lambda), '\n',
        'bone_lower_hu: ', str(args.bone_lower_hu), '\n',
        'soft_mask_lambda: ', str(args.soft_mask_lambda), '\n',
        'soft_window: [', str(args.soft_window_low), ', ', str(args.soft_window_high), '] HU\n',
        'ssim_lambda: ', str(args.ssim_lambda), '\n',
        'use_adapter: ', "yes" if args.use_adapter else "no", '\n',
        'adapter_hidden_channels: ', str(args.adapter_hidden_channels), '\n',
    ])

    exp_state = ''.join(exp_state_list)
    print(exp_state)
    logs_path = os.path.join(args.logs_path, args.name)
    os.makedirs(logs_path, exist_ok=True)
    f_exp = open(logs_path + '/exp_state.txt', mode='a')
    f_exp.write(exp_state)
    f_exp.close()

    return args,conf
