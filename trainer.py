import os.path
import torch.utils.data
from models.render import *
from util.util_func import *
import datetime
from models.loss import *

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

        # epoch
        self.is_train = args.is_train
        self.num_epochs = args.epochs
        self.resume_name = args.resume_name  # specify the resume epoch
        if not self.is_train:
            self.num_epochs = self.num_epochs + 1

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

        # lr scheduler & optimizer
        init_lr = conf.get_float('lr_sche.init_lr')
        step_size = conf.get_float('lr_sche.step_size') # 每 50 个 epoch 衰减一次
        gamma = conf.get_float('lr_sche.gamma')         # 每次衰减为原来的 0.5
        self.G_optim = torch.optim.Adam(self.G_render.parameters(), lr=init_lr, )   # Adam 优化器，只优化 G_render（编码器 + 聚合器 + 3D 解码器）的全部参数
        self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.G_optim, step_size=step_size,
                                                              gamma=gamma)  # StepLR，它是"按调用 step() 的次数"来衰减的。在 start() 里每个 epoch 结束调用一次 self.G_lr_scheduler.step()

        # loss
        self.mse_loss = torch.nn.L1Loss(reduction='mean')

        # load weights & optimizer & iterator
        self.begin_epochs = 0
        os.makedirs("%s/ckpt_history" % (self.checkpoints_path,), exist_ok=True)
        self.latest_model_path = "%s/ckpt_latest" % (self.checkpoints_path,)        # 永远覆盖写"最新"权重
        self.history_model_path = "%s/ckpt_history/ckpt_" % (self.checkpoints_path,)# 按 epoch 归档的历史权重
        if args.resume: 
            self.load_ckpt(self.resume_name)    # 断电加载
        
    def save_ckpt(self, epoch):
        data = {
            'iter': epoch + 1,
            'G_render': self.G_render.state_dict(),
            'G_optim': self.G_optim.state_dict(),
            'G_lr_scheduler': self.G_lr_scheduler.state_dict(),
        }
        torch.save(data, self.latest_model_path)
        if (epoch % self.save_interval == 0) or epoch == self.num_epochs - 1:
            torch.save(data, self.history_model_path + str(epoch))

    def load_ckpt(self, resume_name=None):
        data = None
        if resume_name is None:
            if os.path.exists(self.latest_model_path):
                data = torch.load(self.latest_model_path, map_location=self.device)
        else:
            if os.path.exists(os.path.join(self.history_model_path, resume_name)):
                data = torch.load(os.path.join(self.history_model_path, resume_name), map_location=self.device)
        if data is not None:
            if 'G_render' in data: self.G_render.load_state_dict(data['G_render'])
            if 'iter' in data: self.begin_epochs = data['iter']
            if 'G_optim' in data: self.G_optim.load_state_dict(data['G_optim'])
            if 'G_lr_scheduler' in data: self.G_lr_scheduler.load_state_dict(data['G_lr_scheduler'])

    def train_step(self, data,):
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
        self.G_render.encoder(src_images, src_poses)    # model的ResEncoder
        # 3d volume decoding
        volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                           volume_origin=volume_origin, volume_phy=volume_phy,
                                           scale=self.G_render.decoder.scale, device=device)                    # 获得反投影后的体素空间表示

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

        # 2d ray batch loss
        if self.mse_lambda_2d > 0:
            pix_inds = torch.randint(0, self.args.nviews * H * W, (self.ray_batch_size,))
            images_gt_all = src_images.reshape(-1, 1)
            proj_gt = images_gt_all[pix_inds]
            src_rays = get_rays(src_poses, H, W)
            proj_rays = src_rays.view(-1, src_rays.shape[-1])[pix_inds].to(device=device)
            proj_predict = composite(rays=proj_rays, volume=volume_predict, volume_origin=volume_origin,
                                        volume_phy=volume_phy, render_step_size=render_step_size, 
                                        chunksize=self.chunksize).reshape(proj_gt.shape)
            if self.expnorm:
                proj_predict = torch.exp(-proj_predict/self.divide)
            mse_loss_2d = self.mse_loss(proj_predict, proj_gt) * self.mse_lambda_2d
            loss_dict['mse_loss_2d'] = round(mse_loss_2d.item(), 8)
            G_loss += mse_loss_2d

        loss_dict['G_loss'] = round(G_loss.item(), 8)

        # update model
        G_loss.backward()
        self.G_optim.step()

        # first set G_render to eval state, calculate the PSNR, and turn it back to train state
        self.G_render.eval()
        with torch.no_grad():
            self.G_render.encoder(src_images, src_poses)
            volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                               volume_origin=volume_origin, volume_phy=volume_phy,
                                               scale=self.G_render.decoder.scale, device=device)
            volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
            # 3d ssim calculation is too slow, so we only calculate psnr
            loss_dict['psnr_3d_clamp'] = round(get_psnr(data_norm(volume_predict_clamp), data_norm(volume_gt)), 8)
        self.G_render.train()
        return loss_dict

    def test_step(self, data):
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
        self.G_render.encoder(src_images, src_poses)
        # 3d volume decoding
        volume_predict = predict_3d_volume(model=self.G_render, volume_resolution=volume_resolution,
                                           volume_origin=volume_origin, volume_phy=volume_phy,
                                           scale=self.G_render.decoder.scale, device=device)
        
        # metrics calculation
        volume_predict_clamp = torch.clamp(volume_predict, self.clamp_min, self.clamp_max)
        loss_dict = {}
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

    def start(self):
        
        if self.is_train:
            for epoch in range(self.begin_epochs, self.num_epochs):

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
                for train_data in self.train_data_loader:
                    train_losses = self.train_step(train_data)
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

                # epoch psnr
                train_psnr_3d_clamp = train_psnr_3d_clamp / train_batch
                now = datetime.datetime.now()
                f_train_psnr = open(self.logs_path + '/train_metric.txt', mode='a')
                f_train_psnr.write(
                    now.strftime('%Y-%m-%d %H:%M:%S') + ' Epoch:' + str(epoch) + ' train_psnr_3d_clamp:' + str(train_psnr_3d_clamp) + '\n')
                f_train_psnr.close()
                print("*** train:", now.strftime('%Y-%m-%d %H:%M:%S'), "Epoch:", epoch, 'train_psnr_3d_clamp:', str(train_psnr_3d_clamp))

                # network saving
                print("saving network & optimizer")
                self.save_ckpt(epoch)
                
                # validate with the val dataset
                if ((epoch % self.val_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    print('Network validating')
                    val_batch = 0
                    val_psnr_3d_clamp = 0
                    val_ssim_3d_clamp = 0
                    for val_data in self.val_data_loader:
                        self.G_render.eval()
                        with torch.no_grad():
                            val_losses = self.test_step(val_data)
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

                # test with the test dataset
                if ((epoch % self.test_interval == 0) and (epoch > 0)) or epoch == self.num_epochs - 1:
                    print('Network Testing')
                    test_batch = 0
                    test_psnr_3d_clamp = 0
                    test_ssim_3d_clamp = 0
                    for test_data in self.test_data_loader:
                        self.G_render.eval()
                        with torch.no_grad():
                            test_losses = self.test_step(test_data)
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
