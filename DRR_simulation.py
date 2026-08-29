import numpy as np
import SimpleITK as sitk
import os
import json
from models.render import ct2mu, angle2vec, get_rays, composite
from tqdm import tqdm
import argparse
import torch

def GeometryProduction(args, niipath, projpath):
    """
    Projection Geometry Configuration File Production
    """

    start, end, num = args.start, args.end, args.num
    sad, sid = args.sad, args.sid

    # default projection resolution
    proj_resolution = [512, 512]

    os.makedirs(projpath,exist_ok=True)
    path_list = os.listdir(niipath)     # 列出当前niipath目录下所有的文件名：['X2313838.nii.gz',......,'X2612041.nii.gz']

    for file_cur in path_list:                              # file_cur是单个文件名，如X2313838.nii.gz
        image_path = os.path.join(niipath,file_cur)         # 文件夹的路径+当前文件名，获得完整的路径
        file_name = file_cur[0:-7]                          # 抛弃.nii.gz，只取X2313838
        output_path = os.path.join(projpath,file_name)      # 生成输出针对当前文件image域的投影域输出路径

        image = sitk.ReadImage(image_path)
        volume_resolution = np.asarray(image.GetSize())     # .GetSize()获取尺寸例如 (512, 512, 150)  -> (X, Y, Z)。np.asarray，转成npy数组
        volume_spacing = np.asarray(image.GetSpacing())     # 它表示图像中每一个体素（Voxel）在真实三维物理世界中对应的实际大小。(0.5, 0.5, 1.0)表示（x，y，z）方向上的间距
        volume_phy = volume_spacing * (volume_resolution)   # 获得物理距离
        isocenter = np.asarray([0, 0, 0])                   # X射线源（Source）和平板探测器（Detector）围绕其进行圆弧旋转运动的旋转中心轴。这个二者的交点设为世界坐标原点
        volume_origin = isocenter - volume_phy / 2          # 获得image图像中的坐下原点的实际物理坐标

        proj_phy = volume_phy * sid / sad                   # 第一步：计算放大倍率由于射线是锥形束发散的，物体在探测器上的投影会被放大。放大倍数（Magnification） = sid / sad（因为物体在等中心处，距离源为 sad）。因此，volume_phy * sid / sad 计算的是物体投射到探测器平面上所占的物理范围（毫米大小）。
        proj_phy = proj_phy[-2:]                            # 获得Y，Z。探测器上的水平方向（宽度）对应世界坐标系的 Y轴，垂直方向（高度）对应世界坐标系的 Z轴。而 X轴 方向是射线的飞行深度方向，它在二维投影图上没有对应的像素宽度。
        proj_spacing = volume_spacing * sid / sad  # nominal spacing
        proj_spacing = proj_spacing[-2:]                    # 获得真实的投影的距离
        
        step = (end - start) / num
        angles = np.arange(start, end, step)                # 计算角度步长，生成一个等间距分布的角度列表。例如 num=20，则会生成 [0, 18, 36, ..., 342] 度

        params = {
            'obj_index': file_name,
            'start': start,
            'end': end,
            'angle_per_view': step,
            'N_views': num,
            'sad': sad,
            'sid': sid,
            'volume_resolution': volume_resolution.tolist(),
            'volume_spacing': volume_spacing.tolist(),
            'volume_origin': volume_origin.tolist(),
            'volume_phy': volume_phy.tolist(),
            'proj_resolution': proj_resolution,
            'proj_spacing': proj_spacing.tolist(),
            'proj_phy': proj_phy.tolist(),
        }

        frames = []

        cnt = 0
        for angle in tqdm(angles, desc='Projection Geometry Production'):   # tqdm用于生成实时的进度条
            angle *= np.pi / 180  # degree to radian
            vec = angle2vec(angle, 0, isocenter, sid, sad, proj_spacing[0], proj_spacing[1])    # proj_spaceing的0和1分别表示平板探测器列方向和行方向的像素物理间距

            frame = {
                'file': str(cnt).zfill(4),  # 生成 '0000', '0001' ... 作为文件名索引
                'vec': vec.tolist(),        # 将numpy数组转为list方便保存为json，这样做的好处是，后续进行投影计算时，不需要反复调用三角函数（sin/cos），直接读取这个列表里的数值即可进行快速的光线追踪。
                                            # 'vec' 包含了以下4组关键几何信息（每组3个坐标值，共12个数值）：
                                            # X射线源位置（Source Position）：[x, y, z]。告诉你当前的X光机灯泡在三维空间哪个点。
                                            # 探测器中心位置（Detector Center Position）：[x, y, z]。告诉你平板探测器正中心在哪个点。
                                            # 探测器水平方向基向量（ u方向）：[x, y, z]。告诉你探测器上“从左到右”的方向在三维空间中的指向（即论文公式2中的 u_i，但没有乘像素间距）。
                                            # 探测器垂直方向基向量（v方向）：[x, y, z]。告诉你探测器上“从上到下”的方向在三维空间中的指向（即论文公式2中的 vi）。
            }
            cnt = cnt + 1
            frames.append(frame)

        params['frames'] = frames

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'transforms.json'), 'w') as f:  # 把刚才生成的包含 frames（所有角度位姿）、volume_spacing、proj_spacing 的 params 大字典，格式化地写进了这个文本文件。
            json.dump(params, f, indent=4)                                  # indent指定tab键的缩进级别。
        
        gt_image = sitk.GetArrayFromImage(image)                            # 把SimpleITK对象转成NumPy数组（此时顺序变为了 (Z, Y, X)）
        gt_image = ct2mu(gt_image)                                          # 将CT值（HU，亨氏单位） 转换为 线性衰减系数（μ，mu值），因为DRR模拟的线积分需要真实的 μ 值，而不是HU显示值
        gt_image = np.clip(gt_image, 0, gt_image.max())                     # 将所有小于0的值强制设为0。为什么这么做？ 在CT中，空气、肺组织的HU为负值，对应的衰减系数μ 极低或为零。
                                                                            # 在正向投影（模拟X光）中，负衰减系数没有物理意义（射线穿过空气不会变亮）。把负值裁掉，相当于强制将空气/低密度区域的衰减设为0，确保DRR模拟时，X射线只在骨头、软组织等高密度区域产生衰减，避免伪影。
        gt_image = sitk.GetImageFromArray(gt_image)                         # 把处理过的NumPy数组（已经是正确的 μ 值且裁掉了负值）变回SimpleITK图像对象。

        sitk.WriteImage(gt_image, os.path.join(output_path, 'gt_volume.nii.gz'))    # 保存裁剪后的结果

        print('Finish geometry production for', file_name)

def ProjectionGeneration(args, projpath):
    """
    DRR Projection Production
    """

    device = 'cuda:0'
    path_list = os.listdir(projpath)
    factor = 0.5                    # uniform sampling factor。决定了模拟X射线穿过三维体素时，沿着射线路径每走多远才采集一个采样点
    chunksize = 65536               # DRR模拟中，每个像素对应一条从X射线源出发的射线。chunksize 决定了一次性并行处理多少条射线。

    for file_cur in path_list:
        output_path = os.path.join(projpath, file_cur)
        data_path = os.path.join(output_path, 'gt_volume.nii.gz')                                               # 经过HU值裁剪后的体素图像路径
        with open(os.path.join(output_path, 'transforms.json')) as f:
            camera_paras = json.load(f)
        W, H = camera_paras['proj_resolution']
        Nframes = camera_paras['N_views']

        volume_phy = torch.tensor(camera_paras['volume_phy']).to(device)
        volume_origin = torch.tensor(camera_paras['volume_origin']).to(device)
        volume_spacing = torch.min(torch.tensor(camera_paras['volume_spacing'])).to(device).to(torch.float32)   # 获得体素间距中最小的哪个
        render_step_size = volume_spacing * factor                                                              # 最小间距与采样频率相乘，获得采样步长
        volume = sitk.ReadImage(data_path)                                                                      # 打开经过HU值裁剪后的体素图像路径
        volume_array = sitk.GetArrayFromImage(volume)                                                           # nii.gz -> ndarray
        volume_tensor = torch.tensor(volume_array).to(device)                                                   # ndarray -> tensor张量
        vecs = []
        for i in range(Nframes):
            frame = camera_paras['frames'][i]
            vec = torch.tensor(frame['vec']).to(device)                                                         # 获取关键的几何参数信息
            vecs.append(vec)
        vecs = torch.stack(vecs).to(device)
        cam_rays = get_rays(vecs, H, W)                                                                         # 生成光线池，shape 是 [N, H, W, 6]。
                                                                                                                # N（第0维）：视角数量。即你生成了多少张DRR投影（例如20张）。这对应你 vecs 张量的行数。
                                                                                                                # H（第1维）：图像高度（像素行数，对应三维空间的 Y 轴方向）。
                                                                                                                # W（第2维）：图像宽度（像素列数，对应三维空间的 Z 轴方向）。
                                                                                                                # 6（第3维）：每条光线的参数向量。

        projs = []
        for i in tqdm(range(Nframes), desc='Projection Generation'):
            frame = camera_paras['frames'][i]                                                                   
            rays = cam_rays[i, ...]                                                                             # (1) 取出第 i 张投影的所有光线
            rays = rays.reshape(-1, rays.shape[-1])                                                             # (2) 压平成 (H*W, 6)
            projection = composite(rays, volume_tensor, volume_origin, volume_phy, render_step_size, chunksize=chunksize)   # (3) 执行物理积分（DRR核心计算）
            projection = projection.reshape(H, W)                                                               # (4) 恢复为二维图像矩阵
            projs.append(projection)                                                                            # 循环结束后，projs 列表里存放了 N 个形状为 (H, W) 的张量。
        projs = torch.stack(projs)                                                                              # 沿着新创建的第0维堆叠，生成一个形状为 (N, H, W) 的三维张量。这个张量相当于把20张二维X光片叠在一起，形成了一个“投影体积（Projection Volume）”。第 i 张切片，就是第 i 个角度的DRR图像。
        projs = projs.cpu().detach().numpy()                                                                    # (1) 转NumPy。标准PyTorch转NumPy三步走（从GPU搬回CPU、切断梯度追踪、转为NumPy数组）。
        projs = sitk.GetImageFromArray(projs)                                                                   # 将形状为 (N, H, W) 的NumPy数组转换为SimpleITK图像
        sitk.WriteImage(projs, os.path.join(output_path, 'proj.nii.gz'))                                        # SimpleITK 的 GetImageFromArray 默认将输入视为 (Z, Y, X)。
                                                                                                                # 因此，在这个NIfTI文件里，Z轴方向代表的是“不同的扫描角度”，而不是解剖学上的头脚方向！

        print('Finish projection generation for', file_cur)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Projection Geometry Configuration File Production')

    parser.add_argument('--start', type=int, default=0, help='Start angle')
    parser.add_argument('--end', type=int, default=360, help='End angle')
    parser.add_argument('--num', type=int, default=20, help='Number of angles')
    parser.add_argument('--sad', type=float, default=1000, help='Source-to-axis distance (SAD) | 500 for dental, 1000 for spine')
    parser.add_argument('--sid', type=float, default=1500, help='Source-to-image distance (SID) | 700 for dental, 1500 for spine')
    parser.add_argument('--datapath', type=str, default='./dataset/head', help='Path to input NIfTI files')

    args = parser.parse_args()
    niipath = os.path.join(args.datapath, 'raw_volume')
    projpath = os.path.join(args.datapath, 'syn_data')

    # geometry files production
    GeometryProduction(args, niipath, projpath)

    # projection simulation
    ProjectionGeneration(args, projpath)
