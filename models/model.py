import torch
import torch.nn as nn
from models.ResEncoder import ResEncoder
from models.SRGAN import generator
from models.aggregator import adafusor, localfusor, meanfusor, varfusor

# Main Model
class model(nn.Module):
    def __init__(self, model_conf=None, device=None):
        super(model, self).__init__()
        self.device = device
        self.encoder_conf = model_conf['encoder']
        self.decoder_conf = model_conf['SRGAN.generator']
        self.last_layer = model_conf['last_layer']
        self.fusion = model_conf['fusion']
        self.encoder = ResEncoder(self.encoder_conf).to(device)
        self.decoder = generator(self.decoder_conf).to(device)

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

    def query_volume_latent(self, xyz_world):
        x,y,z = xyz_world.shape[:3]                                 # 下采样后的尺寸 [x,y,z]=[X/4,Y/4,Z/4]
        points = xyz_world.contiguous().reshape(-1,3)               # 所有点坐标
        pnts_split = torch.split(points,100000)                     # 分块省显存
        h = []
        for pnts in pnts_split:
            latent = self.encoder.queryfeature(pnts)                # ① encoder 特征回投影 [nviews, C, npts]
            if self.fusion=='max':
                h.append(torch.max(latent, dim=0)[0])               # ② 聚合多视角特征 [C, npts]
            elif self.fusion=='mean':
                h.append(torch.mean(latent, dim=0))                 # ② 聚合多视角特征 [C, npts]
            else:
                h.append(self.aggregator(latent))                   # ② 聚合多视角特征 [C, npts]
            
        h = torch.cat(h,dim=1)
        return h.reshape(1,-1,x,y,z)                                # ③ 低分辨率特征体积 [1, C, X/4, Y/4, Z/4]

    def forward(self, xyz_world, return_latent=False):
        latent = self.query_volume_latent(xyz_world)
        outputs = self.decoder(latent)[0,0,:,:,:].transpose(0,2)    # align with ITK-SNAP display format。 这里实际上是拿到针对单个点的，从其在不同视角上的对应点的特征向量，这些特征向量是在维度上进行拼接的
                                                                    # ④ self.decoder(outputs)：★ decoder 3D 上采样 [1, 1, X, Y, Z]
                                                                    # [0,0,:,:,:].transpose(0,2)：⑤ 对齐 ITK-SNAP 显示格式
        outputs = self.last_layer_act(outputs)                      # ⑥ GELU/ReLU 激活 → 非负 μ
        if return_latent:
            return outputs, latent
        return outputs                                              # 返回真正的体素空间表示
