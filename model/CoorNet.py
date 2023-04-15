"""
该网络相对于ResNet.py来说，将StructureModule的Block变为了四个，并且在每个Block扩大模型的视野。
"""


import torch
from torch import nn
from torch.nn.functional import softmax
from config import device, eps

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, kernel_size=3, strides=1):
        super(Residual, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.norm1 = nn.InstanceNorm2d(input_channels, affine=True)
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size, padding=self.padding, stride=strides)
        self.norm2 = nn.InstanceNorm2d(num_channels, affine=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=self.padding, stride=strides)

        use_1x1conv = False if input_channels == num_channels else True

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size, padding=self.padding, stride=strides)

        else:
            self.conv3 = None

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.conv1(self.norm1(X)))
        Y = self.conv2(self.norm2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals):
    blk = []
    for _ in range(num_residuals):
        blk.append(Residual(input_channels, num_channels))
    return blk


class PreNet(nn.Module):
    def __init__(self, num_1d_blocks=8, embed_channels=64, num_block=8, resnet_dim=128):
        super(PreNet, self).__init__()
        self.embed_channels = embed_channels
        self.block = num_block
        self.num_1d_blocks = num_1d_blocks

        self.relu = nn.LeakyReLU(inplace=False)
        self.linear_1 = nn.Linear(2560, 512)
        self.linear_2 = nn.Linear(512, 256)
    
        self.conv_1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv_2d = nn.Conv2d(in_channels=512, out_channels=self.embed_channels, kernel_size=3, padding=1)
    
    @staticmethod
    def embed_2_2d(embed):
        # embed [batch_size, 256, L]
        L = embed.shape[-1]
        embed_1 = embed.unsqueeze(dim=3).expand(-1, -1, -1, L)
        embed_2 = embed.unsqueeze(dim=2).expand(-1, -1, L, -1)
        embed = torch.concat((embed_1, embed_2), dim=1)  # (batch_size,512, L,L)
        return embed
    
    def forward(self, embed, atten):  # embed:[batch_size, L, 2560]; atten:[batch_size, 41,L,L]
        # embed reduction
        embed = self.relu(self.linear_1(embed))
        embed = self.linear_2(embed)  # (batch_size,L,256)
        embed = embed.permute(0, 2, 1)  # (batch_size,256,L)

        # 1D ResNet
        for _ in range(self.num_1d_blocks):
            res = self.conv_1d(embed)  # (batch_size,256,L)
            res = self.relu(res)
            embed = res + embed

        # embed to 2d
        embed = self.embed_2_2d(embed)  # (batch_size, 256, L) -> (batch_size, 512, L,L)
        embed = self.conv_2d(embed)  # (batch_size, embed_channels, L,L)

        x2d = torch.concat((embed, atten), dim=1)  # (batch_size, 105, L, L)

        return x2d

        
class Block(nn.Module):
    def __init__(self, x2d_dim=105, clip_num=25, resnet_dim=128, num_blocks=8):
        super().__init__()
        coor_dim = 3
        cmb_dim = x2d_dim + clip_num
        self.norm_x2d = nn.InstanceNorm2d(x2d_dim)

        self.b1 = nn.Sequential(nn.Conv2d(cmb_dim, resnet_dim, kernel_size=3, padding=1), nn.ReLU())  # dim: dim_in->resnet_dim
        self.b2 = nn.Sequential(*resnet_block(resnet_dim, resnet_dim, num_blocks))  # dim: resnet_dim->resnet_dim
        self.b3 = nn.Sequential(*resnet_block(resnet_dim, x2d_dim, 1))  # dim: resnet_dim -> dim_out
        self.b4 = nn.Sequential(nn.Conv2d(x2d_dim, resnet_dim, kernel_size=3, padding=1), nn.ReLU())  # dim: dim_in->resnet_dim
        self.b5 = nn.Sequential(*resnet_block(resnet_dim, resnet_dim, num_blocks))  # dim: resnet_dim->resnet_dim
        self.b6 = nn.Sequential(*resnet_block(resnet_dim, coor_dim, 1))  # dim: resnet_dim -> coor_dim

        self.cmb_2_x2d_I = nn.Sequential(self.b1, self.b2, self.b3)
        self.x2d_2_coor_I = nn.Sequential(self.b4, self.b5, self.b6)


    @staticmethod
    def func_coor_clip(coor):
        trunc = list(range(4,36,2))
        L = coor.shape[1]
        batch_size = coor.shape[0]
        coor = coor.reshape(-1, 3)
        coor = coor ** 2
        coor = coor.sum(-1) ** 0.5 + 1e-7
        coor_clip_list = []
        for i in trunc:
            coor_trunc = coor - i
            coor_trunc = softmax(coor_trunc, dim=0)
            coor_trunc = coor_trunc.reshape(batch_size, L,L)
            coor_clip_list.append(coor_trunc)
        coor_clip = torch.stack(coor_clip_list, dim=1)
        return coor_clip
    
    @staticmethod
    def norm_min_max(coor):
        coor_k_max, _ = torch.topk(coor, k=20, dim=1)
        coor_k_min, _ = torch.topk(coor, k=20, dim=1,largest=False)

        coor_k_max = coor_k_max[:,-1,:].unsqueeze_(1)
        coor_k_min = coor_k_min[:,-1,:].unsqueeze_(1)

        coor_diff = coor_k_max - coor_k_min
        coor_norm = (coor - coor_k_min) / (coor_diff + eps)

        return coor_norm
    
    @staticmethod
    def norm_mean(coor):
        coor_k_max, _ = torch.topk(coor, k=20, dim=1)
        coor_k_min, _ = torch.topk(coor, k=20, dim=1,largest=False)

        coor_k_max = coor_k_max[:,-1,:].unsqueeze_(1)
        coor_k_min = coor_k_min[:,-1,:].unsqueeze_(1)

        coor_diff = coor_k_max - coor_k_min

        coor_mean = coor.mean(dim=1)
        coor_mean.unsqueeze_(1)
        coor_norm = (coor - coor_mean) / (coor_diff + eps)

        return coor_norm

    @staticmethod
    def norm(coor):
        coor_k_max, _ = torch.topk(coor, k=20, dim=1)
        coor_k_min, _ = torch.topk(coor, k=20, dim=1,largest=False)

        coor_k_max = coor_k_max[:,-1,:].unsqueeze_(1)
        coor_k_min = coor_k_min[:,-1,:].unsqueeze_(1)

        coor_diff = coor_k_max - coor_k_min
        coor_norm = coor / (coor_diff + 1e-5)

        return coor_norm
    

    def forward(self, coor,x2d, batch_size,coor_ls, L):
        
        def coor_2_clip(coor):
            coor_clip = self.func_coor_clip(coor)
            coor_norm = self.norm(coor).reshape(batch_size, L,L,3)
            coor_norm = torch.permute(coor_norm, (0,3,1,2))
            coor_norm_mean = self.norm_mean(coor).reshape(batch_size, L,L,3)
            coor_norm_mean = torch.permute(coor_norm_mean, (0,3,1,2))
            coor_norm_max_min = self.norm_min_max(coor).reshape(batch_size, L,L,3)
            coor_norm_max_min = torch.permute(coor_norm_max_min, (0,3,1,2))
            coor_clip = torch.concat((coor_clip, coor_norm, coor_norm_max_min, coor_norm_mean), dim=1)
            return coor_clip
        
        coor_clip = coor_2_clip(coor)
        cmb = torch.concat((coor_clip, x2d), dim=1)
        x2d_tmp = self.cmb_2_x2d_I(cmb)
        x2d = (x2d + x2d_tmp) / 2
        x2d = self.norm_x2d(x2d)
        coor_clip_tmp = self.x2d_2_coor_I(x2d)
        coor_clip_tmp = torch.permute(coor_clip_tmp, (0,2,3,1))
        coor = (coor + coor_clip_tmp) / 2
        # coor_ = coor.reshape(batch_size, L*L, 3)
        coor_ls.append(coor)

        return coor, x2d

class CoorNet(nn.Module):
    def __init__(self, x2d_dim=105):
        super(CoorNet, self).__init__()
        self.form_x2d = PreNet()
        self.norm_x2d = nn.InstanceNorm2d(x2d_dim)

        self.block1 = Block()
        self.block2 = Block()
        self.block3 = Block()
        self.block4 = Block()

    def forward(self, embed, atten):
        coor_ls = []
        x2d = self.form_x2d(embed, atten)
        x2d = self.norm_x2d(x2d)
        batch_size = x2d.shape[0]
        L = x2d.shape[-1]
        coor = torch.zeros(batch_size, L,L,3, device=device)

        # coor,x2d已准备就绪，开始进入循环
        coor, x2d = self.block1(coor,x2d, batch_size, coor_ls, L)
        coor, x2d = self.block2(coor,x2d, batch_size, coor_ls, L)
        coor, x2d = self.block3(coor,x2d, batch_size, coor_ls, L)
        coor, x2d = self.block4(coor,x2d, batch_size, coor_ls, L)

        ### 思路: 将中间的量写入一个列表当中，然后将列表中的值返回出来，以便在train的过程中记录
        return coor_ls, x2d
        # coor (B,L,L,C); x2d (B,105,L,L)


if __name__ == "__main__":
    net = CoorNet()

    # 使用这种方式可以比较优雅的debug；只不过模型的
    def hook(children_model, input, output):
        print("hooking………………")
        print(children_model)
        print(output.shape)
    # 必须用get("b1")的方式获取，以索引的方式获取无效
    h1 = (net._modules["block3"])._modules.get("b1")[1].register_forward_hook(hook)   

    atten = torch.randn(4,41,129,129)
    embed = torch.randn(4,129,2560)

    net = net.to(device)
    atten = atten.to(device)
    embed = embed.to(device)

    true_coor_ls, x2d_pred = net(embed, atten)
    h1.remove()  # 记得最后将hook移除

    # print(true_coor_ls[0].shape, x2d_pred.shape, len(true_coor_ls))
    # print(true_coor_ls[0])
    # torch.Size([4, 129, 129, 3]) torch.Size([4, 105, 129, 129]) 4
    # pred与label都保持这种形状：[4, 129, 129, 3]
