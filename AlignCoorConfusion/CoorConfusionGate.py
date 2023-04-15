import torch
from torch import nn
from config import device
from einops import rearrange
from utils.AlignCoorConfusion.axis_attention import BiasRowAttention, ColAttention
import numpy as np

def row_attn_block(block_num=4):
    blks = []
    for _ in range(block_num):
        # 对(B,3,64,L)中L的维度做attention, 做attention的过程中, 将pred_x2d_coor作为bias加入(仿照AlphaFold2)
        blks.append(BiasRowAttention(in_dim=3, q_k_dim=3, device=device))
    return blks


def col_attn_block(block_num=4):
    blks = []   
    for _ in range(block_num):
        # 对(B,3,64,L)中64的维度做attention, 在这个过程中不加bias，因为具体的信息在上一步已经融合，这一步只负责生成坐标
        blks.append(ColAttention(in_dim=3, q_k_dim=3, device=device))  
    return blks

class coorConfuse(nn.Module):
    """
    pred_coor (B,64,L,3)
    pred_x2d  (B,105,L,L)
    return:
    confused_coor (B,4,L,3)
    """
    def __init__(self):
        super().__init__()
        self.col_attn = nn.Sequential(*col_attn_block())
        self.row_attn = nn.Sequential(*row_attn_block())
        self.assist_linear_1 = nn.Linear(64*2, 256)
        self.assist_linear_2 = nn.Linear(256, 128)
        self.assist_linear_3 = nn.Linear(128, 64)

        self.confusion_linear_1 = nn.Linear(64,256)
        self.confusion_linear_2 = nn.Linear(256,128)
        self.confusion_linear_3 = nn.Linear(128,4)
        self.relu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=-2)
        self.alpha = nn.Parameter(torch.tensor(10.).to(device).requires_grad_())
        self.cycle_num = 1
        # self.cycle_num = np.random.randint(1,10)


    def forward(self,pred_coor, pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score):
        ### lddt_score是不变的，不参与循环
        lddt_score = torch.tile(lddt_score[:,None,:,:], (1,3,1,1))
        lddt_score = self.softmax(lddt_score)
        
        pred_coor = rearrange(pred_coor, "b h w c -> b c h w")
        pred_coor_r_attn = rearrange(pred_coor_r_attn, "b h w c -> b c h w")
        pred_coor_c_attn = rearrange(pred_coor_c_attn, "b h w c -> b c h w")

        print("cycle_num: ", self.cycle_num)
        for cycle in range(self.cycle_num):
            pred_coor_r_attn, _ = self.row_attn((pred_coor_r_attn, pred_x2d))
            pred_coor_c_attn = self.col_attn(pred_coor_c_attn)
            pred_coor_r_attn = self.softmax(pred_coor_r_attn)
            
            assist_info = torch.concat((lddt_score, pred_coor_r_attn), dim=-2)
            assist_info = assist_info.permute(0,1,3,2)
            assist_info = self.relu(self.assist_linear_1(assist_info))
            assist_info = self.relu(self.assist_linear_2(assist_info))
            assist_info = self.assist_linear_3(assist_info)
            assist_info = assist_info.permute(0,1,3,2)
            assist_info = assist_info * self.alpha
            feature = pred_coor_c_attn * assist_info
            pred_coor = pred_coor + feature

        pred_coor = rearrange(pred_coor, "b c h w -> b c w h")
        confused_coor = self.relu(self.confusion_linear_1(pred_coor))
        confused_coor = self.relu(self.confusion_linear_2(confused_coor))
        confused_coor = self.confusion_linear_3(confused_coor)

        return confused_coor

if __name__ == "__main__":
    """
    pred_coor = torch.randn(64,91,3).to(device)
    pred_x2d = torch.randn(105,91,91).to(device)
    lddt_score = torch.randn(64,91).to(device)

    return: 返回四套坐标
    confused_coor: torch.Size([4, 91, 3])
    """
    import time
    beg = time.time()

    pred_coor = torch.randn(64,91,3).to(device)
    pred_x2d = torch.randn(105,91,91).to(device)
    lddt_score = torch.randn(1,64,91).to(device)

    pred_coor = pred_coor.unsqueeze(0)
    pred_coor_c_attn = torch.clone(pred_coor)
    pred_coor_r_attn = torch.clone(pred_coor)
    pred_x2d = pred_x2d.unsqueeze(0)

    coor_confuse = coorConfuse().to(device)
    confused_coor = coor_confuse(pred_coor, pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score)
    confused_coor = confused_coor.squeeze().permute(2,1,0)
    print(confused_coor.shape)
    print("inference time: ", time.time()-beg)

    def count_parameters(model):
        return sum(param.numel() for param in model.parameters())
    count_parameters(coor_confuse)  # 203w
    