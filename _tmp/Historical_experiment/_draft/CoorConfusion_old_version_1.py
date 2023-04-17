### 在这个版本中，还没有将坐标Align到一起，就使用了Attention，意义不大


import torch
from torch import nn
from model.ResNet import resnet_block
    
from main import train_ds
from torch.utils.data import DataLoader
from config import device, NUM_BLOCKS
from scripts.cal_lddt_multiseq import cal_lddt
from torch.utils.tensorboard import SummaryWriter
from utils import weight_init
from einops import rearrange
from utils.AlignCoorConfusion.axis_attention import BiasRowAttention, ColAttention

def attn_block(block_num=5):
    """
    将行注意力和列注意力写成一个块;
    每融合一次行信息, 融合两次坐标信息
    """
    blks = []
    for _ in range(block_num):
        # 对(B,3,64,L)中L的维度做attention, 做attention的过程中, 将pred_x2d_coor作为bias加入(仿照AlphaFold2)
        blks.append(BiasRowAttention(in_dim=3, q_k_dim=3, device=device))
        for coor_confusion_step in range(2):
            # 对(B,3,64,L)中64的维度做attention, 在这个过程中不加bias，因为具体的信息在上一步已经融合，这一步只负责生成坐标
            blks.append(ColAttention(in_dim=3, q_k_dim=3, device=device))  
    return blks

# 整体的思路就是：
# 对BLL3中的1维做Attention, 可以融合这一维度的坐标信息。网络应该可以学到足够的参数，来融合出最好的坐标
# pLDDT将作为Loss

class coorConfusionAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(*attn_block())
        self.coor_confusion1 = nn.Linear(64,16)
        self.coor_confusion2 = nn.Linear(16,4)

    def forward(self,pred_coor,pred_x2d):
        """
        pred_coor (B,64,L,3)
        pred_x2d  (B,105,L,L)
        return:
        confused_coor (B,4,L,3)
        """
        pred_coor = torch.permute(pred_coor, (0,3,1,2))   # (B,3,64,L)
        (pred_coor, pred_x2d) = self.attn((pred_coor, pred_x2d))
        pred_coor = torch.permute(pred_coor, (0,1,3,2))   # (B,3,L,64)
        confused_coor = self.coor_confusion1(pred_coor)   # (B,3,L,16)
        confused_coor = self.coor_confusion2(confused_coor)  # (B,3,L,4)
        confused_coor = torch.permute(confused_coor, (0,3,2,1))  # (B,4,L,3)

        return confused_coor

class coorConfusionAlign(nn.Module):
    def __init__(self):
        super.__init__()
        self.rotation = nn.Parameter()
        self.translation = nn.Parameter()

# ### 下面计算L套坐标的plddt, 选出64套, 形成(B,64,L,3), 便于接下来的操作
import numpy as np

pred_coor = torch.randn((2,91,91,3)).to(device)
cal_plddt = torch.load("utils/plddt_checkpoints/Full_train/epoch7_mark.pt")  # 训练好之后将效果最好的模型填充进去

pred_x2d = torch.randn((2,105,91,91)).to(device)
label = torch.randn((2,91,3)).to(device)    # 使用的是(L,3)的原始坐标