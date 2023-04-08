import torch
from torch import nn
from config import device
from einops import rearrange
from utils.axis_attention import BiasRowAttention, ColAttention

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
        self.confusion_linear_1 = nn.Linear(64*3,256)
        self.confusion_linear_2 = nn.Linear(256,128)
        self.confusion_linear_3 = nn.Linear(128,4)
        self.activate1 = nn.LeakyReLU(0.1)
        self.activate2 = nn.LeakyReLU(0.1)
        self.activate3 = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=)


    def forward(self, pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score):
        pred_coor_r_attn = rearrange(pred_coor_r_attn, "b h w c -> b c h w")
        pred_coor_c_attn = rearrange(pred_coor_c_attn, "b h w c -> b c h w")
        pred_coor_r_attn, _ = self.row_attn((pred_coor_r_attn, pred_x2d))
        pred_coor_c_attn = self.col_attn(pred_coor_c_attn)
        pred_coor_r_attn = self.softmax(pred_coor_r_attn)
        lddt_score = torch.tile(lddt_score[:,None,:,:], (1,3,1,1))
        feature = torch.cat((pred_coor_c_attn, pred_coor_r_attn, lddt_score), 2)
        feature = rearrange(feature, "b c h w -> b c w h")
        confused_coor = self.activate1(self.confusion_linear_1(feature))
        confused_coor = self.activate2(self.confusion_linear_2(confused_coor))
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

    pred_coor_c = torch.clone(pred_coor).unsqueeze(0)
    pred_coor_r = torch.clone(pred_coor).unsqueeze(0)
    pred_x2d = pred_x2d.unsqueeze(0)

    model = coorConfuse()
    model.load_state_dict(torch.load("/home/rotation3/complex-coor-pred/utils/AlignCoorConfusion/checkpoints/demo-1/epoch0.pt"))
    model.device()
    confused_coor = model(pred_coor_r, pred_coor_c, pred_x2d, lddt_score)
    confused_coor = confused_coor.squeeze().permute(2,1,0)
    print(confused_coor.shape)
    print("inference time: ", time.time()-beg)

    # def count_parameters(model):
    #     return sum(param.numel() for param in model.parameters())
    # count_parameters(coor_confuse)  # 200w