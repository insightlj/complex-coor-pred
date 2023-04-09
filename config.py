import torch
import os
from torch import nn
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"
# os.environ["CUDA_VISIBLE_DEVICES"]= "3"
# os.environ["CUDA_VISIBLE_DEVICES"]= "4"

BATCH_SIZE = 1
BATCH_SIZE_1 = 3
BATCH_SIZE_2 = 2
BATCH_SIZE_3 = 2
BATCH_SIZE_4 = 1

eps = 1e-7
EPOCH_NUM = 35
ACC_STEPS = 4   # 将batch做梯度累加的数量
NUM_BLOCKS = 4   # CoorNet设计的Blocks的数量（即输出loss的数量）
LOSS_TRUNC = 10   # 限制两个点之间的距离不会出现特别大的离群值

BLOCK_COOR_TRUNC = [10,20,30,40]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

run_name = "CoorNet_demo"
error_file_name = run_name + ".error"
net_pt = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"
# os.system("tensorboard --logdir %s" % "/home/rotation3/complex-coor-pred/model/checkpoint/" + run_name)


class Coor_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, pred, label, num_block=None):
        pred = pred.unsqueeze(-2) - pred.unsqueeze(-3)
        label = label.unsqueeze(-2) - label.unsqueeze(-3)

        if num_block == None:
            loss = ((((pred - label) ** 2 ).sum(dim=-1) + eps)**0.5).mean()
    
        else:
            # effective_len = len(pred[pred<=BLOCK_COOR_TRUNC[num_block]])/ 3
            # pred[pred>BLOCK_COOR_TRUNC[num_block]] = 0
            # label[pred>BLOCK_COOR_TRUNC[num_block]] = 0
            pred_dist = ((pred ** 2).sum(dim=-1, keepdims=True)) ** 0.5
            effective_len = len(pred_dist[pred_dist<BLOCK_COOR_TRUNC[num_block]])
            pred_dist = pred_dist.repeat(1,1,1,1,3)
            label_dist =((label ** 2).sum(dim=-1, keepdims=True)) ** 0.5
            label_dist = label_dist.repeat(1,1,1,1,3)

            pred_masked = torch.where(pred_dist<BLOCK_COOR_TRUNC[num_block], pred, 0)
            label_masked = torch.where(pred_dist<BLOCK_COOR_TRUNC[num_block], label, 0)
            loss = ((((pred_masked - label_masked) ** 2 ).sum(dim=-1) + eps)**0.5).sum() / effective_len

        return loss

loss_fn = Coor_loss()