import torch
import os
from torch import nn
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"
os.environ["CUDA_VISIBLE_DEVICES"]= "3"
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

name = "CoorNet_VII"
error_file_name = name + ".error"
net_pt = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"