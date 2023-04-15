import torch
import numpy as np

label = torch.load("/home/rotation3/complex-coor-pred/data/label.npy")
coor = label

coor_k_max, _ = torch.topk(coor, k=20, dim=1)
coor_k_min, _ = torch.topk(coor, k=20, dim=1,largest=False)

coor_k_max = coor_k_max[:,-1,:].unsqueeze_(1)
coor_k_min = coor_k_min[:,-1,:].unsqueeze_(1)
print(coor_k_max.shape)

coor_diff = coor_k_max - coor_k_min
coor_norm = coor / (coor_diff + 1e-5)

coor_norm
