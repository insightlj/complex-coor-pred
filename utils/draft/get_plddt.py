import torch
from torch.utils.data import DataLoader
from main import train_ds, test_ds
from config import device, NUM_BLOCKS
from utils.cal_lddt import getLDDT

import sys
sys.path.append("/home/rotation3/complex-coor-pred/")


project_name = "Full_train"
plddt_epoch = 11
net_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"
eps = 1e-7
local_step = 0
global_step = 0
total_loss = 0
avg_loss = 0

from utils.pLDDT import pLDDT
get_pLDDT = torch.load("utils/plddt_checkpoints/" + project_name + "/epoch" + str(plddt_epoch) + ".pt")
get_pLDDT.eval()
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
net_pt = torch.load(net_pt_name, map_location=device)
for param in net_pt.parameters():
    param.requires_grad = False

for data in test_dataloader:
    local_step += 1
    global_step += 1
    embed, atten, coor_label, L = data
    embed = embed.to(device)
    atten = atten.to(device)
    coor_label = coor_label.to(device)
    L = L.to(device)
    pred_coor_ls, pred_x2d = net_pt(embed, atten)
    pred = pred_coor_ls[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

    predcadist = pred.unsqueeze(-2) - pred.unsqueeze(-3)   
    predcadist = ((predcadist**2).sum(dim=-1) + eps) ** 0.5   # predcadist  (N,L,L,L)
    label = ((coor_label**2).sum(dim=-1) + eps) ** 0.5
    lddt = getLDDT(predcadist,label)
    plddt = get_pLDDT(pred, pred_x2d)
    loss = (((plddt - lddt)**2).sum() + eps) ** 0.5
    break

lddt = lddt.squeeze().detach().cpu().numpy()
plddt = plddt.squeeze().detach().cpu().numpy()

import matplotlib.pyplot as plt
plt.hist(lddt[0])