import random
import torch
import numpy as np
from config import device
from data.MyData import Data
from tools.cal_lddt_multiseq import cal_lddt
from pLDDT.pLDDT import pLDDT

train_data_path = '/home/rotation3/complex-coor-pred/data/train22310.3besm2.h5'
test_data_path = '/home/rotation3/complex-coor-pred/data/valid2000.3besm2.h5'
xyz_path = '/home/rotation3/complex-coor-pred/data/xyz.h5'
sorted_train_file = "/home/rotation3/complex-coor-pred/data/sorted_train_list.txt"
test_file = "/home/rotation3/complex-coor-pred/data/valid_list.txt"

net_pt_name = 'model/checkpoint/CoorNet_VII/epoch34.pt'
pLDDT_name = 'pLDDT/plddt_checkpoints/Full_train/epoch7_mark.pt'
# net_pt_name = '/home/rotation3/coor-pred/model/checkpoint/l2_II/epoch29.pt'
net_pt = torch.load(net_pt_name)
net_pt.eval()

pLDDT = torch.load(pLDDT_name)
pLDDT.eval()

train_mode = False
test_ls = range(1988)

lddt_ls = []
for index in test_ls:
    if train_mode:
        path = train_data_path
        file_list = sorted_train_file
    else:
        path = test_data_path
        file_list = test_file

    ds = Data(path, xyz_path, file_list, train_mode=False)   #此处的train_mode控制蛋白会不会被tunc
    embed, atten, coor_label, L, pdb_index = ds[index]
    embed = embed.to(device)
    atten = atten.to(device)
    coor_label = coor_label.to(device)
    embed.unsqueeze_(0)
    atten.unsqueeze_(0)
    coor_label.unsqueeze_(0)

    with torch.no_grad():
        pred_coor_4_steps, pred_x2d = net_pt(embed, atten)
        pred_coor = pred_coor_4_steps[-1]   # 取出最后一个Block预测出的coor
        # pred_coor = net_pt(embed, atten)
        # pred_coor = pred_coor.reshape(1,L,L,3)

    pred_lddt = pLDDT(pred_coor, pred_x2d)

    eps = 1e-5
    pred_coor = pred_coor.unsqueeze(-2) - pred_coor.unsqueeze(-3)
    pred_coor = ((pred_coor**2).sum(dim=-1) + eps) ** 0.5
    coor_label = ((coor_label**2).sum(dim=-1) + eps) ** 0.5
    # print(pred_coor.shape, coor_label.shape)
    lddt = cal_lddt(pred_coor, coor_label)
    # print(lddt.shape, pred_lddt.shape)

    lddt = lddt.squeeze()
    lddt = (lddt.mean(dim=-1)).max()
    lddt = lddt.item()
    lddt_ls.append(lddt)
    print(pred_lddt, lddt)

print(np.mean(lddt_ls))
# np.save('/home/rotation3/complex-coor-pred/scripts/test_ResNet_lddt.npy', np.array(lddt_ls))
