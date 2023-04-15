# function: 根据plddt进行采样，对lddt增加的接受采样，对lddt减小的拒绝采样
# 结果发现，采样空间太大太大了，根本搜索不到合适的序列
# Author: Jun Li

import h5py
import torch
import numpy as np
from config import device
from torch.nn.functional import softmax
from scripts.cal_lddt_tensor import cal_lddt
from main import train_ds
from torch.utils.data import DataLoader

protein_index = 0
train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
train_file = h5py.File("utils/AlignCoorConfusion/h5py_data/train_dataset.h5py", "r")
pred_coor = torch.from_numpy(np.array(train_file["protein"+str(protein_index)]["pred_coor"], dtype=np.float32)).squeeze().to(device)
lddt_score = torch.from_numpy(np.array(train_file["protein"+str(protein_index)]["lddt_score"], dtype=np.float32)).squeeze().to(device)

sample1 = torch.exp((lddt_score - 0.5) * 10)
sample2 = softmax((lddt_score - 0.5) * 10, dim=0)
sample3 = softmax(sample1, dim=0)

L = lddt_score.shape[-2]
sample1.t_()
sample2.t_()
sample3.t_()
for data in train_dataloader:
    _,_,label,_ = data
    label = label[:,1,:,:].squeeze().to(device)
    pre_lddt_ls = []
    for chain in pred_coor:
        pre_lddt = cal_lddt(chain, label)
        pre_lddt_ls.append(pre_lddt.item())
    pre_lddt = torch.tensor(pre_lddt_ls).mean()
    while True:
        index = torch.multinomial(sample2, num_samples=1).squeeze()
        sample_coor_ls = []
        for i in range(L):
            sample_coor_ls.append((pred_coor[index[i], i, :]))
        sample_chain = torch.stack(sample_coor_ls)
        lddt = cal_lddt(label, sample_chain)
        print("lddt", lddt)
        if lddt > pre_lddt:
            print("lddt", lddt)
            print("improve", lddt - pre_lddt)
            print("index", index)
            break
    break

