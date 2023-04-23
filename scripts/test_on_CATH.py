from data.label_generate import label_generate
from torch.utils.data import Dataset
from tools.cal_lddt_tensor import cal_lddt
from tools.cal_phipsi import PsiPhi
import numpy as np
import os
import torch
import h5py
from config import device

class Data(Dataset):
    def __init__(self, data_path, xyz_path, filename, train_mode):
        self.index =  [x.strip().split(",")[0] for x in os.popen('cat '+ filename)]   
        self.train_mode = train_mode
        self.coor = h5py.File(xyz_path, "r")
        self.embed_atten = h5py.File(data_path, "r")

    def __getitem__(self, idx):
        pdb_index = self.index[idx]
        gap = self.coor[pdb_index]['gap'][:]
        coor = self.coor[pdb_index]["xyz"][np.where(gap > 0)[0]]  # [L, 4, 3], 其中L是序列长度，4代表四个原子，顺序是CA， C， N和CB
        # embed = self.embed_atten['embed2560'][pdb_index][0, np.where(gap > 0)[0]]
        embed = self.embed_atten['embed2560'][pdb_index][np.where(gap > 0)[0],:]
        contact = self.embed_atten['contacts'][pdb_index][:, :, np.where(gap > 0)[0]][:, np.where(gap > 0)[0], :]
        atten = self.embed_atten['att40'][pdb_index][:, :, np.where(gap > 0)[0]][:, np.where(gap > 0)[0], :]

        coor = torch.from_numpy(coor)
        embed = torch.from_numpy(embed)
        contact = torch.from_numpy(contact)
        atten = torch.from_numpy(atten)
        atten = torch.concat((contact, atten), dim=0)

        L = embed.shape[0]
        INF = 1e5
        a, trunc_point = INF, INF
        coor_label = label_generate(coor, a, trunc_point, self.train_mode)

        return embed, atten, coor_label, L, pdb_index
        # embed:[L,2560], atten:[41,L,L], coor_label:[L,L,3]

    def __len__(self):
        return len(self.index)
    

"""
plot_file.h5
<HDF5 dataset "identity": shape (), type "|O">
<HDF5 dataset "lddt": shape (), type "<f4">
<HDF5 dataset "phipsi": shape (82, 2), type "<f4">
<HDF5 dataset "pred_coor": shape (83, 3), type "<f4">
<HDF5 dataset "target_tokens": shape (1, 83), type "|i1">
"""

train_data_path = '/home/rotation3/complex-coor-pred/data/train22310.3besm2.h5'
test_data_path = '/home/rotation3/complex-coor-pred/data/valid2000.3besm2.h5'
xyz_path = '/home/rotation3/complex-coor-pred/data/xyz.h5'
sorted_train_file = "/home/rotation3/complex-coor-pred/data/sorted_train_list.txt"
test_file = "/home/rotation3/complex-coor-pred/data/valid_list.txt"

plot_file = h5py.File("/home/rotation3/complex-coor-pred/data/plot_file.h5", "a")

# max_lddt_tensor = torch.load("/home/rotation3/complex-coor-pred/plot/max_lddt_ls_train.pt")
# index = torch.topk(max_lddt_tensor, 40)[1]
initial_lddt = np.load("/home/rotation3/complex-coor-pred/AlignCoorSample/log/initial_lddt.npy")
finally_lddt = np.load("/home/rotation3/complex-coor-pred/AlignCoorSample/log/final_lddt.npy")
diff = finally_lddt - initial_lddt
diff = torch.tensor(diff)
value = diff.topk(10)[0]
index = (diff.topk(10)[1]) * 40

for statue in [True]:
    train_mode = statue
    if train_mode:
        name = "train"
        path = train_data_path
        file_list = sorted_train_file
    else:
        name = "test"
        path = test_data_path
        file_list = test_file

    coor = h5py.File("/home/rotation3/complex-coor-pred/data/xyz.h5", "r")
    pdb_index_ls = []
    ds = Data(path, xyz_path, file_list, train_mode=False)   #此处的train_mode控制蛋白会不会被tunc
    for i in index:
        data = ds[i]
        embed, atten, coor_label, L, pdb_index = data   # pdb_index
        pdb_index_ls.append(pdb_index)
        embed = embed.to(device)
        atten = atten.to(device)
        coor_label = coor_label[0].to(device)
        embed.unsqueeze_(0)
        atten.unsqueeze_(0)

        net_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch16.pt"
        net_pt = torch.load(net_pt_name)

        with torch.no_grad():
            pred_coor_4_steps, pred_x2d = net_pt(embed, atten)
            pred_coor = pred_coor_4_steps[-1][0]   # 取出最后一个Block预测出的coor
        
        lddt_ls =  []
        for chain in pred_coor:
            lddt = (cal_lddt(chain, coor_label)).item()
            lddt_ls.append(lddt)
        lddt_tensor = torch.tensor(lddt_ls)
        max_index = torch.argmax(lddt_tensor)
        pred_coor = pred_coor[max_index]   # pred_coor
        coor_label = coor_label[max_index]  # coor_label
        print(pred_coor.shape)
        lddt = (lddt_tensor[max_index]).item()   # lddt

        gap = coor[pdb_index]['gap'][:]
        coor_label = coor[pdb_index]["xyz"][np.where(gap > 0)[:3]]
        phipsi = PsiPhi(torch.from_numpy(coor_label))

        if pdb_index in plot_file.keys():
            del plot_file[pdb_index]
        # if "lddt" in protein.keys():
        #     del protein["lddt"]
        # if "phipsi" in protein.keys():
        #     del protein["phipsi"]
        # if "pred_coor" in protein.keys():
        #     del protein["pred_coor"]
        # if "identity" in protein.keys():
        #     del protein["identity"]
        
        protein = plot_file.create_group(pdb_index)
        protein["lddt"] = lddt
        protein["pred_coor"] = pred_coor.cpu()
        protein["identity"] = pdb_index
        protein["phipsi"] = phipsi.cpu()

print(pdb_index_ls)