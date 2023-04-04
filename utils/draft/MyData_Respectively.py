import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data.label_generate import label_generate

data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
sorted_train_file = "/home/rotation3/example/sorted_train_list.txt"
test_file = "/home/rotation3/example/valid_list.txt"
trunc_points = np.load("/home/rotation3/complex-coor-pred/data/trunc_points.npy")

index = [x.strip().split(",")[0] for x in os.popen('cat ' + sorted_train_file)]
data_sum = len(index)
quarter_data_sum = data_sum // 4


class MyData(Dataset):
    def __init__(self, data_path, xyz_path, filename, part: '0,1,2,3', train_mode):
        assert part in (0, 1, 2, 3, False)
        index = [x.strip().split(",")[0] for x in os.popen('cat ' + filename)]
        data_sum = len(index)
        quarter_data_sum = data_sum // 4
        quarter_data_sum = int(quarter_data_sum)
        trunc_points = np.load("data/trunc_points.npy")

        self.part = part
        self.index = index[quarter_data_sum * self.part: quarter_data_sum * (self.part + 1)]
        self.trunc_point = trunc_points[self.part]
        self.data_sum = len(self.index)
        self.train_mode = train_mode
        self.coor = h5py.File(xyz_path, "r")
        self.embed_atten = h5py.File(data_path, "r")

    def __getitem__(self, idx):
        pdb_index = self.index[idx]
        gap = self.coor[pdb_index]['gap'][:]
        coor = self.coor[pdb_index]["xyz"][np.where(gap > 0)[0]]  # [L, 4, 3], 其中L是序列长度，4代表四个原子，顺序是CA， C， N和CB
        embed = self.embed_atten[pdb_index]['token_embeds'][0, np.where(gap > 0)[0]]
        atten = self.embed_atten[pdb_index]['feature_2D'][0, :, np.where(gap > 0)[0]][:, :, np.where(gap > 0)[0]]

        coor = torch.from_numpy(coor)
        embed = torch.from_numpy(embed)
        atten = torch.from_numpy(atten)

        L = embed.shape[0]

        trunc_point = int(self.trunc_point)
        if self.train_mode and L > trunc_point:
            a = random.randint(0, L - trunc_point - 1)
            b = a + trunc_point
            embed = embed[a:b]
            atten = atten[:, a:b, a:b]
            L = trunc_point
        else:
            INF = 99999
            a = INF

        coor_label = label_generate(coor, a, self.train_mode)

        return embed, atten, coor_label, L
        # embed:[L,2560], atten:[41,L,L], coor_label:[L,L,3]

    def __len__(self):
        return self.data_sum


if __name__ == "__main__":
    from torch.utils.data import ConcatDataset

    data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
    xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
    sorted_train_file = "/home/rotation3/example/sorted_train_list.txt"
    test_file = "/home/rotation3/example/valid_list.txt"
    trunc_points = np.load("/home/rotation3/complex-coor-pred/data/trunc_points.npy")

    train_ds1 = MyData(data_path, xyz_path, sorted_train_file, part=0, train_mode=True)
    train_ds2 = MyData(data_path, xyz_path, sorted_train_file, part=1, train_mode=True)
    train_ds3 = MyData(data_path, xyz_path, sorted_train_file, part=2, train_mode=True)
    train_ds4 = MyData(data_path, xyz_path, sorted_train_file, part=3, train_mode=True)
    train_ds = ConcatDataset((train_ds1, train_ds2, train_ds3, train_ds4))

    for i in (0, 50, 100, 156, 158, 10000, 15200, 15660, 20000, 20000):
        embed, atten, coor_label, length = train_ds[i]
        print(length)
