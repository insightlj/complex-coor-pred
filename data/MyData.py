import os
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data.label_generate import label_generate
from torch.utils.data.sampler import Sampler
from config import BATCH_SIZE_1, BATCH_SIZE_2, BATCH_SIZE_3, BATCH_SIZE_4


class MyData(Dataset):
    def __init__(self, data_path, xyz_path, filename, train_mode):
        self.whole_index =  [x.strip().split(",")[0] for x in os.popen('cat '+ filename)]   
        # 此时已经根据序列的长度对x进行排序，所以加下来的事情就都明白了
        self.data_sum = len(self.whole_index)
        self.quarter_data_sum = (self.data_sum // 4 + 1)   
        # 在此做 +1 的处理，是为了防止下面计算part的时候出错。如果part=4的话，会报错
        self.trunc_points = np.array([64,99,136,195])   
        # 四等份中，分别取最短的蛋白质长度
        self.train_mode = train_mode
        self.coor = h5py.File(xyz_path, "r")
        self.embed_atten = h5py.File(data_path, "r")

    def __getitem__(self, idx):
        part = idx // self.quarter_data_sum
        part_index = self.whole_index[self.quarter_data_sum*part : self.quarter_data_sum*(part+1)]
        trunc_point = self.trunc_points[part]
        pdb_index = part_index[idx % self.quarter_data_sum]
        gap = self.coor[pdb_index]['gap'][:]
        coor = self.coor[pdb_index]["xyz"][np.where(gap > 0)[0]]  # [L, 4, 3], 其中L是序列长度，4代表四个原子，顺序是CA， C， N和CB
        embed = self.embed_atten[pdb_index]['token_embeds'][0, np.where(gap > 0)[0]]
        atten = self.embed_atten[pdb_index]['feature_2D'][0, :, np.where(gap > 0)[0]][:, :, np.where(gap > 0)[0]]

        coor = torch.from_numpy(coor)
        embed = torch.from_numpy(embed)
        atten = torch.from_numpy(atten)

        L = embed.shape[0]
        
        trunc_point = int(trunc_point+1e-3)
        if self.train_mode and L > trunc_point:
            a = random.randint(0, L-trunc_point-1)
            b = a + trunc_point
            embed = embed[a:b]
            atten = atten[:, a:b, a:b]
            L = trunc_point

        else:
            INF = 99999
            a = INF

        coor_label = label_generate(coor, a, trunc_point, self.train_mode)

        return embed, atten, coor_label, L
        # embed:[L,2560], atten:[41,L,L], coor_label:[L,L,3]

    def __len__(self):
        return self.data_sum

class MyBatchSampler(Sampler):
    def __init__(self, sorted_train_file):
        index =  [x.strip().split(",")[0] for x in os.popen('cat '+ sorted_train_file)]
        data_sum = len(index)
        self.quarter_data_sum = data_sum // 4
        indices_1 = list(range(self.quarter_data_sum*0, self.quarter_data_sum*1-BATCH_SIZE_1+1, BATCH_SIZE_1)) # drop_last
        indices_2 = list(range(self.quarter_data_sum*1, self.quarter_data_sum*2-BATCH_SIZE_2+1, BATCH_SIZE_2))
        indices_3 = list(range(self.quarter_data_sum*2, self.quarter_data_sum*3-BATCH_SIZE_3+1, BATCH_SIZE_3))
        indices_4 = list(range(self.quarter_data_sum*3, self.quarter_data_sum*4-BATCH_SIZE_4+1, BATCH_SIZE_4))
        indices = indices_1 + indices_2 + indices_3 + indices_4
        indices = np.array(indices)
        self.indices = indices
        self.length = len(indices)
        

    def __iter__(self):
        batch = []
        ls = list(range(self.length))
        random.shuffle(ls)
        ls = self.indices[ls]

        for idx in ls:
            if idx < self.quarter_data_sum * 1:
                for i in range(BATCH_SIZE_1):
                    batch.append(idx+i)
            elif idx < self.quarter_data_sum * 2:
                for i in range(BATCH_SIZE_2):
                    batch.append(idx+i)
            elif idx < self.quarter_data_sum * 3:
                for i in range(BATCH_SIZE_3):
                    batch.append(idx+i)
            else:
                batch.append(idx)
            
            yield batch
            batch = []
    
    def __len__(self):
        return None


if __name__ == "__main__":
    data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
    xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
    sorted_train_file = "/home/rotation3/example/sorted_train_list.txt"
    test_file = "/home/rotation3/example/valid_list.txt"
    trunc_points = np.load("/home/rotation3/complex-coor-pred/data/trunc_points.npy")
    
    dataset = MyData(data_path, xyz_path, sorted_train_file, train_mode=True)
    batch_sampler = MyBatchSampler()
    train_dl = DataLoader(dataset, batch_sampler=batch_sampler)

    # for i in (100,1000,2000,5100,10000,15000,16000,20000):
    #     embed, atten, coor_label, length = dataset[i]
    #     print(length)
    #
    # for i in train_dl:
    #     embed, atten, coor_label, length = i
    #     print(embed.shape, atten.shape, coor_label.shape, length)