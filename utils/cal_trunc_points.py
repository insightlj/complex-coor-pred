import os
import numpy as np
import h5py
import csv

def cal_trunc_points():
    xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
    train_file = "/home/rotation3/example/train_list.txt"
    sorted_train_file = "/home/rotation3/example/sorted_train_list.txt"

    index = [x.strip() for x in os.popen('cat '+ train_file)]
    xyz = h5py.File(xyz_path, "r")

    # 所以修改index的顺序即可，也就是修改train_file中蛋白质的顺序
    num = len(index)
    temp_num = num

    pdb_length_dict = dict()
    for i in range(temp_num):
        pdb_index = index[i]
        gap = xyz[pdb_index]['gap'][:]
        coor = xyz[pdb_index]["xyz"][np.where(gap > 0)[0]]  # [L, 4, 3], 其中L是序列长度，4代表四个原子，顺序是CA， C， N和CB
        L = coor.shape[0]
        pdb_length_dict[pdb_index] = L

    pdb_length_dict = sorted(pdb_length_dict.items(), key=lambda x:x[1])

    with open(sorted_train_file, "w") as csv_file:
        writer = csv.writer(csv_file)
        for pdb_index, L in pdb_length_dict:
            writer.writerow([pdb_index, L])
    csv_file.close()

    ls = []
    for i in pdb_length_dict:
        ls.append(i[1])
    ls_np = np.array(ls)

    trunc_points = np.percentile(ls_np, (25, 50, 75), interpolation='midpoint')
    trunc_points = np.floor(trunc_points-1)
    trunc_points = np.append(np.array([64]),trunc_points)

    return trunc_points

trunc_points = cal_trunc_points()
np.save("/home/rotation3/complex-coor-pred/data/trunc_points", trunc_points)
