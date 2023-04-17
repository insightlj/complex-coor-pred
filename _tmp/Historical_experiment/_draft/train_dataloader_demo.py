from data.MyData import MyData
from torch.utils.data import DataLoader


train_data_path = '/home/rotation3/complex-coor-pred/data/train22310.3besm2.h5'
xyz_path = '/home/rotation3/complex-coor-pred/data/xyz.h5'
sorted_train_file = "/home/rotation3/complex-coor-pred/data/sorted_train_list.txt"
train_ds = MyData(train_data_path, xyz_path, sorted_train_file, train_mode=False)
train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
train_dataloader = iter(train_dataloader)

for i in train_dataloader:
    print(len(i))

# import h5py
# f = h5py.File(train_data_path, "r")
# ls = list(f["att40"].keys())

# ls_pre = []
# with open(sorted_train_file, "r") as fi:
#     for i in fi:
#         ls_pre.append(i.split(",")[0])
#     print(len(ls_pre))

# not_exist_protein_ls = []
# for i in ls_pre:
#     if i in ls:
#         print(i, "is in the ls")
#         pass
#     else:
#         not_exist_protein_ls.append(i)
# print(not_exist_protein_ls.__len__())