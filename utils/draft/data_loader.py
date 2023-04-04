from main import train_ds
from data.MyData import MyData
from torch.utils.data import DataLoader

# train_ds = MyData('/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5', 
#                   '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5',
#                   "/home/rotation3/example/sorted_train_list.txt",
#                   train_mode=False)
train_dl = DataLoader(train_ds, batch_size=1, shuffle=False)
q = 0
for i in train_dl:
    _, _, label, L = i
    print(L)
    q += 1
    print(q)
