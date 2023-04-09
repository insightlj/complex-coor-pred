import torch
from torch import nn


def getFapeLoss(diff, dclamp=10, ratio=0.1, lossformat='dclamp'):
    ### true label nan has been masked
    ### diff.shape=(N,3,L,L)
    ### return fapeLoss,realFape

    eps = 1e-8
    L = diff.shape[-1]
    diff = diff[:, :, :, None, :] - diff[:, :, :, :, None]  # (N,3,64,L,L)  # 计算出坐标之间的差值
    diff = torch.sqrt(torch.sum(diff ** 2, dim=1) + eps) # diff实为fapeNLLL (N,64,L,L)

    realFape = torch.mean(diff)  # FapeLoss和lddt的思想相似，均计算原子之间相对坐标的预测准确度

    if lossformat == 'dclamp':
        mask = torch.as_tensor(diff <= dclamp, dtype=torch.float32)  # (N,64,L,L)
        mask = mask * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == 'ReluDclamp':
        diff = ratio * nn.ReLU(inplace=True)(diff - dclamp) + dclamp - nn.ReLU(inplace=True)(
            dclamp - diff)  # 考虑换成mask形式
        mask = torch.ones_like(diff) * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == 'NoDclamp':
        mask = torch.ones_like(diff) * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == 'probDclamp':
        maskdclamp = torch.as_tensor(diff <= dclamp, dtype=torch.float32)  # (N,L,L,L)
        maskdclamp = maskdclamp * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        maskprob = torch.as_tensor(torch.rand([L, L]) >= (1 - ratio),
                                   dtype=torch.float32)  # uniform in [0,1), 按照概率ratio提取dclamp之外的残基对
        maskprob = torch.triu(maskprob, diagonal=1)
        maskprob = maskprob + maskprob.permute(1, 0)
        maskprob = maskprob.to(diff.device)
        mask = maskprob * (1 - maskdclamp) + maskdclamp
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    del mask
    return fapeLoss, realFape


if __name__ == "__main__":
    import sys
    sys.path.append("/home/rotation3/complex-coor-pred/")

    import torch
    import h5py
    import time
    from main import test_ds
    from torch.utils.data import DataLoader
    from config import NUM_BLOCKS,device,eps
    import os
    from data.MyData import MyData

    data_path = '/export/disk1/hujian/cath_database/esm2_3B_targetEmbed.h5'
    xyz_path = '/export/disk1/hujian/Model/Model510/GAT-OldData/data/xyz.h5'
    sorted_train_file = "/home/rotation3/example/sorted_train_list.txt"
    test_file = "/home/rotation3/example/valid_list.txt"

    train_ds = MyData(data_path, xyz_path, sorted_train_file, train_mode=True)

    
    # load data
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch16.pt"
    train_dataloader = DataLoader(train_ds, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    net_pt = torch.load(model_pt_name, map_location=device)

    from torch.utils.tensorboard import SummaryWriter
    with torch.no_grad():
        i = -1
        total_fapeloss = 0
        for data in train_dataloader:
            beg = time.time()
            i += 1
            embed, atten, coor_label, L = data
            embed = embed.to(device)
            atten = atten.to(device)
            coor_label = coor_label.to(device)
            L = L.to(device)
            pred_coor_4_blocks, pred_x2d = net_pt(embed, atten)
            pred_coor = pred_coor_4_blocks[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

            diff = pred_coor - coor_label
            diff = diff.permute(0,3,1,2)
            print(diff.shape)
            fapeloss, realfape = getFapeLoss(diff)
            print(fapeloss, realfape)





if __name__ == '__main __':
    ### 计算训练集和测试集的平均fapeloss
    import sys
    sys.path.append("/home/rotation3/complex-coor-pred/")

    import torch
    import h5py
    import time
    from main import train_ds, test_ds
    from torch.utils.data import DataLoader
    from config import NUM_BLOCKS,device,eps
    import os

    
    # load data
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch16.pt"
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    net_pt = torch.load(model_pt_name, map_location=device)

    from torch.utils.tensorboard import SummaryWriter
    fapeloss_record = SummaryWriter("/home/rotation3/complex-coor-pred/logs/CoorNet_VII/fapeloss/epoch16/train")
    with torch.no_grad():
        i = -1
        total_fapeloss = 0
        for data in train_dataloader:
            beg = time.time()
            i += 1
            embed, atten, coor_label, L = data
            embed = embed.to(device)
            atten = atten.to(device)
            coor_label = coor_label.to(device)
            L = L.to(device)
            pred_coor_4_blocks, pred_x2d = net_pt(embed, atten)
            pred_coor = pred_coor_4_blocks[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

            diff = pred_coor - coor_label
            diff = diff.permute(0,3,1,2)
            fapeloss, realfape = getFapeLoss(diff)
            total_fapeloss += fapeloss
            avg_fapeloss = total_fapeloss / (i+1)
            fapeloss_record.add_scalar("avg_fapeloss", avg_fapeloss, i)

    fapeloss_record = SummaryWriter("/home/rotation3/complex-coor-pred/logs/CoorNet_VII/fapeloss/epoch16/test")
    with torch.no_grad():
        i = -1
        total_fapeloss = 0
        for data in test_dataloader:
            beg = time.time()
            i += 1
            embed, atten, coor_label, L = data
            embed = embed.to(device)
            atten = atten.to(device)
            coor_label = coor_label.to(device)
            L = L.to(device)
            pred_coor_4_blocks, pred_x2d = net_pt(embed, atten)
            pred_coor = pred_coor_4_blocks[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

            diff = pred_coor - coor_label
            diff = diff.permute(0,3,1,2)
            fapeloss, realfape = getFapeLoss(diff)
            total_fapeloss += fapeloss
            avg_fapeloss = total_fapeloss / (i+1)
            fapeloss_record.add_scalar("avg_fapeloss", avg_fapeloss, i)