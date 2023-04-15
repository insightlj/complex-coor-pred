from scripts.cal_fapeloss import getFapeLoss

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
            