# function: 先将预测出的 L3 转化为 LL3, 然后再计算FapeLoss
# Author: Jun Li

import torch
import numpy as np
from torch import nn
from config import device
import h5py
from utils.AlignCoorConfusion.CoorConfusion import coorConfuse

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

from torch.utils.data import DataLoader
from main import train_ds, test_ds
from torch.utils.tensorboard import SummaryWriter
from utils.init_parameters import weight_init
train_file = h5py.File("utils/AlignCoorConfusion/h5py_data/train_dataset.h5py")
test_file = h5py.File("utils/AlignCoorConfusion/h5py_data/test_dataset.h5py")

train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

coor_confuse = coorConfuse().to(device)
coor_confuse.apply(weight_init)
opt = torch.optim.Adam(coor_confuse.parameters(), lr=1e-3)


### train model
num_epochs = 10
train_epoch_record = SummaryWriter("./utils/AlignCoorConfusion/logs/train_epoch_record")
train_record = SummaryWriter("./utils/AlignCoorConfusion/logs/train_record")
for epoch in range(num_epochs):
    i = -1
    global_step = -1
    total_fapeloss = 0
    total_loss = 0
    for data in train_dataloader:
        i += 1
        global_step += 1
        pred_coor = torch.from_numpy(np.array(train_file["protein"+str(global_step)]["aligned_chains"], dtype=np.float32)).to(device)
        pred_x2d = torch.from_numpy(np.array(train_file["protein"+str(global_step)]["pred_x2d"], dtype=np.float32)).to(device)
        lddt_score = torch.from_numpy(np.array(train_file["protein"+str(global_step)]["lddt_score"], dtype=np.float32)).to(device)
        indices = torch.from_numpy(np.array(train_file["protein"+str(global_step)]["indices"], dtype=np.compat.long)).cpu()
        indices = indices.squeeze()

        import time
        beg = time.time()
        pred_coor_c_attn = pred_coor.unsqueeze(0)
        pred_coor_r_attn = pred_coor.unsqueeze(0)

        confused_coor = coor_confuse(pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score)
        confused_coor = confused_coor.squeeze().permute(2,1,0)
        # print("inference time: ", time.time()-beg)

        ### compute label & loss
        rotationMatrix = torch.from_numpy(np.array(train_file["protein"+str(global_step)]["rotation_matrix"],dtype=np.float32)).to(device)
        rotationMatrix = torch.inverse(rotationMatrix)
        rotationMatrix = torch.concat((torch.eye(3, device=device)[None,:,:], rotationMatrix), dim=0)

        translationVector = torch.from_numpy(np.array(train_file["protein"+str(global_step)]["translation_matrix"],dtype=np.float32)).to(device)
        translationVector = torch.concat((torch.zeros(1,3, device=device), translationVector), dim=0)

        L = confused_coor.shape[-2]
        pred_coor_tmp = torch.tile(confused_coor[:,None,:,:], (1,64,1,1)) - torch.tile(translationVector[None,:,None,:], (1,1,L,1))

        pred_coor = torch.einsum("bchw, cwq -> bchq", pred_coor_tmp, rotationMatrix)

        _, _, full_label_coor, _ = data

        ### 还要把预测出来的pred_coor保存下来。。。害，算了算了，做一次label，后面就都可以用了
        pre_pred_coor = torch.from_numpy(np.array(train_file["protein"+str(global_step)]["pred_coor"], dtype=np.float32))
        label_coor = full_label_coor[:,indices,:,:]
        
        pre_pred_coor = pre_pred_coor[:,indices,:,:]
        pre_pred_coor.to(device)    
        pre_diff = pre_pred_coor - label_coor
        pre_diff = pre_diff.permute(0,3,1,2)
        pre_fapeloss, pre_realfape = getFapeLoss(pre_diff)

        label_coor = label_coor.to(device)
        diff = pred_coor - label_coor
        diff = diff.permute(0,3,1,2)
        fapeloss, realfape = getFapeLoss(diff)

        loss = fapeloss - pre_fapeloss  
        loss = torch.exp(loss)
        loss = torch.clamp(loss, max=10)
        # 也可以尝试sigmoid函数
        # 当advance是负值的时候，代表着预测水平的进步；当为正值的时候，代表的fapeloss的提升

        total_fapeloss += fapeloss.item()  # kpi
        total_loss += loss.item()  # loss to optim
        avg_fapeloss = total_fapeloss / (global_step + 1)
        avg_loss = total_loss / (global_step + 1)

        optim_loss = loss / 5
        optim_loss.backward()
        train_record.add_scalar("fapeloss", avg_fapeloss,global_step)
        # train_record.add_scalar("advance", advance.item(),global_step)
        train_record.add_scalar("loss", avg_loss,global_step)
        
        if global_step%100 == 0:
            print("global_step %d" % global_step)
            print("fapeloss", avg_fapeloss)
            # print("advance", advance.item())
            print("loss", avg_loss)

        if i == 5:
            opt.step()
            opt.zero_grad()
            i = -1

    train_epoch_record.add_scalar("fapeloss", avg_fapeloss,epoch+1)
    # train_epoch_record.add_scalar("advance", advance.item(),epoch+1)
    train_epoch_record.add_scalar("loss", avg_loss,epoch+1)

train_file.close()
test_file.close()
