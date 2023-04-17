# function: 测试coor_confuse模型的性能
# Author: Jun Li

import sys
sys.path.append("/home/rotation3/complex-coor-pred/")

import torch
import numpy as np
from torch import nn
from config import device
import h5py

from torch.utils.data import DataLoader
from main import train_ds, test_ds
from torch.utils.tensorboard import SummaryWriter
from scripts.cal_fapeloss import getFapeLoss
from scripts.cal_lddt_tensor import cal_lddt

train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

from AlignCoorConfusion.CoorConfusionGate import coorConfuse
coor_confuse = coorConfuse().to(device)
NAME = "gate_II"
EPOCH = 33
model_pt_name = "/home/rotation3/complex-coor-pred/utils/AlignCoorConfusion/checkpoints/" + NAME + "/epoch"+ str(EPOCH) +".pt"
# model_pt_name = "/home/rotation3/complex-coor-pred/utils/AlignCoorConfusion/checkpoints/demo-1/epoch0.pt"
coor_confuse.load_state_dict(torch.load(model_pt_name))

### test model
test_epoch_record = SummaryWriter("./utils/AlignCoorConfusion/logs/" + NAME + "/test_epoch_record")
test_record = SummaryWriter("./utils/AlignCoorConfusion/logs/" + NAME + "/test_record")
lddt_record = SummaryWriter("./utils/AlignCoorConfusion/logs/" + NAME + "/lddt")

with torch.no_grad():
    with h5py.File("AlignCoorConfusion/h5py_data/test_dataset.h5py", "r") as test_file:
        global_step = -1
        count = 0
        total_fapeloss = 0
        total_loss = 0
        total_lddt_max = 0
        total_lddt_mean = 0
        total_pre_lddt_mean = 0
        total_pre_lddt_max = 0
        for data in test_dataloader:
            global_step += 1
            pred_coor = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["aligned_chains"], dtype=np.float32)).to(device)
            pred_x2d = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["pred_x2d"], dtype=np.float32)).to(device)
            lddt_score = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["lddt_score"], dtype=np.float32)).to(device)
            indices = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["indices"], dtype=np.compat.long)).cpu()
            indices = indices.squeeze()

            pred_coor_c_attn = pred_coor.unsqueeze(0)
            pred_coor_r_attn = pred_coor.unsqueeze(0)
            pred_coor = pred_coor.unsqueeze(0)

            confused_coor = coor_confuse(pred_coor, pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score)
            confused_coor = confused_coor.squeeze().permute(2,1,0)
            # print("inference time: ", time.time()-beg)

            ### compute label & loss
            rotationMatrix = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["rotation_matrix"],dtype=np.float32)).to(device)
            rotationMatrix = torch.inverse(rotationMatrix)
            rotationMatrix = torch.concat((torch.eye(3, device=device)[None,:,:], rotationMatrix), dim=0)

            translationVector = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["translation_matrix"],dtype=np.float32)).to(device)
            translationVector = torch.concat((torch.zeros(1,3, device=device), translationVector), dim=0)

            L = confused_coor.shape[-2]
            pred_coor_tmp = torch.tile(confused_coor[:,None,:,:], (1,64,1,1)) - torch.tile(translationVector[None,:,None,:], (1,1,L,1))
            pred_coor = torch.einsum("bchw, cwq -> bchq", pred_coor_tmp, rotationMatrix)

            _, _, full_label_coor, _ = data

            ### 还要把预测出来的pred_coor保存下来。。。害，算了算了，做一次label，后面就都可以用了
            pre_pred_coor = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["pred_coor"], dtype=np.float32))
            # label_coor = full_label_coor[:,indices[:4],:,:]
            # pre_pred_coor = pre_pred_coor[:,:4,:,:]
            label_coor = full_label_coor[:,indices,:,:]
            pre_pred_coor.to(device) 
            pre_diff = pre_pred_coor - label_coor
            pre_diff = pre_diff.permute(0,3,1,2)
            pre_fapeloss, pre_realfape = getFapeLoss(pre_diff)

            print(label_coor.shape, pred_coor.shape)
            label_coor = label_coor.to(device)
            diff = pred_coor - label_coor
            diff = diff.permute(0,3,1,2)
            fapeloss, realfape = 10000,10000
            for sdiff in diff:
                fapeloss_tmp, realfape_tmp = getFapeLoss(sdiff.unsqueeze(0))
                if fapeloss_tmp < fapeloss:
                    fapeloss = fapeloss_tmp
                    realfape = realfape_tmp

            loss = fapeloss - pre_fapeloss
            loss = torch.exp(loss)
            loss = torch.clamp(loss, max=10)
            # 也可以尝试sigmoid函数
            # 当advance是负值的时候，代表着预测水平的进步；当为正值的时候，代表的fapeloss的提升

            total_fapeloss += fapeloss.item()  # kpi
            total_loss += loss.item()  # loss to optim
            avg_fapeloss = total_fapeloss / (global_step + 1)
            avg_loss = total_loss / (global_step + 1)

            test_record.add_scalar("fapeloss", avg_fapeloss,global_step)
            test_record.add_scalar("loss", avg_loss,global_step)
            
            print("global_step %d" % global_step)
            print("fapeloss", avg_fapeloss)
            print("loss", avg_loss)
            

            # 64_lddt
            pre_lddt_score = lddt_score.squeeze().mean(dim=-1, keepdim=False)
            pre_top4_lddt_score, _ = torch.topk(pre_lddt_score, k=4)
            pre_lddt_mean = pre_top4_lddt_score.mean().item()
            pre_lddt_max = pre_top4_lddt_score.max().item()
            total_pre_lddt_mean += pre_lddt_mean
            total_pre_lddt_max += pre_lddt_max

            # 4_lddt
            single_label = (label_coor.squeeze())[0,:,:]  # 在所有的label中选择第一条序列作为label(经过测试，任意两条之间的lddt值均为1，所以选哪一条序列都无所谓)
            top4_lddt_score = []
            pred_coor = pred_coor[:,0,:,:]
            for chain in pred_coor:
                lddt = cal_lddt(single_label, chain)
                top4_lddt_score.append(lddt.item())
            lddt_mean = torch.mean(torch.tensor(top4_lddt_score)).item()
            lddt_max = torch.max(torch.tensor(top4_lddt_score)).item()
            total_lddt_mean += lddt_mean
            total_lddt_max += lddt_max

            if lddt_max > pre_lddt_max:
                count += 1
            print("pre_lddt_max:", pre_lddt_max)
            print("lddt_max:", lddt_max)
            print("_____________________________")
            lddt_record.add_scalars("lddt_mean",
                                    {"pre_lddt_mean":total_pre_lddt_mean/(global_step+1),
                                    "lddt_mean":total_lddt_mean/(global_step+1)},
                                    global_step)
            lddt_record.add_scalars("lddt_max",
                                    {"pre_lddt_max":total_pre_lddt_max/(global_step+1),
                                    "lddt_max":total_lddt_max/(global_step+1)},
                                    global_step)
        print("max_lddt improvement percent:", count / global_step)
        # epoch33: max_lddt improvement percent: 0.051836940110719675



            
