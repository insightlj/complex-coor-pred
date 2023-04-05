# function: 测试coor_confuse模型的性能
# Author: Jun Li

import torch
import numpy as np
from torch import nn
from config import device
import h5py
from utils.AlignCoorConfusion.CoorConfusion import coorConfuse

from torch.utils.data import DataLoader
from main import train_ds, test_ds
from torch.utils.tensorboard import SummaryWriter
from utils.init_parameters import weight_init
from utils.fapeloss import getFapeLoss
from utils.cal_lddt_single import cal_lddt

train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

coor_confuse = coorConfuse().to(device)
coor_confuse = torch.load(model_pt_name)
coor_confuse.eval()

### test model
test_epoch_record = SummaryWriter("./utils/AlignCoorConfusion/logs/test_epoch_record")
test_record = SummaryWriter("./utils/AlignCoorConfusion/logs/test_record")
lddt_record = SummaryWriter("./utils/AlignCoorConfusion/log/lddt")

with h5py.File("utils/AlignCoorConfusion/h5py_data/test_dataset.h5py", "r") as test_file:
    with torch.no_grad():
        global_step = -1
        total_fapeloss = 0
        total_loss = 0
        for data in test_dataloader:
            global_step += 1
            pred_coor = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["aligned_chains"], dtype=np.float32)).to(device)
            pred_x2d = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["pred_x2d"], dtype=np.float32)).to(device)
            lddt_score = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["lddt_score"], dtype=np.float32)).to(device)
            indices = torch.from_numpy(np.array(test_file["protein"+str(global_step)]["indices"], dtype=np.compat.long)).cpu()
            indices = indices.squeeze()

            import time
            beg = time.time()
            pred_coor_c_attn = pred_coor.unsqueeze(0)
            pred_coor_r_attn = pred_coor.unsqueeze(0)

            confused_coor = coor_confuse(pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score)
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

            test_record.add_scalar("fapeloss", avg_fapeloss,global_step)
            # train_record.add_scalar("advance", advance.item(),global_step)
            test_record.add_scalar("loss", avg_loss,global_step)
            
            if global_step%100 == 0:
                print("global_step %d" % global_step)
                print("fapeloss", avg_fapeloss)
                # print("advance", advance.item())
                print("loss", avg_loss)
            

            # 64_lddt
            pre_lddt_score = lddt_score.squeeze().mean(dim=-1, keepdim=False)
            pre_top4_lddt_score, _ = torch.topk(pre_lddt_score, k=4)
            pre_lddt_mean = pre_top4_lddt_score.mean()
            pre_lddt_max = pre_top4_lddt_score.max()

            # 4_lddt
            print(single_label.shape)
            print(pred_coor.shape)
            single_label = label_coor[0,:,:]
            top4_lddt_score = []
            for chain in pred_coor:
                top4_lddt_score.append(cal_lddt(single_label, chain))
            lddt_mean = top4_lddt_score.mean()
            lddt_max = top4_lddt_score.max()

            lddt_record.add_scalars("lddt_mean",
                                    {"pre_lddt_mean":pre_lddt_mean,
                                    "lddt_mean":lddt_mean},
                                    global_step)
            lddt_record.add_scalars("lddt_max",
                                    {"pre_lddt_mean":pre_lddt_max,
                                    "lddt_mean":lddt_max},
                                    global_step)

            
