# function: 先将预测出的 L3 转化为 LL3, 然后再计算FapeLoss
# Author: Jun Li


############ _______init________ ##############
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--device", type=int)
parser.add_argument("--model", type=str)
FLAGS = parser.parse_args()
NAME = FLAGS.name
model = FLAGS.model
device_index = FLAGS.device

import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]= str(device_index)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if model=="basic":
    from utils.AlignCoorConfusion.CoorConfusion import coorConfuse
elif model=="gate":
    from utils.AlignCoorConfusion.CoorConfusion_gate import coorConfuse
else:
    raise ValueError("wrong input of model! choose one between basic/gate")

import sys
sys.path.append("/home/rotation3/complex-coor-pred/")

import numpy as np
import os
import h5py
from utils.AlignCoorConfusion.assist_class import SeedSampler
from utils.fapeloss import getFapeLoss
from torch.utils.data import DataLoader
from main import train_ds
from torch.utils.tensorboard import SummaryWriter
from utils.init_parameters import weight_init
from utils.set_seed import seed_torch
#########################################


### define model
coor_confuse = coorConfuse().to(device)
# coor_confuse.load_state_dict(torch.load("utils/AlignCoorConfusion/checkpoints/basic/epoch0.pt"))
coor_confuse.apply(weight_init)
opt = torch.optim.Adam(coor_confuse.parameters(), lr=1e-3)

### load Data & SummaryWriter
train_file = h5py.File("utils/AlignCoorConfusion/h5py_data/train_dataset.h5py")
train_epoch_record = SummaryWriter("./utils/AlignCoorConfusion/logs/" + NAME + "/train_epoch_record")

num_epochs = 10
for epoch in range(0, num_epochs):
    train_record = SummaryWriter("./utils/AlignCoorConfusion/logs/" + NAME +"/train_record/epoch" + str(epoch))

    # use seed to set specific order on train_dataloader&train_file(h5py)
    seed = epoch  # 干脆把epoch当成seed好了
    seed_torch(seed)
    total_num = len(train_ds)
    seed_random_ls = torch.randperm(total_num)   # for train_file(h5py)
    mySampler = SeedSampler(train_ds, seed)  # for train_dataloader
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False, sampler=mySampler)
    train_dataloader = iter(train_dataloader)


    i = -1   # 用于梯度累加，记录梯度回传达到5个的时间点
    global_step = -1
    total_loss = 0
    total_fapeloss = 0
    total_realfape = 0
    for index in seed_random_ls:
        i += 1
        index = int(index)
        global_step += 1
        pred_coor = torch.from_numpy(np.array(train_file["protein"+str(index)]["aligned_chains"], dtype=np.float32)).to(device)
        pred_x2d = torch.from_numpy(np.array(train_file["protein"+str(index)]["pred_x2d"], dtype=np.float32)).to(device)
        lddt_score = torch.from_numpy(np.array(train_file["protein"+str(index)]["lddt_score"], dtype=np.float32)).to(device)
        indices = torch.from_numpy(np.array(train_file["protein"+str(index)]["indices"], dtype=np.compat.long)).cpu()
        indices = indices.squeeze()

        data = next(train_dataloader)
        _, _, full_label_coor, _ = data
        if full_label_coor.shape[-2] != lddt_score.shape[-1]:
            print(full_label_coor.shape, lddt_score.shape)
            raise ValueError("随机数失败……")

        import time
        beg = time.time()
        pred_coor_c_attn = pred_coor.unsqueeze(0)
        pred_coor_r_attn = pred_coor.unsqueeze(0)


        if model=="basic":
            confused_coor = coor_confuse(pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score)
        elif model=="gate":
            pred_coor = pred_coor.unsqueeze(0)
            confused_coor = coor_confuse(pred_coor, pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score)
        else:
            raise ValueError("wrong input of model! choose one between basic/gate")
        confused_coor = confused_coor.squeeze().permute(2,1,0)

        ### compute label & loss
        rotationMatrix = torch.from_numpy(np.array(train_file["protein"+str(index)]["rotation_matrix"],dtype=np.float32)).to(device)
        rotationMatrix = torch.inverse(rotationMatrix)
        rotationMatrix = torch.concat((torch.eye(3, device=device)[None,:,:], rotationMatrix), dim=0)

        translationVector = torch.from_numpy(np.array(train_file["protein"+str(index)]["translation_matrix"],dtype=np.float32)).to(device)
        translationVector = torch.concat((torch.zeros(1,3, device=device), translationVector), dim=0)

        L = confused_coor.shape[-2]
        pred_coor_tmp = torch.tile(confused_coor[:,None,:,:], (1,64,1,1)) - torch.tile(translationVector[None,:,None,:], (1,1,L,1))
        pred_coor = torch.einsum("bchw, cwq -> bchq", pred_coor_tmp, rotationMatrix)

        
        # 可以当网络训练即将结束的时候，再调用带有pre_fapeloss的loss
        # pre_pred_coor = torch.from_numpy(np.array(train_file["protein"+str(index)]["pred_coor"], dtype=np.float32))
        # pre_pred_coor.to(device)    
        # pre_diff = pre_pred_coor - label_coor
        # pre_diff = pre_diff.permute(0,3,1,2)
        # pre_fapeloss, pre_realfape = getFapeLoss(pre_diff, dclamp=50)

        label_coor = full_label_coor[:,indices,:,:]
        label_coor = label_coor.to(device)
        label_coor = torch.tile(label_coor,(4,1,1,1))
        diff = pred_coor - label_coor
        diff = diff.permute(0,3,1,2)
        fapeloss_10A, _ = getFapeLoss(diff, dclamp=10)
        fapeloss_25A, _ = getFapeLoss(diff, dclamp=25)
        fapeloss_50A, realfape = getFapeLoss(diff, dclamp=50)

        loss = (fapeloss_10A + fapeloss_25A + fapeloss_50A) / 3
        # 设计这种loss可以给diff中较小的值更高的权重，减小diff较大的部分对loss的影响
        loss = torch.clamp(loss, max=25)

        ### 网络即将收敛时的loss
        # loss = torch.exp(fapeloss_10A - pre_fapeloss)
        # loss = torch.clamp(loss, 10)
        # 当advance是负值的时候，代表着预测水平的进步；当为正值的时候，代表的fapeloss的提升

        total_fapeloss += fapeloss_10A.item()  # kpi
        total_loss += loss.item()  # loss to optim
        total_realfape += realfape.item()
        avg_fapeloss = total_fapeloss / (global_step + 1)
        avg_realfape = total_realfape / (global_step + 1)
        avg_loss = total_loss / (global_step + 1)

        optim_loss = loss / 5
        optim_loss.backward()
        train_record.add_scalar("loss", avg_loss,global_step)
        train_record.add_scalar("fapeloss_10A", avg_fapeloss,global_step)
        train_record.add_scalar("realfape", avg_realfape, global_step)
        
        if global_step in range(2000) or global_step%100 == 0:
            print("global_step %d" % global_step)
            print("loss", avg_loss)
            print("fapeloss_10A", avg_fapeloss)
            print("realfape", avg_realfape)

        if i == 5:
            opt.step()
            opt.zero_grad()
            i = -1
    
    
    if not os.path.exists("utils/AlignCoorConfusion/checkpoints/" +  NAME):
        os.mkdir("utils/AlignCoorConfusion/checkpoints/" +  NAME)
        PATH = "utils/AlignCoorConfusion/checkpoints/" +  NAME +"/epoch" + str(epoch) + ".pt"
        torch.save(coor_confuse.state_dict(), PATH)
    ### 加载模型参数
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))

    train_epoch_record.add_scalar("fapeloss_10A", avg_fapeloss,epoch+1)
    train_epoch_record.add_scalar("realfape", avg_realfape,epoch+1)
    train_epoch_record.add_scalar("loss", avg_loss,epoch+1)


train_file.close()
