# function: 先将预测出的 L3 转化为 LL3, 然后再计算FapeLoss
# Author: Jun Li


########################## _______init________ #############################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--model", type=str)
FLAGS = parser.parse_args()
NAME = FLAGS.name
model = FLAGS.model
import sys; sys.path.append("/home/rotation3/complex-coor-pred/")
if model=="basic":from AlignCoorConfusion.CoorConfusion import coorConfuse
elif model=="gate":from AlignCoorConfusion.CoorConfusionGate import coorConfuse
else:raise ValueError("wrong input of model! choose one between basic/gate")
#############################IMPORT##########################################
import os, sys, h5py, torch, numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import device
from torch.utils.data import DataLoader
from AlignCoorConfusion.assist_class import SeedSampler
from tools.cal_fapeloss import getFapeLoss
from utils import weight_init, seed_torch
###################################################################################


### define model
coor_confuse = coorConfuse().to(device)
# coor_confuse.apply(weight_init)
coor_confuse.load_state_dict(torch.load("AlignCoorConfusion/checkpoints/gate_I/epoch14.pt"))

### load Data & SummaryWriter
train_file = h5py.File("AlignCoorConfusion/h5py_data/train_dataset.h5py")
train_epoch_record = SummaryWriter("./AlignCoorConfusion/logs/" + NAME + "/train_epoch_record")

from torch.utils.data import Dataset
from data.MyData import MyData
train_data_path = '/home/rotation3/complex-coor-pred/data/train22310.3besm2.h5'
xyz_path = '/home/rotation3/complex-coor-pred/data/xyz.h5'
sorted_train_file = "/home/rotation3/complex-coor-pred/data/sorted_train_list.txt"
train_ds = MyData(train_data_path, xyz_path, sorted_train_file, train_mode=False)

num_epochs = 50
for epoch in range(15, num_epochs):
    if epoch < 25:
        learning_rate = 1e-3
    else:
        learning_rate = 1e-4
    opt = torch.optim.Adam(coor_confuse.parameters(), lr=learning_rate)
    train_record = SummaryWriter("./AlignCoorConfusion/logs/" + NAME +"/train_record/epoch" + str(epoch))

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
    total_diff_exp_loss = 0
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
        R = torch.from_numpy(np.array(train_file["protein"+str(index)]["rotation_matrix"],dtype=np.float32)).to(device)
        R = torch.inverse(R)
        R = torch.concat((torch.eye(3, device=device)[None,:,:], R), dim=0)

        t = torch.from_numpy(np.array(train_file["protein"+str(index)]["translation_matrix"],dtype=np.float32)).to(device)
        t = torch.concat((torch.zeros((1,1,3), device=device), t), dim=0)

        L = confused_coor.shape[-2]
        e_pred_coor_ls = []
        for chain in confused_coor:
            chain = torch.tile(chain.unsqueeze(0), (64,1,1))
            tmp_pred_coor = chain - t
            e_pred_coor = torch.einsum("nlc,ncq->nlq", tmp_pred_coor, R)
            e_pred_coor_ls.append(e_pred_coor)
        pred_coor = torch.stack(e_pred_coor_ls, dim=0)

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
        optim_loss = loss / 5

        ### 网络即将收敛时的loss; 可以当网络训练即将结束的时候，再调用带有pre_fapeloss的loss
        if epoch >= 35:
            pre_pred_coor = torch.from_numpy(np.array(train_file["protein"+str(index)]["pred_coor"], dtype=np.float32))
            pre_pred_coor = pre_pred_coor.to(device)
            label_coor = label_coor.to(device)
            label_coor = label_coor[0]
            pre_diff = pre_pred_coor - label_coor
            pre_diff = pre_diff.permute(0,3,1,2)
            pre_fapeloss, pre_realfape = getFapeLoss(pre_diff, dclamp=10)
            diff_exp_loss = torch.exp(fapeloss_10A - pre_fapeloss)
            diff_exp_loss = torch.clamp(loss, 20)
            total_diff_exp_loss += diff_exp_loss.item()
            avg_diff_exp_loss = total_diff_exp_loss / (global_step+1)
            train_record.add_scalar("diff_exp_loss", avg_diff_exp_loss, global_step+1)
            print("diff_exp_loss", diff_exp_loss)
            optim_loss = diff_exp_loss / 5
            # 当advance是负值的时候，代表着预测水平的进步；当为正值的时候，代表的fapeloss的提升

        total_fapeloss += fapeloss_10A.item()  # kpi
        total_loss += loss.item()  # loss to optim
        total_realfape += realfape.item()
        avg_fapeloss = total_fapeloss / (global_step + 1)
        avg_realfape = total_realfape / (global_step + 1)
        avg_loss = total_loss / (global_step + 1)

        optim_loss.backward()
        train_record.add_scalar("loss", avg_loss,global_step+1)
        train_record.add_scalar("fapeloss_10A", avg_fapeloss,global_step+1)
        train_record.add_scalar("realfape", avg_realfape, global_step+1)
        
        if global_step in range(2000) or global_step%100 == 0:
            print("global_step %d" % global_step)
            print("loss", avg_loss)
            print("fapeloss_10A", avg_fapeloss)
            print("realfape", avg_realfape)

        if i == 5:
            opt.step()
            opt.zero_grad()
            i = -1
    
    
    if not os.path.exists("AlignCoorConfusion/checkpoints/" +  NAME):
        os.mkdir("AlignCoorConfusion/checkpoints/" +  NAME)
    ### 所以现在保存的每一步都是最后一个步骤的模型 艹
    PATH = "AlignCoorConfusion/checkpoints/" +  NAME +"/epoch" + str(epoch) + ".pt"
    torch.save(coor_confuse.state_dict(), PATH)
    ### 加载模型参数
    # the_model = TheModelClass(*args, **kwargs)
    # the_model.load_state_dict(torch.load(PATH))

    train_epoch_record.add_scalar("fapeloss_10A", avg_fapeloss,epoch+1)
    train_epoch_record.add_scalar("realfape", avg_realfape,epoch+1)
    train_epoch_record.add_scalar("loss", avg_loss,epoch+1)


train_file.close()
