# function: 先将预测出的 L3 转化为 LL3, 然后再计算FapeLoss
# Author: Jun Li

import torch
import numpy as np
import os
from config import device
import h5py
from utils.AlignCoorConfusion.CoorConfusion import coorConfuse
from utils.fapeloss import getFapeLoss
from torch.utils.data import DataLoader
from main import train_ds, test_ds
from torch.utils.tensorboard import SummaryWriter
from utils.init_parameters import weight_init
from utils.set_seed import seed_torch


class SeedSampler():
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed
    def __iter__(self):
        seed_torch(self.seed)
        seed_random_ls = torch.randperm(len(self.data_source))
        return iter(seed_random_ls)
    def __len__(self):
        return len(self.data_source)


### define model
NAME = "basic"
coor_confuse = coorConfuse().to(device)
coor_confuse.apply(weight_init)
opt = torch.optim.Adam(coor_confuse.parameters(), lr=1e-3)

### load Data & SummaryWriter
train_file = h5py.File("utils/AlignCoorConfusion/h5py_data/train_dataset.h5py")
train_epoch_record = SummaryWriter("./utils/AlignCoorConfusion/logs/" + NAME + "/train_epoch_record")

num_epochs = 10
for epoch in range(num_epochs):
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
    total_fapeloss = 0
    total_loss = 0
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

        confused_coor = coor_confuse(pred_coor_r_attn, pred_coor_c_attn, pred_x2d, lddt_score)
        confused_coor = confused_coor.squeeze().permute(2,1,0)
        # print("inference time: ", time.time()-beg)

        ### compute label & loss
        rotationMatrix = torch.from_numpy(np.array(train_file["protein"+str(index)]["rotation_matrix"],dtype=np.float32)).to(device)
        rotationMatrix = torch.inverse(rotationMatrix)
        rotationMatrix = torch.concat((torch.eye(3, device=device)[None,:,:], rotationMatrix), dim=0)

        translationVector = torch.from_numpy(np.array(train_file["protein"+str(index)]["translation_matrix"],dtype=np.float32)).to(device)
        translationVector = torch.concat((torch.zeros(1,3, device=device), translationVector), dim=0)

        L = confused_coor.shape[-2]
        pred_coor_tmp = torch.tile(confused_coor[:,None,:,:], (1,64,1,1)) - torch.tile(translationVector[None,:,None,:], (1,1,L,1))

        pred_coor = torch.einsum("bchw, cwq -> bchq", pred_coor_tmp, rotationMatrix)


        ### 还要把预测出来的pred_coor保存下来。。。害，算了算了，做一次label，后面就都可以用了
        pre_pred_coor = torch.from_numpy(np.array(train_file["protein"+str(index)]["pred_coor"], dtype=np.float32))
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
        train_record.add_scalar("loss", avg_loss,global_step)
        
        if global_step in range(2000) or global_step%100 == 0:
            print("global_step %d" % global_step)
            print("fapeloss", avg_fapeloss)
            print("loss", avg_loss)

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

    train_epoch_record.add_scalar("fapeloss", avg_fapeloss,epoch+1)
    train_epoch_record.add_scalar("loss", avg_loss,epoch+1)

train_file.close()
