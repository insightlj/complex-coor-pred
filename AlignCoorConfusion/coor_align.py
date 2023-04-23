# funtion: 将2-64条序列Align到第一条序列上. 这个过程没有参数, 使用的是scipy.optimize提供的BFGS和Powell
# Author: Jun Li
"""
H5py数据结构与Shape:
<HDF5 dataset "aligned_chains": shape (64, 64, 3), type "<f4">
<HDF5 dataset "indices": shape (1, 64), type "<i8">
<HDF5 dataset "lddt_score": shape (1, 64, 64), type "<f4">
<HDF5 dataset "pred_coor": shape (1, 64, 64, 3), type "<f4">
<HDF5 dataset "pred_x2d": shape (1, 105, 64, 64), type "<f4">
<HDF5 dataset "rotation_matrix": shape (63, 3, 3), type "<f4">
<HDF5 dataset "translation_matrix": shape (63, 1, 3), type "<f4">   # 与之前唯一的区别
"""

import sys
sys.path.append("/home/rotation3/complex-coor-pred/")

import numpy as np
import torch

def select_chains(lddt_score, num_chains=64):
    lddt_score_chains = lddt_score.sum(dim=2)
    _, indices = lddt_score_chains.topk(num_chains)
    return indices

def svd_align(A, B):
    """ 使用svd的方法进行坐标的Align
    
    :param A: model_chain
    :param B: other_chain
    :return: aligned_chain
    """
    centroid_A = A.mean(-2)
    centroid_B = B.mean(-2)
    AA = A - centroid_A.unsqueeze(-2)
    BB = B - centroid_B.unsqueeze(-2)
    H = torch.matmul(BB.transpose(-2, -1), AA)
    U, S, V = torch.svd(H, some=False)
    R = torch.matmul(V, U.transpose(-2,-1))
    t = -torch.matmul(R, centroid_B.unsqueeze(-1)) + centroid_A.unsqueeze(-1)
    R = R.transpose(-2,-1)
    t = t.reshape(1, 3)
    B = torch.matmul(B, R) + t
    global rotation_matrix_ls
    global translation_matrix_ls
    rotation_matrix_ls.append(R)
    translation_matrix_ls.append(t)
    return B

def align_chain(NL3):
    # array
    # input(NL3): 同一蛋白不同坐标系下的表示
    # output(NL3): 将这些蛋白从坐标上Align到一起
    num_chains = NL3.shape[0]
    model_chain = NL3[0]
    chain_ls = []
    chain_ls.append(model_chain)
    for index in range(1,num_chains):
        other_chain = NL3[index]
        chain_ls.append(svd_align(model_chain, other_chain))
    aligned_chains = torch.stack(chain_ls, dim=0)
    return aligned_chains

def lddtGate(lddt_score):
    """
    fucntion: compute lddt to gate
    type: tensor
    input: lddt_score, BNL
    output: BNL
    max_lddt_chain = lddt_score.mean(dim=2).argmax()
    """
    lddt_gate = torch.clone(lddt_score)
    lddt_gate[lddt_gate<0.5] = 0
    lddt_gate[lddt_gate>=0.5] = 1
    lddt_gate[0,0,:] = 1  
    # 保证每个位点都至少有一个可用的值，所以对lddt值最大的那一条链全部保留
    # 由于在做topk的过程中做了排序，所以第一个的lddt值最高
    return lddt_gate

from tools.cal_lddt_multiseq import cal_lddt
def get_trueLDDT(pred, label):
    """
    pred: BLL3
    label: BLL3
    return truelddt_score BLL
    """
    pred = pred.unsqueeze(-2) - pred.unsqueeze(-3)
    pred = ((pred**2).sum(dim=-1) + eps) ** 0.5
    label = ((label**2).sum(dim=-1) + eps) ** 0.5
    truelddt_score = cal_lddt(pred, label)
    return truelddt_score
    
if __name__ == '__main__':
    import torch
    import h5py
    import time
    from main import train_ds, test_ds
    from torch.utils.data import DataLoader
    from config import NUM_BLOCKS,device,eps
    
    # load lddt compute model
    from pLDDT.pLDDT import pLDDT
    get_pLDDT = torch.load("/home/rotation3/complex-coor-pred/pLDDT/plddt_checkpoints/Full_train/epoch7_mark.pt")
    
    # load data
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch16.pt"
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    net_pt = torch.load(model_pt_name, map_location=device)

    with h5py.File("AlignCoorConfusion/h5py_data/train_dataset.h5py", "a") as train_file:
        with torch.no_grad():
            i = -1
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

                # compute lddt
                # plddt_score = get_pLDDT(pred_coor, pred_x2d)   # 在测试的时候使用pLDDT, 因为测试就是为了完全模仿真实情况
                truelddt_score = get_trueLDDT(pred_coor, coor_label)   # 在训练的时候使用trueLDDT(或许可以两个混着用？相当于数据增强，毕竟之前的pLDDT中，应该保存了LDDT之外的信息。但是我做筛选，使用的就是LDDT，其他是不用的。所以还是越准确越好)
                lddt_score = truelddt_score

                # select top 64 highest lddt chains
                # return new pred_coor & lddt_score
                indices = select_chains(lddt_score)
                pred_coor = pred_coor[0,indices,:]
                lddt_score = lddt_score[0,indices,:]

                # !!!!COOR ALIGN!!!! 
                # 将pred转化为numpy, 从而进行align, 得到align_chains
                pred = pred_coor.squeeze()
                rotation_matrix_ls = []
                translation_matrix_ls = []
                aligned_chains = align_chain(pred)
                rotation_matrix = torch.stack(rotation_matrix_ls, dim=0)
                translation_matrix = torch.stack(translation_matrix_ls, dim=0)
                
                # # 根据lddt_score做mask, 筛选掉lddt比较低的部分, 得到gated_aligned_chains
                # lddt_gate = lddtGate(lddt_score)
                # lddt_gate = lddt_gate.bool().squeeze().cpu()
                # lddt_gate = torch.tile(lddt_gate[:,:,None], (1,1,3))
                # gated_aligned_chains = torch.masked_fill(aligned_chains, ~lddt_gate, value=0)

                
                protein = train_file.create_group("protein" + str(i))
                protein["aligned_chains"] = aligned_chains.cpu()
                protein["rotation_matrix"] = rotation_matrix.cpu()
                protein["translation_matrix"] = translation_matrix.cpu()
                protein["indices"] = indices.cpu()
                protein["lddt_score"] = lddt_score.cpu()
                protein["pred_x2d"] = pred_x2d.cpu()
                protein["pred_coor"] = pred_coor.cpu()
                # if "indices" in train_file["protein" + str(i)].keys():
                #     del train_file["protein" + str(i)]["indices"]
                print("protein{} saved! time usage:{}".format(i, time.time()-beg))

    ### ____test_dataset save____
    with torch.no_grad():
        i = -1
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

            # compute lddt
            plddt_score = get_pLDDT(pred_coor, pred_x2d)   # 在测试的时候使用pLDDT, 因为测试就是为了完全模仿真实情况
            # truelddt_score = get_trueLDDT(pred_coor, coor_label)   # 在训练的时候使用trueLDDT(或许可以两个混着用？相当于数据增强，毕竟之前的pLDDT中，应该保存了LDDT之外的信息。但是我做筛选，使用的就是LDDT，其他是不用的。所以还是越准确越好)
            lddt_score = plddt_score

            # select top 64 highest lddt chains
            # return new pred_coor & lddt_score
            indices = select_chains(lddt_score)
            pred_coor = pred_coor[0,indices,:]
            lddt_score = lddt_score[0,indices,:]

            
            # !!!!COOR ALIGN!!!! 
            # 将pred转化为numpy, 从而进行align, 得到align_chains
            pred = pred_coor.squeeze()
            rotation_matrix_ls = []
            translation_matrix_ls = []
            aligned_chains = align_chain(pred)
            rotation_matrix = torch.stack(rotation_matrix_ls)
            translation_matrix = torch.stack(translation_matrix_ls)
            
            # # 根据lddt_score做mask, 筛选掉lddt比较低的部分, 得到gated_aligned_chains 
            # lddt_gate = lddtGate(lddt_score) 
            # lddt_gate = lddt_gate.bool().squeeze().cpu() 
            # lddt_gate = torch.tile(lddt_gate[:,:,None], (1,1,3)) 
            # gated_aligned_chains = torch.masked_fill(aligned_chains, ~lddt_gate, value=0) 
            

            with h5py.File("AlignCoorConfusion/h5py_data/test_dataset.h5py", "a") as test_file:
                protein = test_file.create_group("protein" + str(i))
                protein["aligned_chains"] = aligned_chains.cpu()     # NL3
                protein["rotation_matrix"] = rotation_matrix.cpu()    # N33
                protein["translation_matrix"] = translation_matrix.cpu()  # N3
                protein["indices"] = indices.cpu()
                protein["lddt_score"] = lddt_score.cpu()
                protein["pred_x2d"] = pred_x2d.cpu()
                protein["pred_coor"] = pred_coor.cpu()
            print("protein{} saved! time usage:{}".format(i, time.time()-beg))