"""
label['T1096-D1'].keys()        <KeysViewHDF5 ['target_tokens', 'xyz']>
<HDF5 dataset "target_tokens": shape (1, 255), type "|i1">
<HDF5 dataset "xyz": shape (255, 3, 3), type "<f4">

embed_attn['T1096-D1'].keys()   <KeysViewHDF5 ['feature_2D', 'target_tokens', 'token_embeds']>
<HDF5 dataset "feature_2D": shape (1, 41, 255, 255), type "<f4">
<HDF5 dataset "target_tokens": shape (1, 255), type "|i1">
<HDF5 dataset "token_embeds": shape (1, 255, 2560), type "<f4">
"""

"""这三个蛋白质存在于embed_attn中，但是不存在于label中
T1027; T1044; T1064"""


import torch
import h5py
import time
import random
from config import device
from tools.align_svd import align_svd
from tools.cal_lddt_tensor import cal_lddt as cal_ture_lddt

from pLDDT.pLDDT import pLDDT
# 之前的pLDDT模型是针对L套坐标训练的，所以需要将拓展成L套。
# 之前的pLDDT模型是针对L套坐标训练的，所以需要将拓展成L套。
pLDDT = torch.load("/home/rotation3/complex-coor-pred/pLDDT/plddt_checkpoints/Full_train/epoch7_mark.pt",
                              map_location=device)
def cal_residue_lddt(pLDDT, pred_coor, pred_x2d):
    """
    :param pred_coor: [L,3]
    :param pred_x2d: [1,105,L,L]
    """
    L = pred_coor.shape[0]
    pred_coor = torch.tile(pred_coor[None][None], (1,L,1,1))
    lddt_residue = pLDDT(pred_coor, pred_x2d)
    return lddt_residue.squeeze().mean(dim=0)

def cal_lddt(pLDDT, pred_coor, pred_x2d):
    return cal_residue_lddt(pLDDT, pred_coor, pred_x2d).mean()

pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch16.pt"
model = torch.load(pt_name, map_location=device)

label = h5py.File("/export/disk4/for_Lijun/CASP/CASP14_91_domains_coords_seq.h5")
embed_attn = h5py.File("/export/disk4/for_Lijun/CASP/CASP14.h5")

label_list = label.keys()
input_list = embed_attn.keys()

initial_lddt_dict = {}
lddt_dict = {}
best_pred_coor_dict = {}

with torch.no_grad():
    for target in label_list:
    # for target in ["T1076-D1"]:
        embed = torch.from_numpy(embed_attn[target]["token_embeds"][:]).to(device)
        attn = torch.from_numpy(embed_attn[target]["feature_2D"][:]).to(device)
        if target in ["T1076-D1"]:
            embed = embed[:,1:,:]
            attn = attn[:,:,1:,1:]

        plabel = torch.from_numpy(label[target]["xyz"][:,1,:]).to(device)
        print("protein:", target)
        pred_coor, pred_x2d = model(embed, attn)
        print(pred_coor[3].shape)

        pred_coor = pred_coor[3][0]
        chain_lddt_ls = []
        for chain in pred_coor:
            # print(chain.shape, pred_x2d.shape)
            chain_lddt =  cal_lddt(pLDDT, chain, pred_x2d)
            # chain_lddt =  cal_lddt(chain, plabel)
            chain_lddt_ls.append(chain_lddt)
            lddt_tensor = torch.tensor(chain_lddt_ls)
        initial_lddt_max = lddt_tensor.max()
        initial_lddt_dict[target] = initial_lddt_max.item()
        
        indices = (lddt_tensor.topk(64)[1])
        pred_coor = pred_coor[indices]

        # 计算per_residue_lddt
        lddt_residue_ls = []
        for chain in pred_coor:
            lddt_residue = cal_residue_lddt(pLDDT, chain, pred_x2d)
            lddt_residue_ls.append(lddt_residue)
        lddt_score = torch.stack(lddt_residue_ls)
        # print(lddt_score.shape)   # torch.Size([64, 193])

        index = lddt_score.argmax(axis=0)
        
        L = len(index)
        best_chain = pred_coor[0]  # chain代表一整条链；seq代表一个序列片断
        best_chain_guarantee = best_chain.clone()
        stand_lddt = initial_lddt_max
        step = 0
        stop_step = 0
        beg = time.time()
        while step<500:
            if stop_step>25:
                break
            step += 1
            stop_step += 1
            # print("step {} begin".format(step+1))
            CLAMP = random.choice(range(3,51,2))   # 这只是一种妥协的做法。其实可以通过另外一种方法将其更精确的Align到一起。
            half_clamp = int((CLAMP-1)/2)
            for i in range(L):
                if index[i] == 0:
                    continue
                if i <= half_clamp:
                    seq4align = pred_coor[index[i]][:CLAMP]
                    model4align = best_chain[:CLAMP]
                elif (L-i) <= half_clamp:
                    seq4align = pred_coor[index[i]][-CLAMP:]
                    model4align = best_chain[-CLAMP:]
                else:
                    seq4align = pred_coor[index[i]][i-half_clamp:i+half_clamp+1]
                    model4align = best_chain[i-half_clamp:i+half_clamp+1]
                align_seq = align_svd(model4align, seq4align)
                best_chain_tmp = best_chain.clone()
                best_chain_tmp[i] = align_seq[half_clamp+1]
                lddt = cal_lddt(pLDDT, best_chain_tmp, pred_x2d)
                # lddt = cal_lddt(plabel, best_chain_tmp)   # 这里先使用真实的lddt，如果实验结果比较好，就再使用pLDDT
                if lddt > (stand_lddt+1e-5):
                    best_chain = best_chain_tmp.clone()
                    # print("improve!", (lddt-stand_lddt).item())
                    stand_lddt = lddt
                    # print("improved_lddt", stand_lddt.item())
                    stop_step = 0
                elif lddt > (stand_lddt-1e-3):
                    best_chain = best_chain_tmp.clone()
                    # print("about…", lddt-stand_lddt)
        final_lddt = cal_lddt(pLDDT, best_chain, pred_x2d)
        # final_lddt = cal_lddt(plabel, best_chain)
        if final_lddt < initial_lddt_max:
            best_chain = best_chain_guarantee
            final_lddt = initial_lddt_max
        lddt_dict[target] = final_lddt.item()
        best_pred_coor_dict[target] = best_chain

        print("{}, initial_lddt:{}, final_lddt:{}".format(target, initial_lddt_dict[target], lddt_dict[target]))
        print("true lddt:{}".format(cal_ture_lddt(best_chain, plabel)))


