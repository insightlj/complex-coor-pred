# Function: 将之前Align到一条链上的思路改成随机Align到任意含有极大值的链上
# 结果并不Work

import sys
sys.path.append("/home/rotation3/complex-coor-pred/")

import h5py
import time
import random
import torch
import numpy as np
from tools.align_svd import align_svd
from tools.cal_lddt_tensor import cal_lddt
from toolsom_dataset import sample_only

protein_index_ls = []
initial_lddt_ls = []
final_lddt_ls = []
ideal_lddt_ls = []
print("===nohup_begin===")
for protein_index in range(0,20000,40):
    # print("=========protein{}================".format(protein_index))
    train_file = h5py.File("/home/rotation3/complex-coor-pred/AlignCoorConfusion/h5py_data/train_dataset.h5py", "r")
    protein = train_file["protein"+str(protein_index)]

    _,_,label,_ = sample_only(train_mode=True, index=protein_index)
    label = label[0]
    label = torch.from_numpy(np.array(label))
    pred_coor = torch.from_numpy(np.array(protein["pred_coor"]).squeeze())
    rotation_matrix = torch.from_numpy(np.array(protein["rotation_matrix"]).squeeze())
    trasslation_matrix = torch.from_numpy(np.array(protein["translation_matrix"]).squeeze())
    lddt_score = torch.from_numpy(np.array(protein["lddt_score"]).squeeze())

    stand_lddt = lddt_score[0].mean()  # 0.848
    initial_stand_lddt = stand_lddt.clone()
    lddt_score.mean()  # 0.782
    idea_lddt = lddt_score.max(axis=0).values.mean()   # 0.891

    index = lddt_score.argmax(axis=0)   # protein的每个位置对应的最大值的位置
    L = len(index)
    best_chain = pred_coor[0]  # chain代表一整条链；seq代表一个序列片断
    best_chain_guarantee = best_chain.clone() 

    select_index_ls = torch.unique(index)
    # for s_index in select_index_ls:
    step = 0
    stop_step = 0
    beg = time.time()
    for epoch in range(10):
        if stop_step>100:
            print("stop at {} step".format(step))
            break

        for s_index in select_index_ls:
            good_chain = pred_coor[s_index]
            for _ in range(50):
                step += 1
                stop_step += 1
                for i in range(L):   # 遍历一条序列的每一个残基位点
                    CLAMP = random.choice(range(3,51,2))   # 这只是一种妥协的做法。其实可以通过另外一种方法将其更精确的Align到一起。
                    half_clamp = int((CLAMP-1)/2)   
                    if index[i] == s_index:
                        continue
                    if i <= half_clamp:
                        seq4align = pred_coor[index[i]][:CLAMP]
                        model4align = good_chain[:CLAMP]
                    elif (L-i) <= half_clamp:
                        seq4align = pred_coor[index[i]][-CLAMP:]
                        model4align = good_chain[-CLAMP:]
                    else:
                        seq4align = pred_coor[index[i]][i-half_clamp:i+half_clamp+1]
                        model4align = good_chain[i-half_clamp:i+half_clamp+1]
                    align_seq = align_svd(model4align, seq4align)
                    good_chain_tmp = good_chain.clone()
                    good_chain_tmp[i] = align_seq[half_clamp+1]
                    lddt = cal_lddt(label, good_chain_tmp)   # 这里先使用真实的lddt，如果实验结果比较好，就再使用pLDDT
                    if lddt > (stand_lddt+1e-5):
                        pred_coor[s_index] = good_chain_tmp.clone()
                        # print("improve!", (lddt-stand_lddt).item())
                        stand_lddt = lddt
                        # print("improved_lddt", stand_lddt.item())
                        stop_step = 0
                    elif lddt > (stand_lddt-1e-3):
                        pred_coor[s_index] = good_chain_tmp.clone()
                        # print("about…", lddt-stand_lddt)

    # 在迭代结束后，选择一条lddt最高的链作为最终结果
    lddt_ls = []
    for chain in pred_coor:
        lddt = cal_lddt(label, chain)
        lddt_ls.append(lddt.item())
    lddt_argmax = lddt_ls.index(max(lddt_ls))
    best_chain = pred_coor[lddt_argmax]

    print("===METRIC===")
    final_lddt = cal_lddt(label, best_chain)
    if final_lddt < initial_stand_lddt:
        best_chain = best_chain_guarantee
        final_lddt = initial_stand_lddt
    print("protein:", protein_index)
    print("initial lddt:", initial_stand_lddt)
    print("final lddt:", final_lddt)
    print("ideal lddt", idea_lddt)
    print("time useage", time.time()-beg)
    print("===END===\n")

    protein_index_ls.append(protein_index)
    initial_lddt_ls.append(initial_stand_lddt)
    final_lddt_ls.append(final_lddt)
    ideal_lddt_ls.append(idea_lddt)
    

# align_result = open("/home/rotation3/complex-coor-pred/AlignCoorSample/log/log_multiseq.txt","w",encoding="utf-8")
# align_result.write("index\tinitial_lddt\tfinal_lddt\tideal_lddt\n")
# for i in range(len(protein_index_ls)):
#     align_result.write(str(protein_index_ls[i]) + "\t")
#     align_result.write(str(initial_lddt_ls[i]) + "\t")
#     align_result.write(str(final_lddt_ls[i]) + "\t")
#     align_result.write(str(ideal_lddt_ls[i]) + "\n")
# align_result.close()
