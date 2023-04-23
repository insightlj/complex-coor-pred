import sys
sys.path.append("/home/rotation3/complex-coor-pred/")

import h5py
import time
import random
import torch
import numpy as np
from tools.align_svd import align_svd
from tools.cal_lddt_tensor import cal_lddt
from tools.sample_from_dataset import sample_only

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
    lddt_score = torch.from_numpy(np.array(protein["lddt_score"]).squeeze())

    stand_lddt = lddt_score[0].mean()  # 0.848
    initial_stand_lddt = stand_lddt.clone()
    lddt_score.mean()  # 0.782
    idea_lddt = lddt_score.max(axis=0).values.mean()   # 0.891

    index = lddt_score.argmax(axis=0)   # protein的每个位置对应的最大值的位置
    L = len(index)
    best_chain = pred_coor[0]  # chain代表一整条链；seq代表一个序列片断
    best_chain_guarantee = best_chain.clone()
    step = 0
    stop_step = 0
    beg = time.time()
    while step<500:
        if stop_step>25:
            break
        step += 1
        stop_step += 1
        # print("step {} begin".format(step+1))
        if L%2 == 0:
            m = L-1
        else:
            m = L
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
            lddt = cal_lddt(label, best_chain_tmp)   # 这里先使用真实的lddt，如果实验结果比较好，就再使用pLDDT
            if lddt > (stand_lddt+1e-5):
                best_chain = best_chain_tmp.clone()
                # print("improve!", (lddt-stand_lddt).item())
                stand_lddt = lddt
                # print("improved_lddt", stand_lddt.item())
                stop_step = 0
            elif lddt > (stand_lddt-1e-3):
                best_chain = best_chain_tmp.clone()
                # print("about…", lddt-stand_lddt)
        


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
    

align_result = open("/home/rotation3/complex-coor-pred/AlignCoorSample/log/log.txt","w",encoding="utf-8")
align_result.write("index\tinitial_lddt\tfinal_lddt\tideal_lddt\n")
for i in range(len(protein_index_ls)):
    align_result.write(str(protein_index_ls[i]) + "\t")
    align_result.write(str(initial_lddt_ls[i]) + "\t")
    align_result.write(str(final_lddt_ls[i]) + "\t")
    align_result.write(str(ideal_lddt_ls[i]) + "\n")
align_result.close()
