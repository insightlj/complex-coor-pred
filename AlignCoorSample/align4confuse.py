import h5py
import time
import random
import numpy as np
from scripts.align_svd import align
from scripts.cal_lddt_numpy import cal_lddt
from scripts.sample_from_dataset import sample_only


print("===nohup_begin===")
for protein_index in range(0,20000,40):
    # print("=========protein{}================".format(protein_index))
    train_file = h5py.File("/home/rotation3/complex-coor-pred/AlignCoorConfusion/h5py_data/train_dataset.h5py", "r")
    protein = train_file["protein"+str(protein_index)]

    _,_,label,_ = sample_only(train_mode=False, index=protein_index)
    label = label[0]
    label = np.array(label)
    pred_coor = np.array(protein["pred_coor"]).squeeze()
    rotation_matrix = np.array(protein["rotation_matrix"]).squeeze()
    trasslation_matrix = np.array(protein["translation_matrix"]).squeeze()
    lddt_score = np.array(protein["lddt_score"]).squeeze()

    stand_lddt = lddt_score[0].mean()  # 0.848
    initial_stand_lddt = stand_lddt.copy()
    lddt_score.mean()  # 0.782
    idea_lddt = lddt_score.max(axis=0).mean()   # 0.891

    index = lddt_score.argmax(axis=0)   # protein的每个位置对应的最大值的位置
    L = len(index)
    best_chain = pred_coor[0]  # chain代表一整条链；seq代表一个序列片断
    step = 0
    beg = time.time()
    if step<=10:
        step += 1
        # print("step {} begin".format(step+1))
        if L%2 == 0:
            m = L-1
        else:
            m = L
        CLAMP = random.choice([3,5,7,15,21,51,m])
        for i in range(L):
            if index[i] == 0:
                continue
            half_clamp = int((CLAMP-1)/2)
            if i <= half_clamp:
                seq4align = pred_coor[index[i]][:CLAMP]
                model4align = best_chain[:CLAMP]
            elif (L-i) <= half_clamp:
                seq4align = pred_coor[index[i]][-CLAMP:]
                model4align = best_chain[-CLAMP:]
            else:
                seq4align = pred_coor[index[i]][i-half_clamp:i+half_clamp+1]
                model4align = best_chain[i-half_clamp:i+half_clamp+1]
            align_seq = align(model4align, seq4align)
            best_chain_tmp = best_chain.copy()
            best_chain_tmp[i] = align_seq[half_clamp+1]
            lddt = cal_lddt(label, best_chain_tmp)   # 这里先使用真实的lddt，如果实验结果比较好，就再使用pLDDT
            if lddt > (stand_lddt+1e-5):
                best_chain = best_chain_tmp.copy()
                # print("improve!", lddt-stand_lddt)
                stand_lddt = lddt
                # print("improved_lddt", stand_lddt)
            elif lddt > (stand_lddt-1e-4):
                best_chain = best_chain_tmp.copy()
                # print("about…", lddt-stand_lddt)
        

    print("===METRIC===")
    print("protein:", protein_index)
    print("initial lddt:", initial_stand_lddt)
    print("final lddt:", cal_lddt(label, best_chain))
    print("ideal lddt", idea_lddt)
    print("time useage", time.time()-beg)
    print("===END===\n")
