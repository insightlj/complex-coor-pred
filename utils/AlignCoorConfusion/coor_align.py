# funtion: 将2-64条序列Align到第一条序列上. 这个过程没有参数, 使用的是scipy.optimize提供的BFGS和Powell
# Author: Jun Li

import numpy as np
from scipy import optimize as opt

def select_chains(lddt_score, num_chains=64):
    lddt_score_chains = lddt_score.sum(dim=2)
    _, indices = lddt_score_chains.topk(num_chains)
    return indices

def cal_r_matrix(b,c,d):
    a,b,c,d = np.array([1,b,c,d])/np.sqrt(1+b**2+c**2+d**2)
    r_matrix = np.array([a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c,
                        2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b,
                        2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]).reshape(3,3)
    return r_matrix

def Align(model_chain, other_chain):
    """
    model_chain [L,3]  #第一条序列
    other_chain [N-1,L,3]   #剩下的所有条序列中的一条
    
    return aligned_chains [N,L,3]
    """
    L = model_chain.shape[0]
    def minize_L2Loss(x):
        b,c,d = x[0], x[1], x[2]
        r_matrix = cal_r_matrix(b,c,d)
        L2Loss = ((model_chain - other_chain @ r_matrix - (x[3],x[4],x[5])) ** 2 ).mean()
        return L2Loss

    res = opt.minimize(minize_L2Loss, (0,0,0,0,0,0), method = 'BFGS')
    x = res.x
    
    if x[0]>5 or x[0]<-5:
        # print("BFGS失效, 尝试Powell")
        bounds = np.array([[-1,1],[-1,1],[-1,1],[None,None],[None,None],[None,None]])
        res = opt.minimize(minize_L2Loss, (0,0,0,0,0,0), method = 'Powell',bounds=bounds)
        x = res.x

    global rotation_matrix_ls, translation_matrix_ls
    r_matrix = cal_r_matrix(x[0],x[1],x[2])
    rotation_matrix_ls.append(r_matrix)
    translation_matrix_ls.append(np.array([x[3],x[4],x[5]]))

    aligned_chain = other_chain @ r_matrix - (x[3],x[4],x[5])
    return aligned_chain

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
        chain_ls.append(Align(model_chain, other_chain))
    chain_array = np.array(chain_ls)
    return chain_array

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

from utils.cal_lddt import getLDDT
def get_trueLDDT(pred, label):
    """
    pred: BLL3
    label: BLL3
    return truelddt_score BLL
    """
    pred = pred.unsqueeze(-2) - pred.unsqueeze(-3)
    pred = ((pred**2).sum(dim=-1) + eps) ** 0.5
    label = ((label**2).sum(dim=-1) + eps) ** 0.5
    truelddt_score = getLDDT(pred, label)
    return truelddt_score
    
if __name__ == '__main__':
    import torch
    import h5py
    import time
    from main import train_ds, test_ds
    from torch.utils.data import DataLoader
    from config import NUM_BLOCKS,device,eps
    
    # load lddt compute model
    from utils.pLDDT import pLDDT
    get_pLDDT = torch.load("/home/rotation3/complex-coor-pred/utils/plddt_checkpoints/Full_train/epoch7_mark.pt")
    
    # load data
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    net_pt = torch.load(model_pt_name, map_location=device)

    with h5py.File("utils/AlignCoorConfusion/h5py_data/train_dataset.h5py", "a") as train_file:
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
                # indices = select_chains(lddt_score).cpu()
                
                # pred_coor = pred_coor[0,indices,:]
                indices = train_file["protein" + str(i)]["indices"]
                lddt_score = lddt_score[0,indices,:].cpu()

                # __保存lddt_score和pred_x2d中，暂时不需要，注释掉__
                # !!!!COOR ALIGN!!!! 
                # 将pred转化为numpy, 从而进行align, 得到align_chains
                # pred = ((pred_coor.squeeze())).cpu().numpy()
                # rotation_matrix_ls = []
                # translation_matrix_ls = []
                # chain_array = align_chain(pred)
                # aligned_chains = torch.from_numpy(chain_array)
                # rotation_matrix = torch.from_numpy(np.array(rotation_matrix_ls))
                # translation_matrix = torch.from_numpy(np.array(translation_matrix_ls))
                
                # # 根据lddt_score做mask, 筛选掉lddt比较低的部分, 得到gated_aligned_chains
                # lddt_gate = lddtGate(lddt_score)
                # lddt_gate = lddt_gate.bool().squeeze().cpu()
                # lddt_gate = torch.tile(lddt_gate[:,:,None], (1,1,3))
                # gated_aligned_chains = torch.masked_fill(aligned_chains, ~lddt_gate, value=0)

                
                # protein = train_file.create_group("protein" + str(i))
                # protein["aligned_chains"] = aligned_chains
                # protein["rotation_matrix"] = rotation_matrix
                # protein["translation_matrix"] = translation_matrix
                # if "indices" in train_file["protein" + str(i)].keys():
                #     del train_file["protein" + str(i)]["indices"]
                if "lddt_score" in train_file["protein" + str(i)].keys():
                    del train_file["protein" + str(i)]["lddt_score"]
                if "pred_x2d" in train_file["protein" + str(i)].keys():
                    del train_file["protein" + str(i)]["pred_x2d"]
                # train_file["protein" + str(i)]["indices"] = indices
                train_file["protein" + str(i)]["lddt_score"] = lddt_score
                train_file["protein" + str(i)]["pred_x2d"] = pred_x2d.cpu()
                if i < 500:
                    print(lddt_score.shape)
                # if "aligned_chains" in train_file["protein" + str(i)].keys():
                #     del train_file["protein" + str(i)]["aligned_chains"] 
                # train_file["protein" + str(i)]["aligned_chains"] = aligned_chains.cpu()
                print("protein{} saved! time usage:{}".format(i, time.time()-beg))

    ### ____test_dataset save____
    # with torch.no_grad():
    #     i = -1
    #     for data in test_dataloader:
    #         beg = time.time()
    #         i += 1
    #         embed, atten, coor_label, L = data
    #         embed = embed.to(device)
    #         atten = atten.to(device)
    #         coor_label = coor_label.to(device)
    #         L = L.to(device)
    #         pred_coor_4_blocks, pred_x2d = net_pt(embed, atten)
    #         pred_coor = pred_coor_4_blocks[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor

    #         # compute lddt
    #         plddt_score = get_pLDDT(pred_coor, pred_x2d)   # 在测试的时候使用pLDDT, 因为测试就是为了完全模仿真实情况
    #         # truelddt_score = get_trueLDDT(pred_coor, coor_label)   # 在训练的时候使用trueLDDT(或许可以两个混着用？相当于数据增强，毕竟之前的pLDDT中，应该保存了LDDT之外的信息。但是我做筛选，使用的就是LDDT，其他是不用的。所以还是越准确越好)
    #         lddt_score = torch.clone(plddt_score)

    #         # select top 64 highest lddt chains
    #         # return new pred_coor & lddt_score
    #         indices = select_chains(lddt_score).cpu()
            
    #         pred_coor = pred_coor[0,indices,:]
    #         lddt_score = lddt_score[0,indices,:].cpu()

            
    #         # !!!!COOR ALIGN!!!! 
    #         # 将pred转化为numpy, 从而进行align, 得到align_chains
    #         pred = ((pred_coor.squeeze())).cpu().numpy()
    #         print(pred.shape)   # (L,L,3)
    #         rotation_matrix_ls = []
    #         translation_matrix_ls = []
    #         chain_array = align_chain(pred)
    #         aligned_chains = torch.from_numpy(chain_array)
    #         rotation_matrix = torch.from_numpy(np.array(rotation_matrix_ls))
    #         translation_matrix = torch.from_numpy(np.array(translation_matrix_ls))
            
    #         # # 根据lddt_score做mask, 筛选掉lddt比较低的部分, 得到gated_aligned_chains
    #         # lddt_gate = lddtGate(lddt_score)
    #         # lddt_gate = lddt_gate.bool().squeeze().cpu()
    #         # lddt_gate = torch.tile(lddt_gate[:,:,None], (1,1,3))
    #         # gated_aligned_chains = torch.masked_fill(aligned_chains, ~lddt_gate, value=0)
            

    #         # with h5py.File("utils/AlignCoorConfusion/h5py_data/test_dataset.h5py", "a") as test_file:
    #         #     # protein = test_file.create_group("protein" + str(i))
    #         #     # protein["aligned_chains"] = aligned_chains     # LL3
    #         #     # protein["rotation_matrix"] = rotation_matrix    # L33
    #         #     # protein["translation_matrix"] = translation_matrix  # L3
                
    #         #     if "indices" in test_file["protein" + str(i)].keys():
    #         #         del test_file["protein" + str(i)]["indices"]
    #         #     if "lddt_score" in test_file["protein" + str(i)].keys():
    #         #         del test_file["protein" + str(i)]["lddt_score"]
    #         #     if "pred_x2d" in test_file["protein" + str(i)].keys():
    #         #         del test_file["protein" + str(i)]["pred_x2d"]
    #         #     test_file["protein" + str(i)]["indices"] = indices
    #         #     test_file["protein" + str(i)]["lddt_score"] = lddt_score
    #         #     test_file["protein" + str(i)]["pred_x2d"] = pred_x2d.cpu()
            
    #         print("protein{} saved! time usage:{}".format(i, time.time()-beg)) 



if __name__ == "__main_ _":
    import h5py
    import torch
    import h5py
    import time
    from main import train_ds, test_ds
    from torch.utils.data import DataLoader
    from config import NUM_BLOCKS,device,eps
    # load data
    model_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt"
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False)
    net_pt = torch.load(model_pt_name, map_location=device)

    with torch.no_grad():
        i = -1
        for data in train_dataloader:
            beg = time.time()
            i += 1
            embed, atten, coor_label, L = data
            print(L)
            embed = embed.to(device)
            atten = atten.to(device)
            coor_label = coor_label.to(device)
            L = L.to(device)
            pred_coor_4_blocks, pred_x2d = net_pt(embed, atten)
            pred_coor = pred_coor_4_blocks[NUM_BLOCKS-1]   # 取出最后一个Block预测出的coor
            print(pred_coor.shape)  

            with h5py.File("utils/AlignCoorConfusion/h5py_data/train_dataset.h5py", "a") as f:
                if "pred_coor" in f["protein" + str(i)].keys():
                    del f["protein" + str(i)]["pred_coor"] 
                f["protein" + str(i)]["pred_coor"] = pred_coor.cpu()
            print("protein{} saved! time usage:{}".format(i, time.time()-beg))