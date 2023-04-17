from utils import cal_r_matrix
from scipy import optimize as opt
import numpy as np


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

    aligned_chain = other_chain @ r_matrix + (x[3],x[4],x[5]) 
    return aligned_chain