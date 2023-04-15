import h5py
import numpy as np
from scipy import optimize as opt
from scripts.cal_lddt_numpy import cal_lddt 

def cal_r_matrix(b,c,d):
    a,b,c,d = np.array([1,b,c,d])/np.sqrt(1+b**2+c**2+d**2)
    r_matrix = np.array([a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c,
                        2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b,
                        2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]).reshape(3,3)
    return r_matrix

def Align(model_chain, other_chain):
    """
    model_chain [L,3]  #第一条序列
    other_chain [L,3]   #剩下的所有条序列中的一条
    
    return aligned_chains [L,3]
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

    r_matrix = cal_r_matrix(x[0],x[1],x[2])

    aligned_chain = other_chain @ r_matrix + (x[3],x[4],x[5])
    return aligned_chain


chain4align = np.load("chain.npy")
model4align = np.load("model.npy")
aligned_chain = Align(model4align, chain4align)
print(chain4align, model4align, aligned_chain)