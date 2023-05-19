# Function: 计算L套坐标的lddt
# Author: Yaoguang Xing

import torch
eps = 1e-7

def cal_lddt(predcadist,truecadist):
    """ 根据CAdist计算L套坐标的LDDT
    
    :param predcadist: [B,L,L,L]
    :param truecadist: [B,L,L]
    :return lddt:  [B,L,L]
    :variables
        s: minimum sequence separation. lDDT original paper: default s=0
        t: threshold [0.5,1,2,4] the same ones used to compute the GDT-HA score
        Dmax: inclusion radius,far definition,according to lDDT paper   
    """
    N,L,L,L=predcadist.shape
    truecadist=torch.tile(truecadist[:,None,:,:],(1,L,1,1))
    
    Dmax=15.0
    maskfar=torch.as_tensor(truecadist<=Dmax,dtype=torch.float32) # (N,L,L,L)
    
    s=0  #  lDDT original paper: default s=0
    a=torch.arange(L).reshape([1,L]).to(maskfar.device)
    maskLocal=torch.as_tensor(torch.abs(a-a.T)>=s,dtype=torch.float32) # (L,L)
    maskLocal=torch.tile(maskLocal[None,None,:,:],(N,L,1,1))
    fenmu=maskLocal*maskfar

    Ratio=0
    t=[0.5,1,2,4] # the same ones used to compute the GDT-HA score
    for t0 in t:
        preserved=torch.as_tensor(torch.abs(truecadist-predcadist)<t0,dtype=torch.float32)
        fenzi=maskLocal*maskfar*preserved
        Ratio+=torch.sum(fenzi,dim=3)/(torch.sum(fenmu,dim=3)+eps)
    lddt=Ratio/4.0  # (N,L,L)  range (0,1]
    return lddt


if __name__ == "__main__":
    pred = torch.randn(4,91,91,3) * 100
    label = torch.randn(4,91,91,3) * 100

    pred = pred.unsqueeze(-2) - pred.unsqueeze(-3)
    pred = ((pred**2).sum(dim=-1) + eps) ** 0.5
    label = ((label**2).sum(dim=-1) + eps) ** 0.5
    lddt = cal_lddt(pred, label)