# authot: Jun Li
# file: coorAlign.py
# function: 通过torch提供的梯度下降的方法将所有的序列Align到一起; 
# BUT FAILURE!!!! torch的梯度下降似乎不适合这种情况. 可以了解一下BFGS

import torch
from torch import nn


### loss
def loss_fn(pred_coor):
    pred_coor = torch.permute(pred_coor, (2,1,0))
    loss = 0
    for pred_xyz in pred_coor:
        for position in pred_xyz:
            loss += torch.var(position)
    return loss

### compute rotation matrix
def to_matrix(r_vector):
    b,c,d = r_vector
    a,b,c,d = torch.tensor([1,b,c,d])/torch.sqrt(1+b**2+c**2+d**2)
    r_matrix = torch.tensor([a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c,
                             2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b,
                             2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]).reshape(3,3)
    return r_matrix



torch.manual_seed(1)

class Align(nn.Module):
    def __init__(self):
        super(Align, self).__init__()
        r_vector = torch.randn(64,3)
        t_vector = torch.randn(64,3)
        self.r_vector_param = nn.Parameter(r_vector, requires_grad=True)
        self.t_vector_param = nn.Parameter(t_vector, requires_grad=True)

    def forward(self, pred_coor):
        r_ls = []
        for i in self.r_vector_param:
            r_ls.append(to_matrix(i))
        r_matrix =torch.stack(r_ls, dim=0)

        pred_coor = torch.einsum("jkm,jmn->jkn", pred_coor, r_matrix)\
                    + torch.tile(self.t_vector_param.unsqueeze(1), dims=(1,91,1))

        return pred_coor

pred_coor = torch.randn(1,64,91,3)
align = Align()
opt = torch.optim.SGD(align.parameters(), lr=1e-4)

for i in range(10000):
    pred_coor.squeeze_()
    pred_coor = align(pred_coor)
    l = loss_fn(pred_coor)
    l.backward(retain_graph=True)
    opt.step()
    opt.zero_grad()
    print(l.item())

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# count_parameters(align)



### 或者将所有的序列Align到第一条序列上, 这个时候就需要有TM-score的辅助了
