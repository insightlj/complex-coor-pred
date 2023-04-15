import torch

def align(protA, protB, rigid_idx):
    N = protA.size(0)
    A, B = protA[:, rigid_idx[0]:rigid_idx[-1]], protB[:, rigid_idx[0]:rigid_idx[-1]]
    centroid_A = A.mean(dim=-2)
    centroid_B = B.mean(dim=-2)
    AA = A - centroid_A.unsqueeze(dim=-2)
    BB = B - centroid_B.unsqueeze(dim=-2)
    H = torch.matmul(AA.transpose(dim0=-2, dim1=-1), BB)
    U, S, V = torch.svd(H, some=False)
    R = torch.matmul(V, U.transpose(dim0=-2, dim1=-1))
    t = -torch.matmul(R, centroid_A.unsqueeze(-1)) + centroid_B.unsqueeze(-1)
    protA = torch.matmul(protA, R.transpose(dim0=-2, dim1=-1)) + t.reshape(N, 1, 3)
    return protA