import torch
import numpy as np
eps = 1e-5

def PsiPhi(atoms):
    # input：局部坐标[L,3,3]
    # output: 二面角 [L-1, 2]

    def psi(CA, C, N):
        a = N[1:] - C[:-1]
        b = C - CA
        c = N - CA
        ab = torch.cross(a, b[:-1])
        bc = torch.cross(b[:-1], c[:-1])
        ca = torch.cross(c[:-1], a)
        
        cos_ca_b = torch.sum(ca * b[:-1], dim=-1) / (torch.linalg.norm(ca, dim=-1) * torch.linalg.norm(b[:-1], dim=-1) + eps)
        cospsi = torch.sum(ab * bc, dim=-1)/(torch.linalg.norm(ab, dim=-1) * torch.linalg.norm(bc, dim=-1) + eps)
        cospsi = np.pi - torch.arccos(torch.clamp(cospsi, max=1, min=-1))
        return (cos_ca_b / abs(cos_ca_b)) * cospsi

    def phi(CA, C, N):
        b = C - CA
        c = N - CA
        d = C[:-1] - N[1:]
        bc = torch.cross(b[1:], c[1:])
        cd = torch.cross(c[1:], d)
        bd = torch.cross(b[1:], d)
        cos_bd_c = torch.sum(bd * c[1:], dim=-1) / (torch.linalg.norm(bd, dim=-1) * torch.linalg.norm(c[1:], dim=-1) + eps)
        cosphi = torch.sum(bc * cd, dim=-1) / (torch.linalg.norm(bc, dim=-1) * torch.linalg.norm(cd, dim=-1) + eps)
        cosphi = np.pi - torch.arccos(torch.clamp(cosphi, max=1, min=-1))
        return (cos_bd_c / abs(cos_bd_c)) * cosphi
    N, CA, C = atoms[:, 2], atoms[:, 0], atoms[:, 1]
    return torch.stack([psi(CA, C, N), phi(CA, C, N)], dim=1)
