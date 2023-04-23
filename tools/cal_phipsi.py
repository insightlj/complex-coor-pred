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

# if __name__ == "__main__":

    # import h5py
    # label_xyz = h5py.File("/export/disk5/chenyinghui/database/xyz.h5", "r")

    # validlist=[line.strip() for line in open('/export/disk4/for_Lijun/data/validlist_tmp.txt')]
    # trainlist=[line.strip() for line in open('/export/disk4/for_Lijun/data/trainlist_tmp.txt')]

    # # 当前目标：计算出 label_dihedral_angle并保存在h5py文件中
    # with h5py.File("/export/disk4/for_Lijun/data/label.valid1988.h5", "w") as f:
    #     for domain in validlist:
    #         xyz = torch.tensor(np.array(label_xyz[domain]["xyz"]))
    #         dihedral_angle = PsiPhi(xyz)
    #         dihedral_angle = np.array(dihedral_angle)
    #         f.create_dataset(domain, data=dihedral_angle, dtype=dihedral_angle.dtype)
    
    # with h5py.File("/export/disk4/for_Lijun/data/label.train22197.h5", "w") as f:
    #     for domain in trainlist:
    #         xyz = torch.tensor(np.array(label_xyz[domain]["xyz"]))
    #         dihedral_angle = PsiPhi(xyz)
    #         dihedral_angle = np.array(dihedral_angle)
    #         f.create_dataset(domain, data=dihedral_angle, dtype=dihedral_angle.dtype)
            
            

    # label_train = h5py.File("/export/disk4/for_Lijun/data/label.train22310.h5", "r")
    # label_test = h5py.File("/export/disk4/for_Lijun/data/label.valid2000.h5", "r")


    # blackList=[line.strip() for line in open('/export/disk4/for_Lijun/data/blacklist109-cadist-localangle-has-nan.txt')]  # blacklist109-localangle-has-nan.txt后来修改为blacklist109-cadist-localangle-has-nan.txt
    # validlist=[line.strip() for line in open('/export/disk4/for_Lijun/data/valid2000.filt.caths35v42.txt') if line.strip()!='3j7yp00' and line.strip() not in blackList]
    # trainlist=[line.strip() for line in open('/export/disk4/for_Lijun/data/train22310.filt.caths35v42.txt') if line.strip()!='3j7yp00' and line.strip() not in blackList]

    # ls = list(label_xyz.keys())
    # print(all([i in ls for i in validlist]))
    # print(all([i in ls for i in trainlist]))
    
    # ls_vaild = [i in ls for i in validlist]
    # ls_train = [i in ls for i in trainlist]
    
    # ### 经验证，有1个验证数据不在label_xyz中；有14个训练数据不在label_xyz中
    # print(len(ls_vaild) - sum(ls_vaild))   # 1
    # print(len(ls_train) - sum(ls_train))   # 14

    # ### 下面找出这些数据的名称叫什么，然后修改一下索引文件
    # tmp_train = []
    # for i in trainlist:
    #     if i not in ls:
    #         tmp_train.append(i)
    
    # tmp_test = []
    # for i in validlist:
    #     if i not in ls:
    #         tmp_test.append(i)

    # import numpy as np
    # validlist_tmp = list(set(validlist) - set(tmp_test))
    # trainlist_tmp = list(set(trainlist) - set(tmp_train))
    # print(len(validlist_tmp))
    # print(len(trainlist_tmp))

    # ### 我直接将修改之后的trainlist和validlist写入txt文件中，以便读取
    # with open("data/trainlist_tmp.txt", "w") as f:
    #     for i in trainlist_tmp:
    #         f.writelines(i)
    #         f.writelines("\n")

    # with open("data/validlist_tmp.txt", "w") as f:
    #     for i in validlist_tmp:
    #         f.writelines(i)
    #         f.writelines("\n")