import torch

def NormVec(V):
    eps = 1e-10
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= (torch.norm(axis_x, dim=-1).unsqueeze(1) + eps)
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= (torch.norm(axis_z, dim=-1).unsqueeze(1) + eps)
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y /= (torch.norm(axis_y, dim=-1).unsqueeze(1) + eps)
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec

def comp_feature(atoms):
    rotation = NormVec(atoms)
    r = torch.inverse(rotation)
    
    xyz_CA = torch.einsum('a b i, a i j -> a b j', atoms[:, 1].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)
    xyz_C  = torch.einsum('a b i, a i j -> a b j', atoms[:, 2].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)
    xyz_N  = torch.einsum('a b i, a i j -> a b j', atoms[:, 0].unsqueeze(0) - atoms[:, 1].unsqueeze(1), r)
    
    N_CA_C = torch.stack([xyz_N, xyz_CA, xyz_C], dim=-2)
    return N_CA_C

def label_generate(pre_coor, a, trunc_point, train_mode=True):
    # pre_coor: [L,4,3]
    L = pre_coor.shape[0]
    if train_mode and L > trunc_point:
        pre_coor = pre_coor[a:a+trunc_point, :,:]

    key_atoms = pre_coor[:,:4,:]
    transform_xyz = comp_feature(key_atoms)
    # print(transform_xyz.shape)
    coor = transform_xyz[:,:,1,:].squeeze()
    # print(coor.shape)

    return coor    # [L, L, 3]

if __name__ == "__main__":
    label = label_generate(torch.randn(129,4,3), a=12, trunc_point=99)