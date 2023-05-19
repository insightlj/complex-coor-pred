import h5py
import torch

plot_file = h5py.File("full_atoms/plot_file.h5")
full_atom_file = h5py.File("/home/rotation3/complex-coor-pred/data/plot_file.h5","a")
target_ls = ['2dq0A01', 'T1087-D1', 'T1024-D1', '4i0uA03', '3pm9A04', 'T1065s2-D1', 'T1025-D1', '2zhjA03', 'T1101-D1', 'T1024-D2', 'T1083-D1', '3ngxA02', 'T1073-D1', 'T1084-D1', 'T1070-D4']

for target in target_ls:
    psiphi = torch.from_numpy(plot_file[target]["phipsi"][:])
    pred_coor = torch.from_numpy(plot_file[target]["pred_coor"][:])
    print(psiphi.shape, pred_coor.shape)

    """
    full_atoms = func(pred_coor,psiphi)
    根据psiphi和pred_coor计算全原子模型
    """

    full_atom_file.create_dataset(target,data=full_atoms)
    
plot_file.close()
full_atom_file.close()


