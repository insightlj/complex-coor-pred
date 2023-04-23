import h5py

plot_file = h5py.File("/home/rotation3/complex-coor-pred/data/plot_file.h5")
full_atom_file = h5py.File("/home/rotation3/complex-coor-pred/data/full_atom_file.h5","a")
target_ls = ['T1024-D1','T1024-D2','T1024-D1','T1101-D1','2dq0A01','2zhjA03','3ngxA02','4i0uA03','3pm9A04','4dwdA01','4uexA00']

for target in plot_file.keys():
# for target in target_ls:
    psiphi = plot_file[target]["phipsi"][:]
    pred_coor = plot_file[target]["pred_coor"][:]
    print(psiphi.shape, pred_coor.shape)

    """
    full_atoms = func(pred_coor,psiphi)
    根据psiphi和pred_coor计算全原子模型
    """

    full_atom_file.create_dataset(target,data=full_atoms)
    
plot_file.close()
full_atom_file.close()
