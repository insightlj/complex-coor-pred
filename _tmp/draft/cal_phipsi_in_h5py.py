from tools.cal_phipsi import PsiPhi
import h5py, torch

plot_file = h5py.File("/home/rotation3/complex-coor-pred/data/plot_file.h5", "a")
target_ls = list(plot_file.keys())

label = h5py.File("/export/disk4/for_Lijun/CASP/CASP14_91_domains_coords_seq.h5")

for target in target_ls:
    plabel = torch.from_numpy(label[target]["xyz"][:])
    phipsi = PsiPhi(plabel)
    plot_file[target]["phipsi"] = phipsi

