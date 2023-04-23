import h5py


plot_file = h5py.File("/home/rotation3/complex-coor-pred/data/plot_file.h5")
plot_file.keys()
TEST = ['1j5wB02', '1r73A00', '2aplA01', '2dq0A01', '2e2aA00', '2ondA00', '2p0tA02', '2wy7Q00', '2zhjA03', '3csxB00', '3j7aU02', '3kawB00', '3ngxA02', '3pm9A04', '3u1kB04', '4dwdA01', '4i0uA03', '4lupA00', '4nqwA01', '4v1gA00']
CASP = ['T1024-D1', 'T1024-D2', 'T1025-D1', 'T1065s2-D1', 'T1070-D4', 'T1073-D1', 'T1083-D1', 'T1084-D1', 'T1087-D1', 'T1101-D1']


ls = []
for i in plot_file.keys():
    if i not in TEST and i not in CASP:
        ls.append(i)