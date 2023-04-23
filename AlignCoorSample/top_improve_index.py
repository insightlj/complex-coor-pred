import torch
import numpy as np

initial_lddt = np.load("/home/rotation3/complex-coor-pred/AlignCoorSample/log/initial_lddt.npy")
finally_lddt = np.load("/home/rotation3/complex-coor-pred/AlignCoorSample/log/final_lddt.npy")

diff = finally_lddt - initial_lddt
diff = torch.tensor(diff)
value = diff.topk(10)[0]
index = (diff.topk(10)[1]) * 40

