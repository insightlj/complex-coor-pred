# Function: 验证生成的h5py文件是否正确
import h5py
import torch

f = h5py.File("/home/rotation3/complex-coor-pred/AlignCoorConfusion/h5py_data/train_dataset.h5py", "r")

# for index in range(0, 15):
#     print((torch.from_numpy(f["protein0"]["aligned_chains"][0,:,:])
#             - torch.from_numpy(f["protein0"]["aligned_chains"][index,:,:])).abs().mean().item())

# 按道理说，pred_coor经过平移和旋转，应该可以得到aligned_chains
# 下面尝试复原这一步骤
a = f["protein1000"]
aligned_chains = torch.from_numpy(a["aligned_chains"][:])
pred_coor = torch.from_numpy(a["pred_coor"][:])
r = torch.from_numpy(a["rotation_matrix"][:])
t = torch.from_numpy(a["translation_matrix"][:])

# 经验证，从pred_coor到aligned_chains完全正确！
r0 = torch.eye(3).unsqueeze(0)
t0 = torch.zeros(3).unsqueeze(0).unsqueeze(0)
R = torch.concat((r0, r), dim=0)
T = torch.concat((t0, t), dim=0)
e_aligned_chains = torch.matmul(pred_coor, R) + T
print((e_aligned_chains-aligned_chains).abs().mean())

# 下面开始验证从aligned_chains到pred_coor, 验证成功！
L = aligned_chains.shape[-2]
R_inv = torch.inverse(R)
print(aligned_chains.shape, T.shape)
tmp_pred_coor = aligned_chains - T
e_pred_coor = torch.einsum("nlc,ncq->nlq", tmp_pred_coor,R_inv)
print((e_pred_coor - pred_coor).abs().mean())


# 下面就是将单个序列的Align转化为confusion得来的4条序列的Align. 暂时没有成功Align
L = aligned_chains.shape[-2]
R_inv = torch.inverse(R)
confused_chains = aligned_chains[:4]

confused_chains_ls = []
for chain in confused_chains:
    chain = torch.tile(chain.unsqueeze(0), (64,1,1))
    print(chain.shape, T.shape)
    tmp_pred_coor = chain - T
    e_pred_coor = torch.einsum("nlc,ncq->nlq", tmp_pred_coor,R_inv)
    diff = e_pred_coor - pred_coor
    confused_chains_ls.append(e_pred_coor)

e_pred_coor = torch.stack(confused_chains_ls, dim=0)
# print((e_pred_coor - pred_coor).abs().mean())

# pred_coor_tmp = torch.tile(aligned_chains[:,None,:,:], (1,64,1,1)) - torch.tile(T[None,:,:,:], (1,1,L,1))
# print(pred_coor_tmp.shape)
# print(R_inv.shape)
# e_pred_coor = torch.matmul(pred_coor_tmp, R_inv)
# # e_pred_coor = torch.einsum("bchw, cwq -> bchq", pred_coor_tmp, R_inv)
# print((e_pred_coor - pred_coor).abs().mean())