"""
label['T1096-D1'].keys()        <KeysViewHDF5 ['target_tokens', 'xyz']>
<HDF5 dataset "target_tokens": shape (1, 255), type "|i1">
<HDF5 dataset "xyz": shape (255, 3, 3), type "<f4">

embed_attn['T1096-D1'].keys()   <KeysViewHDF5 ['feature_2D', 'target_tokens', 'token_embeds']>
<HDF5 dataset "feature_2D": shape (1, 41, 255, 255), type "<f4">
<HDF5 dataset "target_tokens": shape (1, 255), type "|i1">
<HDF5 dataset "token_embeds": shape (1, 255, 2560), type "<f4">
"""

"""这三个蛋白质存在于embed_attn中，但是不存在于label中
T1027; T1044; T1064"""

import torch
import h5py
from config import device
from tools.cal_lddt_tensor import cal_lddt

pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch16.pt"
model = torch.load(pt_name, map_location=device)

label = h5py.File("/export/disk4/for_Lijun/CASP/CASP14_91_domains_coords_seq.h5")
embed_attn = h5py.File("/export/disk4/for_Lijun/CASP/CASP14.h5")

label_list = list(label.keys())
input_list = list(embed_attn.keys())

lddt_ls = []

plot_file = h5py.File("data/plot_file.h5", "a")
with torch.no_grad():
    for i in [68, 89,  0, 67, 60, 69, 53,  2,  1, 59]:
        target = label_list[i]
    # for target in label_list:
    # for target in ["T1076-D1"]:
        embed = torch.from_numpy(embed_attn[target]["token_embeds"][:]).to(device)
        attn = torch.from_numpy(embed_attn[target]["feature_2D"][:]).to(device)
        # if target in ["T1076-D1"]:
        #     embed = embed[:,1:,:]
        #     attn = attn[:,:,1:,1:]

        plabel = torch.from_numpy(label[target]["xyz"][:,1,:]).to(device)
        print("protein:", target)
        print(embed.shape, plabel.shape)
        pred_coor, pred_x2d = model(embed, attn)
        print(pred_coor[3].shape)

        chain_lddt_ls = []
        for chain in pred_coor[3][0]:
            chain_lddt = cal_lddt(chain, plabel)
            chain_lddt_ls.append(chain_lddt)
        lddt_tensor = torch.tensor(chain_lddt_ls)
        # topk_lddt = (lddt_tensor.topk(10)[0]).mean()
        lddt = lddt_tensor.max()
        print(lddt)
        lddt_ls.append(lddt.item())

        # h5py
        target_protein = plot_file.create_group(target)
        target_protein["pred_coor"] = (pred_coor[3][0][torch.argmax(lddt_tensor)]).cpu()
        target_protein["lddt"] = lddt.cpu()
        target_protein["target_tokens"] = (embed_attn[target]["target_tokens"][:])
        target_protein["identity"] = target

plot_file.close()

# lddt_ls = torch.tensor(lddt_ls)
# print(lddt_ls)
# print(lddt_ls.mean())

"""
topk_lddt = (lddt_ls.topk(10)[0])
tensor([0.7736, 0.6768, 0.6684, 0.6657, 0.6425, 0.6383, 0.6358, 0.6271, 0.6126, 0.6120])

topk_index = (lddt_ls.topk(10)[1])
tensor([68, 89,  0, 67, 60, 69, 53,  2,  1, 59])
"""