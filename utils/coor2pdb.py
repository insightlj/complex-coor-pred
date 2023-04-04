from utils.demo_of_model_and_eval import demo_pred_label

pred, label = demo_pred_label(demo_size=1, net_pt_name="/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt")
print(pred.shape)
print(label.shape)


import torch
pred_coor = pred[0,:,:]
label_coor = label[0,:,:]

def generate_pdb(pdb_name="example.pdb"):
    with open(pdb_name, "w") as pdb_file:
        pdb_file.write("MODEL  1\n")
        L = pred_coor.shape[0]
        for i in range(L-1):
            x,y,z = pred_coor[i,:]
            pdb_file.write("ATOM")
            pdb_file.write("%7d"%(i+1))
            pdb_file.write("  CA  ")
            pdb_file.write("GLY A")
            pdb_file.write("%4d"%(i+1))
            pdb_file.write("    ")
            pdb_file.write("%8.3f"%(x))
            pdb_file.write("%8.3f"%(y))
            pdb_file.write("%8.3f"%(z))
            pdb_file.write("  1.00 97.50           C  \n")
        pdb_file.write(f"TER {L}  GLY A {L} \n")
        pdb_file.write(f"ENDMDL\n")
        pdb_file.write("END\n")





# if __name__ == '__ main__':
#     from utils.demo_of_model_and_eval import demo_pred_label

#     ls_pred, ls_label = demo_pred_label(demo_size=10, 
#                                         net_pt_name="/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_VII/epoch34.pt",
#                                         shuffle=False)
