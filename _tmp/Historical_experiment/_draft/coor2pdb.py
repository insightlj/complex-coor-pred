def generate_pdb(coor, pdb_name):
    """ 将coor坐标转化为PDB文件
    
    :coor: [L,3]
    :pdb_name: pdb_name
    """
    with open(pdb_name, "w") as pdb_file:
        pdb_file.write("MODEL  1\n")
        L = coor.shape[0]
        for i in range(L-1):
            x,y,z = coor[i,:]
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