import h5py

plot_file = h5py.File("/home/rotation3/complex-coor-pred/data/plot_file_CASP.h5")
target_ls = ['T1024-D1', 'T1024-D2', 'T1025-D1', 'T1065s2-D1', 'T1070-D4', 'T1073-D1', 'T1083-D1', 'T1084-D1', 'T1087-D1', 'T1101-D1']

token2seq_dict = {0:'A', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I', 8:'K', 9:'L', 10:'M', 11:'N', 12:'P', 
                13:'Q', 14:'R', 15:'S', 16:'T', 17:'V', 18:'W', 19:'Y', 20:'X', 21:'O', 22:'U', 23:'B', 24:'Z', 25:'-', 26:'.', 
                27:'<mask>', 28: '<pad>',}

DICT = {
    'A':'ALA',
    'R':'ARG',
    'N':'ASN',
    'D':'ASP',
    'C':'CYS',
    'Q':'GLN',
    'E':'GLU',
    'G':'GLY',
    'H':'HIS',
    'I':'ILE',
    'L':'LEU',
    'K':'LYS',
    'M':'MET',
    'F':'PHE',
    'P':'PRO',
    'S':'SER',
    'T':'THR',
    'W':'TRP',
    'Y':'TYR',
    'V':'VAL',
    "O":'PYL',
    'X':'GLY',
    'U':'SEC',
    'B':'ASP',
    'Z':'GLU'
    }

for target in target_ls:
    print(plot_file[target].keys())
    print(token2seq_dict[plot_file[target]['target_tokens'][0][0]])

    pdb_name = "/home/rotation3/complex-coor-pred/plot/CASP/" + target + ".pdb"
    coor = plot_file[target]['pred_coor']
    tokens = plot_file[target]['target_tokens'][0]

    with open(pdb_name, "w") as pdb_file:
        pdb_file.write("MODEL  1\n")
        L = coor.shape[0]
        for i in range(L-2):
            P = token2seq_dict[tokens[i]]
            if P in ["-", ".", "<mask>", "<pad>"]:
                continue
            ABC = DICT[P]
            x,y,z = coor[i,:]
            pdb_file.write("ATOM")
            pdb_file.write("%7d"%(i+1))
            pdb_file.write("  CA  ")
            pdb_file.write(ABC + " A")
            pdb_file.write("%4d"%(i+1))
            pdb_file.write("    ")
            pdb_file.write("%8.3f"%(x))
            pdb_file.write("%8.3f"%(y))
            pdb_file.write("%8.3f"%(z))
            pdb_file.write("  1.00 97.50           C  \n")
        P = token2seq_dict[tokens[L-1]]
        if P not in ["-", ".", "<mask>", "<pad>"]:
            ABC = DICT[P]
            pdb_file.write(f"TER {L}  "+ ABC +" A {L} \n")
            pdb_file.write(f"ENDMDL\n")
            pdb_file.write("END\n")
