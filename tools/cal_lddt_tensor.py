import torch
eps=1e-7

def get_distogram(pos):
    """ 计算类似于距离图的[L,L,3]矩阵
    
    :param pos: [L,3]
    :return distogram: [L,L,3]
    """
    dist_matrics = pos.reshape([1,-1,3])-pos.reshape([-1,1,3])
    distogram = (torch.sum(dist_matrics ** 2, axis=2) + eps) ** 0.5
    return distogram

def cal_lddt(ref, model, cutoff=15.):
    """ 计算整条序列的lddt

    ref: Native Structure
    model: Evaluated Structure
    cutoff: unit - angstrom
    """
    ref = get_distogram(ref)
    model = get_distogram(model)
    residue_length = model.shape[0]
    boolean_contact = ref<cutoff
    diff = model[boolean_contact] - ref[boolean_contact]
    contact_num1 = torch.sum( torch.abs(diff) < 0.5 ) - residue_length
    contact_num2 = torch.sum( torch.abs(diff) < 1.  ) - residue_length
    contact_num3 = torch.sum( torch.abs(diff) < 2.  ) - residue_length
    contact_num4 = torch.sum( torch.abs(diff) < 4.  ) - residue_length
    origin_contact = torch.sum( boolean_contact ) - residue_length
    lddt = (contact_num1 + contact_num2 + contact_num3 + contact_num4) / (origin_contact * 4)
    return lddt

def cal_residue_lddt(ref, model, cutoff=15.):
    """ 计算序列各个残基的lddt

    ref: Native Structure
    model: Evaluated Structure
    cutoff: unit - angstrom
    """
    ref = get_distogram(ref)
    model = get_distogram(model)
    residue_length = model.shape[0]
    boolean_contact = ref<cutoff
    residue_lddt = torch.zeros((residue_length,))
    for resi in range(residue_length):
        diff = model[resi][boolean_contact[resi]] - ref[resi][boolean_contact[resi]]
        contact_num1 = torch.sum( torch.abs(diff) < 0.5 ) - 1
        contact_num2 = torch.sum( torch.abs(diff) < 1.  ) - 1
        contact_num3 = torch.sum( torch.abs(diff) < 2.  ) - 1
        contact_num4 = torch.sum( torch.abs(diff) < 4.  ) - 1
        origin_contact = torch.sum( boolean_contact[resi] ) - 1
        lddt = (contact_num1 + contact_num2 + contact_num3 + contact_num4) / (origin_contact * 4)
        residue_lddt[resi] = lddt
    return residue_lddt

if __name__ == "__main__":
    label = torch.randn(91,3)
    pred = torch.randn(91,3)
    
    lddt = cal_lddt(label, pred)
    per_resi_lddt= cal_residue_lddt(label, pred)
    print("Global lDDT:", lddt)
    print("Per-residue lDDT:\n", per_resi_lddt)