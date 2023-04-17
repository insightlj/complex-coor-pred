import torch


def svd_align(A, B):
    """ 使用svd的方法进行坐标的Align
    
    :param A: model_chain
    :param B: other_chain
    :return: aligned_chain
    """
    centroid_A = A.mean(-2)
    centroid_B = B.mean(-2)
    AA = A - centroid_A.unsqueeze(-2)
    BB = B - centroid_B.unsqueeze(-2)
    H = torch.matmul(BB.transpose(-2, -1), AA)
    U, S, V = torch.svd(H, some=False)
    R = torch.matmul(V, U.transpose(-2,-1))
    t = -torch.matmul(R, centroid_B.unsqueeze(-1)) + centroid_A.unsqueeze(-1)
    B = torch.matmul(B, R.transpose(-2,-1)) + t.reshape(1, 3)
    return B


if __name__ == '__main__':
    import torch
    from scripts.sample_from_dataset import sample_only
    from AlignCoorConfusion.coor_align import svd_align as sa
    _,_,coor_label,L = sample_only(train_mode=True)
    chain1 = coor_label[0].unsqueeze(0)
    chain2 = coor_label[1].unsqueeze(0)
    aligned = svd_align(chain1, chain2)
    print((aligned - chain1))

    a = sa(chain1, chain2)
    print(a - chain1)