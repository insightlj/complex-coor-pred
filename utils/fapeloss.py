import torch
from torch import nn


def getFapeLoss(diff, dclamp=10, ratio=0.1, lossformat='dclamp'):
    ### true label nan has been masked
    ### diff.shape=(N,3,L,L)
    ### return fapeLoss,realFape

    eps = 1e-8
    L = diff.shape[-1]
    diff = diff[:, :, :, None, :] - diff[:, :, :, :, None]  # (N,3,64,L,L)  # 计算出坐标之间的差值
    diff = torch.sqrt(torch.sum(diff ** 2, dim=1) + eps) # diff实为fapeNLLL (N,64,L,L)

    realFape = torch.mean(diff)  # FapeLoss和lddt的思想相似，均计算原子之间相对坐标的预测准确度

    if lossformat == 'dclamp':
        mask = torch.as_tensor(diff <= dclamp, dtype=torch.float32)  # (N,64,L,L)
        mask = mask * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == 'ReluDclamp':
        diff = ratio * nn.ReLU(inplace=True)(diff - dclamp) + dclamp - nn.ReLU(inplace=True)(
            dclamp - diff)  # 考虑换成mask形式
        mask = torch.ones_like(diff) * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == 'NoDclamp':
        mask = torch.ones_like(diff) * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)
    elif lossformat == 'probDclamp':
        maskdclamp = torch.as_tensor(diff <= dclamp, dtype=torch.float32)  # (N,L,L,L)
        maskdclamp = maskdclamp * (torch.ones([L, L]) - torch.eye(L)).to(diff.device)
        maskprob = torch.as_tensor(torch.rand([L, L]) >= (1 - ratio),
                                   dtype=torch.float32)  # uniform in [0,1), 按照概率ratio提取dclamp之外的残基对
        maskprob = torch.triu(maskprob, diagonal=1)
        maskprob = maskprob + maskprob.permute(1, 0)
        maskprob = maskprob.to(diff.device)
        mask = maskprob * (1 - maskdclamp) + maskdclamp
        fapeLoss = torch.sum(diff * mask) / (torch.sum(mask) + eps)

    del mask

    return fapeLoss, realFape