import torch
from torch import nn
from config import eps, BLOCK_COOR_TRUNC

class Coor_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, pred, label, num_block=None):
        pred = pred.unsqueeze(-2) - pred.unsqueeze(-3)
        label = label.unsqueeze(-2) - label.unsqueeze(-3)

        if num_block == None:
            loss = ((((pred - label) ** 2 ).sum(dim=-1) + eps)**0.5).mean()
    
        else:
            # effective_len = len(pred[pred<=BLOCK_COOR_TRUNC[num_block]])/ 3
            # pred[pred>BLOCK_COOR_TRUNC[num_block]] = 0
            # label[pred>BLOCK_COOR_TRUNC[num_block]] = 0
            pred_dist = ((pred ** 2).sum(dim=-1, keepdims=True)) ** 0.5
            effective_len = len(pred_dist[pred_dist<BLOCK_COOR_TRUNC[num_block]])
            pred_dist = pred_dist.repeat(1,1,1,1,3)
            label_dist =((label ** 2).sum(dim=-1, keepdims=True)) ** 0.5
            label_dist = label_dist.repeat(1,1,1,1,3)

            pred_masked = torch.where(pred_dist<BLOCK_COOR_TRUNC[num_block], pred, 0)
            label_masked = torch.where(pred_dist<BLOCK_COOR_TRUNC[num_block], label, 0)
            loss = ((((pred_masked - label_masked) ** 2 ).sum(dim=-1) + eps)**0.5).sum() / effective_len

        return loss