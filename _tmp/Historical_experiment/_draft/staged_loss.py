import torch
from scripts.sample_from_dataset import demo_pred_label

DEMO_SIZE = 1
TRUNC = 28
net_pt_name = "/home/rotation3/complex-coor-pred/model/checkpoint/CoorNet_demo/epoch0.pt"
pred, label = demo_pred_label(demo_size=DEMO_SIZE, net_pt_name=net_pt_name)
pred = pred.unsqueeze(0)
label = label.unsqueeze(0)

pred = pred.unsqueeze(-2) - pred.unsqueeze(-3)
label = label.unsqueeze(-2) - label.unsqueeze(-3)
print(pred.shape, label.shape)
# pred/label: [b,l,l,l,3]

pred_dist = ((pred ** 2).sum(dim=-1, keepdims=True)) ** 0.5
effective_len = len(pred_dist[pred_dist < TRUNC])
pred_dist = pred_dist.repeat(1, 1, 1, 1, 3)
label_dist = ((label ** 2).sum(dim=-1, keepdims=True)) ** 0.5
label_dist = label_dist.repeat(1, 1, 1, 1, 3)
print(pred_dist.shape, label_dist.shape)

mask = torch.where(pred_dist >= TRUNC)

pred_masked = torch.where(pred_dist < TRUNC, pred, 0)
label_masked = torch.where(pred_dist < TRUNC, label, 0)
print(pred_masked.shape, label_masked.shape)

loss = ((((pred_masked - label_masked) ** 2).sum(dim=-1) + 1e-7) ** 0.5).sum() / effective_len
print(loss)

loss_pri = ((((pred - label) ** 2).sum(dim=-1) + 1e-7) ** 0.5).mean()
print(loss_pri)
