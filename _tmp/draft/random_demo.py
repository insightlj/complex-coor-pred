import torch

torch.manual_seed(10)
torch.abs(torch.tensor(-9))
print(torch.rand(1))

torch.manual_seed(10)
print(torch.rand(1))


torch.manual_seed(10)
print(torch.rand(2))