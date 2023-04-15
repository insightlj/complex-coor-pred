import torch
from _tmp.draft.set_seed import seed_torch

class SeedSampler():
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed
    def __iter__(self):
        seed_torch(self.seed)
        seed_random_ls = torch.randperm(len(self.data_source))
        return iter(seed_random_ls)
    def __len__(self):
        return len(self.data_source)
    
