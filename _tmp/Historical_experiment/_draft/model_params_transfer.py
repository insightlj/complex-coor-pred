import torch
from torch import nn

def load_model_and_params_transfer(model_pt_name, to_transfer_model):

    model = torch.load(model_pt_name)
    model_params = nn.Module.state_dict(model)
    nn.Module.load_state_dict(to_transfer_model, model_params)