import torch
from torch import nn


def rnn_param_init(
    tensor: nn.Parameter, min_rad, max_rad, transform="softplus", epsilon=1e-8
):
    with torch.no_grad():
        tensor.uniform_(min_rad**2 + epsilon, max_rad**2 + epsilon).log_().mul_(0.5)
        if transform == "softplus":
            return tensor.neg_().exp_().sub_(1.0).log_()
        else:
            raise NotImplementedError()


def precompute_freqs_cis(model_dim, end_pos, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, model_dim, 2).float() / model_dim))
    time_steps = torch.arange(end_pos, device=freqs.device)
    return torch.polar(torch.ones_like(freqs), torch.outer(time_steps, freqs).float())
