import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SelfAttPooling(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x, keepdims=False):
        n_state = x.size(-1) // self.num_heads
        attention_weights = F.softmax(torch.matmul(x, x.transpose(-2, -1)) * (1.0 / np.sqrt(n_state)), dim=-1)
        return torch.max(torch.matmul(attention_weights, x), dim=-2, keepdim=keepdims)[0]
