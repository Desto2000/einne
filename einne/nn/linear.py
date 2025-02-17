import math

import einops
import torch
from torch import nn


class BlockDiagonalLinear(nn.Module):
    def __init__(self, width, num_blocks, w_init_variance_scale=1.0):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.block_width = width // num_blocks
        self.w = nn.Parameter(
            torch.empty([num_blocks, self.block_width, self.block_width])
        )
        self.w_init_variance_scale = w_init_variance_scale
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        torch.nn.init.normal_(self.w, mean=0.0, std=std)

    def forward(self, x):
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w)
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)
