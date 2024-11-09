import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=1000):
        super().__init__()
        positional_encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    def forward(self, input_tensor):
        return input_tensor + self.positional_encoding.to(input_tensor.device)[:, :input_tensor.size(1)]


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)

    def forward(self, x):
        return self.position_encoding(self.embedding(x))
