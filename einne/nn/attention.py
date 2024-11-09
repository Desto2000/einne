import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import einne.nn.functional as fe
from einne.nn.gating import RG, TokenSelection


class RGSAttention(nn.Module):
    def __init__(self, model_dim, num_heads=4, weight_init_variance_scale=1.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attention_space = nn.Parameter(torch.empty(model_dim, model_dim))
        self.spatial_attention = nn.Linear(model_dim, model_dim, bias=False)
        self.input_silhouette = nn.Linear(model_dim, model_dim, bias=False)
        self.output_head = nn.Linear(model_dim, model_dim, bias=False)
        self.recurrent_gating = RG(model_dim, num_heads, weight_init_variance_scale)
        self.feature_selection = TokenSelection(model_dim, num_heads)
        self.probability_activation = fe.ProbabilisticActivation()
        self.probability_activation2 = fe.ProbabilisticActivation()
        self.output_layer = nn.Linear(model_dim, model_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.attention_space)

    def compute_attention_space(self, input_tensor):
        spatial_attention_output = self.spatial_attention(
            torch.einsum("batch_len_dim,dim_dim->batch_len_dim", input_tensor, self.attention_space)
        )
        gated_input = self.recurrent_gating(input_tensor, spatial_attention_output)
        sparsity = self.probability_activation(self.input_silhouette(input_tensor))
        return gated_input * sparsity * self.feature_selection(spatial_attention_output)

    def calculate_attention_scores(self, query, key, value):
        attention_weights = self.probability_activation2(
            torch.matmul(query, key.transpose(-2, -1)) * (1.0 / np.sqrt(query.size(-1) // self.num_heads))
        )
        return torch.matmul(attention_weights, value)

    def forward(self, input_tensor):
        output_head = F.gelu(self.output_head(input_tensor))
        attention_space_output = self.compute_attention_space(input_tensor)
        context = self.calculate_attention_scores(
            attention_space_output, attention_space_output, attention_space_output
        ) * output_head
        return self.output_layer(context)
