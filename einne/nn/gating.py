import torch
import torch.nn.functional as F
from torch import nn

import einne.nn.functional as fe
from einne.nn.linear import BlockDiagonalLinear
from einne.nn.pooling import SelfAttPooling


class TokenSelection(nn.Module):
    def __init__(self, model_dim, num_selections):
        super().__init__()
        self.model_dim = model_dim
        self.num_selections = num_selections
        self.classifiers = nn.ModuleList(
            [nn.Linear(model_dim, 1) for _ in range(num_selections)]
        )
        self.token_weights = nn.Parameter(torch.empty((num_selections, 1)))
        self.prelu = nn.PReLU()
        self.probability_activation = nn.Softmax(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.token_weights, nonlinearity="linear")

    def forward(self, input_tensor):
        token_scores = torch.stack(
            [classifier(input_tensor).squeeze(-1) for classifier in self.classifiers]
        )
        prelu_scores = self.prelu(token_scores)
        token_scores = F.celu(prelu_scores) * F.silu(prelu_scores) + F.gelu(
            prelu_scores
        )
        combined_scores = torch.einsum(
            "s,s...->...",
            F.selu(self.token_weights.squeeze(-1)),
            token_scores,
        )
        top_k_indices = torch.topk(
            combined_scores, k=input_tensor.size(1), dim=-1
        ).indices
        importance = torch.ones_like(combined_scores).scatter_add(
            -1, top_k_indices, token_scores.sum(0)
        )
        return 1 + self.probability_activation(importance).unsqueeze(-1)


class RG(nn.Module):
    def __init__(self, model_dim, num_heads, weight_init_variance_scale=1.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.input_gate = BlockDiagonalLinear(
            model_dim, num_heads, weight_init_variance_scale
        )
        self.a_gate = BlockDiagonalLinear(
            model_dim, num_heads, weight_init_variance_scale
        )
        self.state_pool = SelfAttPooling(num_heads)
        self.a_param = nn.Parameter(torch.empty([model_dim]))
        self.reset_parameters()

    def reset_parameters(self):
        fe.rnn_param_init(self.a_param, min_rad=0.9, max_rad=0.999)

    def forward(self, x, h):
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        log_a = -8.0 * F.softplus(self.a_param) * gate_a
        a = torch.exp(log_a)
        gated_x = x * gate_x
        normalized_x = gated_x * fe.SqrtBoundDerivative.apply(1 - torch.exp(2 * log_a))
        h = self.state_pool(h, keepdims=True)
        return a * h + normalized_x
