import math

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_MAX_SQRT_GRADIENT = 1000.0


class BlockDiagonalLinear(nn.Module):
    def __init__(self, width, num_blocks, w_init_variance_scale=1.0):
        super().__init__()
        self.width = width
        self.num_blocks = num_blocks
        self.block_width = width // num_blocks
        self.w = nn.Parameter(torch.empty([num_blocks, self.block_width, self.block_width]))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(self.w_init_variance_scale / self.block_width)
        torch.nn.init.normal_(self.w, mean=0.0, std=std)

    def forward(self, x):
        x = einops.rearrange(x, "... (h i) -> ... h i", h=self.num_blocks)
        y = torch.einsum("... h i, h i j -> ... h j", x, self.w)
        return einops.rearrange(y, "... h j -> ... (h j)", h=self.num_blocks)


def rnn_param_init(tensor, min_radius, max_radius, transform="softplus", epsilon=1e-8):
    with torch.no_grad():
        tensor.uniform_(min_radius ** 2 + epsilon, max_radius ** 2 + epsilon).log_().mul_(0.5)
        if transform == "softplus":
            return tensor.neg_().exp_().sub_(1.0).log_()
        else:
            raise NotImplementedError()


def get_attention_mask(size, attention_mode, local_attention_context=None):
    if attention_mode == 'all':
        mask = torch.tril(torch.ones([size, size]))
    elif attention_mode == 'local':
        context = min(size - 1, local_attention_context - 1)
        mask = torch.tril(torch.ones([size, size]), context)
    else:
        raise ValueError('Not yet implemented')
    return mask.view(1, 1, size, size)


def split_heads(tensor, num_heads):
    return torch.transpose(split_states(tensor, num_heads), 0, 2, 1, 3)


def merge_heads(tensor):
    return merge_states(torch.transpose(tensor, 0, 2, 1, 3))


def split_states(tensor, num_states):
    return torch.reshape(tensor, tensor.size()[:-1] + [num_states, tensor.size(-1) // num_states])


def merge_states(tensor):
    return torch.reshape(tensor, tensor.size()[:-2] + [np.prod(tensor.size()[-2:])])


def attention_impl(query, key, value, num_heads, attention_mode, local_attention_context=None):
    query, key, value = split_heads(query, num_heads), split_heads(key, num_heads), split_heads(value, num_heads)
    mask = get_attention_mask(key.size(2), attention_mode, local_attention_context).float()
    attention_weights = torch.matmul(query, key.transpose(-2, -1)) * (1.0 / np.sqrt(query.size(-1)))
    attention_weights = F.softmax(attention_weights * mask + -1e9 * (1 - mask), dim=-1)
    return merge_heads(torch.matmul(attention_weights, value))


class SqrtBoundDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / torch.sqrt(torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT ** 2)))


class SelfAttPooling(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x, keepdims=False):
        n_state = x.size(-1) // self.num_heads
        attention_weights = F.softmax(torch.matmul(x, x.transpose(-2, -1)) * (1.0 / np.sqrt(n_state)), dim=-1)
        return torch.max(torch.matmul(attention_weights, x), dim=-2, keepdim=keepdims)[0]


def clip_gradients(grads, clip_value=1.0):
    return torch.clamp(grads, -clip_value, clip_value)


class ProbabilisticActivation(nn.Module):
    def __init__(self, activation="ripple", transform_type="hybrid", alpha=1.0, beta=1.0, num_experts=4):
        super().__init__()
        assert activation in ["ripple", "ripplemoe", "softmax"]
        assert transform_type in ["digamma", "gamma", "bessel", "hybrid"]
        self.activation = activation
        self.transform_type = transform_type
        self.register_buffer('activation_history', torch.zeros(1000))
        self.register_buffer('gradient_history', torch.zeros(1000))
        self.current_idx = 0
        if activation == "ripple":
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        elif activation == "ripplemoe":
            self.weights = nn.Parameter(torch.ones(num_experts) / num_experts)
            self.alphas = nn.Parameter(torch.repeat_interleave(torch.tensor(alpha), num_experts))
            self.beta = nn.Parameter(torch.tensor(beta))
        if transform_type == "hybrid":
            self.mix_params = nn.Parameter(torch.ones(2) / 2)

    def update_history(self, activations, gradients):
        idx = self.current_idx % 1000
        self.activation_history[idx] = activations.mean()
        self.gradient_history[idx] = gradients.mean()
        self.current_idx += 1

    def ripple_transform(self, data):
        if self.transform_type == 'digamma':
            return torch.special.digamma(data + 1)
        elif self.transform_type == 'gamma':
            return torch.special.gammainc(data + 1, data + 2)
        elif self.transform_type == 'bessel':
            return torch.special.i0(data)
        elif self.transform_type == 'hybrid':
            mix = F.softmax(self.mix_params, dim=0)
            return mix[0] * torch.special.digamma(data + 1) + mix[1] * torch.special.i0(data)

    def ripple(self, data, temperature=1.0):
        norm = torch.special.erf(data / (data.max() * self.alpha))
        transformed = self.ripple_transform(norm * self.beta)
        gradients = clip_gradients(torch.autograd.grad(outputs=transformed.sum(), inputs=data, create_graph=True)[0])
        self.update_history(transformed.detach(), gradients.detach())
        return F.softmin(gradients / temperature, dim=-1)

    def ripplemoe(self, data, temperature=1.0):
        outputs = [self.ripple_transform(torch.special.erf(data / (data.max() * alpha)) * self.beta) for alpha in
                   self.alphas]
        mix = torch.stack([weight * output for weight, output in zip(F.softmax(self.weights, dim=0), outputs)])
        gradients = clip_gradients(torch.autograd.grad(outputs=mix.sum(), inputs=data, create_graph=True)[0])
        self.update_history(mix.detach().mean(0), gradients.detach())
        return F.softmin(gradients / temperature, dim=-1)

    def forward(self, input_tensor, temperature=1.0):
        if self.training:
            if self.transform_type == "hybrid":
                self.mix_entropy = -(F.softmax(self.mix_params, dim=0) * F.log_softmax(self.mix_params, dim=0)).sum()
            if self.activation == "ripplemoe":
                self.weight_entropy = -(F.softmax(self.weights, dim=0) * F.log_softmax(self.weights, dim=0)).sum()
        if self.activation == "ripple":
            return self.ripple(input_tensor, temperature)
        elif self.activation == "ripplemoe":
            return self.ripplemoe(input_tensor, temperature)
        elif self.activation == "softmax":
            output = F.softmax(input_tensor / temperature, dim=-1)
            gradients = torch.autograd.grad(outputs=output.sum(), inputs=input_tensor, create_graph=True)[0]
            self.update_history(output.detach(), gradients.detach())
            return output

    def get_state(self):
        return {
            'activation'    : self.activation,
            'transform_type': self.transform_type,
            'alpha'         : self.alpha.item() if hasattr(self, 'alpha') else None,
            'beta'          : self.beta.item(),
            'mix_params'    : self.mix_params.tolist() if hasattr(self, 'mix_params') else None,
            'weights'       : self.weights.tolist() if hasattr(self, 'weights') else None,
            'alphas'        : self.alphas.tolist() if hasattr(self, 'alphas') else None}


class TokenSelection(nn.Module):
    def __init__(self, model_dim, num_selections):
        super().__init__()
        self.model_dim = model_dim
        self.num_selections = num_selections
        self.classifiers = nn.ModuleList([nn.Linear(model_dim, 1) for _ in range(num_selections)])
        self.token_weights = nn.Parameter(torch.empty((num_selections, 1)))
        self.prelu = nn.PReLU()
        self.probability_activation = ProbabilisticActivation(activation="ripplemoe", num_experts=num_selections)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.token_weights, nonlinearity="linear")

    def forward(self, input_tensor):
        token_scores = torch.stack([classifier(input_tensor).squeeze(-1) for classifier in self.classifiers])
        prelu_scores = self.prelu(token_scores)
        token_scores = F.celu(prelu_scores) * F.silu(prelu_scores) + F.gelu(prelu_scores)
        combined_scores = torch.einsum(
            'selection,selection...->...', F.selu(self.token_weights.squeeze(-1)), token_scores
        )
        top_k_indices = torch.topk(combined_scores, k=input_tensor.size(1), dim=-1).indices
        importance = torch.ones_like(combined_scores).scatter_add(-1, top_k_indices, token_scores.sum(0))
        return 1 + self.probability_activation(importance).unsqueeze(-1)


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


class RG(nn.Module):
    def __init__(self, model_dim, num_heads, weight_init_variance_scale=1.0):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.input_gate = BlockDiagonalLinear(model_dim, num_heads, weight_init_variance_scale)
        self.a_gate = BlockDiagonalLinear(model_dim, num_heads, weight_init_variance_scale)
        self.state_pool = SelfAttPooling(num_heads)
        self.a_param = nn.Parameter(torch.empty([model_dim]))
        self.reset_parameters()

    def reset_parameters(self):
        rnn_param_init(self.a_param, min_rad=0.9, max_rad=0.999)

    def forward(self, x, h):
        gate_x = torch.sigmoid(self.input_gate(x))
        gate_a = torch.sigmoid(self.a_gate(x))
        log_a = -8.0 * F.softplus(self.a_param) * gate_a
        a = torch.exp(log_a)
        gated_x = x * gate_x
        normalized_x = gated_x * SqrtBoundDerivative.apply(1 - torch.exp(2 * log_a))
        h = self.state_pool(h, keepdims=True)
        return a * h + normalized_x


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
        self.probability_activation = ProbabilisticActivation()
        self.probability_activation2 = ProbabilisticActivation()
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


def precompute_freqs_cis(model_dim, end_pos, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, model_dim, 2).float() / model_dim))
    time_steps = torch.arange(end_pos, device=freqs.device)
    return torch.polar(torch.ones_like(freqs), torch.outer(time_steps, freqs).float())
