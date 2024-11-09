import numpy as np
import torch
import torch.nn.functional as F


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
