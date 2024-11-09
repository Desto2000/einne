import torch

_MAX_SQRT_GRADIENT = 1000.0


def clip_gradients(grads, clip_value=1.0):
    return torch.clamp(grads, -clip_value, clip_value)


class SqrtBoundDerivative(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / torch.sqrt(torch.clip(4.0 * x, min=1 / (_MAX_SQRT_GRADIENT ** 2)))
