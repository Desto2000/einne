import torch
import torch.nn.functional as F
from torch import nn

import einne.nn.functional as fe


class ProbabilisticActivation(nn.Module):
    def __init__(
        self,
        activation="ripple",
        transform_type="hybrid",
        alpha=1.0,
        beta=1.0,
        num_experts=4,
    ):
        super().__init__()
        assert activation in ["ripple", "ripplemoe", "softmax"]
        assert transform_type in ["digamma", "gamma", "bessel", "hybrid"]
        self.activation = activation
        self.transform_type = transform_type
        self.register_buffer("activation_history", torch.zeros(1000))
        self.register_buffer("gradient_history", torch.zeros(1000))
        self.current_idx = 0
        if activation == "ripple":
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        elif activation == "ripplemoe":
            self.weights = nn.Parameter(torch.ones(num_experts) / num_experts)
            self.alphas = nn.Parameter(
                torch.repeat_interleave(torch.tensor(alpha), num_experts)
            )
            self.beta = nn.Parameter(torch.tensor(beta))
        if transform_type == "hybrid":
            self.mix_params = nn.Parameter(torch.ones(2) / 2)

    def update_history(self, activations, gradients):
        idx = self.current_idx % 1000
        self.activation_history[idx] = activations.mean()
        self.gradient_history[idx] = gradients.mean()
        self.current_idx += 1

    def ripple_transform(self, data):
        if self.transform_type == "digamma":
            return torch.special.digamma(data + 1)
        elif self.transform_type == "gamma":
            return torch.special.gammainc(data + 1, data + 2)
        elif self.transform_type == "bessel":
            return torch.special.i0(data)
        elif self.transform_type == "hybrid":
            mix = F.softmax(self.mix_params, dim=0)
            return mix[0] * torch.special.digamma(data + 1) + mix[1] * torch.special.i0(
                data
            )

    def ripple(self, data, temperature=1.0):
        norm = torch.special.erf(data / (data.max() * self.alpha))
        transformed = self.ripple_transform(norm * self.beta)
        gradients = fe.clip_gradients(
            torch.autograd.grad(
                outputs=transformed.sum(), inputs=data, create_graph=True
            )[0]
        )
        self.update_history(transformed.detach(), gradients.detach())
        return F.softmin(gradients / temperature, dim=-1)

    def ripplemoe(self, data, temperature=1.0):
        outputs = [
            self.ripple_transform(
                torch.special.erf(data / (data.max() * alpha)) * self.beta
            )
            for alpha in self.alphas
        ]
        mix = torch.stack(
            [
                weight * output
                for weight, output in zip(F.softmax(self.weights, dim=0), outputs)
            ]
        )
        gradients = fe.clip_gradients(
            torch.autograd.grad(outputs=mix.sum(), inputs=data, create_graph=True)[0]
        )
        self.update_history(mix.detach().mean(0), gradients.detach())
        return F.softmin(gradients / temperature, dim=-1)

    def forward(self, input_tensor, temperature=1.0):
        if self.training:
            if self.transform_type == "hybrid":
                self.mix_entropy = -(
                    F.softmax(self.mix_params, dim=0)
                    * F.log_softmax(self.mix_params, dim=0)
                ).sum()
            if self.activation == "ripplemoe":
                self.weight_entropy = -(
                    F.softmax(self.weights, dim=0) * F.log_softmax(self.weights, dim=0)
                ).sum()
        if self.activation == "ripple":
            return self.ripple(input_tensor, temperature)
        elif self.activation == "ripplemoe":
            return self.ripplemoe(input_tensor, temperature)
        elif self.activation == "softmax":
            output = F.softmax(input_tensor / temperature, dim=-1)
            gradients = torch.autograd.grad(
                outputs=output.sum(), inputs=input_tensor, create_graph=True
            )[0]
            self.update_history(output.detach(), gradients.detach())
            return output

    def get_state(self):
        # flake:
        return {
            "activation": self.activation,
            "transform_type": self.transform_type,
            "alpha": self.alpha.item() if hasattr(self, "alpha") else None,
            "beta": self.beta.item(),
            "mix_params": (
                self.mix_params.tolist() if hasattr(self, "mix_params") else None
            ),
            "weights": self.weights.tolist() if hasattr(self, "weights") else None,
            "alphas": self.alphas.tolist() if hasattr(self, "alphas") else None,
        }
