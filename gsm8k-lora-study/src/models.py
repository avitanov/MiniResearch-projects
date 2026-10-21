from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen linear layer with trainable low-rank LoRA adapters.

    The wrapped linear layer remains frozen while the adapter matrices learn
    an additive low-rank update. The adapter uses:

    - `A` with shape `(in_features, rank)`
    - `B` with shape `(rank, out_features)`

    The forward pass is:

        linear(x) + (x @ A @ B) * (alpha / rank)
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        """Initialize the LoRA wrapper around an existing linear layer.

        Args:
            linear: The base linear layer to freeze and adapt.
            rank: The LoRA bottleneck rank.
            alpha: The LoRA scaling hyperparameter.

        Returns:
            None.
        """
        super().__init__()
        if rank <= 0:
            msg = "rank must be a positive integer"
            raise ValueError(msg)

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.A = nn.Parameter(torch.empty(linear.in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, linear.out_features))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # Start B at zero so the LoRA branch contributes nothing initially and
        # the wrapped module exactly matches the pretrained linear layer.
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the frozen linear layer and scaled LoRA update.

        Args:
            x: Input tensor whose final dimension matches the linear layer.

        Returns:
            The base linear output plus the low-rank adapter contribution.
        """
        base_output = self.linear(x)
        lora_update = (x @ self.A) @ self.B
        return base_output + lora_update * self.scaling
