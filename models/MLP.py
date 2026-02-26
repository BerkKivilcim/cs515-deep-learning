# models/MLP.py
"""
MLP model for MNIST.

CHANGES vs tutorial repo:
- Flatten artık nn.Flatten() (HW requirement).
- Activation parametric: ReLU or GELU.
- BN and Dropout toggles.
- Type hints + docstring.
"""

from __future__ import annotations

from typing import List
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    A configurable MLP for MNIST classification.

    Input: (B, 1, 28, 28)
    Output: logits (B, 10)
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        activation: str = "relu",
        dropout: float = 0.3,
        use_bn: bool = True,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be 'relu' or 'gelu'")

        act_layer = nn.ReLU if activation == "relu" else nn.GELU

        layers: List[nn.Module] = []

        # ----- CHANGED: nn.Flatten() (instead of view) -----
        layers.append(nn.Flatten())

        in_dim = 28 * 28  # MNIST
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            # ----- HW asks BN placement; we keep BN before activation (common practice) -----
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.net(x)