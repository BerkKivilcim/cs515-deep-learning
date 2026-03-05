# models/MLP.py
"""
Configurable MLP model for MNIST digit classification.

This module defines an MLP (Multi-Layer Perceptron) architecture that:
- Accepts MNIST images of shape (B, 1, 28, 28),
- Flattens them to (B, 784),
- Applies a configurable stack of Linear (+ optional BatchNorm) + Activation (+ optional Dropout),
- Outputs classification logits of shape (B, num_classes).

Design choices
--------------
- Flattening is implemented via `nn.Flatten()` (rather than `view`) for clarity
  and compatibility with typical homework/project requirements.
- BatchNorm (if enabled) is applied before the activation function, which is a
  common and stable practice for MLPs.
- Dropout is applied after the activation function.

Activation options
------------------
- "relu":    nn.ReLU
- "gelu":    nn.GELU
- "sigmoid": nn.Sigmoid

Notes
-----
- The final layer produces *logits* (unnormalized scores). For training with
  `nn.CrossEntropyLoss`, you should pass logits directly (do NOT apply softmax
  in the model).
"""

from __future__ import annotations

from typing import Sequence, Literal

import torch
import torch.nn as nn


ActivationName = Literal["relu", "gelu", "sigmoid"]


class MLP(nn.Module):
    """
    A configurable MLP (Multi-Layer Perceptron) for MNIST classification.

    Parameters
    ----------
    hidden_sizes:
        Sizes of hidden layers (e.g., [256, 512, 256]).
        If empty, the model becomes a single linear classifier (784 -> num_classes).
    activation:
        Activation function name. One of: {"relu", "gelu", "sigmoid"}.
    dropout:
        Dropout probability in [0, 1]. Set to 0.0 to disable dropout.
    use_bn:
        If True, applies BatchNorm1d after each Linear layer and before activation.
    num_classes:
        Number of output classes. For MNIST digits, this is typically 10.

    Input shape
    -----------
    x: torch.Tensor
        Shape (B, 1, 28, 28), where B is the batch size.

    Output shape
    ------------
    torch.Tensor
        Logits of shape (B, num_classes).

    Raises
    ------
    ValueError
        If `activation` is not one of {"relu", "gelu", "sigmoid"}.
    ValueError
        If `dropout` is not in the range [0, 1].
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        activation: ActivationName = "relu",
        dropout: float = 0.3,
        use_bn: bool = True,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if activation not in ("relu", "gelu", "sigmoid"):
            raise ValueError("activation must be one of: 'relu', 'gelu', 'sigmoid'.")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout}.")

        # Map activation name -> module class
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        else:
            act_layer = nn.Sigmoid

        layers: list[nn.Module] = []

        # Flatten (B, 1, 28, 28) -> (B, 784)
        layers.append(nn.Flatten())

        in_dim = 28 * 28  # MNIST flattened dimension

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(h)))

            # BatchNorm placement: Linear -> BN -> Activation (common practice)
            if use_bn:
                layers.append(nn.BatchNorm1d(int(h)))

            layers.append(act_layer())

            # Dropout after activation
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

            in_dim = int(h)

        # Final classification layer: outputs logits (B, num_classes)
        layers.append(nn.Linear(in_dim, int(num_classes)))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Output logits of shape (B, num_classes).
        """
        return self.net(x)