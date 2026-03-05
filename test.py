# test.py
"""
Test evaluation utilities for MNIST.

This module provides a single helper function, `test_model`, which evaluates a
trained model on the official MNIST test split and reports:
- Average cross-entropy loss
- Classification accuracy

Notes
-----
- The model is assumed to output logits of shape (B, num_classes).
- This function uses `torch.no_grad()` to speed up evaluation and reduce memory usage.
- MNIST is downloaded automatically if not present under `data_dir`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@torch.no_grad()
def test_model(
    model: nn.Module,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate a model on the MNIST test split.

    Parameters
    ----------
    model:
        Trained model to evaluate. Must output logits (unnormalized scores).
    data_dir:
        Root directory to download/cache MNIST.
    batch_size:
        Batch size for the test DataLoader.
    num_workers:
        Number of worker processes for the DataLoader.
    device:
        Device on which evaluation is performed.

    Returns
    -------
    (avg_loss, accuracy):
        - avg_loss: Average cross-entropy loss across the test set.
        - accuracy: Fraction of correctly classified samples in [0, 1].

    Notes
    -----
    - Uses `nn.CrossEntropyLoss`, so do NOT apply softmax in the model.
    - Assumes MNIST input images are shaped (B, 1, 28, 28) and labels are (B,).
    """
    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc