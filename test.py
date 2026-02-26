# test.py
"""
Test evaluation on MNIST test split.

CHANGES vs tutorial repo:
- best_model.pt yüklenir ve test accuracy hesaplanır.
- Type hints + docstrings.
"""

from __future__ import annotations

from pathlib import Path
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
    Evaluate model on MNIST test set.
    Returns (avg_loss, accuracy).
    """
    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tf)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

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

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)