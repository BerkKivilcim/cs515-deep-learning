# main.py
"""
Entry point.

CHANGES vs tutorial repo:
- parameters.py dataclass config kullanılır.
- device selection düzgün (auto/cuda/cpu).
- training + test akışı net.
- model config (activation/bn/dropout/hidden_sizes) CLI ile gelir.
"""

from __future__ import annotations

from pathlib import Path
import torch

from parameters import build_config_from_cli
from models import MLP
from train import run_training, resolve_device
from test import test_model

import sys
print("EXECUTABLE:", sys.executable)


def main() -> None:
    cfg = build_config_from_cli()

    # For debugging: which interpreter?
    import sys
    print("EXECUTABLE:", sys.executable)

    # IMPORTANT: Your RTX 5060 needs a PyTorch build supporting sm_120.
    # If you still see "sm_120 not compatible", update torch (cu124+) in the SAME interpreter.

    # Build model
    model = MLP(
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        dropout=cfg.dropout,
        use_bn=cfg.use_bn,
    )
    print(model)

    # Train
    summary = run_training(model, cfg)
    print("Training summary:", summary)

    # Load best checkpoint for test
    device = resolve_device(cfg.device)
    model = model.to(device)

    ckpt_path = Path(cfg.out_dir) / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"[INFO] Loaded checkpoint: {ckpt_path}")

    test_loss, test_acc = test_model(
        model=model,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=device,
    )
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}")


if __name__ == "__main__":
    main()