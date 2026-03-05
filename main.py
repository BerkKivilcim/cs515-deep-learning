# main.py
"""
Project entry point for training and evaluating an MLP on MNIST.

This script performs the following high-level steps:
1) Parses CLI arguments into a configuration dataclass (see `parameters.py`).
2) Builds an MLP model according to the configuration (see `models.py`).
3) Trains the model and saves checkpoints/metrics (see `train.py`).
4) Loads the best checkpoint (if available) and runs a final test evaluation (see `test.py`).

Notes
-----
- The checkpoint file is expected at: `{cfg.out_dir}/best_model.pt`.
- The checkpoint is assumed to be a dict with a `"model"` key that stores the
  `state_dict` of the model (i.e., `{"model": model.state_dict(), ...}`).
- For CUDA compatibility, ensure that your PyTorch build supports your GPU
  compute capability. If you observe errors such as "sm_120 not compatible",
  update PyTorch (e.g., CUDA 12.4+ builds) in the same interpreter environment.

Modules
-------
- `parameters.build_config_from_cli`: CLI -> configuration dataclass.
- `models.MLP`: MLP definition.
- `train.run_training`: training loop and checkpointing.
- `train.resolve_device`: resolves device spec (auto/cuda/cpu) to `torch.device`.
- `test.test_model`: final test evaluation.

Typical usage
-------------
python main.py --help

Example:
python main.py --hidden_sizes 256,512,1024 --activation gelu --dropout 0.3
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from parameters import build_config_from_cli
from models import MLP
from train import run_training, resolve_device
from test import test_model

import os
import sys


def _maybe_print_executable() -> None:
    """
    Optionally print the active Python interpreter path for debugging.

    This is useful in multi-env setups where the wrong interpreter might be used
    (e.g., VS Code selecting a different conda env).

    Control:
    - Set environment variable `PRINT_EXECUTABLE=1` to enable.
      Example: PRINT_EXECUTABLE=1 python main.py ...
    """
    if os.getenv("PRINT_EXECUTABLE", "0") == "1":
        print("[DEBUG] EXECUTABLE:", sys.executable)


def _load_best_checkpoint_if_available(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
) -> bool:
    """
    Load the best model checkpoint into `model` if it exists.

    Parameters
    ----------
    model:
        The model instance to load weights into.
    ckpt_path:
        Path to the checkpoint file (expected format: dict with key `"model"`).
    device:
        Target device for `map_location` during torch.load.

    Returns
    -------
    bool
        True if a checkpoint was found and loaded successfully; otherwise False.

    Raises
    ------
    KeyError
        If the checkpoint exists but does not contain the `"model"` key.
    RuntimeError
        If the state dict is incompatible with the current model architecture.
    """
    if not ckpt_path.exists():
        return False

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise KeyError(
            f"Checkpoint at '{ckpt_path}' does not contain key 'model'. "
            "Expected format: {'model': state_dict, ...}."
        )

    model.load_state_dict(ckpt["model"])
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return True


def main() -> None:
    """
    Main entry point.

    Workflow
    --------
    1) Parse CLI arguments into a config.
    2) Build and print the model.
    3) Train the model.
    4) Load best checkpoint (if available).
    5) Run test evaluation and print metrics.
    """
    _maybe_print_executable()

    # Parse configuration from CLI (dataclass-like object).
    cfg = build_config_from_cli()

    # Resolve device (e.g., "auto" -> cuda if available, else cpu).
    device = resolve_device(cfg.device)

    # Build model.
    model = MLP(
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        dropout=cfg.dropout,
        use_bn=cfg.use_bn,
    ).to(device)

    print(model)

    # Train model (expected to handle saving best_model.pt to cfg.out_dir).
    summary = run_training(model, cfg)
    print("Training summary:", summary)

    # Load best checkpoint before final test.
    ckpt_path = Path(cfg.out_dir) / "best_model.pt"
    _load_best_checkpoint_if_available(model=model, ckpt_path=ckpt_path, device=device)

    # Final test evaluation.
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