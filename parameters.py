# parameters.py
"""
Argument parsing and configuration dataclass for MNIST MLP experiments.

This module defines:
- `TrainConfig`: a frozen dataclass holding all experiment settings
  (reproducibility, data paths, training hyperparameters, regularization,
  scheduler, model architecture, visualization/logging, evaluation).
- `_parse_int_list`: helper to parse comma-separated integer lists.
- `build_config_from_cli`: CLI -> `TrainConfig` builder using argparse.

Design goals
------------
- Provide a single source of truth for experiment parameters.
- Improve reproducibility by storing config in a structured dataclass.
- Support ablation studies via CLI flags (hidden sizes, activation, BN, dropout,
  L1/L2 regularization, scheduler variants, etc.).

Notes
-----
- This file intentionally keeps paths as strings (`data_dir`, `out_dir`) to avoid
  coupling config to filesystem operations. Downstream code can cast to `Path`.
- The dataclass is `frozen=True` to prevent accidental modifications during
  training.

Typical usage
-------------
from parameters import build_config_from_cli
cfg = build_config_from_cli()
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Literal


DeviceName = Literal["auto", "cuda", "cpu"]
ActivationName = Literal["relu", "gelu", "sigmoid"]
SchedulerType = Literal["step", "plateau"]
ThresholdMode = Literal["abs", "rel"]


@dataclass(frozen=True)
class TrainConfig:
    """
    Configuration for training and evaluating an MLP on MNIST.

    Attributes
    ----------
    seed:
        Random seed for reproducibility.
    data_dir:
        Root directory to download/cache MNIST.
    out_dir:
        Output directory for checkpoints, curves, and summaries.

    device:
        Device selection strategy:
        - "auto": use CUDA if available, else CPU
        - "cuda": force CUDA
        - "cpu":  force CPU

    batch_size:
        Batch size for training/validation/test DataLoaders.
    num_workers:
        Number of DataLoader worker processes.
    epochs:
        Maximum number of training epochs.
    lr:
        Learning rate.

    weight_decay:
        L2 regularization strength applied via optimizer `weight_decay`.
        (Commonly referred to as "weight decay".)
    l1_lambda:
        L1 regularization coefficient added to the loss as:
            loss_total = loss_data + l1_lambda * sum(|w|)

    early_stop_patience:
        Number of epochs to wait for validation-loss improvement before stopping.
    early_stop_min_delta:
        Minimum decrease in validation loss required to count as an improvement.

    use_scheduler:
        Whether to enable learning-rate scheduling.
    scheduler_type:
        Scheduler type:
        - "step": StepLR style (step_size, gamma)
        - "plateau": ReduceLROnPlateau style (plateau_* params)
    step_size:
        For StepLR: number of epochs between LR reductions.
    gamma:
        For StepLR: multiplicative LR decay factor.
        For Plateau: can be interpreted as `factor` (if you use it that way downstream).

    plateau_patience:
        For ReduceLROnPlateau: number of epochs with no improvement before reducing LR.
    plateau_threshold:
        For ReduceLROnPlateau: threshold for measuring improvement.
    plateau_threshold_mode:
        For ReduceLROnPlateau: "abs" or "rel".
    plateau_cooldown:
        For ReduceLROnPlateau: cooldown epochs after LR reduction.
    plateau_min_lr:
        For ReduceLROnPlateau: minimum LR.

    hidden_sizes:
        Hidden layer sizes (e.g., [512, 256, 128]).
    activation:
        Activation function name: "relu" | "gelu" | "sigmoid".
    use_bn:
        Whether to use BatchNorm1d between Linear and activation.
    dropout:
        Dropout probability in [0, 1]. Set to 0.0 to disable dropout.

    save_curves:
        Whether to save train/val/test curves as PNG files.
    export_torchviz:
        Whether to export the computation graph via torchviz.
    torchviz_format:
        Output format for torchviz ("png" | "pdf" | "svg").
    log_every:
        Mini-batch logging frequency (steps). 0 disables step-level logs.

    val_split:
        Fraction of the training set used for validation (e.g., 0.1).
    """

    # --- Reproducibility / paths ---
    seed: int = 42
    data_dir: str = "./data"
    out_dir: str = "./outputs"

    # --- Device ---
    device: DeviceName = "auto"

    # --- Data / training ---
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 30
    lr: float = 1e-3

    # --- Regularization ---
    weight_decay: float = 0.0  # L2 via optimizer weight_decay
    l1_lambda: float = 0.0     # L1 added to loss

    # --- Early stopping (val loss) ---
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0

    # --- Scheduler ---
    use_scheduler: bool = True
    scheduler_type: SchedulerType = "step"  # "step" | "plateau"

    step_size: int = 10
    gamma: float = 0.5  # StepLR gamma; can be interpreted as Plateau factor downstream

    # Plateau-specific
    plateau_patience: int = 2
    plateau_threshold: float = 0.002
    plateau_threshold_mode: ThresholdMode = "abs"  # "abs" | "rel"
    plateau_cooldown: int = 0
    plateau_min_lr: float = 0.0

    # --- Model (MLP) ---
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: ActivationName = "relu"
    use_bn: bool = True
    dropout: float = 0.3
    optimizer: str = "adam"

    # --- Visualization / logging ---
    save_curves: bool = True
    export_torchviz: bool = True
    torchviz_format: Literal["png", "pdf", "svg"] = "png"
    log_every: int = 100

    # --- Evaluation ---
    val_split: float = 0.1


def _parse_int_list(s: str) -> List[int]:
    """
    Parse a comma-separated list of integers.

    Parameters
    ----------
    s:
        A string such as "512,256,128" (whitespace is allowed).

    Returns
    -------
    List[int]
        Parsed integer list. Returns an empty list if the input is empty.

    Raises
    ------
    ValueError
        If any token cannot be converted to an integer.

    Examples
    --------
    >>> _parse_int_list("512, 256, 128")
    [512, 256, 128]
    >>> _parse_int_list("")
    []
    """
    s = s.strip()
    if not s:
        return []

    parts = [p.strip() for p in s.split(",")]
    if any(p == "" for p in parts):
        raise ValueError(f"Invalid hidden_sizes string: '{s}'. Example: '512,256,128'")

    return [int(p) for p in parts]


def build_config_from_cli(argv: List[str] | None = None) -> TrainConfig:
    """
    Build a `TrainConfig` from command-line arguments.

    Parameters
    ----------
    argv:
        Optional list of arguments (excluding program name). If None, argparse
        reads from sys.argv automatically. This is useful for unit tests.

    Returns
    -------
    TrainConfig
        Frozen configuration dataclass.

    Notes
    -----
    This function uses paired boolean flags for some options:
    - `--use_scheduler` / `--no_scheduler`
    - `--use_bn` / `--no_bn`
    - `--save_curves` / `--no_curves`
    - `--export_torchviz` / `--no_torchviz`

    Default behavior matches the dataclass defaults. Supplying the "no_*"
    flags overrides them.
    """
    p = argparse.ArgumentParser(description="CS515 HW1a - MLP on MNIST")

    # Paths & seed
    p.add_argument("--seed", type=int, default=TrainConfig.seed, help="Random seed.")
    p.add_argument("--data_dir", type=str, default=TrainConfig.data_dir, help="MNIST data root directory.")
    p.add_argument("--out_dir", type=str, default=TrainConfig.out_dir, help="Output directory for artifacts.")

    # Device
    p.add_argument("--device", type=str, default=TrainConfig.device, choices=["auto", "cuda", "cpu"],
                   help="Device selection: auto/cuda/cpu.")

    # Train
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size, help="Batch size.")
    p.add_argument("--num_workers", type=int, default=TrainConfig.num_workers, help="DataLoader workers.")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="Max epochs.")
    p.add_argument("--lr", type=float, default=TrainConfig.lr, help="Learning rate.")

    # Regularization
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay,
                   help="L2 regularization via optimizer weight_decay.")
    p.add_argument("--l1_lambda", type=float, default=TrainConfig.l1_lambda,
                   help="L1 regularization coefficient added to loss.")

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=TrainConfig.early_stop_patience,
                   help="Early stopping patience (epochs).")
    p.add_argument("--early_stop_min_delta", type=float, default=TrainConfig.early_stop_min_delta,
                   help="Minimum val-loss improvement to reset patience.")

    # Scheduler
    p.add_argument("--scheduler_type", type=str, default=TrainConfig.scheduler_type, choices=["step", "plateau"],
                   help="Scheduler type: step or plateau.")

    p.add_argument("--use_scheduler", action="store_true", help="Enable LR scheduler.")
    p.add_argument("--no_scheduler", action="store_true", help="Disable LR scheduler.")

    p.add_argument("--step_size", type=int, default=TrainConfig.step_size,
                   help="StepLR step size (epochs).")
    p.add_argument("--gamma", type=float, default=TrainConfig.gamma,
                   help="StepLR gamma (or Plateau factor if used downstream).")

    p.add_argument("--plateau_patience", type=int, default=TrainConfig.plateau_patience,
                   help="ReduceLROnPlateau patience.")
    p.add_argument("--plateau_threshold", type=float, default=TrainConfig.plateau_threshold,
                   help="ReduceLROnPlateau threshold.")
    p.add_argument("--plateau_threshold_mode", type=str, default=TrainConfig.plateau_threshold_mode,
                   choices=["abs", "rel"], help="ReduceLROnPlateau threshold mode.")
    p.add_argument("--plateau_cooldown", type=int, default=TrainConfig.plateau_cooldown,
                   help="ReduceLROnPlateau cooldown.")
    p.add_argument("--plateau_min_lr", type=float, default=TrainConfig.plateau_min_lr,
                   help="ReduceLROnPlateau minimum LR.")

    # Model
    p.add_argument(
        "--hidden_sizes",
        type=str,
        default="512,256,128",
        help="Comma-separated hidden sizes, e.g. '512,256,128'.",
    )
    p.add_argument("--activation", type=str, default=TrainConfig.activation, choices=["relu", "gelu", "sigmoid"],
                   help="Activation function.")
    p.add_argument("--use_bn", action="store_true", help="Enable BatchNorm.")
    p.add_argument("--no_bn", action="store_true", help="Disable BatchNorm.")
    p.add_argument("--dropout", type=float, default=TrainConfig.dropout, help="Dropout probability in [0,1].")

    # Viz
    p.add_argument("--save_curves", action="store_true", help="Enable saving curve plots.")
    p.add_argument("--no_curves", action="store_true", help="Disable saving curve plots.")
    p.add_argument("--export_torchviz", action="store_true", help="Enable torchviz graph export.")
    p.add_argument("--no_torchviz", action="store_true", help="Disable torchviz graph export.")
    p.add_argument("--torchviz_format", type=str, default=TrainConfig.torchviz_format, choices=["png", "pdf", "svg"],
                   help="Torchviz output format.")
    p.add_argument("--log_every", type=int, default=TrainConfig.log_every,
                   help="Log every N steps (0 disables step logs).")

    # Val split
    p.add_argument("--val_split", type=float, default=TrainConfig.val_split,
                   help="Validation split fraction from training data.")
    
    p.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"]
    )

    args = p.parse_args(argv)

    # ---- Resolve paired boolean flags while keeping defaults ----
    # If neither flag is provided, fall back to TrainConfig defaults.
    use_scheduler = TrainConfig.use_scheduler
    if args.use_scheduler:
        use_scheduler = True
    if args.no_scheduler:
        use_scheduler = False

    use_bn = TrainConfig.use_bn
    if args.use_bn:
        use_bn = True
    if args.no_bn:
        use_bn = False

    save_curves = TrainConfig.save_curves
    if args.save_curves:
        save_curves = True
    if args.no_curves:
        save_curves = False

    export_torchviz = TrainConfig.export_torchviz
    if args.export_torchviz:
        export_torchviz = True
    if args.no_torchviz:
        export_torchviz = False

    hidden_sizes = _parse_int_list(args.hidden_sizes)
    if not hidden_sizes:
        hidden_sizes = TrainConfig.hidden_sizes  # default

    return TrainConfig(
        seed=args.seed,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        use_scheduler=use_scheduler,
        scheduler_type=args.scheduler_type,
        step_size=args.step_size,
        gamma=args.gamma,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        plateau_threshold_mode=args.plateau_threshold_mode,
        plateau_cooldown=args.plateau_cooldown,
        plateau_min_lr=args.plateau_min_lr,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        use_bn=use_bn,
        dropout=args.dropout,
        save_curves=save_curves,
        export_torchviz=export_torchviz,
        torchviz_format=args.torchviz_format,
        log_every=args.log_every,
        val_split=args.val_split,
        optimizer=args.optimizer,
    )