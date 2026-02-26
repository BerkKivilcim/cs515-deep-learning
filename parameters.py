# parameters.py
"""
Argument parsing + configuration dataclasses for HW1a.

CHANGES vs tutorial repo:
- dict yerine dataclass kullanıldı (HW requirement).
- device selection: auto/cuda/cpu gibi seçenekler.
- ablation parametreleri eklendi (activation, bn, dropout, l1, l2, hidden sizes).
- viz (torchviz) ve curve plot seçenekleri eklendi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import argparse


@dataclass(frozen=True)
class TrainConfig:
    # --- Reproducibility / paths ---
    seed: int = 42
    data_dir: str = "./data"
    out_dir: str = "./outputs"

    # --- Device ---
    # "auto" -> cuda varsa cuda, yoksa cpu
    device: str = "auto"

    # --- Data / training ---
    batch_size: int = 128
    num_workers: int = 2
    epochs: int = 30
    lr: float = 1e-3

    # --- Regularization ---
    # L2 -> optimizer weight_decay
    weight_decay: float = 0.0
    # L1 -> loss’a eklenir
    l1_lambda: float = 0.0

    # --- Early stopping (val loss) ---
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0

    # --- Scheduler ---
    # step scheduler
    use_scheduler: bool = True
    step_size: int = 10
    gamma: float = 0.5

    # --- Model (MLP) ---
    hidden_sizes: List[int] = None  # e.g., [512,256,128]
    activation: str = "relu"        # "relu" or "gelu"
    use_bn: bool = True
    dropout: float = 0.3

    # --- Visualization / logging ---
    save_curves: bool = True
    export_torchviz: bool = True
    torchviz_format: str = "png"    # png/pdf/svg
    log_every: int = 100            # mini-batch log frequency

    # --- Evaluation ---
    val_split: float = 0.1


def _parse_int_list(s: str) -> List[int]:
    """
    Parse '512,256,128' -> [512, 256, 128]
    """
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",")]


def build_config_from_cli() -> TrainConfig:
    """
    Create TrainConfig from argparse CLI.
    """
    p = argparse.ArgumentParser(description="CS515 HW1a - MLP on MNIST")

    # Paths & seed
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./outputs")

    # Device
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # Train
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)

    # Regularization
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--l1_lambda", type=float, default=0.0)

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=5)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)

    # Scheduler
    p.add_argument("--use_scheduler", action="store_true")
    p.add_argument("--no_scheduler", action="store_true")
    p.add_argument("--step_size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.5)

    # Model
    p.add_argument("--hidden_sizes", type=str, default="512,256,128",
                   help="Comma separated, e.g. '512,256,128'")
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    p.add_argument("--use_bn", action="store_true")
    p.add_argument("--no_bn", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)

    # Viz
    p.add_argument("--save_curves", action="store_true")
    p.add_argument("--no_curves", action="store_true")
    p.add_argument("--export_torchviz", action="store_true")
    p.add_argument("--no_torchviz", action="store_true")
    p.add_argument("--torchviz_format", type=str, default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--log_every", type=int, default=100)

    # Val split
    p.add_argument("--val_split", type=float, default=0.1)

    args = p.parse_args()

    # ----- CHANGED: boolean flags resolution (so defaults behave well) -----
    use_scheduler = True
    if args.use_scheduler:
        use_scheduler = True
    if args.no_scheduler:
        use_scheduler = False

    use_bn = True
    if args.use_bn:
        use_bn = True
    if args.no_bn:
        use_bn = False

    save_curves = True
    if args.save_curves:
        save_curves = True
    if args.no_curves:
        save_curves = False

    export_torchviz = True
    if args.export_torchviz:
        export_torchviz = True
    if args.no_torchviz:
        export_torchviz = False

    hidden_sizes = _parse_int_list(args.hidden_sizes)
    if not hidden_sizes:
        hidden_sizes = [512, 256, 128]

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
        step_size=args.step_size,
        gamma=args.gamma,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        use_bn=use_bn,
        dropout=args.dropout,
        save_curves=save_curves,
        export_torchviz=export_torchviz,
        torchviz_format=args.torchviz_format,
        log_every=args.log_every,
        val_split=args.val_split,
    )