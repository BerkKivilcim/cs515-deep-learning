# train.py
"""
Training utilities for MNIST classification using an MLP.

This module provides:
- Reproducibility utilities (seeding).
- Device selection (cpu/cuda/auto).
- DataLoader construction for train/validation/test splits.
- Training loop with:
  - optional L1 regularization (added to loss),
  - optional L2 regularization via optimizer weight_decay,
  - optional learning-rate scheduler,
  - early stopping based on validation loss,
  - curve plotting (loss/accuracy),
  - optional computation graph export via torchviz.

Notes
-----
- The training set is split into train/validation subsets using `random_split`.
  For strict reproducibility of the split, you should pass a seeded `generator`
  to `random_split`. This file keeps your original behavior, but a small
  improvement suggestion is included in docstrings below.
- The best checkpoint is saved as `{cfg.out_dir}/best_model.pt` and is expected
  to contain at least a `"model"` key storing the model state_dict.

File outputs
------------
- config snapshot:  `{cfg.out_dir}/config.txt`
- best checkpoint:  `{cfg.out_dir}/best_model.pt`
- curves:           `{cfg.out_dir}/curves_loss.png` and `{cfg.out_dir}/curves_acc.png` (if enabled)
- summary:          `{cfg.out_dir}/summary.txt`

Dependencies
------------
- torch, torchvision, matplotlib
- torchviz (optional): `pip install torchviz` and system Graphviz `dot` executable

Typical usage
-------------
This module is typically called from `main.py`:

    summary = run_training(model, cfg)
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Protocol, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class TrainConfigLike(Protocol):
    """
    Minimal configuration interface required by `run_training`.

    This Protocol avoids importing a concrete TrainConfig dataclass here,
    preventing potential import cycles. Any object with the following
    attributes is accepted.

    Attributes
    ----------
    seed: int
        Random seed for reproducibility.
    device: str
        Device selection: "auto", "cuda", or "cpu".
    out_dir: str
        Output directory for checkpoints and logs.
    data_dir: str
        Root directory for MNIST download/cache.
    batch_size: int
        Batch size for DataLoaders.
    num_workers: int
        Number of DataLoader workers.
    val_split: float
        Fraction of training set to use for validation (e.g., 0.1).
    epochs: int
        Maximum number of epochs to train.
    lr: float
        Learning rate for the optimizer.
    weight_decay: float
        L2 regularization strength via optimizer weight_decay.
    use_scheduler: bool
        Whether to use a learning-rate scheduler.
    step_size: int
        StepLR step size (epochs).
    gamma: float
        StepLR decay factor.
    export_torchviz: bool
        Whether to export the computation graph via torchviz.
    torchviz_format: str
        Output format for torchviz render (e.g., "png", "pdf").
    l1_lambda: float
        L1 regularization coefficient; 0 disables L1 penalty.
    log_every: int
        Print training status every N steps; 0 disables step logging.
    early_stop_patience: int
        Number of epochs to wait without improvement before stopping.
    early_stop_min_delta: float
        Minimum validation-loss improvement required to reset patience.
    save_curves: bool
        Whether to save loss/accuracy curves as PNG files.
    """

    seed: int
    device: str

    out_dir: str
    data_dir: str
    batch_size: int
    num_workers: int
    val_split: float

    epochs: int
    lr: float
    weight_decay: float

    use_scheduler: bool
    step_size: int
    gamma: float

    export_torchviz: bool
    torchviz_format: str

    l1_lambda: float
    log_every: int

    early_stop_patience: int
    early_stop_min_delta: float

    save_curves: bool


def seed_everything(seed: int) -> None:
    """
    Seed PyTorch random number generators for reproducibility.

    Parameters
    ----------
    seed:
        Random seed value.

    Notes
    -----
    - This seeds Torch CPU RNG via `torch.manual_seed`.
    - This seeds all CUDA devices via `torch.cuda.manual_seed_all`.
    - For fully deterministic behavior, you may also need:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
      However, this can slow down training.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """
    Resolve a user-provided device string to a `torch.device`.

    Parameters
    ----------
    device_str:
        One of: "cuda", "cpu", or "auto".

    Returns
    -------
    torch.device
        The resolved torch device.

    Examples
    --------
    >>> resolve_device("cuda")
    device(type='cuda')
    >>> resolve_device("auto")
    device(type='cuda')  # if available, else cpu
    """
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")

    # "auto": choose CUDA if available, else CPU
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation splits of MNIST.

    Parameters
    ----------
    data_dir:
        Root directory to download/cache MNIST.
    batch_size:
        Batch size for both loaders.
    num_workers:
        Number of worker processes for DataLoader.
    val_split:
        Fraction of the training set to reserve for validation.

    Returns
    -------
    (train_loader, val_loader):
        A tuple containing DataLoaders for the training and validation splits.

    Notes
    -----
    - Uses `random_split` without an explicit generator; the split is random
      per run unless global RNG is controlled. If you want the split to be
      fully reproducible across runs, consider:

        g = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)

      This module keeps your original behavior to avoid changing experiments.
    """
    tf = transforms.Compose([transforms.ToTensor()])
    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=tf)

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    """
    Evaluate a model over a dataset split.

    Parameters
    ----------
    model:
        The neural network to evaluate.
    loader:
        DataLoader providing batches of (images, labels).
    device:
        Device on which evaluation is performed.
    loss_fn:
        Loss function used to compute average loss (e.g., CrossEntropyLoss).

    Returns
    -------
    (avg_loss, accuracy):
        Average loss over the dataset and classification accuracy in [0, 1].

    Notes
    -----
    - Uses `torch.no_grad()` to reduce memory usage and speed up evaluation.
    - Assumes classification logits output (N, C) and labels (N,).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
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


def l1_penalty(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 penalty (sum of absolute values) over all model parameters.

    Parameters
    ----------
    model:
        The neural network whose parameters will be regularized.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the L1 norm of all parameters.

    Notes
    -----
    - This is typically added to the data loss as: loss + lambda * l1_penalty(model).
    - The penalty is computed on the same device as the model parameters.
    """
    device = next(model.parameters()).device
    l1 = torch.tensor(0.0, device=device)
    for p in model.parameters():
        l1 = l1 + p.abs().sum()
    return l1


def export_model_graph_torchviz(
    model: nn.Module,
    device: torch.device,
    out_path_no_ext: Path,
    fmt: str = "png",
) -> None:
    """
    Export the model computation graph using torchviz.

    Parameters
    ----------
    model:
        The neural network to visualize.
    device:
        Device used to create the dummy input and run a forward pass.
    out_path_no_ext:
        Output file path WITHOUT extension. Example: out_dir / "mlp_graph"
    fmt:
        Output format passed to graphviz render (e.g., "png", "pdf").

    Requirements
    ------------
    - Python package: `torchviz` (pip install torchviz)
    - System Graphviz installed and `dot` executable available in PATH.

    Notes
    -----
    - This performs a single forward pass with a dummy MNIST-shaped input
      (1, 1, 28, 28) to build the computation graph.
    """
    try:
        from torchviz import make_dot
    except Exception:
        print("[WARN] torchviz not available. Install via: pip install torchviz")
        return

    model.eval()
    x = torch.randn(1, 1, 28, 28, device=device)
    y = model(x)
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.render(str(out_path_no_ext), format=fmt, cleanup=True)
    print(f"[INFO] torchviz graph saved: {out_path_no_ext}.{fmt}")


def plot_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    test_accs: List[float],
    out_path: Path,
) -> None:
    """
    Plot loss and accuracy curves and save them as PNG files.

    Parameters
    ----------
    train_losses, val_losses:
        Per-epoch loss values for training and validation.
    train_accs, val_accs, test_accs:
        Per-epoch accuracy values for training, validation, and test.
    out_path:
        Base output path (file name is used as stem). Two files are written:
        `<stem>_loss.png` and `<stem>_acc.png`.

    Notes
    -----
    - Uses matplotlib default styling.
    - Saves figures at 200 DPI.
    """
    epochs = list(range(1, len(train_losses) + 1))

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_loss.png"), dpi=200)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_accs, label="train_acc")
    plt.plot(epochs, val_accs, label="val_acc")
    plt.plot(epochs, test_accs, label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_acc.png"), dpi=200)
    plt.close()


def run_training(
    model: nn.Module,
    cfg: TrainConfigLike,  # Protocol-based typing to avoid import cycles
) -> Dict[str, float]:
    """
    Execute the full MNIST training pipeline.

    Steps
    -----
    1) Seed RNGs.
    2) Resolve device and create output directory.
    3) Build train/val loaders and test loader.
    4) Configure loss, optimizer, and optional scheduler.
    5) Train for up to `cfg.epochs` with early stopping on validation loss.
    6) Save best model checkpoint and (optionally) plots and torchviz graph.
    7) Load best checkpoint and return summary statistics.

    Parameters
    ----------
    model:
        The model to train.
    cfg:
        Training configuration (see TrainConfigLike for required fields).

    Returns
    -------
    Dict[str, float]
        Summary dictionary containing:
        - best_epoch
        - best_val_loss
        - final_val_loss
        - final_val_acc
    """
    seed_everything(cfg.seed)

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility.
    (out_dir / "config.txt").write_text(str(asdict(cfg)), encoding="utf-8")

    # Data loaders
    train_loader, val_loader = get_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )
    # Test loader (standard MNIST test split)
    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = model.to(device)

    # Classification loss for MNIST.
    loss_fn = nn.CrossEntropyLoss()

    # L2 regularization is typically implemented via optimizer weight_decay.
    # Optimizer options are SGD and Adam
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
  
        )
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )

    # Optional learning-rate scheduler.
    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )

    # Curves for plotting
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    test_accs: List[float] = []

    # Early stopping bookkeeping
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    best_ckpt_path = out_dir / "best_model.pt"

    # Optional: export computation graph with torchviz
    if cfg.export_torchviz:
        export_model_graph_torchviz(
            model=model,
            device=device,
            out_path_no_ext=out_dir / "mlp_graph",
            fmt=cfg.torchviz_format,
        )

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            # Optional L1 regularization (adds |w| penalty to the loss).
            if cfg.l1_lambda > 0.0:
                loss = loss + cfg.l1_lambda * l1_penalty(model)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                print(
                    f"Epoch {epoch}/{cfg.epochs} | step {step}/{len(train_loader)} | loss={loss.item():.4f}"
                )

        # Epoch-level metrics
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        _test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)

        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

        # Early stopping: monitor validation loss improvements
        improved = (best_val_loss - val_loss) > cfg.early_stop_min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, best_ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(
                    f"[EARLY STOP] No val_loss improvement for {cfg.early_stop_patience} epochs. "
                    f"Best epoch={best_epoch}, best_val_loss={best_val_loss:.4f}"
                )
                break

    # Save curves (optional)
    if cfg.save_curves:
        plot_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            test_accs=test_accs,
            out_path=out_dir / "curves.png",
        )
        print(f"[INFO] Curves saved under: {out_dir}")

    # Load best model checkpoint before computing final summary
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    final_val_loss, final_val_acc = evaluate(model, val_loader, device, loss_fn)

    summary: Dict[str, float] = {
        "best_epoch": float(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "final_val_acc": float(final_val_acc),
    }

    (out_dir / "summary.txt").write_text(str(summary), encoding="utf-8")
    return summary