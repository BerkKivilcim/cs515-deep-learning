# train.py
"""
Training utilities: loaders, train loop, early stopping, scheduler, curves, torchviz.

CHANGES vs tutorial repo:
- Early stopping (val loss) eklendi.
- Loss/acc curve logging + plot.
- Torchviz export eklendi.
- L1 regularization eklendi (loss’a ek).
- Device handling train loop içinde net.
- Type hints + docstrings.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



def seed_everything(seed: int) -> None:
    """
    Seed torch for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    """
    Resolve device string to torch.device.
    """
    if device_str == "cuda":
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders for MNIST.
    """
    tf = transforms.Compose([transforms.ToTensor()])

    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=tf)

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
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
    Evaluate average loss and accuracy.
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

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def l1_penalty(model: nn.Module) -> torch.Tensor:
    """
    Compute L1 penalty over all parameters.
    """
    l1 = torch.tensor(0.0, device=next(model.parameters()).device)
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
    Export computation graph with torchviz.

    Requirements:
    - pip install torchviz
    - graphviz installed on system and in PATH (dot command).
    """
    try:
        from torchviz import make_dot
    except Exception as e:
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
    Plot train/val loss and accuracy curves into a single PNG.
    """
    epochs = list(range(1, len(train_losses) + 1))

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
    cfg,  # TrainConfig (import cycle olmasın diye burada type yazmadım)
) -> Dict[str, float]:
    """
    Full training pipeline: loaders -> train loop -> early stopping -> save best.
    Returns summary metrics.
    """
    seed_everything(cfg.seed)

    device = resolve_device(cfg.device)
    print(f"Using device: {device}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    (out_dir / "config.txt").write_text(str(asdict(cfg)), encoding="utf-8")

    train_loader, val_loader = get_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
    )

    tf = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
)

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    # ----- CHANGED: L2 regularization via weight_decay (cfg.weight_decay) -----
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    scheduler = None
    if cfg.use_scheduler:
        # ----- CHANGED: scheduler exists and is configurable -----
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )

    # Curves for plotting
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    test_accs: List[float] = []

    # Early stopping
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    best_ckpt_path = out_dir / "best_model.pt"

    # Optional: torchviz export at start (architecture)
    if cfg.export_torchviz and device.type == "cuda":
        export_model_graph_torchviz(
            model=model,
            device=device,
            out_path_no_ext=out_dir / "mlp_graph",
            fmt=cfg.torchviz_format,
        )
    elif cfg.export_torchviz and device.type != "cuda":
        # CPU’da da export edebilirsin, ama graphviz yoksa hata vermesin diye try/except zaten var
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

            # ----- CHANGED: L1 regularization (optional) -----
            if cfg.l1_lambda > 0.0:
                loss = loss + cfg.l1_lambda * l1_penalty(model)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            if cfg.log_every > 0 and step % cfg.log_every == 0:
                print(f"Epoch {epoch}/{cfg.epochs} | step {step}/{len(train_loader)} | loss={loss.item():.4f}")

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
        test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)

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

        # ----- CHANGED: early stopping on val_loss -----
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

    # Plot curves
    if cfg.save_curves:
        plot_curves(
            train_losses=train_losses,
            val_losses=val_losses,
            train_accs=train_accs,
            val_accs=val_accs,
            test_accs=test_accs,  # NEW
            out_path=out_dir / "curves.png",
        )
        print(f"[INFO] Curves saved under: {out_dir}")

    # Load best model before returning
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    final_val_loss, final_val_acc = evaluate(model, val_loader, device, loss_fn)

    summary = {
        "best_epoch": float(best_epoch),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(final_val_loss),
        "final_val_acc": float(final_val_acc),
    }
    (out_dir / "summary.txt").write_text(str(summary), encoding="utf-8")
    return summary