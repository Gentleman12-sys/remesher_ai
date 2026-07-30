"""
Train MeshGNN on the processed graph dataset (dataset/processed/*.pt,
produced by preprocessing/generate_pairs.py).

Usage:
    python training/train.py \
        --processed_dir dataset/processed \
        --epochs 100 --batch_size 8 --lr 0.0001 \
        --checkpoint_dir checkpoints

Saves:
    checkpoints/best_model.pt   -- state_dict of the best validation epoch
    checkpoints/last_model.pt   -- state_dict of the final epoch
    checkpoints/training_curves.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.mesh_gnn import MeshGNN
from training.dataset import MeshPairDataset
from training.losses import CombinedLoss
from training.metrics import edge_accuracy, edge_precision_recall_f1, epoch_summary


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)

    batch_metrics = []
    for batch in loader:
        batch = batch.to(device)

        with torch.set_grad_enabled(is_train):
            scores = model(batch.x, batch.edge_index, batch.edge_attr)
            loss, components = loss_fn(scores, batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        acc = edge_accuracy(scores.detach(), batch.y)
        prf = edge_precision_recall_f1(scores.detach(), batch.y)
        batch_metrics.append({**components, "accuracy": acc, **prf})

    return epoch_summary(batch_metrics)


def plot_curves(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(history["train_total"], label="train")
    axes[0].plot(history["val_total"], label="val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Edge Classification Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    axes[2].plot(history["val_precision"], label="precision")
    axes[2].plot(history["val_recall"], label="recall")
    axes[2].plot(history["val_f1"], label="f1")
    axes[2].set_title("Edge Prediction Quality (val)")
    axes[2].set_xlabel("epoch")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the AI-MeshOptimizer GNN.")
    parser.add_argument("--processed_dir", default="dataset/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected -- training will run on CPU (this will be slow for large datasets).")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = MeshPairDataset(args.processed_dir)
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
    print(f"Dataset: {len(dataset)} meshes -> {n_train} train / {n_val} val")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MeshGNN(hidden_dim=args.hidden_dim, gat_heads=args.gat_heads, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CombinedLoss()

    history = {
        "train_total": [], "val_total": [],
        "train_accuracy": [], "val_accuracy": [],
        "val_precision": [], "val_recall": [], "val_f1": [],
    }
    best_val_loss = float("inf")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        train_stats = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_stats = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step()

        history["train_total"].append(train_stats["total"])
        history["val_total"].append(val_stats["total"])
        history["train_accuracy"].append(train_stats["accuracy"])
        history["val_accuracy"].append(val_stats["accuracy"])
        history["val_precision"].append(val_stats["precision"])
        history["val_recall"].append(val_stats["recall"])
        history["val_f1"].append(val_stats["f1"])

        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_stats['total']:.4f} val_loss={val_stats['total']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f} val_f1={val_stats['f1']:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_stats["total"] < best_val_loss:
            best_val_loss = val_stats["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                checkpoint_dir / "best_model.pt",
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": args.epochs, "args": vars(args)},
        checkpoint_dir / "last_model.pt",
    )
    plot_curves(history, checkpoint_dir / "training_curves.png")
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoints and plots saved to {checkpoint_dir}/")


if __name__ == "__main__":
    main()
