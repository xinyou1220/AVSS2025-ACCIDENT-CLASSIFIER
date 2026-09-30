"""Train the frame-level accident CNN on an image folder (``<split>/<label>/*.jpg``).

Example:
    python train_cnn.py --train-dir data/cnn/training --val-dir data/cnn/validation --depth 4
"""
import argparse
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from accident.cnn import AccidentCNN  # noqa: E402
from accident.data import load_image_folder  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train-dir", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--output-dir", default="runs/cnn")
    p.add_argument("--depth", type=int, default=4, help="number of conv blocks (4 or 5 in our experiments)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ("loss", "acc")):
        ax.plot(epochs, history[f"train_{metric}"], "b-", label="train")
        ax.plot(epochs, history[f"val_{metric}"], "r-", label="validation")
        ax.set(title=metric, xlabel="epoch")
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_dir = os.path.join(args.output_dir, "every_epoch")
    os.makedirs(epoch_dir, exist_ok=True)

    size = (args.image_size, args.image_size)
    train_set = TensorDataset(*load_image_folder(args.train_dir, size))
    val_set = TensorDataset(*load_image_folder(args.val_dir, size))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    print(f"train={len(train_set)} val={len(val_set)} device={device}")

    model = AccidentCNN(depth=args.depth).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[key].append(value)
        print(f"[{epoch:03d}/{args.epochs}] {time.time() - start:.1f}s "
              f"train acc {train_acc:.4f} loss {train_loss:.4f} | val acc {val_acc:.4f} loss {val_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(epoch_dir, f"epoch_{epoch:03d}.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))

    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))
    print(f"Best validation loss {best_val_loss:.4f}; weights in {args.output_dir}")


if __name__ == "__main__":
    main()
