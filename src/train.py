
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import get_model


def load_data(data_dir: Path):
    """Load the ``.pt`` tensors created by ``preProcessing.py``.
    Returns ``train_loader`` and ``val_loader``.
    """
    X_train = torch.load(data_dir / "X_train.pt")
    y_train = torch.load(data_dir / "y_train.pt")
    X_val = torch.load(data_dir / "X_val.pt")
    y_val = torch.load(data_dir / "y_val.pt")

    # Convert to binary labels: COVID (original label 1) -> 1, others -> 0
    y_train_bin = (y_train == 1).long()
    y_val_bin = (y_val == 1).long()

    train_dataset = TensorDataset(X_train, y_train_bin)
    val_dataset = TensorDataset(X_val, y_val_bin)
    return train_dataset, val_dataset


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
    checkpoint_path: Path,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"TrainLoss: {epoch_loss:.4f} "
            f"ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f}"
        )

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Đã lưu mô hình tại {checkpoint_path}")

def main():
    # Đặt seed để tái tạo kết quả
    from utils import seed_everything
    seed_everything(42)
    parser = argparse.ArgumentParser(description="Train COVID‑19 binary classifier")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    args = parser.parse_args()

    data_dir = Path("Data") / "Processed"
    train_dataset, val_dataset = load_data(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = get_model(device)

    checkpoint_path = Path("checkpoints") / "best_model.pth"
    train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        checkpoint_path=checkpoint_path,
    )

if __name__ == "__main__":
    main()