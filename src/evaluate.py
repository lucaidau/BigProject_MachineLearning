import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from model import get_model


def load_test_data(data_dir: Path):
    X_test = torch.load(data_dir / "X_test.pt")
    y_test = torch.load(data_dir / "y_test.pt")
    # Chuyển đổi nhị phân: nhãn 1 (COVID) -> 1, còn lại -> 0
    y_test_bin = (y_test == 1).long()
    test_dataset = TensorDataset(X_test, y_test_bin)
    return test_dataset


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    acc = correct / total if total > 0 else 0.0
    print(f"Độ chính xác trên tập kiểm tra: {acc:.4f} ({correct}/{total})")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phân loại nhị phân COVID-19")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị sử dụng",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(Path("checkpoints") / "best_model.pth"),
        help="Đường dẫn đến checkpoint của mô hình đã lưu",
    )
    args = parser.parse_args()

    data_dir = Path("Data") / "Processed"
    test_dataset = load_test_data(data_dir)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = get_model(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Đã tải trọng số mô hình từ {args.checkpoint}")

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()