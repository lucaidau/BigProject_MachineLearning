import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_numpy() -> Tuple[np.ndarray, np.ndarray]:
    """Load the pre‑saved NumPy arrays.

    Returns
    -------
    X : np.ndarray, shape (N, H, W, C)
        Image data (uint8).
    y : np.ndarray, shape (N,)
        Integer labels.
    """
    processed_dir = Path("Data") / "Processed"
    x_path = processed_dir / "x_resized_masked.npy"
    y_path = processed_dir / "y_labels.npy"
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def _prepare_tensors(
    X: np.ndarray, y: np.ndarray, val_size: float = 0.1, test_size: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the data and convert to ``torch.Tensor``.

    The function returns six tensors: ``X_train, y_train, X_val, y_val, X_test, y_test``.
    Images are normalised to ``[0, 1]`` and have a channel dimension added
    (``(N, 1, H, W)``) because the model expects a single‑channel input.
    """
    # First split off the test set, then split the remaining part into train/val.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    val_ratio = val_size / (1 - test_size)  # proportion of the remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=val_ratio,
        stratify=y_tmp,
        random_state=42,
    )

    # Normalise to float32 in [0, 1] and add channel dim.
    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        arr = arr.astype(np.float32) / 255.0  # scale
        if arr.ndim == 3:  # (H, W, C) → (1, H, W)
            arr = arr[..., np.newaxis]
        arr = np.moveaxis(arr, -1, 1)  # (N, C, H, W)
        return torch.from_numpy(arr)

    X_train_t = _to_tensor(X_train)
    X_val_t = _to_tensor(X_val)
    X_test_t = _to_tensor(X_test)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))
    y_val_t = torch.from_numpy(y_val.astype(np.int64))
    y_test_t = torch.from_numpy(y_test.astype(np.int64))
    return (
        X_train_t,
        y_train_t,
        X_val_t,
        y_val_t,
        X_test_t,
        y_test_t,
    )


def save_processed_tensors(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    out_dir: Path = Path("Data") / "Processed",
) -> None:
    """Save the split tensors to ``.pt`` files.

    The function creates ``out_dir`` if necessary and stores the six tensors under obvious
    filenames (e.g. ``X_train.pt``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(X_train, out_dir / "X_train.pt")
    torch.save(y_train, out_dir / "y_train.pt")
    torch.save(X_val, out_dir / "X_val.pt")
    torch.save(y_val, out_dir / "y_val.pt")
    torch.save(X_test, out_dir / "X_test.pt")
    torch.save(y_test, out_dir / "y_test.pt")
    print(f"Saved processed tensors to {out_dir}")


def main() -> None:
    """Entry‑point used by ``python -m src.preProcessing``.

    It loads the raw ``.npy`` files, splits the data, and writes the ``.pt`` tensors.
    """
    X, y = _load_numpy()
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = _prepare_tensors(X, y)
    print(
        f"Split sizes – train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )
    save_processed_tensors(
        X_train, y_train, X_val, y_val, X_test, y_test
    )


if __name__ == "__main__":
    main()
