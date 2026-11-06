import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def _load_numpy() -> tuple[np.ndarray, np.ndarray]:
    # Use memory-mapped loading to avoid high RAM usage for large .npy files
    processed_dir = Path("Data") / "Processed"
    X = np.load(processed_dir / "x_resized_masked.npy", mmap_mode='r')
    y = np.load(processed_dir / "y_labels.npy", mmap_mode='r')
    processed_dir = Path("Data") / "Processed"
    X = np.load(processed_dir / "x_resized_masked.npy")
    y = np.load(processed_dir / "y_labels.npy")
    return X, y


def _binary_labels(y: np.ndarray) -> np.ndarray:
    return (y == 1).astype(np.int64)


def _flatten_images(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def _split_data(
    X: np.ndarray, y: np.ndarray, val_size: float = 0.1, test_size: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=val_ratio,
        stratify=y_tmp,
        random_state=42,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def _train_and_report(model, X_train, y_train, X_val, y_val, X_test, y_test, name: str):
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Validation accuracy : {accuracy_score(y_val, val_pred):.4f}")
    print(f"Test accuracy       : {accuracy_score(y_test, test_pred):.4f}")
    print("\nClassification report (test set):")
    print(
        classification_report(
            y_test, test_pred, target_names=["Non‑COVID", "COVID"]
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline classical ML models on the extracted COVID‑19 X‑ray data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["svm", "logreg", "knn", "all"],
        help="Which model(s) to train",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load raw NumPy arrays, convert to binary, flatten, split
    # ---------------------------------------------------------------
    X_raw, y_raw = _load_numpy()
    print(f"Loaded raw data – X shape: {X_raw.shape}, y shape: {y_raw.shape}")
    y_bin = _binary_labels(y_raw)
    X_flat = _flatten_images(X_raw)

    X_train, y_train, X_val, y_val, X_test, y_test = _split_data(X_flat, y_bin)
    print(
        f"Dataset split – train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    # ---------------------------------------------------------------
    # Train selected models
    # ---------------------------------------------------------------
    if args.model in ("logreg", "all"):
        logreg = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs"
        )
        _train_and_report(
            logreg, X_train, y_train, X_val, y_val, X_test, y_test, "Logistic Regression"
        )

    if args.model in ("knn", "all"):
        knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
        _train_and_report(
            knn, X_train, y_train, X_val, y_val, X_test, y_test, "K‑Nearest Neighbours (k=5)"
        )

    if args.model in ("svm", "all"):
        svm = SVC(kernel="linear", C=1.0, class_weight="balanced")
        _train_and_report(
            svm, X_train, y_train, X_val, y_val, X_test, y_test, "Support Vector Machine (Linear)"
        )


if __name__ == "__main__":
    main()

