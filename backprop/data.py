from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_california_housing(
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load California Housing from sklearn, normalize features (StandardScaler fit on
    train only), and return (train_loader, val_loader, test_loader).

    No data leakage: scaler is fit only on the train split.

    Args:
        val_size: Fraction of data for validation (e.g. 0.1 for 10%).
        test_size: Fraction of data for test (e.g. 0.1 for 10%).
        seed: Random seed for reproducible splits.
        batch_size: DataLoader batch size (default 32).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data = fetch_california_housing()
    X: np.ndarray = data.data          # shape (N, 8)
    y: np.ndarray = data.target        # shape (N,)

    # First split off the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Then split val from the remaining train+val pool.
    # Adjust val fraction relative to the trainval pool size.
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction_of_trainval, random_state=seed
    )

    # Fit scaler on train only — no data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def _make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False)
    test_loader = _make_loader(X_test, y_test, shuffle=False)

    return train_loader, val_loader, test_loader
