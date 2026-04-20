from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def add_target(df: pd.DataFrame, horizon: int, thr: float) -> pd.DataFrame:
    out = df.copy()
    out["future_close"] = out["Close"].shift(-horizon)
    out["future_ret"] = (out["future_close"] - out["Close"]) / (out["Close"] + 1e-12)
    out["Target"] = (out["future_ret"] >= thr).astype(int)
    out = out.dropna().copy()
    return out


def time_splits(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = df.index[:n_train]
    val_idx = df.index[n_train : n_train + n_val]
    test_idx = df.index[n_train + n_val :]
    return train_idx, val_idx, test_idx


def make_windows_aligned(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    dates_1d: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make windows and align y/dates to the LAST index of each window."""
    Xw, yw, dw = [], [], []
    for i in range(window - 1, len(X_2d)):
        Xw.append(X_2d[i - window + 1 : i + 1])
        yw.append(y_1d[i])
        dw.append(dates_1d[i])
    return np.asarray(Xw), np.asarray(yw), np.asarray(dw)

