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


def make_windows_grouped(
    X_2d: np.ndarray,
    y_1d: np.ndarray,
    dates_1d: np.ndarray,
    groups_1d: np.ndarray,
    window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make windows strictly within a single group (ticker).

    Returns:
      Xw, yw, dw, gw, iw
    where:
      dw = date of the last element of each window,
      gw = group (ticker) of the last element of each window,
      iw = index of the last element in the original 1D arrays.

    Important: expects X_2d/y_1d/dates_1d/groups_1d to be sorted so that
    rows of the same group are contiguous (e.g., sort by [ticker, date]).
    """
    Xw, yw, dw, gw, iw = [], [], [], [], []

    start = 0
    n = int(len(X_2d))
    groups_1d = np.asarray(groups_1d)

    while start < n:
        g = groups_1d[start]
        end = start
        while end < n and groups_1d[end] == g:
            end += 1

        for i in range(start + window - 1, end):
            Xw.append(X_2d[i - window + 1 : i + 1])
            yw.append(y_1d[i])
            dw.append(dates_1d[i])
            gw.append(groups_1d[i])
            iw.append(i)

        start = end

    return (
        np.asarray(Xw),
        np.asarray(yw),
        np.asarray(dw),
        np.asarray(gw),
        np.asarray(iw),
    )
