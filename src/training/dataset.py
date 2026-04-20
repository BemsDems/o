from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config.settings import CFG, BASE_FEATURES
from src.data.loaders import cbr_key_rate_range, load_candles_moexalgo, moex_iss_dividends
from src.features.engineering import build_features
from src.features.target import add_target, make_windows_aligned, time_splits


def prepare_dataset_once() -> Dict[str, Any]:
    """Load data + build features + make windows once (shared across seeds)."""
    print("Loading market series from MOEX...")
    px = load_candles_moexalgo(CFG["TICKER"], CFG["START"], CFG["END"])
    usd = load_candles_moexalgo("USD000UTSTOM", CFG["START"], CFG["END"])
    imo = load_candles_moexalgo("IMOEX", CFG["START"], CFG["END"])
    print(f"{CFG['TICKER']}:", px.shape, "USD:", usd.shape, "IMOEX:", imo.shape)

    print("\nLoading CBR key rate...")
    key_rate = cbr_key_rate_range(CFG["START"], CFG["END"])
    print("Key rate rows:", len(key_rate))

    if CFG.get("USE_DIVIDENDS", False):
        print("\nLoading MOEX dividends...")
        divs = moex_iss_dividends(CFG["TICKER"])
        print("Div rows:", len(divs))
    else:
        print("\nSkipping dividends: USE_DIVIDENDS=False")
        divs = pd.DataFrame(columns=["date", "dividend_rub", "currency"])

    print("\nBuilding features (Russian sources only)...")
    feat = build_features(px, usd, imo, key_rate, divs)
    feat = add_target(feat, CFG["HORIZON"], CFG["THR_MOVE"])
    print("Final dataset:", feat.shape)
    print("Class share (BUY=1):", float(feat["Target"].mean().round(3)))

    FEATURES = [c for c in BASE_FEATURES if c in feat.columns]

    print("\nSELECTED FEATURES:")
    print(FEATURES)
    print(f"Признаков используется: {len(FEATURES)}")

    train_idx, val_idx, test_idx = time_splits(feat, CFG["TRAIN_FRAC"], CFG["VAL_FRAC"])

    scaler = RobustScaler()
    X_all_2d = feat[FEATURES].values
    y_all = feat["Target"].values.astype(int)
    dates_all = feat.index.values
    future_ret_all = feat["future_ret"].values.astype(float)

    X_train_raw_2d = feat.loc[train_idx, FEATURES].values

    clip_q = float(CFG.get("CLIP_Q", 0.0) or 0.0)
    if 0.0 < clip_q < 0.5:
        lo = np.nanquantile(X_train_raw_2d, clip_q, axis=0)
        hi = np.nanquantile(X_train_raw_2d, 1 - clip_q, axis=0)
        X_all_2d = np.clip(X_all_2d, lo, hi)
        scaler.fit(np.clip(X_train_raw_2d, lo, hi))
    else:
        lo = None
        hi = None
        scaler.fit(X_train_raw_2d)

    X_all_scaled = scaler.transform(X_all_2d)

    Xw, yw, dw = make_windows_aligned(X_all_scaled, y_all, dates_all, CFG["WINDOW"])
    future_ret_w = future_ret_all[CFG["WINDOW"] - 1 :]

    train_mask = np.isin(dw, train_idx.values)
    val_mask = np.isin(dw, val_idx.values)
    test_mask = np.isin(dw, test_idx.values)

    X_train, y_train = Xw[train_mask], yw[train_mask]
    X_val, y_val = Xw[val_mask], yw[val_mask]
    X_test, y_test = Xw[test_mask], yw[test_mask]

    print("\nWindows shapes:")
    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print("class_weight:", class_weight)

    return {
        "feat": feat,
        "px": px,
        "FEATURES": FEATURES,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "dw": dw,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "future_ret_w": future_ret_w,
        "class_weight": class_weight,
        "clip_lo": lo,
        "clip_hi": hi,
    }

