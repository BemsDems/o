from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from src.config.settings import CFG, BASE_FEATURES, FUND_FEATURES
from src.data.loaders import cbr_key_rate_range, load_candles_moexalgo, moex_iss_dividends
from src.data.fundamentals import (
    add_fundamental_features_past_only,
    combine_fundamental_sources,
    fetch_smartlab_financials,
    normalize_fundamentals,
)
from src.features.engineering import build_features
from src.features.target import add_target, make_windows_grouped


def prepare_dataset_once_panel() -> Dict[str, Any]:
    """Load panel data + build features + make grouped windows once (shared across seeds).

    Key property: windows are created strictly inside each ticker segment
    (no cross-ticker boundary leakage).
    """
    print("Loading common market series from MOEX...")
    usd = load_candles_moexalgo("USD000UTSTOM", CFG["START"], CFG["END"])
    imo = load_candles_moexalgo("IMOEX", CFG["START"], CFG["END"])
    print("USD:", usd.shape, "IMOEX:", imo.shape)

    print("\nLoading CBR key rate...")
    key_rate = cbr_key_rate_range(CFG["START"], CFG["END"])
    print("Key rate rows:", len(key_rate))

    px_map: Dict[str, pd.DataFrame] = {}
    panel_parts = []

    tickers = list(CFG.get("TICKERS") or [])
    if not tickers:
        tickers = [str(CFG["TICKER"])]

    for ticker in tickers:
        print(f"\nLoading stock: {ticker}")
        px = load_candles_moexalgo(ticker, CFG["START"], CFG["END"])
        if px.empty:
            print(f"Skip {ticker}: empty candles")
            continue

        px_map[str(ticker)] = px.copy()

        if CFG.get("USE_DIVIDENDS", False):
            divs = moex_iss_dividends(ticker)
            print(f"Div rows for {ticker}:", len(divs))
        else:
            divs = pd.DataFrame(columns=["date", "dividend_rub", "currency"])

        feat_i = build_features(px, usd, imo, key_rate, divs)

        # Fundamentals (past-only attachment, no leakage)
        if CFG.get("USE_FUNDAMENTALS", False):
            fund_msfo = fetch_smartlab_financials(
                ticker=ticker,
                report_type="MSFO",
                freq="q",
            )
            fund_rsbu = fetch_smartlab_financials(
                ticker=ticker,
                report_type="RSBU",
                freq="q",
            )

            fund = combine_fundamental_sources(
                normalize_fundamentals(fund_msfo),
                normalize_fundamentals(fund_rsbu),
            )

            feat_i = add_fundamental_features_past_only(
                feat_i,
                fund,
                ticker=ticker,
                lag_days=int(CFG.get("FUND_LAG_DAYS", 1)),
            )

        feat_i = add_target(feat_i, int(CFG["HORIZON"]), float(CFG["THR_MOVE"]))
        if feat_i.empty:
            print(f"Skip {ticker}: empty feature dataset after target")
            continue

        feat_i = feat_i.copy()
        feat_i["ticker"] = str(ticker)
        feat_i = feat_i.reset_index().rename(columns={feat_i.index.name or "index": "date"})
        panel_parts.append(feat_i)

    if not panel_parts:
        raise ValueError("Panel dataset is empty: no tickers produced usable data.")

    feat = pd.concat(panel_parts, ignore_index=True)
    feat["date"] = pd.to_datetime(feat["date"], errors="coerce")
    feat = feat.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    print("\nPanel dataset:", feat.shape)
    print("Rows by ticker:")
    print(feat["ticker"].value_counts().sort_index().to_string())
    print("Class share (BUY=1):", float(feat["Target"].mean().round(3)))

    print("\nTarget share by ticker:")
    print(
        feat.groupby("ticker")["Target"]
        .mean()
        .sort_index()
        .to_string()
    )

    unique_dates = np.array(sorted(feat["date"].unique()))
    n_dates = int(len(unique_dates))
    n_train = int(n_dates * float(CFG["TRAIN_FRAC"]))
    n_val = int(n_dates * float(CFG["VAL_FRAC"]))

    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val :])

    feat["split"] = np.where(
        feat["date"].isin(train_dates),
        "train",
        np.where(feat["date"].isin(val_dates), "val", "test"),
    )

    # Ticker identity features (one-hot). This is critical for a multi-ticker
    # classifier so the model can learn per-ticker baselines/regimes.
    ticker_dummies = pd.get_dummies(feat["ticker"], prefix="ticker", dtype=float)
    feat = pd.concat([feat, ticker_dummies], axis=1)

    fund_core_cols = [
        "roe",
        "pb_ratio",
        "net_margin",
        "value_quality",
        "log_revenue",
        "log_net_income",
        "eps",
    ]
    present_fund_core = [c for c in fund_core_cols if c in feat.columns]

    if present_fund_core:
        print("\nFUNDAMENTAL COVERAGE BY TICKER (non-null share):")
        cov = feat.groupby("ticker")[present_fund_core].apply(lambda x: x.notna().mean())
        print(cov.to_string())

        # Make sure fund columns are numeric and fill NaNs with TRAIN median (past-only).
        for c in present_fund_core:
            feat[c] = pd.to_numeric(feat[c], errors="coerce")
            feat[c] = feat[c].replace([np.inf, -np.inf], np.nan)

            med = feat.loc[feat["split"] == "train", c].median()
            if pd.isna(med):
                med = 0.0
            feat[c] = feat[c].fillna(med)

    ticker_features = sorted(ticker_dummies.columns.tolist())
    fund_features_present = [c for c in FUND_FEATURES if c in feat.columns]

    FEATURES = (
        [c for c in BASE_FEATURES if c in feat.columns]
        + fund_features_present
        + ticker_features
    )
    print("\nSELECTED FEATURES:")
    print(FEATURES)
    print(f"Признаков используется: {len(FEATURES)}")
    print(f"Fundamental features added: {fund_features_present}")
    print(f"Ticker features added: {ticker_features}")

    if CFG.get("USE_FUNDAMENTALS", False):
        expected_fund = {"roe", "pb_ratio", "net_margin", "value_quality"}
        missing_fund = sorted(expected_fund - set(FEATURES))
        if missing_fund:
            print(f"WARNING: some expected fundamental features are missing: {missing_fund}")

    scaler = RobustScaler()

    X_all_2d = feat[FEATURES].values
    y_all = feat["Target"].values.astype(int)
    dates_all = feat["date"].values
    tickers_all = feat["ticker"].astype(str).values
    split_all = feat["split"].astype(str).values
    future_ret_all = feat["future_ret"].values.astype(float)

    train_rows = split_all == "train"
    X_train_raw_2d = X_all_2d[train_rows]

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

    Xw, yw, dw, gw, iw = make_windows_grouped(
        X_all_scaled,
        y_all,
        dates_all,
        tickers_all,
        int(CFG["WINDOW"]),
    )

    split_w = split_all[iw]
    future_ret_w = future_ret_all[iw]

    train_mask = split_w == "train"
    val_mask = split_w == "val"
    test_mask = split_w == "test"

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
        "px_map": px_map,
        "FEATURES": FEATURES,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "dw": dw,
        "group_w": gw,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "future_ret_w": future_ret_w,
        "class_weight": class_weight,
        "clip_lo": lo,
        "clip_hi": hi,
    }
