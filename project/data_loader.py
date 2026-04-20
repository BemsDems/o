from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
from moexalgo import Ticker

from project.config import CFG


def _resolve_end_date(end: str | None) -> str:
    if end is None:
        return datetime.now().strftime("%Y-%m-%d")
    return str(end)


def fetch_moex_candles(secid: str, start: str, end: str | None) -> pd.DataFrame:
    end_resolved = _resolve_end_date(end)
    raw = Ticker(secid).candles(start=str(start), end=str(end_resolved), period="1D")
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    df = df.dropna(subset=["begin"]).drop_duplicates(subset=["begin"]).set_index("begin")

    keep = ["close", "high", "low", "volume"]
    df = df[keep].sort_index()
    df = df.rename(columns={"close": "CLOSE", "high": "HIGH", "low": "LOW", "volume": "VOLUME"})
    df["secid"] = str(secid)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features_one(df: pd.DataFrame, *, secid: str = "") -> pd.DataFrame:
    """Feature engineering — 20 features (15 base + 5 new)."""
    out = df.copy()

    n0 = len(out)
    if secid:
        print(f"  candles: {n0}")

    # Log-returns
    for lag in (1, 2, 3, 5, 10):
        price_ratio = out["CLOSE"] / out["CLOSE"].shift(lag)
        price_ratio = price_ratio.clip(0.5, 2.0)
        out[f"logret_{lag}"] = np.log(price_ratio)
        out[f"logret_{lag}"] = out[f"logret_{lag}"].replace([np.inf, -np.inf], np.nan)

    # Trend: SMA
    out["sma_20"] = out["CLOSE"].rolling(20).mean()
    out["sma_50"] = out["CLOSE"].rolling(50).mean()
    out["sma_200"] = out["CLOSE"].rolling(200).mean()
    out["trend_up_20"] = (out["CLOSE"] > out["sma_20"]).astype(int)
    out["trend_up_50"] = (out["CLOSE"] > out["sma_50"]).astype(int)
    out["trend_up_200"] = (out["CLOSE"] > out["sma_200"]).astype(int)

    # Volume
    out["vol_ma_20"] = out["VOLUME"].rolling(20).mean()
    out["vol_rel"] = out["VOLUME"] / (out["vol_ma_20"] + 1e-9)
    out["vol_rel"] = out["vol_rel"].clip(0.1, 3.0)
    out["vol_spike"] = (out["vol_rel"] > 2.0).astype(int)

    # RSI
    out["rsi_14"] = compute_rsi(out["CLOSE"], 14)
    out["rsi_14"] = out["rsi_14"].clip(0.0, 100.0)
    out["rsi_oversold"] = (out["rsi_14"] < 30).astype(int)
    out["rsi_overbought"] = (out["rsi_14"] > 70).astype(int)

    # Price position
    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_pos_20"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-9)
    out["price_pos_20"] = out["price_pos_20"].clip(0.0, 1.0)

    # Volatility
    out["volatility_20"] = out["logret_1"].rolling(20).std()
    out["volatility_20"] = out["volatility_20"].clip(0.0, 0.1)

    # MACD (12/26/9)
    ema_12 = out["CLOSE"].ewm(span=12, adjust=False).mean()
    ema_26 = out["CLOSE"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_norm"] = (macd_line / out["CLOSE"]).clip(-0.05, 0.05)
    out["macd_hist_norm"] = ((macd_line - macd_signal) / out["CLOSE"]).clip(-0.03, 0.03)
    out["macd_cross_up"] = (
        (macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1))
    ).astype(int)

    # Bollinger Bands (20, 2σ)
    bb_mid = out["CLOSE"].rolling(20).mean()
    bb_std = out["CLOSE"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_pos"] = ((out["CLOSE"] - bb_lower) / (bb_upper - bb_lower + 1e-9)).clip(0.0, 1.0)
    out["bb_width"] = ((bb_upper - bb_lower) / (bb_mid + 1e-9)).clip(0.0, 0.3)

    if secid:
        print(f"  after indicators: {len(out)}")
        print(f"  NaN sma_200: {int(out['sma_200'].isna().sum())}")

    out = out.replace([np.inf, -np.inf], np.nan)

    critical_cols = [
        "logret_1",
        "logret_10",
        "sma_20",
        "vol_ma_20",
        "rsi_14",
        "price_pos_20",
    ]
    out = out.dropna(subset=critical_cols).copy()

    # Fill long-window indicators
    out["sma_200"] = out["sma_200"].ffill()
    out["trend_up_200"] = out["trend_up_200"].fillna(0).astype(int)

    if secid:
        n1 = len(out)
        share = (n1 / n0 * 100.0) if n0 else 0.0
        print(f"  after dropna(critical): {n1} ({share:.1f}%)")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)

    if secid:
        ok = (not out.isnull().any().any())
        print(f"  final NaN check: {ok}")

    return out


def make_target(close: pd.Series, horizon: int, thr: float) -> Tuple[pd.Series, pd.Series]:
    fwd = close.shift(-horizon)
    fwd_ret = (fwd - close) / close
    y = (fwd_ret >= thr).astype(int)
    return y, fwd_ret


@dataclass
class MultiDataset:
    X: np.ndarray
    y: np.ndarray
    fwd_ret: np.ndarray
    dates: np.ndarray
    secids: np.ndarray


def build_multi_ticker_dataset() -> Tuple[MultiDataset, List[str]]:
    rows: List[pd.DataFrame] = []
    for secid in CFG["TICKERS"]:
        print(f"Loading {secid}...")
        df = fetch_moex_candles(secid, str(CFG["START"]), CFG["END"])
        print(f"  loaded rows: {len(df)}")
        if df.empty:
            print("  -> empty, skip")
            continue

        df_feat = build_features_one(df, secid=secid)

        min_rows = max(250, int(CFG["SEQ_LEN"]) + int(CFG["HORIZON"]) + 50)
        if len(df_feat) < min_rows:
            print(f"  -> too few rows after features ({len(df_feat)} < {min_rows}), skip")
            continue

        y, fwd_ret = make_target(
            df_feat["CLOSE"], int(CFG["HORIZON"]), float(CFG["THR_MOVE"])
        )

        h = int(CFG["HORIZON"])
        df_feat = df_feat.iloc[:-h]
        y = y.iloc[:-h]
        fwd_ret = fwd_ret.iloc[:-h]

        print(f"  final rows (after horizon trim): {len(df_feat)}")

        tmp = df_feat.copy()
        tmp["target"] = y.values
        tmp["fwd_ret"] = fwd_ret.values
        tmp["date"] = tmp.index
        rows.append(tmp.reset_index(drop=True))

    if not rows:
        raise RuntimeError("No tickers loaded. Check MOEX availability / tickers list.")

    full = pd.concat(rows, axis=0).sort_values(["secid", "date"]).reset_index(drop=True)

    feature_cols = [
        "logret_1",
        "logret_2",
        "logret_3",
        "logret_5",
        "trend_up_20",
        "trend_up_200",
        "price_pos_20",
        "volatility_20",
        "macd_norm",
        "macd_hist_norm",
        "macd_cross_up",
        "bb_pos",
    ]
    feature_cols = [c for c in feature_cols if c in full.columns]

    X = full[feature_cols].values
    y = full["target"].astype(int).values
    fwd_ret = full["fwd_ret"].astype(float).values
    dates = pd.to_datetime(full["date"]).values
    secids = full["secid"].astype(str).values

    print(
        f"\nMulti-ticker dataset: X={X.shape}, pos_rate={y.mean():.3%}, tickers={sorted(set(secids))}"
    )
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    return (
        MultiDataset(X=X, y=y, fwd_ret=fwd_ret, dates=dates, secids=secids),
        feature_cols,
    )
