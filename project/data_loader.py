from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
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


# ── Dividend fundamentals (MOEX ISS) ────────────────────────────────────────


def fetch_dividends_moex(secid: str, timeout: int = 15) -> pd.DataFrame:
    """Fetch full dividend history from MOEX ISS.

    Returns DataFrame with columns:
      - registryclosedate: record date (public anchor)
      - value: dividend per share (RUB)

    Endpoint:
      https://iss.moex.com/iss/securities/{secid}/dividends.json
    """
    url = f"https://iss.moex.com/iss/securities/{secid}/dividends.json"
    try:
        r = requests.get(url, timeout=int(timeout))
        r.raise_for_status()
        data = r.json()

        block = data.get("dividends", {})
        cols = block.get("columns", [])
        rows = block.get("data", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=cols)
        if "registryclosedate" not in df.columns or "value" not in df.columns:
            return pd.DataFrame()

        df["registryclosedate"] = pd.to_datetime(df["registryclosedate"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["registryclosedate", "value"]).copy()

        # Filter to RUB only (avoid mixing currencies)
        if "currencyid" in df.columns:
            df["currencyid"] = df["currencyid"].astype(str)
            df = df[df["currencyid"].str.upper() == "RUB"].copy()

        df = df.sort_values("registryclosedate").reset_index(drop=True)
        return df[["registryclosedate", "value"]]
    except Exception as e:
        print(f"  [div] {secid}: fetch failed ({type(e).__name__}: {e})")
        return pd.DataFrame()


def add_dividend_features_past_only(
    price_df: pd.DataFrame,
    div_df: pd.DataFrame,
    *,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Attach dividend-based features WITHOUT leakage.

    Methodology:
    - effective_date = registryclosedate + lag_days (extra safety)
    - For each price date, use only dividends with effective_date <= date
    - Compute TTM dividend sum over past 365 days

    Adds:
      - div_yield_ttm
      - days_since_last_div (normalized to [0,1])
      - div_yield_is_missing
    """
    out = price_df.copy().reset_index()
    out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if div_df is None or div_df.empty:
        out["div_yield_ttm"] = 0.0
        out["days_since_last_div"] = 1.0
        out["div_yield_is_missing"] = 1
        return out.set_index("date")

    d = div_df.copy()
    d["registryclosedate"] = pd.to_datetime(d["registryclosedate"], errors="coerce")
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    d = d.dropna(subset=["registryclosedate", "value"]).sort_values("registryclosedate").reset_index(drop=True)
    if d.empty:
        out["div_yield_ttm"] = 0.0
        out["days_since_last_div"] = 1.0
        out["div_yield_is_missing"] = 1
        return out.set_index("date")

    d["effective_date"] = d["registryclosedate"] + pd.Timedelta(days=int(lag_days))
    d = d.sort_values("effective_date").reset_index(drop=True)

    out_sorted = out.sort_values("date").reset_index(drop=True)

    # last dividend effective date per row
    last_div = pd.merge_asof(
        out_sorted[["date"]],
        d[["effective_date"]].rename(columns={"effective_date": "last_div_date"}),
        left_on="date",
        right_on="last_div_date",
        direction="backward",
    )
    out_sorted["last_div_date"] = last_div["last_div_date"]

    # TTM dividend sum: O(N*M) is fine (N~3k, M~tens)
    eff_dates = d["effective_date"].values
    eff_values = d["value"].values.astype(float)

    def _ttm_sum(row_date: pd.Timestamp) -> float:
        if pd.isna(row_date):
            return 0.0
        lo = row_date - pd.Timedelta(days=365)
        mask = (eff_dates > np.datetime64(lo)) & (eff_dates <= np.datetime64(row_date))
        return float(eff_values[mask].sum())

    out_sorted["ttm_div"] = out_sorted["date"].apply(_ttm_sum)

    out_sorted["div_yield_ttm"] = out_sorted["ttm_div"] / (out_sorted["CLOSE"] + 1e-9)
    out_sorted["div_yield_ttm"] = out_sorted["div_yield_ttm"].clip(0.0, 0.30)

    days_since = (out_sorted["date"] - out_sorted["last_div_date"]).dt.days
    days_since = days_since.fillna(365.0).clip(0, 365)
    out_sorted["days_since_last_div"] = (days_since / 365.0).astype(float)

    out_sorted["div_yield_is_missing"] = out_sorted["last_div_date"].isna().astype(int)
    out_sorted = out_sorted.drop(columns=["last_div_date", "ttm_div"], errors="ignore")
    return out_sorted.set_index("date")


# ── Macro data ──────────────────────────────────────────────────────────────


def _fetch_macro_close(secid: str, start: str, end: str) -> pd.Series:
    """Fetch daily close for a macro instrument.

    Returns a Series indexed by date (begin).
    """
    try:
        raw = Ticker(secid).candles(start=str(start), end=str(end), period="1D")
        df = pd.DataFrame(raw)
        if df.empty:
            print(f"  [macro] {secid}: empty")
            return pd.Series(dtype=float)

        df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
        df = df.dropna(subset=["begin"]).drop_duplicates(subset=["begin"]).set_index("begin").sort_index()
        if "close" not in df.columns:
            print(f"  [macro] {secid}: no 'close' column")
            return pd.Series(dtype=float)

        return df["close"].astype(float)
    except Exception as e:
        print(f"  [macro] {secid}: fetch failed ({type(e).__name__}: {e})")
        return pd.Series(dtype=float)


def fetch_macro_data(start: str, end: str) -> pd.DataFrame:
    """Fetch macro instruments (USD/RUB, Brent proxy, IMOEX) and compute features.

    Designed to be called once, then joined to each stock's feature frame by date.
    Produces 8 macro features:
      usdrub_logret_1, usdrub_logret_5, usdrub_volatility_20,
      brent_logret_1, brent_logret_5,
      imoex_logret_1, imoex_logret_5, imoex_logret_20.
    """
    print("\n=== LOADING MACRO DATA ===")

    # 1) Fetch closes with fallbacks.
    print("Loading USD/RUB (USD000UTSTOM)...")
    usdrub = _fetch_macro_close("USD000UTSTOM", start, end)

    # Brent on MOEX can be represented by different secids depending on the period.
    # We'll try a few common tickers and keep the first non-empty.
    print("Loading Brent proxy...")
    brent = pd.Series(dtype=float)
    for secid in ("BR", "BRJ4", "BRM4", "BRQ4", "BRZ4"):
        brent = _fetch_macro_close(secid, start, end)
        if not brent.empty:
            break

    print("Loading IMOEX...")
    imoex = _fetch_macro_close("IMOEX", start, end)

    # 2) Build aligned close table (ffill/bfill), allowing missing series.
    close_tbl = pd.DataFrame(index=pd.Index([], name="begin"))
    if not usdrub.empty:
        close_tbl["usdrub_close"] = usdrub
    if not brent.empty:
        close_tbl["brent_close"] = brent
    if not imoex.empty:
        close_tbl["imoex_close"] = imoex

    if close_tbl.empty:
        print("  [macro] WARNING: no macro series loaded; macro features will be zeros")
        return pd.DataFrame()

    close_tbl = close_tbl.sort_index().ffill().bfill()

    # 3) Derived features
    out = pd.DataFrame(index=close_tbl.index)

    # USD/RUB
    if "usdrub_close" in close_tbl.columns:
        c = close_tbl["usdrub_close"]
        for lag in (1, 5):
            ratio = (c / c.shift(lag)).clip(0.5, 2.0)
            out[f"usdrub_logret_{lag}"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
        out["usdrub_volatility_20"] = out["usdrub_logret_1"].rolling(20).std().clip(0.0, 0.1)
    else:
        out["usdrub_logret_1"] = 0.0
        out["usdrub_logret_5"] = 0.0
        out["usdrub_volatility_20"] = 0.0

    # Brent (optional). If we failed to fetch a stable proxy, we deliberately
    # DO NOT create placeholder zero columns to avoid "fake" macro features.
    if "brent_close" in close_tbl.columns:
        c = close_tbl["brent_close"]
        for lag in (1, 5):
            ratio = (c / c.shift(lag)).clip(0.5, 2.0)
            out[f"brent_logret_{lag}"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)

    # IMOEX
    if "imoex_close" in close_tbl.columns:
        c = close_tbl["imoex_close"]
        for lag in (1, 5, 20):
            ratio = (c / c.shift(lag)).clip(0.5, 2.0)
            out[f"imoex_logret_{lag}"] = np.log(ratio).replace([np.inf, -np.inf], np.nan)
    else:
        out["imoex_logret_1"] = 0.0
        out["imoex_logret_5"] = 0.0
        out["imoex_logret_20"] = 0.0

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

    print(f"  [macro] features: {list(out.columns)}")
    print(f"  [macro] date range: {out.index.min()} — {out.index.max()}")
    return out


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features_one(df: pd.DataFrame, *, secid: str = "") -> pd.DataFrame:
    """Feature engineering — technical indicators for a single ticker."""
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
    # Load macro series once, then join to each ticker by date.
    macro_df = fetch_macro_data(str(CFG["START"]), str(CFG["END"]))
    macro_cols = list(macro_df.columns) if not macro_df.empty else []

    # Load dividend history once (per ticker).
    use_div_fund = bool(CFG.get("USE_DIVIDEND_FEATURES", True))
    div_map: Dict[str, pd.DataFrame] = {}
    if use_div_fund:
        print("\n=== LOADING DIVIDEND HISTORY (MOEX ISS) ===")
        for secid in CFG["TICKERS"]:
            d = fetch_dividends_moex(secid)
            div_map[str(secid)] = d
            if d is not None and not d.empty:
                print(
                    f"  {secid}: {len(d)} records, "
                    f"range=[{d['registryclosedate'].min().date()} .. {d['registryclosedate'].max().date()}]"
                )
            else:
                print(f"  {secid}: no dividend data")

    rows: List[pd.DataFrame] = []
    for secid in CFG["TICKERS"]:
        print(f"Loading {secid}...")
        df = fetch_moex_candles(secid, str(CFG["START"]), CFG["END"])
        print(f"  loaded rows: {len(df)}")
        if df.empty:
            print("  -> empty, skip")
            continue

        df_feat = build_features_one(df, secid=secid)

        # Attach dividend fundamentals (past-only)
        if use_div_fund:
            df_feat = add_dividend_features_past_only(
                df_feat,
                div_map.get(str(secid), pd.DataFrame()),
                lag_days=int(CFG.get("DIV_LAG_DAYS", 1)),
            )

        # Merge macro features by date index (ffill/bfill to cover missing macro dates).
        if not macro_df.empty:
            df_feat = df_feat.join(macro_df, how="left")
            for col in macro_cols:
                df_feat[col] = df_feat[col].ffill().bfill().fillna(0.0)
            print(f"  after macro merge: {len(df_feat)} rows, macro cols={len(macro_cols)}")

        min_rows = max(250, int(CFG["SEQ_LEN"]) + int(CFG["HORIZON"]) + 50)
        if len(df_feat) < min_rows:
            print(f"  -> too few rows after features ({len(df_feat)} < {min_rows}), skip")
            continue

        horizons = CFG.get("HORIZONS", [int(CFG["HORIZON"])])
        thr_map = CFG.get("THR_MAP", {int(CFG["HORIZON"]): float(CFG["THR_MOVE"])})

        for h in horizons:
            h = int(h)
            thr = float(thr_map.get(h, float(CFG.get("THR_MOVE", 0.03))))

            y_h, fwd_ret_h = make_target(df_feat["CLOSE"], h, thr)

            df_h = df_feat.iloc[:-h].copy()
            y_h = y_h.iloc[:-h]
            fwd_ret_h = fwd_ret_h.iloc[:-h]

            tmp = df_h.copy()
            tmp["target"] = y_h.values
            tmp["fwd_ret"] = fwd_ret_h.values
            tmp["date"] = tmp.index

            # Horizon as a feature (normalized to [0,1]).
            tmp["horizon_norm"] = h / 360.0
            tmp["horizon_days"] = h
            rows.append(tmp.reset_index(drop=True))

            print(
                f"  {secid} h={h}d thr={thr}: {len(df_h)} rows, pos_rate={float(y_h.mean()):.3%}"
            )

    if not rows:
        raise RuntimeError("No tickers loaded. Check MOEX availability / tickers list.")

    full = pd.concat(rows, axis=0).sort_values(["secid", "date"]).reset_index(drop=True)

    # Ensure sequences/splits don't mix different horizons for the same ticker.
    if "horizon_days" in full.columns:
        full["secid"] = full["secid"].astype(str) + "_h" + full["horizon_days"].astype(str)

    technical_cols = [
        "logret_1",
        "logret_2",
        "logret_3",
        "logret_5",
        "logret_10",
        "trend_up_20",
        "trend_up_50",
        "trend_up_200",
        "vol_rel",
        "vol_spike",
        "rsi_14",
        "rsi_oversold",
        "rsi_overbought",
        "price_pos_20",
        "volatility_20",
    ]
    fundamental_cols = ["div_yield_ttm", "days_since_last_div", "div_yield_is_missing"] if use_div_fund else []
    feature_cols = technical_cols + fundamental_cols + macro_cols + ["horizon_norm"]

    # NOTE: technical features must exist; macro features are optional (filled with zeros
    # if the macro series failed to load).
    missing_tech = [c for c in technical_cols if c not in full.columns]
    if missing_tech:
        raise RuntimeError(
            "Missing expected technical feature columns. "
            f"Expected={len(technical_cols)}, missing={missing_tech}. "
            "Check build_features_one() and upstream data columns."
        )

    for c in macro_cols:
        if c not in full.columns:
            full[c] = 0.0

    X = full[feature_cols].values
    y = full["target"].astype(int).values
    fwd_ret = full["fwd_ret"].astype(float).values
    dates = pd.to_datetime(full["date"]).values
    secids = full["secid"].astype(str).values

    print(
        f"\nMulti-ticker dataset: X={X.shape}, pos_rate={y.mean():.3%}, tickers={sorted(set(secids))}"
    )
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"  Technical: {len(technical_cols)} | Fundamental: {len(fundamental_cols)} | Macro: {len(macro_cols)}")

    if use_div_fund and "div_yield_is_missing" in full.columns:
        print("\n=== FUNDAMENTAL COVERAGE BY TICKER (dividends) ===")
        for sec in sorted(full["secid"].astype(str).unique()):
            m = full["secid"].astype(str) == str(sec)
            coverage = float((full.loc[m, "div_yield_is_missing"] == 0).mean())
            avg_yield = float(full.loc[m & (full["div_yield_is_missing"] == 0), "div_yield_ttm"].mean()) if (m.any()) else float("nan")
            print(f"  {sec}: coverage={coverage:.1%}, avg div_yield_ttm={avg_yield:.4f}")
    return (
        MultiDataset(X=X, y=y, fwd_ret=fwd_ret, dates=dates, secids=secids),
        feature_cols,
    )
