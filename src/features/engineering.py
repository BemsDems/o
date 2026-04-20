from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_dividend_past_only_features(df: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    """Non-leaky dividend features based only on already known past dividend events."""
    out = df.copy().sort_index()

    tmp = out.reset_index().rename(columns={out.index.name or out.reset_index().columns[0]: "date"})
    tmp["date"] = pd.to_datetime(tmp["date"])

    # defaults
    tmp["last_dividend"] = 0.0
    tmp["days_since_last_dividend"] = 9999
    tmp["last_div_yield_approx"] = 0.0
    tmp["div_paid_recent_30d"] = 0
    tmp["div_growth_last"] = 0.0
    tmp["div_sum_365d"] = 0.0

    if div is None or div.empty:
        return tmp.set_index("date")

    div2 = div.copy()
    div2["date"] = pd.to_datetime(div2["date"], errors="coerce")
    div2["dividend_rub"] = pd.to_numeric(div2["dividend_rub"], errors="coerce")
    div2 = div2.dropna(subset=["date", "dividend_rub"]).sort_values("date").reset_index(drop=True)

    if div2.empty:
        return tmp.set_index("date")

    # growth of last dividend vs previous one
    div2["prev_dividend"] = div2["dividend_rub"].shift(1)
    div2["div_growth_last_tmp"] = div2["dividend_rub"] / div2["prev_dividend"].replace(0, np.nan) - 1.0
    div2["div_growth_last_tmp"] = (
        div2["div_growth_last_tmp"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # merge last known dividend EVENT date, not just propagated dividend value
    last_known = (
        div2.rename(columns={"date": "last_div_date"})
        [["last_div_date", "dividend_rub", "div_growth_last_tmp"]]
        .sort_values("last_div_date")
        .reset_index(drop=True)
    )

    tmp = pd.merge_asof(
        tmp.sort_values("date"),
        last_known,
        left_on="date",
        right_on="last_div_date",
        direction="backward",
    )

    tmp["last_dividend"] = tmp["dividend_rub"].fillna(0.0)

    tmp["days_since_last_dividend"] = (
        (tmp["date"] - tmp["last_div_date"]).dt.days.fillna(9999).astype(int)
    )

    tmp["div_paid_recent_30d"] = (tmp["days_since_last_dividend"] <= 30).astype(int)

    tmp["last_div_yield_approx"] = (
        tmp["last_dividend"] / tmp["Close"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tmp["div_growth_last"] = tmp["div_growth_last_tmp"].fillna(0.0)

    # sum of dividends paid over the last 365 calendar days (event-based)
    event_days = div2["date"].values.astype("datetime64[D]").astype(np.int64)
    event_vals = div2["dividend_rub"].values.astype(float)
    csum = np.cumsum(event_vals)

    query_days = tmp["date"].values.astype("datetime64[D]").astype(np.int64)
    right_idx = np.searchsorted(event_days, query_days, side="right") - 1
    left_idx = np.searchsorted(event_days, query_days - 365, side="right") - 1

    right_sum = np.where(right_idx >= 0, csum[right_idx], 0.0)
    left_sum = np.where(left_idx >= 0, csum[left_idx], 0.0)
    tmp["div_sum_365d"] = right_sum - left_sum

    tmp = tmp.drop(columns=["dividend_rub", "div_growth_last_tmp", "last_div_date"], errors="ignore")
    return tmp.set_index("date")


def build_features(
    stock: pd.DataFrame,
    usd: pd.DataFrame,
    imo: pd.DataFrame,
    key_rate: pd.DataFrame,
    divs: pd.DataFrame,
) -> pd.DataFrame:
    df = stock.copy()

    # returns
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_2"] = df["Close"].pct_change(2)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)
    df["ret_20"] = df["Close"].pct_change(20)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # SMA distances
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    sma200 = df["Close"].rolling(200).mean()
    df["dist_sma20"] = (df["Close"] - sma20) / (sma20 + 1e-12)
    df["dist_sma50"] = (df["Close"] - sma50) / (sma50 + 1e-12)
    df["trend_up_200"] = (df["Close"] > sma200).astype(int)

    # RSI
    df["rsi_14"] = rsi(df["Close"], 14)

    # Volatility
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()
    df["vol_rel"] = df["vol_20"] / (df["vol_60"] + 1e-12)

    # ATR-like relative range
    df["hl_range"] = (df["High"] - df["Low"]) / (df["Close"].shift(1) + 1e-12)
    df["atr_rel"] = df["hl_range"].rolling(14).mean()

    # Bollinger band diagnostics
    mid = df["Close"].rolling(20).mean()
    sd = df["Close"].rolling(20).std()
    upper = mid + 2 * sd
    lower = mid - 2 * sd
    df["bb_width"] = (upper - lower) / (mid + 1e-12)
    df["bb_pos"] = (df["Close"] - lower) / ((upper - lower) + 1e-12)

    # Volume features
    v20 = df["Volume"].rolling(20).mean()
    df["vol_ratio_5_20"] = df["Volume"].rolling(5).mean() / (v20 + 1e-12)
    df["vol_spike"] = (df["Volume"] > (v20 + 2 * df["Volume"].rolling(20).std())).astype(int)

    # Market / FX context (align to stock calendar first)
    usd_close = usd["Close"].reindex(df.index).ffill()
    imo_close = imo["Close"].reindex(df.index).ffill()

    df["usd_ret_1"] = usd_close.pct_change(1)
    df["usd_ret_5"] = usd_close.pct_change(5)

    df["imoex_ret_1"] = imo_close.pct_change(1)
    df["imoex_ret_5"] = imo_close.pct_change(5)
    df["imoex_ret_20"] = imo_close.pct_change(20)

    df["stock_vs_imoex_5"] = df["ret_5"] - df["imoex_ret_5"]

    # Macro: key rate derivatives only
    if key_rate is None or key_rate.empty:
        df["key_rate"] = np.nan
    else:
        kr = key_rate.copy()
        kr["date"] = pd.to_datetime(kr["date"], errors="coerce")
        kr = kr.dropna().drop_duplicates(subset=["date"]).sort_values("date")
        kr = kr.set_index("date").sort_index()
        kr_daily = kr.reindex(df.index, method="ffill")
        df["key_rate"] = kr_daily["key_rate"]

    df["key_rate_chg"] = df["key_rate"].diff()
    df["rate_rising"] = (df["key_rate_chg"] > 0).astype(int)

    # Dividends (optional, past-only)
    if divs is not None and not divs.empty:
        df = add_dividend_past_only_features(df, divs)
    else:
        df["last_dividend"] = 0.0
        df["days_since_last_dividend"] = 9999
        df["last_div_yield_approx"] = 0.0

    # Soft dividend transforms (more stable than raw sparse dividend features)
    df["days_since_last_dividend_capped"] = df["days_since_last_dividend"].clip(0, 365)
    df["div_decay_90"] = np.exp(-df["days_since_last_dividend_capped"] / 90.0)

    df = df.dropna().copy()
    return df

