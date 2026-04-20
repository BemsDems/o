
"""MOEX TCN baseline (ChatGPT variant).

Colab-ready: MOEX/CBR feature engineering + TCN classifier (single-stock baseline)

Current practical baseline:
- single Russian stock
- binary target
- TCN
- technical + market context + key rate
- no fundamentals in final working version
- dividends disabled in baseline (configurable)
"""
#
# In Colab:
# !pip -q install moexalgo requests pandas numpy scikit-learn tensorflow lxml html5lib keras-tcn

from __future__ import annotations

import os
import gc
import re
import time
from dataclasses import dataclass
import pickle
import random
from io import StringIO
from typing import Optional, Tuple, Dict, Any, Iterable

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from tcn import TCN
from moexalgo import Ticker

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# CONFIG
# ----------------------------
CFG = {
    "TICKER": "GAZP",
    "START": "2015-01-01",
    "END": "2025-12-31",

    "HORIZON": 5,        # target horizon in trading days
    "THR_MOVE": 0.015,   # target: future_return > 1.5%

    "WINDOW": 30,

    # time split (by dates order)
    "TRAIN_FRAC": 0.70,
    "VAL_FRAC": 0.15,  # rest goes to test

    # training
    "EPOCHS": 200,
    "BATCH": 32,
    "LR": 1e-3,
    "PATIENCE": 15,

    # backtest assumptions (simple)
    "FEE": 0.001,        # 0.1% per trade (simplified)
    "NON_OVERLAP": True, # skip next HORIZON days after entry

    # Multi-seed evaluation
    "RUN_SEEDS": [11, 21, 31, 41, 51],

    # Feature clipping (robustness against outliers)
    "CLIP_Q": 0.005,

    # Current practical baseline: no dividends
    "USE_DIVIDENDS": False,

    # Save model/scaler only for single-seed runs
    "SAVE_SINGLE_RUN_ARTIFACTS": False,

}


BASE_FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "log_ret",
    "dist_sma20", "dist_sma50", "trend_up_200", "rsi_14",
    "vol_rel", "bb_width", "bb_pos", "vol_ratio_5_20", "vol_spike",
    "imoex_ret_1", "imoex_ret_5", "imoex_ret_20",
    "stock_vs_imoex_5",
    "key_rate_chg", "rate_rising",
]

# ----------------------------
# RUN FLAGS (to reduce output)
# ----------------------------
SHOW_MODEL_SUMMARY = False
SHOW_TRAIN_VAL_DIAG = False
FIT_VERBOSE = 0




def set_global_seed(seed: int):
    # Best-effort reproducibility across Python/NumPy/TensorFlow.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# HOW TO USE (Fundamentals scaffold)
#
# 1) Quick start via Smart-Lab fallback (no e-disclosure token needed):
#
#   ticker = "SBER"
#
#   smartlab_msfo = fetch_smartlab_financials(ticker=ticker, report_type="MSFO", freq="q")
#   smartlab_rsbu = fetch_smartlab_financials(ticker=ticker, report_type="RSBU", freq="q")
#
#   fund = combine_fundamental_sources(
#       normalize_fundamentals(smartlab_msfo),
#       normalize_fundamentals(smartlab_rsbu),
#   )
#
#   print(fund.head())
#   print(fund.tail())
#
# 2) If you have an e-disclosure token:
#
#   client = EDisclosureClient(
#       token=CFG_FUND.edisclosure_token,
#       base_url=CFG_FUND.edisclosure_base,
#       timeout=CFG_FUND.request_timeout,
#       user_agent=CFG_FUND.user_agent,
#   )
#
#   # disclosures = client.search_disclosures("SBER", "2019-01-01", "2025-12-31")
#   # print(disclosures.head())
#
# 3) How to plug into the current pipeline (no leakage):
#    After candles load and BEFORE add_target(...):
#
#   fund_msfo = fetch_smartlab_financials(CFG["TICKER"], report_type="MSFO", freq="q")
#   fund_rsbu = fetch_smartlab_financials(CFG["TICKER"], report_type="RSBU", freq="q")
#   fund = combine_fundamental_sources(
#       normalize_fundamentals(fund_msfo),
#       normalize_fundamentals(fund_rsbu),
#   )
#
#   feat = build_features(sber, usd, imo, key_rate, divs)
#   feat = add_fundamental_features_past_only(feat, fund, ticker=CFG["TICKER"], lag_days=1)
#   feat = add_target(feat, CFG["HORIZON"], CFG["THR_MOVE"])
#
#   FUND_FEATURES = [
#       "revenue",
#       "net_income",
#       "eps",
#       "roe",
#       "pb_ratio",
#       "net_margin",
#       "value_quality",
#   ]
#   FEATURES = BASE_FEATURES + [c for c in FUND_FEATURES if c in feat.columns]
#
# ============================
# FUNDAMENTALS (diploma scaffold)
# ============================


@dataclass
class FundamentalConfig:
    edisclosure_token: Optional[str] = os.getenv("EDISCLOSURE_TOKEN")
    edisclosure_base: str = "https://gateway.e-disclosure.ru/api"
    request_timeout: int = 30
    sleep_sec: float = 0.4
    user_agent: str = "Mozilla/5.0 (compatible; DiplomaResearchBot/1.0)"


CFG_FUND = FundamentalConfig()


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    )
    return s


class EDisclosureClient:
    """API-first client for e-disclosure.

    IMPORTANT:
    - Exact endpoint paths and params must be taken from your Swagger UI / account.
    - This is a scaffold: plug in real `path` + response mapping.
    """

    def __init__(
        self,
        token: str,
        base_url: str,
        timeout: int = 30,
        user_agent: str = "Mozilla/5.0",
    ):
        if not token:
            raise ValueError("EDISCLOSURE_TOKEN не задан.")
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.s = make_session(user_agent)
        self.s.headers.update({"Authorization": f"Bearer {token}"})

    def _get(self, path: str, params: Optional[dict] = None):
        url = f"{self.base_url}/{path.lstrip('/')}"
        r = self.s.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "json" in ctype.lower():
            return r.json()
        return r.text

    def search_disclosures(
        self,
        ticker: str,
        date_from: str,
        date_to: str,
        limit: int = 200,
    ) -> pd.DataFrame:
        """Search disclosures for given ticker in [date_from, date_to].

        TODO:
        1) Open Swagger UI e-disclosure
        2) Find correct endpoint for searching disclosure events/messages
        3) Replace `path` and `params` below

        Expected output columns:
          publish_date, title, category, issuer_name, ticker, message_id, message_url, attachment_url
        """

        # ===== PLUG FROM SWAGGER =====
        path = "/TODO/search/disclosures"
        params = {
            "ticker": ticker,
            "dateFrom": date_from,
            "dateTo": date_to,
            "limit": int(limit),
        }
        data = self._get(path, params=params)

        # ===== MAP YOUR RESPONSE HERE =====
        rows = []
        items = data.get("items", []) if isinstance(data, dict) else []
        for x in items:
            rows.append(
                {
                    "ticker": ticker,
                    "publish_date": x.get("publishDate"),
                    "title": x.get("title"),
                    "category": x.get("category"),
                    "issuer_name": x.get("issuerName"),
                    "message_id": x.get("id"),
                    "message_url": x.get("url"),
                    "attachment_url": x.get("attachmentUrl"),
                    "source": "e-disclosure",
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
            df = df.sort_values("publish_date").reset_index(drop=True)
        return df


def _to_float_ru(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", "-", "—", "–", "nan", "None"}:
        return np.nan
    s = s.replace(" ", "")
    s = s.replace("%", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-]", "", s)
    if s in {"", "-", "."}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def _smartlab_url(ticker: str, report_type: str = "MSFO", freq: str = "q") -> str:
    # examples:
    # https://smart-lab.ru/q/MAGN/f/q/MSFO/
    # https://smart-lab.ru/q/RTKM/f/q/RSBU/
    return f"https://smart-lab.ru/q/{ticker}/f/{freq}/{report_type}/"


def fetch_smartlab_financials(
    ticker: str,
    report_type: str = "MSFO",
    freq: str = "q",
    timeout: int = 30,
    user_agent: str = "Mozilla/5.0",
) -> pd.DataFrame:
    url = _smartlab_url(ticker, report_type=report_type, freq=freq)
    s = make_session(user_agent)
    html = s.get(url, timeout=timeout).text

    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame()

    # pick the widest table
    tbl = max(tables, key=lambda x: x.shape[1]).copy()
    tbl.columns = [str(c).strip() for c in tbl.columns]

    metric_col = tbl.columns[0]
    tbl[metric_col] = tbl[metric_col].astype(str).str.strip()

    value_cols = [c for c in tbl.columns[1:] if str(c).strip()]
    long_rows = []
    for _, row in tbl.iterrows():
        metric = str(row[metric_col]).strip()
        for col in value_cols:
            long_rows.append(
                {
                    "metric_ru": metric,
                    "period": str(col).strip(),
                    "value_raw": row[col],
                    "ticker": ticker,
                    "source": "smart-lab",
                    "report_type": report_type,
                }
            )

    df = pd.DataFrame(long_rows)
    if df.empty:
        return df

    # publish date row (if present)
    date_mask = df["metric_ru"].str.contains("Дата отчета", case=False, na=False)
    date_map = df[date_mask][["period", "value_raw"]].rename(columns={"value_raw": "publish_date"}).copy()
    if not date_map.empty:
        date_map["publish_date"] = pd.to_datetime(date_map["publish_date"], dayfirst=True, errors="coerce")

    df = df[~date_mask].copy()

    metric_map = {
        "Выручка": "revenue",
        "EBITDA": "ebitda",
        "Чистая прибыль": "net_income",
        "Чистая прибыль н/с": "net_income",
        "EPS": "eps",
        "ROE": "roe",
        "P/B": "pb_ratio",
        "P/BV": "pb_ratio",
        "Чистая рентаб": "net_margin",
        "Чистая маржа": "net_margin",
        "Долг/EBITDA": "debt_ebitda",
        "Долг": "debt",
        "Чистый долг": "net_debt",
    }

    def map_metric(x: str) -> Optional[str]:
        for k, v in metric_map.items():
            if x.startswith(k):
                return v
        return None

    df["metric"] = df["metric_ru"].map(map_metric)
    df = df[df["metric"].notna()].copy()
    df["value"] = df["value_raw"].map(_to_float_ru)

    if not date_map.empty:
        df = df.merge(date_map, on="period", how="left")
    else:
        df["publish_date"] = pd.NaT

    out = (
        df.pivot_table(
            index=["ticker", "period", "publish_date", "source", "report_type"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    out.columns.name = None
    out = out.sort_values(["publish_date", "period"]).reset_index(drop=True)
    return out


def normalize_fundamentals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "period",
                "publish_date",
                "source",
                "revenue",
                "ebitda",
                "net_income",
                "eps",
                "roe",
                "pb_ratio",
                "net_margin",
            ]
        )

    out = df.copy()
    out["publish_date"] = pd.to_datetime(out["publish_date"], errors="coerce")
    out = out.dropna(subset=["publish_date"]).sort_values("publish_date").reset_index(drop=True)

    for c in ["revenue", "ebitda", "net_income", "eps", "roe", "pb_ratio", "net_margin"]:
        if c not in out.columns:
            out[c] = np.nan

    return out[
        [
            "ticker",
            "period",
            "publish_date",
            "source",
            "revenue",
            "ebitda",
            "net_income",
            "eps",
            "roe",
            "pb_ratio",
            "net_margin",
        ]
    ]


def combine_fundamental_sources(*dfs: pd.DataFrame) -> pd.DataFrame:
    parts = [x.copy() for x in dfs if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["publish_date"] = pd.to_datetime(out["publish_date"], errors="coerce")
    out = out.dropna(subset=["publish_date"]).sort_values("publish_date").reset_index(drop=True)

    source_priority = {"e-disclosure": 0, "smart-lab": 1, "ir": 2}
    out["_prio"] = out["source"].map(source_priority).fillna(99)
    out = out.sort_values(["ticker", "period", "_prio", "publish_date"])
    out = out.drop_duplicates(subset=["ticker", "period"], keep="first").drop(columns="_prio")

    return out.reset_index(drop=True)


def add_fundamental_features_past_only(
    price_df: pd.DataFrame,
    fund_df: pd.DataFrame,
    ticker: str,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Attach fundamentals to daily candles WITHOUT leakage.

    Merge logic:
    - fundamentals become effective only after publish_date (+ optional lag_days)
    - merge_asof backward on effective_date.
    """

    out = price_df.copy().sort_index()
    out = out.reset_index().rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"])

    if fund_df is None or fund_df.empty:
        return out.set_index("date")

    f = fund_df[fund_df["ticker"] == ticker].copy()
    if f.empty:
        return out.set_index("date")

    f["publish_date"] = pd.to_datetime(f["publish_date"], errors="coerce")
    f = f.dropna(subset=["publish_date"]).sort_values("publish_date").reset_index(drop=True)

    if lag_days:
        f["effective_date"] = f["publish_date"] + pd.Timedelta(days=int(lag_days))
    else:
        f["effective_date"] = f["publish_date"]

    cols_keep = [
        "effective_date",
        "revenue",
        "ebitda",
        "net_income",
        "eps",
        "roe",
        "pb_ratio",
        "net_margin",
    ]
    cols_keep = [c for c in cols_keep if c in f.columns]

    out = pd.merge_asof(
        out.sort_values("date"),
        f[cols_keep].sort_values("effective_date"),
        left_on="date",
        right_on="effective_date",
        direction="backward",
    )

    out = out.drop(columns=["effective_date"], errors="ignore")

    # Simple derived features
    if "roe" in out.columns and "pb_ratio" in out.columns:
        out["value_quality"] = out["roe"] / out["pb_ratio"].replace(0, np.nan)

    if "net_income" in out.columns and "revenue" in out.columns:
        out["net_margin_calc"] = out["net_income"] / out["revenue"].replace(0, np.nan)

    return out.set_index("date")


# ----------------------------
# 1) DATA LOADERS (Russian sources)
# ----------------------------
def load_candles_moexalgo(ticker: str, start: str, end: str, period: str = "1D") -> pd.DataFrame:
    df = pd.DataFrame(Ticker(ticker).candles(start=start, end=end, period=period))
    if df.empty:
        return df
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.drop_duplicates(subset=["begin"]).set_index("begin").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def cbr_key_rate_range(start: str, end: str) -> pd.DataFrame:
    """CBR key rate history within a range. Returns daily table (date, key_rate)."""
    url = "https://www.cbr.ru/hd_base/KeyRate/"
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": pd.to_datetime(start).strftime("%d.%m.%Y"),
        "UniDbQuery.To": pd.to_datetime(end).strftime("%d.%m.%Y"),
    }

    html = requests.get(url, params=params, timeout=30).text
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame(columns=["date", "key_rate"])

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    # identify date/rate columns (ru)
    date_col = None
    rate_col = None
    for c in df.columns:
        lc = c.lower()
        if date_col is None and "дата" in lc:
            date_col = c
        if rate_col is None and ("став" in lc or "ключ" in lc):
            rate_col = c

    if date_col is None or rate_col is None:
        return pd.DataFrame(columns=["date", "key_rate"])

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[rate_col] = (
        df[rate_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna().sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date", rate_col: "key_rate"})

    # fix if parsed as 1500 instead of 15.00
    if not df.empty and df["key_rate"].max() > 100:
        df["key_rate"] = df["key_rate"] / 100.0

    return df


def moex_iss_dividends(ticker: str) -> pd.DataFrame:
    """Best-effort dividends via MOEX ISS."""
    url = f"https://iss.moex.com/iss/securities/{ticker}/dividends.json"
    j = requests.get(url, params={"iss.meta": "off"}, timeout=30).json()

    div = j.get("dividends", {})
    if not div or not div.get("data"):
        return pd.DataFrame(columns=["date", "dividend_rub", "currency"])

    df = pd.DataFrame(div["data"], columns=div["columns"])
    if "registryclosedate" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "dividend_rub", "currency"])

    out = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(df["registryclosedate"], errors="coerce"),
                "dividend_rub": pd.to_numeric(df["value"], errors="coerce"),
                "currency": df.get("currencyid", "RUB"),
            }
        )
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


# ----------------------------
# 2) FEATURE ENGINEERING
# ----------------------------
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
        div2["div_growth_last_tmp"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
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
    sber: pd.DataFrame,
    usd: pd.DataFrame,
    imo: pd.DataFrame,
    key_rate: pd.DataFrame,
    divs: pd.DataFrame,
) -> pd.DataFrame:
    df = sber.copy()

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
    v60 = df["Volume"].rolling(60).mean()
    df["vol_ratio_5_20"] = df["Volume"].rolling(5).mean() / (v20 + 1e-12)
    df["vol_spike"] = (df["Volume"] > (v20 + 2 * df["Volume"].rolling(20).std())).astype(int)
    # Market / FX context (align to SBER calendar first)
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

    # Dividends (optional features; may be excluded from FEATURES)
    if divs is not None and not divs.empty:
        df = add_dividend_past_only_features(df, divs)
    else:
        df["last_dividend"] = 0.0
        df["days_since_last_dividend"] = 9999
        df["last_div_yield_approx"] = 0.0

    

    # Soft dividend transforms (more stable than raw sparse dividend features)
    # NOTE: uses only past-known information (already past-only merged in add_dividend_past_only_features).
    df["days_since_last_dividend_capped"] = df["days_since_last_dividend"].clip(0, 365)
    df["div_decay_90"] = np.exp(-df["days_since_last_dividend_capped"] / 90.0)
    df = df.dropna().copy()
    return df


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


def make_windows_aligned(X_2d: np.ndarray, y_1d: np.ndarray, dates_1d: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make windows and align y/dates to the LAST index of each window."""
    Xw, yw, dw = [], [], []
    for i in range(window - 1, len(X_2d)):
        Xw.append(X_2d[i - window + 1 : i + 1])
        yw.append(y_1d[i])
        dw.append(dates_1d[i])
    return np.asarray(Xw), np.asarray(yw), np.asarray(dw)


def build_tcn_model(window: int, n_features: int, lr: float) -> tf.keras.Model:
    """
    TCN‑архитектура как замена LSTM для задачи классификации.
    """
    inp = tf.keras.Input(shape=(window, n_features))

    # TCN блок с несколькими уровнями расширенной (dilated) свертки
    x = TCN(
        nb_filters=32,  # число фильтров; можно снизить до 32 для уменьшения переобучения
        kernel_size=3,  # размер окна свертки
        nb_stacks=1,  # 1 стек => 1 “слой” TCN
        dilations=(1, 2, 4, 8),  # расширенная (dilated) свертка
        padding="causal",  # только прошлые данные, без lookahead
        use_skip_connections=True,  # residual‑skip, как в оригинальной TCN
        dropout_rate=0.3,  # dropout внутри TCN‑блока
        return_sequences=False,  # нужен только последний шаг (для классификации)
        activation="relu",
        kernel_initializer="he_normal",
    )(inp)

    # Post‑TCN блок (аналог Dense‑слоя после LSTM)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )
    return model


def non_overlap_pnl(pred: np.ndarray, future_ret: np.ndarray, horizon: int, fee: float) -> Tuple[float, int]:
    """Average return per trade, non-overlapping trades (simple)."""
    i = 0
    trades = []
    while i < len(pred):
        if pred[i] == 1:
            trades.append(float(future_ret[i] - fee))
            i += horizon
        else:
            i += 1
    if not trades:
        return 0.0, 0
    return float(np.mean(trades)), int(len(trades))


def pick_threshold_on_val(
    y_true_val: np.ndarray,
    prob_val: np.ndarray,
    future_ret_val: np.ndarray,
    horizon: int,
    fee: float,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)

    rows = []
    for thr in thresholds:
        pred = (prob_val >= thr).astype(int)

        f1_1 = f1_score(y_true_val, pred, pos_label=1, zero_division=0)
        bal = balanced_accuracy_score(y_true_val, pred)
        mcc = matthews_corrcoef(y_true_val, pred) if len(np.unique(pred)) > 1 else 0.0

        avg_pnl, n_tr = non_overlap_pnl(pred, future_ret_val, horizon, fee)

        MIN_TRADES = 15
        MIN_BUY_RATE = 0.05
        MAX_BUY_RATE = 0.35
        feasible = (n_tr >= MIN_TRADES) and (MIN_BUY_RATE <= pred.mean() <= MAX_BUY_RATE)
        if not feasible:
            avg_pnl = -1e9

        rows.append(
            {
                "thr": float(thr),
                "f1_class1": float(f1_1),
                "balanced_acc": float(bal),
                "mcc": float(mcc),
                "share_buy": float(pred.mean()),
                "avg_trade_ret_nonoverlap": float(avg_pnl),
                "n_trades_nonoverlap": int(n_tr),
            }
        )

    tab = pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)

    # feasible = not those where we forced -1e9
    tab_feas = tab[tab["avg_trade_ret_nonoverlap"] > -1e8].copy()

    if tab_feas.empty:
        # if no threshold passed constraints, fall back to 0.50
        thr_f1 = 0.50
        thr_pnl = 0.50
    else:
        thr_f1 = float(tab_feas.iloc[tab_feas["f1_class1"].values.argmax()]["thr"])
        thr_pnl = float(tab_feas.iloc[tab_feas["avg_trade_ret_nonoverlap"].values.argmax()]["thr"])

    return float(thr_f1), float(thr_pnl), tab


def eval_block(name: str, y_true: np.ndarray, prob: np.ndarray, thr: float):
    pred = (prob >= thr).astype(int)
    print("\n" + "=" * 70)
    print(f"{name} | threshold={thr:.2f}")
    print("=" * 70)

    print(f"Accuracy: {accuracy_score(y_true, pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, pred):.3f}")
    print(f"F1 macro: {f1_score(y_true, pred, average='macro', zero_division=0):.3f}")
    print(f"F1 (BUY=1): {f1_score(y_true, pred, pos_label=1, zero_division=0):.3f}")
    print(f"MCC: {(matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0):.3f}")

    if len(np.unique(y_true)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_true, prob):.3f}")
        print(f"PR-AUC: {average_precision_score(y_true, prob):.3f}")

    cm = confusion_matrix(y_true, pred)
    print("\nConfusion matrix:")
    print(f"TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"FN={cm[1,0]}, TP={cm[1,1]}")

    print("\nClassification report:")
    print(classification_report(y_true, pred, zero_division=0, target_names=["No Growth", "Growth>thr"]))

    print("\nPred distribution:")
    print(f"Pred HOLD=0: {(pred==0).mean():.1%} | Pred BUY=1: {(pred==1).mean():.1%}")

    base0 = np.zeros_like(y_true)
    base1 = np.ones_like(y_true)
    print("\nBaselines:")
    print(f"Always HOLD (0) accuracy: {accuracy_score(y_true, base0):.3f}")
    print(f"Always BUY  (1) accuracy: {accuracy_score(y_true, base1):.3f}")


# ----------------------------
# EXTRA DIAGNOSTICS (probabilities, calibration, drift)
# ----------------------------

def ece_score(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    y_true = y_true.astype(int)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(prob[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def prob_summary(name: str, y_true: np.ndarray, prob: np.ndarray) -> None:
    y_true = y_true.astype(int)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    pos = float(y_true.mean())
    print("\n" + "=" * 70)
    print(f"PROB SUMMARY: {name}")
    print("=" * 70)
    print(f"Samples: {len(y_true)} | Pos rate (BUY=1): {pos:.3f}")
    print(
        f"Prob mean={prob.mean():.3f} std={prob.std():.3f} "
        f"p05={np.quantile(prob,0.05):.3f} p50={np.quantile(prob,0.50):.3f} p95={np.quantile(prob,0.95):.3f}"
    )

    # ranking
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, prob))
        auc_inv = float(roc_auc_score(y_true, 1 - prob))
        ap = float(average_precision_score(y_true, prob))
        print(f"ROC-AUC(prob): {auc:.3f}")
        print(f"ROC-AUC(1-prob): {auc_inv:.3f} (если > ROC-AUC(prob), сигнал мог 'перевернуться')")
        print(f"PR-AUC(AP): {ap:.3f}")

    # probability quality
    ll = float(log_loss(y_true, prob))
    bs = float(brier_score_loss(y_true, prob))
    ece = float(ece_score(y_true, prob, n_bins=10))
    print(f"LogLoss: {ll:.3f} | Brier: {bs:.3f} | ECE(10 bins): {ece:.3f}")

    # baseline logloss from constant probability = pos_rate
    base_prob = np.full_like(prob, fill_value=pos, dtype=float)
    base_ll = float(log_loss(y_true, np.clip(base_prob, 1e-6, 1 - 1e-6)))
    print(f"Baseline LogLoss (const p=pos_rate): {base_ll:.3f}")


def history_summary(history) -> dict:
    h = pd.DataFrame(getattr(history, "history", {}) or {})
    out = {
        "epochs_run": int(len(h)),
        "best_epoch": np.nan,
        "best_val_auc_pr": np.nan,
        "best_val_auc_roc": np.nan,
        "best_val_loss": np.nan,
    }
    if h.empty:
        return out

    if "val_auc_pr" in h.columns:
        out["best_epoch"] = int(h["val_auc_pr"].idxmax()) + 1
        out["best_val_auc_pr"] = float(h["val_auc_pr"].max())
    elif "val_loss" in h.columns:
        out["best_epoch"] = int(h["val_loss"].idxmin()) + 1

    if "val_auc_roc" in h.columns:
        out["best_val_auc_roc"] = float(h["val_auc_roc"].max())
    if "val_loss" in h.columns:
        out["best_val_loss"] = float(h["val_loss"].min())

    return out


def compact_prob_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 1e-6, 1 - 1e-6)

    pos_rate = float(y_true.mean())
    out = {
        "pos_rate": pos_rate,
        "prob_mean": float(prob.mean()),
        "prob_std": float(prob.std()),
        "logloss": float(log_loss(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece10": float(ece_score(y_true, prob, n_bins=10)),
    }

    base_prob = np.full_like(prob, fill_value=pos_rate, dtype=float)
    out["baseline_logloss"] = float(log_loss(y_true, np.clip(base_prob, 1e-6, 1 - 1e-6)))
    out["logloss_gain_vs_baseline"] = float(out["baseline_logloss"] - out["logloss"])

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["roc_auc_inv"] = float(roc_auc_score(y_true, 1 - prob))
        out["pr_auc"] = float(average_precision_score(y_true, prob))
        out["pr_auc_lift"] = float(out["pr_auc"] - pos_rate)
    else:
        out["roc_auc"] = float("nan")
        out["roc_auc_inv"] = float("nan")
        out["pr_auc"] = float("nan")
        out["pr_auc_lift"] = float("nan")

    return out


def make_decile_table(
    y_true: np.ndarray,
    prob: np.ndarray,
    future_ret: np.ndarray | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "p": prob.astype(float)})
    if future_ret is not None:
        df["fret"] = future_ret.astype(float)

    df["decile"] = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        buy_rate=("y", "mean"),
    )
    if future_ret is not None:
        g["avg_future_ret"] = df.groupby("decile")["fret"].mean()

    return g.reset_index()


def backtest_nonoverlap_long_only_stats(prob, dates_signal, close_full, thr, horizon, fee) -> dict:
    dates = pd.to_datetime(dates_signal)
    close_full = close_full.copy()
    close_full.index = pd.to_datetime(close_full.index)

    start_date = dates[0]
    last_date = dates[-1]
    try:
        last_loc = close_full.index.get_loc(last_date)
        end_loc = min(last_loc + horizon, len(close_full.index) - 1)
        end_date = close_full.index[end_loc]
    except KeyError:
        end_date = close_full.index[-1]

    bh_ret = float(close_full.loc[end_date] / close_full.loc[start_date] - 1.0)

    eq = 1.0
    trades = []
    i = 0
    while i < len(prob):
        if prob[i] >= thr:
            d0 = dates[i]
            try:
                loc0 = close_full.index.get_loc(d0)
            except KeyError:
                i += 1
                continue

            loc1 = loc0 + horizon
            if loc1 >= len(close_full.index):
                break

            entry = float(close_full.iloc[loc0])
            exitp = float(close_full.iloc[loc1])
            ret = exitp / entry - 1.0 - fee

            eq *= (1.0 + ret)
            trades.append(ret)
            i += horizon
        else:
            i += 1

    strat = float(eq - 1.0)
    alpha = float(strat - bh_ret)

    return {
        "strategy_return": strat,
        "buyhold_return": bh_ret,
        "alpha": alpha,
        "n_trades": int(len(trades)),
        "winrate": float((sum(1 for x in trades if x > 0) / len(trades)) if trades else 0.0),
        "avg_trade_ret": float((sum(trades) / len(trades)) if trades else 0.0),
        "median_trade_ret": float(np.median(trades) if trades else 0.0),
    }


def threshold_sweep(name: str, y_true: np.ndarray, prob: np.ndarray, thresholds=None, top_k: int = 10) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)

    rows = []
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        rows.append({
            "thr": float(thr),
            "buy_rate": float(pred.mean()),
            "acc": float(accuracy_score(y_true, pred)),
            "bal_acc": float(balanced_accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "f1_buy": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
            "prec_buy": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
            "rec_buy": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0),
        })

    tab = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"THRESHOLD SWEEP: {name} (top by F1-macro)")
    print("=" * 70)
    print(tab.sort_values("f1_macro", ascending=False).head(top_k).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"THRESHOLD SWEEP: {name} (top by MCC)")
    print("=" * 70)
    print(tab.sort_values("mcc", ascending=False).head(top_k).to_string(index=False))

    return tab


def decile_report(name: str, y_true: np.ndarray, prob: np.ndarray, future_ret: Optional[np.ndarray] = None) -> None:
    """Check: higher prob => higher BUY rate and (optionally) higher future return."""
    df = pd.DataFrame({"y": y_true.astype(int), "p": prob.astype(float)})
    if future_ret is not None:
        df["fret"] = future_ret.astype(float)

    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        buy_rate=("y", "mean"),
    )
    if future_ret is not None:
        g["avg_future_ret"] = df.groupby("decile")["fret"].mean()

    print("\n" + "=" * 70)
    print(f"DECILE REPORT: {name}")
    print("=" * 70)
    print(g.reset_index().to_string(index=False))


def psi_1d(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index for one feature (train defines bins)."""
    train = train[~np.isnan(train)]
    test = test[~np.isnan(test)]
    if len(train) < 50 or len(test) < 50:
        return float('nan')

    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(train, qs))
    if len(bins) <= 2:
        return float('nan')

    tr_hist, _ = np.histogram(train, bins=bins)
    te_hist, _ = np.histogram(test, bins=bins)

    tr = tr_hist / max(tr_hist.sum(), 1)
    te = te_hist / max(te_hist.sum(), 1)
    tr = np.clip(tr, 1e-6, None)
    te = np.clip(te, 1e-6, None)

    return float(np.sum((te - tr) * np.log(te / tr)))


def drift_report_features(X_train_2d: np.ndarray, X_test_2d: np.ndarray, feature_names: list, top_k: int = 10) -> None:
    """Feature drift report using PSI(train->test)."""
    psis = []
    for j, name in enumerate(feature_names):
        p = psi_1d(X_train_2d[:, j], X_test_2d[:, j], n_bins=10)
        if p is None or np.isnan(p):
            continue
        psis.append((str(name), float(p)))
    psis.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("FEATURE DRIFT (PSI train→test): top")
    print("=" * 70)
    for n, p in psis[:top_k]:
        flag = " !!!" if p >= 0.25 else (" !!" if p >= 0.10 else "")
        print(f"{n:25s} PSI={p:.3f}{flag}")

    # PSI interpretation:
    # <0.10 — ok
    # 0.10–0.25 — moderate drift
    # >0.25 — strong drift


# ----------------------------
# 5) PIPELINE
# ----------------------------

def alpha_nonoverlap(prob: np.ndarray, dates_signal: np.ndarray, close_full: pd.Series, thr: float, horizon: int, fee: float) -> float:
    """Compute alpha (strategy - buy&hold) for non-overlap long-only rule."""
    out = alpha_nonoverlap_stats(prob, dates_signal, close_full, thr, horizon, fee)
    return float(out["alpha"])


def alpha_nonoverlap_stats(prob, dates_signal, close_full, thr, horizon, fee):
    dates = pd.to_datetime(dates_signal)
    close_full2 = close_full.copy()
    close_full2.index = pd.to_datetime(close_full2.index)

    start_date = dates[0]
    last_date = dates[-1]
    try:
        last_loc = close_full2.index.get_loc(last_date)
        end_loc = min(int(last_loc) + int(horizon), len(close_full2.index) - 1)
        end_date = close_full2.index[end_loc]
    except Exception:
        end_date = close_full2.index[-1]

    bh_ret = float(close_full2.loc[end_date] / close_full2.loc[start_date] - 1.0)

    eq = 1.0
    i = 0
    n_trades = 0
    while i < len(prob):
        if prob[i] >= thr:
            d0 = dates[i]
            try:
                loc0 = close_full2.index.get_loc(d0)
            except KeyError:
                i += 1
                continue

            loc1 = int(loc0) + int(horizon)
            if loc1 >= len(close_full2.index):
                break

            entry = float(close_full2.iloc[int(loc0)])
            exitp = float(close_full2.iloc[int(loc1)])
            ret = exitp / entry - 1.0 - float(fee)

            eq *= (1.0 + ret)
            n_trades += 1
            i += int(horizon)
        else:
            i += 1

    strat_ret = float(eq - 1.0)
    alpha = float(strat_ret - bh_ret)
    return {"alpha": alpha, "n_trades": int(n_trades)}


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

    fund_cols = ["revenue", "net_income", "eps", "roe", "pb_ratio", "net_margin", "value_quality"]
    print("\nFUNDAMENTAL COLUMNS IN DATASET:")
    for c in fund_cols:
        if c in feat.columns:
            nn = int(feat[c].notna().sum())
            mean = float(feat[c].dropna().mean()) if nn > 0 else None
            print(c, "non-null =", nn, "mean =", mean)
        else:
            print(c, "MISSING")

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
    future_ret_w = future_ret_all[CFG["WINDOW"] - 1:]

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


def run_once(run_seed: int, prepared: Dict[str, Any]) -> Dict[str, Any]:
    """Run one full train/eval cycle for a given seed (data is already prepared)."""
    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seed(int(run_seed))

    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_weight = prepared["class_weight"]

    dw = prepared["dw"]
    test_mask = prepared["test_mask"]
    val_mask = prepared["val_mask"]
    future_ret_w = prepared["future_ret_w"]
    px = prepared["px"]

    ckpt_path = f"tcn_best_seed_{int(run_seed)}.weights.h5"

    model = build_tcn_model(CFG["WINDOW"], X_train.shape[2], CFG["LR"])
    if SHOW_MODEL_SUMMARY:
        model.summary()

    class ShortMetrics(tf.keras.callbacks.Callback):
        def __init__(self, every: int = 5):
            super().__init__()
            self.every = every

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            ep = epoch + 1
            if ep == 1 or ep % self.every == 0:
                print(
                    f"ep={ep:03d} "
                    f"loss={logs.get('loss', np.nan):.4f} "
                    f"val_loss={logs.get('val_loss', np.nan):.4f} "
                    f"val_auc_pr={logs.get('val_auc_pr', np.nan):.4f} "
                    f"val_auc_roc={logs.get('val_auc_roc', np.nan):.4f}"
                )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=CFG["PATIENCE"],
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        ShortMetrics(every=5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CFG["EPOCHS"],
        batch_size=CFG["BATCH"],
        class_weight=class_weight,
        shuffle=False,
        callbacks=callbacks,
        verbose=FIT_VERBOSE,
    )

    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    prob_train = model.predict(X_train, verbose=0).reshape(-1)
    prob_val = model.predict(X_val, verbose=0).reshape(-1)
    prob_test = model.predict(X_test, verbose=0).reshape(-1)

    # Orientation calibration on the last part of VAL (no TEST leakage)
    N_CAL = min(120, len(y_val))
    auc_tail = roc_auc_score(y_val[-N_CAL:], prob_val[-N_CAL:]) if len(np.unique(y_val[-N_CAL:])) > 1 else 0.5
    auc_tail_inv = roc_auc_score(y_val[-N_CAL:], 1 - prob_val[-N_CAL:]) if len(np.unique(y_val[-N_CAL:])) > 1 else 0.5
    FLIP = auc_tail_inv > auc_tail

    if SHOW_TRAIN_VAL_DIAG:
        print("\n" + "=" * 70)
        print(f"VAL tail AUC={auc_tail:.3f} | AUC(1-p)={auc_tail_inv:.3f} | FLIP={FLIP}")
        print("=" * 70)

    if FLIP:
        prob_train = 1 - prob_train
        prob_val = 1 - prob_val
        prob_test = 1 - prob_test

    if SHOW_TRAIN_VAL_DIAG:
        prob_summary("TRAIN", y_train, prob_train)
        prob_summary("VAL", y_val, prob_val)
    prob_summary("TEST", y_test, prob_test)

    fret_train = future_ret_w[prepared["train_mask"]]
    fret_val = future_ret_w[prepared["val_mask"]]
    fret_test = future_ret_w[prepared["test_mask"]]

    if SHOW_TRAIN_VAL_DIAG:
        decile_report("TRAIN", y_train, prob_train, fret_train)
        decile_report("VAL", y_val, prob_val, fret_val)
    decile_report("TEST", y_test, prob_test, fret_test)

    X_train_last = X_train[:, -1, :]
    X_test_last = X_test[:, -1, :]
    drift_report_features(X_train_last, X_test_last, prepared["FEATURES"], top_k=12)

    p_psi = psi_1d(prob_train, prob_test, n_bins=10)
    print("\n" + "=" * 70)
    print(f"PROB DRIFT PSI(train→test): {p_psi:.3f} (if >0.25 — strong shift)")
    print("=" * 70)

    thr_f1, thr_pnl, _ = pick_threshold_on_val(y_val, prob_val, fret_val, CFG["HORIZON"], CFG["FEE"])

    hist_info = history_summary(history)
    test_metrics = compact_prob_metrics(y_test, prob_test)
    dec_test = make_decile_table(y_test, prob_test, future_ret=fret_test, n_bins=10)

    dec_spread = float(dec_test.iloc[-1]["buy_rate"] - dec_test.iloc[0]["buy_rate"]) if len(dec_test) >= 2 else float("nan")

    _dates_test = dw[test_mask]
    _close_full = px["Close"]

    bt_f1 = backtest_nonoverlap_long_only_stats(prob_test, _dates_test, _close_full, thr_f1, CFG["HORIZON"], CFG["FEE"])
    bt_pnl = backtest_nonoverlap_long_only_stats(prob_test, _dates_test, _close_full, thr_pnl, CFG["HORIZON"], CFG["FEE"])

    print("\n" + "=" * 70)
    print("=== COMPACT DASHBOARD ===")
    print("=" * 70)
    print(
        f"HISTORY | epochs_run={hist_info['epochs_run']} "
        f"best_epoch={hist_info['best_epoch']} "
        f"best_val_auc_pr={hist_info['best_val_auc_pr']:.4f} "
        f"best_val_auc_roc={hist_info['best_val_auc_roc']:.4f} "
        f"best_val_loss={hist_info['best_val_loss']:.4f}"
    )
    print(
        f"TEST | roc_auc={test_metrics['roc_auc']:.4f} "
        f"pr_auc={test_metrics['pr_auc']:.4f} "
        f"pr_lift={test_metrics['pr_auc_lift']:+.4f} | "
        f"logloss={test_metrics['logloss']:.4f} "
        f"(gain_vs_base={test_metrics['logloss_gain_vs_baseline']:+.4f}) | "
        f"ece10={test_metrics['ece10']:.4f} | "
        f"prob_mean={test_metrics['prob_mean']:.4f}±{test_metrics['prob_std']:.4f} | "
        f"prob_psi(train→test)={p_psi:.3f}"
    )
    print(
        f"inverted_signal={test_metrics['roc_auc_inv'] > test_metrics['roc_auc']} "
        f"(roc_auc(1-p)={test_metrics['roc_auc_inv']:.4f})"
    )
    print(f"decile_spread(buy_rate top-bottom)={dec_spread:+.4f}")
    print("deciles (TEST):")
    print(dec_test.to_string(index=False))

    print("\nBACKTEST TEST (non-overlap, Close):")
    print(
        f"thr_f1={thr_f1:.2f} | strat={bt_f1['strategy_return']:+.2%} "
        f"bh={bt_f1['buyhold_return']:+.2%} alpha={bt_f1['alpha']:+.2%} "
        f"trades={bt_f1['n_trades']} win={bt_f1['winrate']:.1%}"
    )
    print(
        f"thr_pnl={thr_pnl:.2f} | strat={bt_pnl['strategy_return']:+.2%} "
        f"bh={bt_pnl['buyhold_return']:+.2%} alpha={bt_pnl['alpha']:+.2%} "
        f"trades={bt_pnl['n_trades']} win={bt_pnl['winrate']:.1%}"
    )
    print("=" * 70)

    alpha_stats = alpha_nonoverlap_stats(
        prob_test,
        _dates_test,
        _close_full,
        float(thr_pnl),
        int(CFG["HORIZON"]),
        float(CFG["FEE"]),
    )
    alpha_thr_pnl = float(alpha_stats["alpha"])
    n_trades_thr_pnl = int(alpha_stats["n_trades"])

    if bool(CFG.get("SAVE_SINGLE_RUN_ARTIFACTS", False)) and len(CFG.get("RUN_SEEDS", [])) <= 1:
        model.save("tcn_ru_model.keras")
        with open("tcn_ru_scaler.pkl", "wb") as f:
            pickle.dump(prepared["scaler"], f)
        print("\nSaved: tcn_ru_model.keras and tcn_ru_scaler.pkl")

    return {
        "seed": int(run_seed),
        "roc_auc": float(test_metrics["roc_auc"]),
        "pr_auc": float(test_metrics["pr_auc"]),
        "logloss_gain_vs_baseline": float(test_metrics["logloss_gain_vs_baseline"]),
        "prob_psi": float(p_psi),
        "alpha_thr_pnl": float(alpha_thr_pnl),
        "n_trades_thr_pnl": int(n_trades_thr_pnl),
        "best_epoch": int(hist_info["best_epoch"]),
        "thr_f1": float(thr_f1),
        "thr_pnl": float(thr_pnl),
    }


if __name__ == "__main__":
    prepared = prepare_dataset_once()

    results = []
    for sd in CFG.get("RUN_SEEDS", [42]):
        res = run_once(int(sd), prepared)
        results.append(res)
        print(
            f"Seed {sd}: "
            f"roc_auc={res['roc_auc']:.3f} "
            f"pr_auc={res['pr_auc']:.3f} "
            f"ll_gain={res['logloss_gain_vs_baseline']:+.4f} "
            f"psi={res['prob_psi']:.3f} "
            f"alpha@thr_pnl={res['alpha_thr_pnl']:+.2%} "
            f"trades={res['n_trades_thr_pnl']}"
        )

    df = pd.DataFrame(results)

    print("\n=== MULTI-SEED SUMMARY ===")
    with pd.option_context("display.max_columns", 50):
        print(df.to_string(index=False))
        print("\nDescribe:")
        print(df.describe(include="all").to_string())

    num_cols = [
        "roc_auc",
        "pr_auc",
        "logloss_gain_vs_baseline",
        "prob_psi",
        "alpha_thr_pnl",
        "n_trades_thr_pnl",
    ]
    summary = pd.DataFrame({
        "metric": num_cols,
        "mean": [df[c].mean() for c in num_cols],
        "std": [df[c].std(ddof=1) for c in num_cols],
        "min": [df[c].min() for c in num_cols],
        "max": [df[c].max() for c in num_cols],
    })

    print("\n=== AGGREGATED SUMMARY ===")
    print(summary.to_string(index=False))

    df.to_csv("multi_seed_summary.csv", index=False)
    summary.to_csv("multi_seed_aggregated.csv", index=False)
