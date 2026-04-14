"""Colab-ready: Fundamental/Macro data loaders with SAFE execution (Sonnet variant).

Goal: run all blocks; any block failure prints error and continues.

Sources / blocks:
- MOEX ISS (description, dividends, splits, last price)
- CBR USD/RUB daily series and key rate table
- Yahoo Finance fundamentals via `yfinance` (optional)
- Optional APIs: Alpha Vantage, Financial Modeling Prep, EODHD (API keys optional)
- e-disclosure.ru search template (best-effort)

In Colab you may need:
    pip install yfinance beautifulsoup4 lxml pandas numpy requests

Env vars (optional):
- ALPHAVANTAGE_API_KEY
- FMP_API_KEY
- EODHD_API_KEY

"""

from __future__ import annotations

import os
import traceback
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import requests

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 160)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Colab; +https://colab.research.google.com)",
})
TIMEOUT_SECONDS = 30


# -----------------------------
# SAFE helpers
# -----------------------------

def log(msg: str) -> None:
    print(msg)


def safe_run(title: str, fn: Callable[[], Any], default: Any = None) -> Any:
    """Runs fn() safely.

    If error -> prints stack trace and returns default.
    """

    log(f"\n=== {title} ===")
    try:
        out = fn()
        log(f"[OK] {title}")
        return out
    except Exception as e:  # noqa: BLE001
        log(f"[ERROR] {title}: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        return default


def _safe_get(url: str, params: Optional[Dict[str, Any]] = None, *, expect_json: bool = True) -> Any:
    r = SESSION.get(url, params=params, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json() if expect_json else r.text


def _iss_to_df(payload: Any, block_name: str) -> pd.DataFrame:
    if not isinstance(payload, dict) or block_name not in payload:
        return pd.DataFrame()
    block = payload[block_name]
    if not isinstance(block, dict) or "columns" not in block or "data" not in block:
        return pd.DataFrame()
    return pd.DataFrame(block["data"], columns=block["columns"])


def show_df(df: Any, name: str, n: int = 10) -> None:
    if df is None:
        log(f"[{name}] -> None")
        return
    if not isinstance(df, pd.DataFrame):
        log(f"[{name}] -> type={type(df)}")
        return
    log(f"[{name}] -> shape={df.shape}")
    if not df.empty:
        print(df.head(n).to_string(index=False))


# -----------------------------
# 1) MOEX ISS (official)
# -----------------------------

def moex_iss_security_description(secid: str) -> pd.DataFrame:
    url = f"https://iss.moex.com/iss/securities/{secid}.json"
    j = _safe_get(url, params={"iss.meta": "off"})
    return _iss_to_df(j, "description")


def moex_iss_dividends(secid: str) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/statistics/engines/stock/markets/shares/"
        f"securities/{secid}/dividends.json"
    )
    j = _safe_get(url, params={"iss.meta": "off", "iss.only": "dividends"})
    return _iss_to_df(j, "dividends")


def moex_iss_splits(secid: str) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/statistics/engines/stock/markets/shares/"
        f"securities/{secid}/splits.json"
    )
    j = _safe_get(url, params={"iss.meta": "off", "iss.only": "splits"})
    return _iss_to_df(j, "splits")


def moex_iss_last_price(secid: str) -> Optional[float]:
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{secid}.json"
    j = _safe_get(url, params={"iss.meta": "off", "iss.only": "marketdata"})
    md = _iss_to_df(j, "marketdata")
    if md.empty:
        return None

    for col in ["LAST", "LCURRENTPRICE", "LEGALCLOSEPRICE"]:
        if col in md.columns and pd.notna(md.loc[0, col]):
            return float(md.loc[0, col])
    return None


# -----------------------------
# 2) CBR (macro): USD/RUB series + key rate table
# -----------------------------

def cbr_usdrub_daily(start: str = "2020-01-01", end: Optional[str] = None) -> pd.Series:
    import datetime as dt
    from xml.etree import ElementTree as ET

    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    start_dt = pd.to_datetime(start).strftime("%d/%m/%Y")
    end_dt = pd.to_datetime(end).strftime("%d/%m/%Y")

    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
    xml = _safe_get(
        url,
        params={"date_req1": start_dt, "date_req2": end_dt, "VAL_NM_RQ": "R01235"},
        expect_json=False,
    )

    root = ET.fromstring(xml)
    rows = []
    for rec in root.findall("Record"):
        d = rec.attrib.get("Date")
        v = rec.findtext("Value")
        if d and v:
            rows.append((pd.to_datetime(d, dayfirst=True), float(v.replace(",", "."))))

    s = pd.Series(dict(rows)).sort_index()
    s.name = "CBR_USD_RUB"
    return s


def cbr_key_rate_table() -> pd.DataFrame:
    url = "https://www.cbr.ru/hd_base/KeyRate/"
    html = _safe_get(url, expect_json=False)

    tables = pd.read_html(html)
    if not tables:
        return pd.DataFrame()

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    rate_col = None
    for col in df.columns:
        if date_col is None and "дата" in col.lower():
            date_col = col
        if rate_col is None and "став" in col.lower():
            rate_col = col

    if date_col is None or rate_col is None:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[rate_col] = pd.to_numeric(df[rate_col].astype(str).str.replace(",", "."), errors="coerce")
    df = df.dropna().sort_values(date_col).reset_index(drop=True)
    return df.rename(columns={date_col: "date", rate_col: "key_rate"})


# -----------------------------
# 3) Yahoo via yfinance (optional; may be sparse for MOEX)
# -----------------------------

def yahoo_fundamentals(yahoo_ticker: str) -> Dict[str, Any]:
    import yfinance as yf

    t = yf.Ticker(yahoo_ticker)

    out: Dict[str, Any] = {}

    # info
    try:
        out["info"] = t.info
    except Exception as e:  # noqa: BLE001
        out["info_error"] = str(e)

    # statements
    for name, attr in [("financials", "financials"), ("balance_sheet", "balance_sheet"), ("cashflow", "cashflow")]:
        try:
            df = getattr(t, attr)
            if isinstance(df, pd.DataFrame) and not df.empty:
                out[name] = df
        except Exception:
            pass

    return out


# -----------------------------
# 4) Alpha Vantage (needs API key)
# -----------------------------

def alphavantage_overview(symbol: str, api_key: str) -> Dict[str, Any]:
    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
    return _safe_get(url, params=params, expect_json=True)


# -----------------------------
# 5) Financial Modeling Prep (needs API key)
# -----------------------------

def fmp_profile(symbol: str, api_key: str) -> pd.DataFrame:
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
    j = _safe_get(url, params={"apikey": api_key}, expect_json=True)
    return pd.DataFrame(j)


def fmp_ratios(symbol: str, api_key: str, limit: int = 40) -> pd.DataFrame:
    url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
    j = _safe_get(url, params={"apikey": api_key, "limit": limit}, expect_json=True)
    return pd.DataFrame(j)


# -----------------------------
# 6) EODHD fundamentals (needs API key)
# -----------------------------

def eodhd_fundamentals(symbol_with_exchange: str, api_key: str) -> Dict[str, Any]:
    url = f"https://eodhd.com/api/fundamentals/{symbol_with_exchange}"
    return _safe_get(url, params={"api_token": api_key, "fmt": "json"}, expect_json=True)


# -----------------------------
# 7) e-disclosure.ru (template; may break if site changes)
# -----------------------------

def edisclosure_search(query: str) -> pd.DataFrame:
    # Best-effort template.
    from bs4 import BeautifulSoup

    url = "https://www.e-disclosure.ru/poisk-po-kompaniyam"
    html = _safe_get(url, params={"query": query}, expect_json=False)

    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)
        if "/portal/company.aspx?id=" in href:
            links.append({"name": text, "url": "https://www.e-disclosure.ru" + href})

    return pd.DataFrame(links).drop_duplicates()


def main() -> None:
    # -----------------------------
    # RUN ALL BLOCKS (errors won't stop execution)
    # -----------------------------
    secid = "SBER"  # MOEX SECID
    yahoo = "SBER.ME"  # Yahoo ticker

    av_key = os.getenv("ALPHAVANTAGE_API_KEY")  # optional
    fmp_key = os.getenv("FMP_API_KEY")  # optional
    eod_key = os.getenv("EODHD_API_KEY")  # optional

    # --- MOEX ISS ---
    sec_desc = safe_run("MOEX: Security description", lambda: moex_iss_security_description(secid), default=pd.DataFrame())
    show_df(sec_desc, "MOEX description", n=15)

    divs = safe_run("MOEX: Dividends", lambda: moex_iss_dividends(secid), default=pd.DataFrame())
    show_df(divs, "MOEX dividends", n=10)

    splits = safe_run("MOEX: Splits", lambda: moex_iss_splits(secid), default=pd.DataFrame())
    show_df(splits, "MOEX splits", n=10)

    last_price = safe_run("MOEX: Last price", lambda: moex_iss_last_price(secid), default=None)
    log(f"[MOEX last price] {last_price}")

    # --- CBR macro ---
    usdrub = safe_run("CBR: USD/RUB daily series", lambda: cbr_usdrub_daily("2020-01-01"), default=pd.Series(dtype=float))
    if isinstance(usdrub, pd.Series) and not usdrub.empty:
        log(f"[CBR_USD_RUB] points={len(usdrub)} last5:")
        print(usdrub.tail(5))

    keyrate = safe_run("CBR: Key rate table", cbr_key_rate_table, default=pd.DataFrame())
    show_df(keyrate, "CBR key rate", n=10)

    # --- Yahoo fundamentals ---
    yf_data = safe_run("Yahoo (yfinance): fundamentals", lambda: yahoo_fundamentals(yahoo), default={})
    if isinstance(yf_data, dict) and yf_data:
        info = yf_data.get("info", {})
        if isinstance(info, dict) and info:
            keys = [
                "shortName",
                "longName",
                "currency",
                "exchange",
                "quoteType",
                "marketCap",
                "enterpriseValue",
                "trailingPE",
                "forwardPE",
                "priceToBook",
                "dividendYield",
                "trailingAnnualDividendRate",
                "trailingAnnualDividendYield",
                "beta",
                "profitMargins",
                "operatingMargins",
                "totalRevenue",
                "grossProfits",
                "ebitda",
                "netIncomeToCommon",
                "returnOnEquity",
                "debtToEquity",
            ]
            log("yfinance info (selected keys):")
            print({k: info.get(k, None) for k in keys})
        else:
            log("yfinance: .info пустой/недоступен для этого тикера.")

        for k in ["financials", "balance_sheet", "cashflow"]:
            dfk = yf_data.get(k)
            if isinstance(dfk, pd.DataFrame) and not dfk.empty:
                log(f"\n[{k}] shape={dfk.shape}")
                print(dfk.head().to_string())
    else:
        log("yfinance: данных нет (или блок упал и вернул default).")

    # --- Optional APIs (won't crash if no keys) ---
    safe_run(
        "AlphaVantage: OVERVIEW (optional)",
        lambda: alphavantage_overview("SBER.ME", av_key) if av_key else "SKIP (no ALPHAVANTAGE_API_KEY)",
        default="SKIP/ERROR",
    )

    safe_run(
        "FMP: profile & ratios (optional)",
        lambda: {
            "profile": fmp_profile("AAPL", fmp_key).head(3),
            "ratios": fmp_ratios("AAPL", fmp_key, limit=5).head(3),
        }
        if fmp_key
        else "SKIP (no FMP_API_KEY)",
        default="SKIP/ERROR",
    )

    safe_run(
        "EODHD: fundamentals (optional)",
        lambda: eodhd_fundamentals("AAPL.US", eod_key) if eod_key else "SKIP (no EODHD_API_KEY)",
        default="SKIP/ERROR",
    )

    # --- e-disclosure template ---
    edis = safe_run("e-disclosure.ru: search template (optional)", lambda: edisclosure_search("Сбербанк"), default=pd.DataFrame())
    show_df(edis, "e-disclosure search", n=10)

    log("\nALL DONE: all blocks attempted; errors (if any) were printed but execution continued.")


if __name__ == "__main__":
    main()
