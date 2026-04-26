from __future__ import annotations

from io import StringIO

import pandas as pd
import requests
from moexalgo import Ticker


# Map: stock ticker -> MOEX sector index code
# (official MOEX/RTS sector indices exist as separate series).
SECTOR_INDEX_MAP = {
    "SBER": "RTSFN",  # finance
    "T": "RTSFN",
    "VTBR": "RTSFN",
    "MOEX": "RTSFN",

    "GAZP": "RTSOG",  # oil & gas
    "LKOH": "RTSOG",
    "ROSN": "RTSOG",
    "NVTK": "RTSOG",
    "TATN": "RTSOG",

    "GMKN": "RTSMM",  # metals & mining
    "CHMF": "RTSMM",
    "NLMK": "RTSMM",
    "MAGN": "RTSMM",
    "PLZL": "RTSMM",

    "MTSS": "RTSTL",  # telecom
    "RTKM": "RTSTL",

    "IRAO": "RTSEU",  # electric utilities
    "HYDR": "RTSEU",

    "MGNT": "RTSCN",  # consumer / retail
    "X5": "RTSCN",

    "AFLT": "RTSTR",  # transport
}


def load_candles_moexalgo(ticker: str, start: str, end: str, period: str = "1D") -> pd.DataFrame:
    df = pd.DataFrame(Ticker(ticker).candles(start=start, end=end, period=period))
    if df.empty:
        return df
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.drop_duplicates(subset=["begin"]).set_index("begin").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def load_sector_index_moex(index_code: str, start: str, end: str) -> pd.DataFrame:
    """Load MOEX sector index candles (same schema as load_candles_moexalgo)."""
    return load_candles_moexalgo(index_code, start, end)


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


def cbr_ruonia_range(start: str, end: str) -> pd.DataFrame:
    """Best-effort RUONIA history within a range. Returns daily table (date, ruonia)."""
    url = "https://www.cbr.ru/hd_base/ruonia/"
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": pd.to_datetime(start).strftime("%d.%m.%Y"),
        "UniDbQuery.To": pd.to_datetime(end).strftime("%d.%m.%Y"),
    }

    html = requests.get(url, params=params, timeout=30).text
    tables = pd.read_html(StringIO(html))
    if not tables:
        return pd.DataFrame(columns=["date", "ruonia"])

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    val_col = None
    for c in df.columns:
        lc = c.lower()
        if date_col is None and "дата" in lc:
            date_col = c
        if val_col is None and ("ruonia" in lc or "руони" in lc):
            val_col = c

    if date_col is None:
        # fallback: first column looks like date
        date_col = df.columns[0] if len(df.columns) else None
    if val_col is None:
        # fallback: second column is usually the value
        val_col = df.columns[1] if len(df.columns) > 1 else None

    if date_col is None or val_col is None:
        return pd.DataFrame(columns=["date", "ruonia"])

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[val_col] = (
        df[val_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[date_col, val_col]).sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date", val_col: "ruonia"})

    # fix if parsed as 1500 instead of 15.00
    if not df.empty and df["ruonia"].max() > 100:
        df["ruonia"] = df["ruonia"] / 100.0

    return df


def cbr_usd_rate_range(start: str, end: str) -> pd.DataFrame:
    """Best-effort official USD/RUB (CBR) history. Returns daily table (date, usd_cbr)."""
    # CBR has a stable XML endpoint for dynamic currency rates.
    # NOTE: dates are DD/MM/YYYY.
    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
    params = {
        "date_req1": pd.to_datetime(start).strftime("%d/%m/%Y"),
        "date_req2": pd.to_datetime(end).strftime("%d/%m/%Y"),
        # USD: R01235
        "VAL_NM_RQ": "R01235",
    }
    xml = requests.get(url, params=params, timeout=30).text
    try:
        x = pd.read_xml(StringIO(xml))
    except Exception:
        return pd.DataFrame(columns=["date", "usd_cbr"])

    if x is None or x.empty:
        return pd.DataFrame(columns=["date", "usd_cbr"])

    # Expected columns: Date, Value, Nominal (sometimes)
    date_col = None
    for c in x.columns:
        if str(c).lower() in {"date", "@date"}:
            date_col = c
            break
    if date_col is None and "Date" in x.columns:
        date_col = "Date"
    if date_col is None:
        # fallback: first column
        date_col = x.columns[0]

    val_col = "Value" if "Value" in x.columns else (x.columns[1] if len(x.columns) > 1 else x.columns[0])
    nom_col = "Nominal" if "Nominal" in x.columns else None

    out = x[[date_col, val_col] + ([nom_col] if nom_col else [])].copy()
    out[date_col] = pd.to_datetime(out[date_col], dayfirst=True, errors="coerce")
    out[val_col] = (
        out[val_col]
        .astype(str)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    out[val_col] = pd.to_numeric(out[val_col], errors="coerce")

    if nom_col and nom_col in out.columns:
        out[nom_col] = pd.to_numeric(out[nom_col], errors="coerce").replace(0, 1)
        out["usd_cbr"] = out[val_col] / out[nom_col]
    else:
        out["usd_cbr"] = out[val_col]

    out = out.rename(columns={date_col: "date"})
    out = out[["date", "usd_cbr"]].dropna().sort_values("date").reset_index(drop=True)
    return out


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
