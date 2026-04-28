"""Yahoo Finance fundamentals loader for MOEX tickers.

Replaces Smart-Lab parser. Advantages:
- 4+ years of quarterly data (vs 1.5 years from Smart-Lab)
- No HTML scraping — stable API via yfinance
- Anti-leakage: merge_asof(backward) by report_date
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# MOEX ticker -> list of Yahoo candidates (.ME suffix)
YAHOO_MAP: dict[str, list[str]] = {
    "SBER": ["SBER.ME"],
    "T": ["T.ME", "TCSG.ME"],
    "VTBR": ["VTBR.ME"],
    "MOEX": ["MOEX.ME"],
    "GAZP": ["GAZP.ME"],
    "LKOH": ["LKOH.ME"],
    "ROSN": ["ROSN.ME"],
    "NVTK": ["NVTK.ME"],
    "TATN": ["TATN.ME"],
    "YNDX": ["YDEX.ME", "YNDX.ME"],
    "GMKN": ["GMKN.ME"],
    "CHMF": ["CHMF.ME"],
    "NLMK": ["NLMK.ME"],
    "MAGN": ["MAGN.ME"],
    "PLZL": ["PLZL.ME"],
    "MGNT": ["MGNT.ME"],
    "MTSS": ["MTSS.ME"],
    "RTKM": ["RTKM.ME"],
    "IRAO": ["IRAO.ME"],
    "HYDR": ["HYDR.ME"],
    "AFLT": ["AFLT.ME"],
    "OZON": ["OZON.ME"],
    "VKCO": ["VKCO.ME"],
    "POSI": ["POSI.ME"],
    "X5": ["X5.ME", "FIVE.ME"],
    "FIVE": ["FIVE.ME", "X5.ME"],
    "ALRS": ["ALRS.ME"],
    "POLY": ["POLY.ME"],
    "PHOR": ["PHOR.ME"],
    "RUAL": ["RUAL.ME"],
    "FEES": ["FEES.ME"],
    "PIKK": ["PIKK.ME"],
    "SMLT": ["SMLT.ME"],
    "SGZH": ["SGZH.ME"],
    "MTLR": ["MTLR.ME"],
    "AFKS": ["AFKS.ME"],
    "CBOM": ["CBOM.ME"],
    "SNGS": ["SNGS.ME"],
    "BANEP": ["BANEP.ME"],
    "TRNFP": ["TRNFP.ME"],
    "NMTP": ["NMTP.ME"],
    "FLOT": ["FLOT.ME"],
    "TCSG": ["TCSG.ME", "T.ME"],
    "FIXP": ["FIXP.ME"],
    "ENPG": ["ENPG.ME"],
    "LENT": ["LENT.ME"],
    "RASP": ["RASP.ME"],
    "SELG": ["SELG.ME"],
    "BSPB": ["BSPB.ME"],
    "AQUA": ["AQUA.ME"],
    "RNFT": ["RNFT.ME"],
    "MSNG": ["MSNG.ME"],
    "LSRG": ["LSRG.ME"],
    "RENI": ["RENI.ME"],
}

_INCOME_ROWS = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "operating_income": ["Operating Income"],
}

_BALANCE_ROWS = {
    "total_assets": ["Total Assets"],
    "total_equity": [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
    ],
    "total_debt": ["Total Debt"],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
    ],
}

_CASHFLOW_ROWS = {
    "operating_cashflow": ["Operating Cash Flow", "Total Cash From Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
    "free_cashflow": ["Free Cash Flow"],
}

# Keep ONLY the 5 independent fundamental axes.
YAHOO_FUND_FEATURES = [
    "roe_calc",  # profitability (ROE)
    "net_margin",  # sales margin
    "debt_to_equity",  # leverage
    "revenue_growth_yoy",  # growth (YoY)
    "cash_ratio",  # liquidity
]


def _pick_value(stmt_df: pd.DataFrame, possible_rows: list[str], col) -> float:
    if stmt_df is None or stmt_df.empty:
        return np.nan

    idx_map = {str(x): x for x in stmt_df.index}
    idx_lower = {str(x).lower(): x for x in stmt_df.index}

    for name in possible_rows:
        if name in idx_map:
            try:
                return float(stmt_df.loc[idx_map[name], col])
            except Exception:
                pass
        if name.lower() in idx_lower:
            try:
                return float(stmt_df.loc[idx_lower[name.lower()], col])
            except Exception:
                pass

    return np.nan


def _norm_columns(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    new_cols = []
    for c in out.columns:
        try:
            new_cols.append(pd.to_datetime(c))
        except Exception:
            new_cols.append(c)

    out.columns = new_cols
    date_cols = [c for c in out.columns if isinstance(c, pd.Timestamp)]
    return out[date_cols] if date_cols else pd.DataFrame()


def _best_symbol(ticker: str):
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("pip install yfinance") from e

    candidates = YAHOO_MAP.get(ticker, [f"{ticker}.ME"])
    best_payload = None
    best_score = -1

    for symbol in candidates:
        try:
            y = yf.Ticker(symbol)
            info = y.info if isinstance(y.info, dict) else {}

            income = _norm_columns(y.income_stmt)
            balance = _norm_columns(y.balance_sheet)
            cashflow = _norm_columns(y.cashflow)

            score = 0
            score += 5 if not income.empty else 0
            score += 5 if not balance.empty else 0
            score += 5 if not cashflow.empty else 0
            score += len(
                [
                    k
                    for k in ["trailingPE", "priceToBook", "returnOnEquity"]
                    if info.get(k) is not None
                ]
            )

            if score > best_score:
                best_score = score
                best_payload = {
                    "symbol": symbol,
                    "score": score,
                    "income": income,
                    "balance": balance,
                    "cashflow": cashflow,
                }
        except Exception:
            pass
        time.sleep(0.15)

    return best_payload


def fetch_yahoo_fundamentals(ticker: str, cache: dict | None = None) -> pd.DataFrame:
    if cache is not None and ticker in cache:
        return cache[ticker]

    payload = _best_symbol(ticker)
    if payload is None or payload["score"] <= 0:
        print(f"  [Yahoo] {ticker}: no data found")
        empty = pd.DataFrame()
        if cache is not None:
            cache[ticker] = empty
        return empty

    income = payload["income"]
    balance = payload["balance"]
    cashflow = payload["cashflow"]

    all_dates = set()
    for df in [income, balance, cashflow]:
        if isinstance(df, pd.DataFrame) and not df.empty:
            all_dates.update([c for c in df.columns if isinstance(c, pd.Timestamp)])

    if not all_dates:
        print(f"  [Yahoo] {ticker} -> {payload['symbol']}: no report dates")
        empty = pd.DataFrame()
        if cache is not None:
            cache[ticker] = empty
        return empty

    rows = []
    for dt in sorted(all_dates):
        row = {"ticker": ticker, "report_date": pd.to_datetime(dt)}

        for metric, names in _INCOME_ROWS.items():
            row[metric] = _pick_value(income, names, dt)

        for metric, names in _BALANCE_ROWS.items():
            row[metric] = _pick_value(balance, names, dt)

        for metric, names in _CASHFLOW_ROWS.items():
            row[metric] = _pick_value(cashflow, names, dt)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("report_date").reset_index(drop=True)

    out["roe_calc"] = out["net_income"] / out["total_equity"].replace(0, np.nan)
    out["debt_to_equity"] = out["total_debt"] / out["total_equity"].replace(0, np.nan)
    out["net_margin"] = out["net_income"] / out["revenue"].replace(0, np.nan)
    # YoY for quarterly reports: compare to the same quarter a year ago.
    out["revenue_growth_yoy"] = out.groupby("ticker")["revenue"].pct_change(4)
    out["cash_ratio"] = out["cash"] / out["total_assets"].replace(0, np.nan)

    for col in [
        "roe_calc",
        "debt_to_equity",
        "net_margin",
        "revenue_growth_yoy",
        "cash_ratio",
    ]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    n = len(out)
    rng = ""
    if n > 0:
        rng = f" | range=[{out['report_date'].min()} .. {out['report_date'].max()}]"
    print(f"  [Yahoo] {ticker} -> {payload['symbol']}: {n} reports{rng}")

    if cache is not None:
        cache[ticker] = out

    return out


def add_yahoo_features_past_only(
    df_feat: pd.DataFrame,
    fund_df: pd.DataFrame | None,
    lag_days: int = 1,
) -> pd.DataFrame:
    if fund_df is None or fund_df.empty:
        for col in YAHOO_FUND_FEATURES:
            df_feat[col] = 0.0
        return df_feat

    f = fund_df.copy()
    f["report_date"] = pd.to_datetime(f["report_date"])
    if lag_days > 0:
        f["report_date"] = f["report_date"] + pd.Timedelta(days=lag_days)
    f = f.sort_values("report_date")

    d = df_feat.copy()
    d["_date"] = d.index

    merged = pd.merge_asof(
        d.reset_index(drop=True),
        f[["report_date", "roe_calc", "net_margin", "debt_to_equity", "revenue_growth_yoy", "cash_ratio"]],
        left_on="_date",
        right_on="report_date",
        direction="backward",
    )

    # Fill missing fundamentals with 0.0 (no *_is_missing flags by design).
    for col in YAHOO_FUND_FEATURES:
        merged[col] = merged[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    merged = merged.set_index("_date")
    merged.index.name = df_feat.index.name
    merged = merged.drop(columns=["report_date"], errors="ignore")

    return merged
