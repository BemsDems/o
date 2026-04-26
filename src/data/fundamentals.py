from __future__ import annotations

"""Fundamentals scaffold (optional).

Not used in the current practical baseline, but kept for the diploma scaffold.
"""

import re
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.config.settings import CFG_FUND


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

        TODO: plug correct endpoint from Swagger.
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
    # Smart-Lab HTML часто содержит неразрывные пробелы и узкие NBSP, из-за
    # которых числа могут silently парситься в NaN.
    s = s.replace(" ", "")
    s = s.replace("\xa0", "")
    s = s.replace(" ", "")
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

    # max(..., key=shape[1]) хрупко: на странице могут быть другие широкие таблицы.
    # Выбираем таблицу, где в первой колонке встречаются характерные метрики.
    candidate_tables = []
    key_patterns = ["выручка", "ebitda", "чистая прибыль", "eps", "roe", "дата отч"]

    for t in tables:
        tt = t.copy()
        tt.columns = [str(c).strip() for c in tt.columns]
        if tt.empty:
            continue

        first_col = tt.columns[0]
        sample = " ".join(tt[first_col].astype(str).head(20).tolist()).lower().replace("ё", "е")
        score = sum(int(p in sample) for p in key_patterns)
        candidate_tables.append((score, tt.shape[1], tt))

    if not candidate_tables:
        return pd.DataFrame()

    candidate_tables.sort(key=lambda x: (x[0], x[1]), reverse=True)
    tbl = candidate_tables[0][2].copy()

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

    date_mask = df["metric_ru"].str.contains(r"дата отч[её]та|report date", case=False, na=False)
    date_map = df[date_mask][["period", "value_raw"]].rename(columns={"value_raw": "publish_date"}).copy()
    if not date_map.empty:
        date_map["publish_date"] = pd.to_datetime(date_map["publish_date"], dayfirst=True, errors="coerce")

    df = df[~date_mask].copy()

    def _norm_metric_label(x: str) -> str:
        x = str(x).strip().lower()
        x = x.replace("ё", "е")
        x = re.sub(r"\s+", " ", x)
        return x

    metric_map = {
        "выручка": "revenue",
        "ebitda": "ebitda",
        "чистая прибыль н/с": "net_income",
        "чистая прибыль": "net_income",
        "eps": "eps",
        "roe": "roe",
        "p/bv": "pb_ratio",
        "p/b": "pb_ratio",
        "чистая рентаб": "net_margin",
        "чистая маржа": "net_margin",
        "долг/ebitda": "debt_ebitda",
        "чистый долг": "net_debt",
        "долг": "debt",
    }

    def map_metric(x: str) -> Optional[str]:
        x_norm = _norm_metric_label(x)
        for k, v in metric_map.items():
            if x_norm.startswith(k):
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
    - merge_asof backward on effective_date
    """

    out = price_df.copy().sort_index().reset_index()

    # first column after reset_index() is former datetime index
    out = out.rename(columns={out.columns[0]: "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    if out["date"].isna().all():
        raise ValueError("Price date column could not be parsed after reset_index()")

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

    f["effective_date"] = pd.to_datetime(f["effective_date"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

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

    out["fund_age_days"] = (
        (out["date"] - out["effective_date"]).dt.days
        .clip(lower=0)
    )

    out = out.drop(columns=["effective_date"], errors="ignore")

    # derived features
    if "roe" in out.columns and "pb_ratio" in out.columns:
        out["value_quality"] = out["roe"] / out["pb_ratio"].replace(0, np.nan)
        out["value_quality"] = out["value_quality"].replace([np.inf, -np.inf], np.nan)

    if "net_income" in out.columns and "revenue" in out.columns:
        out["net_margin_calc"] = out["net_income"] / out["revenue"].replace(0, np.nan)

    if "net_margin" not in out.columns:
        out["net_margin"] = np.nan
    if "net_margin_calc" in out.columns:
        out["net_margin"] = out["net_margin"].where(out["net_margin"].notna(), out["net_margin_calc"])

    if "revenue" in out.columns:
        out["log_revenue"] = np.sign(out["revenue"]) * np.log1p(np.abs(out["revenue"]))
    else:
        out["log_revenue"] = np.nan

    if "net_income" in out.columns:
        out["log_net_income"] = np.sign(out["net_income"]) * np.log1p(np.abs(out["net_income"]))
    else:
        out["log_net_income"] = np.nan

    fund_core_cols = [
        "roe",
        "pb_ratio",
        "value_quality",
        "eps",
    ]
    for c in fund_core_cols:
        if c not in out.columns:
            out[c] = np.nan
        out[f"{c}_is_missing"] = out[c].isna().astype(int)

    if "fund_age_days" not in out.columns:
        out["fund_age_days"] = np.nan

    if out["date"].dt.year.min() < 1990:
        raise ValueError(
            "Attached price dates look corrupted (<1990). "
            "Check reset_index()/date parsing in add_fundamental_features_past_only()."
        )

    return out.set_index("date")
