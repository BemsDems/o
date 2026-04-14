# Colab-ready: MOEX/CBR feature engineering + LSTM classifier (Sonnet fixed)
# Fixes requested:
# 1) First LSTM uses return_sequences=True (explicit)
# 2) (Optional) Dividends are already merged via merge_asof in past-only manner
#
# In Colab:
# !pip -q install moexalgo requests pandas numpy scikit-learn tensorflow lxml html5lib

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from moexalgo import Ticker

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# CONFIG
# ----------------------------
CFG: Dict[str, Any] = {
    "TICKER": "SBER",
    "START": "2015-01-01",
    "END": None,  # None => today
    "TARGET_HORIZON_DAYS": 5,
    "TARGET_UP_THRESHOLD": 0.01,
    "SEQ_LEN": 60,
    "TRAIN_SPLIT": 0.8,
    "BATCH_SIZE": 64,
    "EPOCHS": 20,
    "LR": 1e-3,
    "SEED": 42,
    "CACHE_DIR": "cache",
}

np.random.seed(CFG["SEED"])
tf.random.set_seed(CFG["SEED"])


def _ensure_cache_dir() -> str:
    os.makedirs(CFG["CACHE_DIR"], exist_ok=True)
    return CFG["CACHE_DIR"]


def _cache_path(name: str) -> str:
    return os.path.join(_ensure_cache_dir(), name)


def fetch_moex_history(secid: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Fetch OHLCV via MOEX ISS history (shares)."""
    url = (
        "https://iss.moex.com/iss/history/engines/stock/markets/shares/"
        f"securities/{secid}.json"
    )

    all_rows = []
    start_pos = 0
    while True:
        params = {
            "from": start,
            "till": end,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE",
            "start": start_pos,
        }
        j = requests.get(url, params=params, timeout=30).json()
        block = j.get("history", {})
        cols = block.get("columns", [])
        data = block.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        start_pos += len(data)
        if len(data) < 100:
            break

    df = pd.DataFrame(all_rows, columns=cols)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"]).sort_values("TRADEDATE").reset_index(drop=True)
    df = df.rename(columns={"TRADEDATE": "date"})
    for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_moex_dividends(secid: str) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/statistics/engines/stock/markets/shares/"
        f"securities/{secid}/dividends.json"
    )
    params = {"iss.meta": "off", "iss.only": "dividends"}
    j = requests.get(url, params=params, timeout=30).json()
    block = j.get("dividends", {})
    df = pd.DataFrame(block.get("data", []), columns=block.get("columns", []))
    if df.empty:
        return df

    # columns vary; normalize a couple of useful ones
    for col in ["registryclosedate", "registryclose", "close_date", "registry_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # try to pick a date column
    date_col = None
    for cand in ["registryclosedate", "registryclose", "close_date", "registry_date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return pd.DataFrame()

    df = df.rename(columns={date_col: "date"})
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    elif "dividendvalue" in df.columns:
        df["dividendvalue"] = pd.to_numeric(df["dividendvalue"], errors="coerce")
        df = df.rename(columns={"dividendvalue": "value"})
    else:
        df["value"] = np.nan

    df = df[["date", "value"]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def fetch_cbr_usdrub(start: str, end: Optional[str]) -> pd.Series:
    # CBR XML dynamic for USD (R01235)
    import datetime as dt
    from xml.etree import ElementTree as ET

    if end is None:
        end = dt.date.today().strftime("%Y-%m-%d")

    start_dt = pd.to_datetime(start).strftime("%d/%m/%Y")
    end_dt = pd.to_datetime(end).strftime("%d/%m/%Y")

    url = "https://www.cbr.ru/scripts/XML_dynamic.asp"
    params = {"date_req1": start_dt, "date_req2": end_dt, "VAL_NM_RQ": "R01235"}
    xml = requests.get(url, params=params, timeout=30).text

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


def make_target(close: pd.Series, horizon: int, thr: float) -> pd.Series:
    fwd = close.shift(-horizon)
    ret = (fwd - close) / close
    y = (ret >= thr).astype(int)
    return y


def add_features(df: pd.DataFrame, usd: pd.Series, divs: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # basic returns
    out["ret_1d"] = out["CLOSE"].pct_change()
    out["ret_5d"] = out["CLOSE"].pct_change(5)

    # merge USD/RUB (past-only via asof)
    usd_df = usd.reset_index().rename(columns={"index": "date", "CBR_USD_RUB": "usd_rub"})
    usd_df["date"] = pd.to_datetime(usd_df["date"])
    out = pd.merge_asof(out.sort_values("date"), usd_df.sort_values("date"), on="date", direction="backward")

    # merge dividends (past-only)
    if isinstance(divs, pd.DataFrame) and not divs.empty:
        dd = divs.copy()
        dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
        dd = dd.dropna(subset=["date"]).sort_values("date")
        out = pd.merge_asof(out.sort_values("date"), dd[["date", "value"]], on="date", direction="backward")
        out = out.rename(columns={"value": "last_dividend"})
    else:
        out["last_dividend"] = np.nan

    out = out.dropna().reset_index(drop=True)
    return out


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
    return np.asarray(Xs), np.asarray(ys)


def build_model(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(CFG["SEQ_LEN"], n_features))

    # explicitly return_sequences=True on the first LSTM
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["LR"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    secid = CFG["TICKER"]

    cache_file = _cache_path(f"dataset_{secid}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            df_feat = pickle.load(f)
    else:
        df_price = fetch_moex_history(secid, CFG["START"], CFG["END"])
        divs = fetch_moex_dividends(secid)
        usd = fetch_cbr_usdrub(CFG["START"], CFG["END"])
        df_feat = add_features(df_price, usd, divs)
        with open(cache_file, "wb") as f:
            pickle.dump(df_feat, f)

    y = make_target(df_feat["CLOSE"], CFG["TARGET_HORIZON_DAYS"], CFG["TARGET_UP_THRESHOLD"]).values

    feature_cols = [c for c in df_feat.columns if c not in {"date"}]
    X = df_feat[feature_cols].values

    split = int(len(X) * CFG["TRAIN_SPLIT"])
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_train, y_train = make_sequences(X_train_scaled, y_train_raw, CFG["SEQ_LEN"])
    X_test, y_test = make_sequences(X_test_scaled, y_test_raw, CFG["SEQ_LEN"])

    # class weights
    classes = np.unique(y_train)
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    model = build_model(X_train.shape[-1])
    model.summary()

    model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=CFG["EPOCHS"],
        batch_size=CFG["BATCH_SIZE"],
        class_weight=class_weight,
        verbose=1,
    )

    proba = model.predict(X_test, verbose=0).reshape(-1)
    pred = (proba >= 0.5).astype(int)

    print("balanced_accuracy:", balanced_accuracy_score(y_test, pred))
    print("f1:", f1_score(y_test, pred))
    print("roc_auc:", roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None)
    print("mcc:", matthews_corrcoef(y_test, pred) if len(np.unique(y_test)) > 1 else None)
    print("confusion_matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    print("average_precision:", average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else None)


if __name__ == "__main__":
    main()
