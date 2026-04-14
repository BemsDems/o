# SONNET MODEL v3.0 - RADICAL SIMPLIFICATION
# Target: stability via simpler target/features/model.
# NOTE: No accuracy target is guaranteed.
#
# In Colab:
# !pip -q install requests pandas numpy scikit-learn tensorflow lxml html5lib

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# SIMPLIFIED CONFIG - FOCUS ON STABILITY
# ----------------------------
CFG: Dict[str, Any] = {
    "TICKER": "SBER",
    "START": "2015-01-01",
    "END": None,

    "TARGET_HORIZON_DAYS": 5,
    "TARGET_UP_THRESHOLD": 0.02,

    "SEQ_LEN": 30,

    "TRAIN_SPLIT": 0.70,
    "VAL_SPLIT": 0.15,

    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "LR": 1e-3,

    "SEED": 42,
    "CACHE_DIR": "cache",

    "DIAGNOSTICS": False,
    "BEST_THR": 0.5,
    "FEE": 0.001,
}

np.random.seed(CFG["SEED"])
tf.random.set_seed(CFG["SEED"])


def _ensure_cache_dir() -> str:
    os.makedirs(CFG["CACHE_DIR"], exist_ok=True)
    return CFG["CACHE_DIR"]


def _cache_path(name: str) -> str:
    return os.path.join(_ensure_cache_dir(), name)


# ----------------------------
# HTTP helpers
# ----------------------------

def _get_json(url: str, *, params: dict | None = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout)
    ct = (r.headers.get("Content-Type") or "").lower()
    try:
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(f"HTTP error for {url} status={r.status_code} ct={ct} preview={preview!r}") from e

    try:
        return r.json()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(f"Non-JSON response for {url} status={r.status_code} ct={ct} preview={preview!r}") from e


# ----------------------------
# DATA LOADING
# ----------------------------

def fetch_moex_history(secid: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Fetch OHLCV via MOEX ISS history."""
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{secid}.json"

    all_rows = []
    start_pos = 0
    cols = []

    while True:
        params = {
            "from": start,
            "till": end,
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,VALUE",
            "start": start_pos,
        }
        j = _get_json(url, params=params, timeout=30)
        block = j.get("history", {})
        cols = block.get("columns", cols)
        data = block.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        start_pos += len(data)
        if len(data) < 100:
            break

    df = pd.DataFrame(all_rows, columns=cols)
    if df.empty:
        return df

    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"]).sort_values("TRADEDATE").reset_index(drop=True)
    df = df.rename(columns={"TRADEDATE": "date"})

    for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def fetch_cbr_usdrub(start: str, end: Optional[str]) -> pd.Series:
    """CBR XML dynamic for USD."""
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


# ----------------------------
# SIMPLIFIED TARGET + FEATURES
# ----------------------------

def make_simple_target(close: pd.Series, horizon: int, thr: float) -> pd.Series:
    """Simple target: future return > threshold."""
    fwd_ret = (close.shift(-horizon) - close) / close
    return (fwd_ret >= thr).astype(int)


def make_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    fwd = close.shift(-horizon)
    return (fwd - close) / close


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_stable_features(df: pd.DataFrame, usd: pd.Series) -> pd.DataFrame:
    """Only stable features (small set)."""

    out = df.copy()

    # RETURNS
    out["ret_1d"] = out["CLOSE"].pct_change(1)
    out["ret_5d"] = out["CLOSE"].pct_change(5)
    out["ret_10d"] = out["CLOSE"].pct_change(10)
    out["ret_20d"] = out["CLOSE"].pct_change(20)
    out["log_ret"] = np.log(out["CLOSE"] / out["CLOSE"].shift(1))

    # TREND
    sma20 = out["CLOSE"].rolling(20).mean()
    sma50 = out["CLOSE"].rolling(50).mean()
    sma200 = out["CLOSE"].rolling(200).mean()
    out["price_vs_sma20"] = out["CLOSE"] / (sma20 + 1e-12) - 1
    out["price_vs_sma50"] = out["CLOSE"] / (sma50 + 1e-12) - 1
    out["trend_up"] = (out["CLOSE"] > sma200).astype(int)

    # MOMENTUM
    out["rsi_14"] = _rsi(out["CLOSE"], 14)

    # MACD histogram
    ema12 = out["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = out["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = macd - macd_signal

    # VOLATILITY / VOLUME
    out["volatility_20"] = out["ret_1d"].rolling(20).std()
    out["volume_ratio"] = out["VOLUME"] / (out["VOLUME"].rolling(20).mean() + 1e-8)

    # CURRENCY (CBR USD/RUB)
    if isinstance(usd, pd.Series) and not usd.empty:
        usd_df = usd.reset_index().rename(columns={"index": "date", "CBR_USD_RUB": "usd_rub"})
        usd_df["date"] = pd.to_datetime(usd_df["date"])
        out = pd.merge_asof(out.sort_values("date"), usd_df.sort_values("date"), on="date", direction="backward")
        out["usd_ret_5d"] = out["usd_rub"].pct_change(5)
    else:
        out["usd_ret_5d"] = 0.0

    # PRICE POSITION
    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_position"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-8)

    # Bollinger position
    bb_mid = out["CLOSE"].rolling(20).mean()
    bb_std = out["CLOSE"].rolling(20).std()
    out["bb_position"] = (out["CLOSE"] - bb_mid) / ((2 * bb_std) + 1e-8)

    out = out.dropna().reset_index(drop=True)
    return out


# ----------------------------
# OPTIMIZED MODEL - SMALLER
# ----------------------------

def build_optimized_model(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(CFG["SEQ_LEN"], n_features))

    x = tf.keras.layers.LSTM(
        32,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        recurrent_regularizer=tf.keras.regularizers.l2(5e-4),
        dropout=0.4,
        recurrent_dropout=0.3,
    )(inputs)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.LSTM(
        16,
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
        recurrent_regularizer=tf.keras.regularizers.l2(5e-4),
        dropout=0.4,
        recurrent_dropout=0.3,
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dense(
        12,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(5e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=CFG["LR"],
            beta_1=0.9,
            beta_2=0.999,
            clipnorm=1.0,
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_improved_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_sonnet_v3.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ----------------------------
# MAIN
# ----------------------------

@dataclass
class Dataset:
    df_feat: pd.DataFrame
    target: pd.Series
    fwd_ret: pd.Series


def load_or_build_v3_dataset(secid: str) -> Dataset:
    cache_file = _cache_path(f"simple_dataset_{secid}_v3.pkl")
    if os.path.exists(cache_file):
        print("Loading cached dataset...")
        with open(cache_file, "rb") as f:
            df_feat = pickle.load(f)
    else:
        print("Building simple stable dataset...")
        df_price = fetch_moex_history(secid, CFG["START"], CFG["END"])
        usd = fetch_cbr_usdrub(CFG["START"], CFG["END"])
        df_feat = add_stable_features(df_price, usd)
        with open(cache_file, "wb") as f:
            pickle.dump(df_feat, f)

    y = make_simple_target(df_feat["CLOSE"], int(CFG["TARGET_HORIZON_DAYS"]), float(CFG["TARGET_UP_THRESHOLD"]))
    fret = make_forward_return(df_feat["CLOSE"], int(CFG["TARGET_HORIZON_DAYS"]))
    return Dataset(df_feat=df_feat, target=y, fwd_ret=fret)


def make_sequences_with_dates(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    fwd_ret: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, ds, rs = [], [], [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
        ds.append(dates[i])
        rs.append(fwd_ret[i])
    return np.asarray(Xs), np.asarray(ys), np.asarray(ds), np.asarray(rs)


def main() -> float:
    print("SONNET MODEL v3.0 - simplified for stability")
    print("=" * 60)

    secid = CFG["TICKER"]
    data = load_or_build_v3_dataset(secid)
    df_feat = data.df_feat.copy()

    horizon = int(CFG["TARGET_HORIZON_DAYS"])
    df_feat = df_feat.iloc[:-horizon].reset_index(drop=True)
    y = data.target.iloc[:-horizon].astype(int).values
    fwd_ret = data.fwd_ret.iloc[:-horizon].astype(float).values

    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "log_ret",
        "price_vs_sma20",
        "price_vs_sma50",
        "trend_up",
        "rsi_14",
        "macd_histogram",
        "volatility_20",
        "volume_ratio",
        "usd_ret_5d",
        "price_position",
        "bb_position",
    ]
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    print(f"Stable features: {len(feature_cols)}")
    print(f"Target balance: {y.mean():.3%} positive / {(1 - y.mean()):.3%} negative")

    X = df_feat[feature_cols].values
    dates = pd.to_datetime(df_feat["date"]).values

    split = int(len(X) * float(CFG["TRAIN_SPLIT"]))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]
    d_train_raw, d_test_raw = dates[:split], dates[split:]
    r_train_raw, r_test_raw = fwd_ret[:split], fwd_ret[split:]

    val_frac = float(CFG["VAL_SPLIT"])
    val_cut = int(len(X_train_raw) * (1.0 - val_frac))
    X_tr_raw, X_val_raw = X_train_raw[:val_cut], X_train_raw[val_cut:]
    y_tr_raw, y_val_raw = y_train_raw[:val_cut], y_train_raw[val_cut:]
    d_tr_raw, d_val_raw = d_train_raw[:val_cut], d_train_raw[val_cut:]
    r_tr_raw, r_val_raw = r_train_raw[:val_cut], r_train_raw[val_cut:]

    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_tr, y_tr, d_tr, r_tr = make_sequences_with_dates(X_tr_scaled, y_tr_raw, d_tr_raw, r_tr_raw, int(CFG["SEQ_LEN"]))
    X_val, y_val, d_val, r_val = make_sequences_with_dates(X_val_scaled, y_val_raw, d_val_raw, r_val_raw, int(CFG["SEQ_LEN"]))
    X_test, y_test, d_test, r_test = make_sequences_with_dates(X_test_scaled, y_test_raw, d_test_raw, r_test_raw, int(CFG["SEQ_LEN"]))

    print(f"Sequences - Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    unique_classes = np.unique(y_tr)
    cw_vals = compute_class_weight("balanced", classes=unique_classes, y=y_tr)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    print("Class weights:", class_weight)

    model = build_optimized_model(X_tr.shape[-1])
    print(f"Model parameters: {model.count_params():,}")

    callbacks = get_improved_callbacks()

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,  # time-series safety
        verbose=2,
    )

    if os.path.exists("best_sonnet_v3.keras"):
        model = tf.keras.models.load_model("best_sonnet_v3.keras")

    # threshold tuning on validation
    prob_val = model.predict(X_val, verbose=0).reshape(-1)
    best_thr = 0.5
    best_acc = -1.0
    for thr in np.arange(0.30, 0.81, 0.05):
        pred_val = (prob_val >= thr).astype(int)
        acc_val = float(accuracy_score(y_val, pred_val))
        if acc_val > best_acc:
            best_acc = acc_val
            best_thr = float(thr)

    print(f"Best threshold (val): {best_thr:.2f} (val_acc={best_acc:.1%})")

    proba = model.predict(X_test, verbose=0).reshape(-1)
    pred = (proba >= best_thr).astype(int)

    acc = float(accuracy_score(y_test, pred))
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")

    print("\n" + "=" * 60)
    print("SONNET v3.0 RESULTS")
    print("=" * 60)

    print(f"Accuracy: {acc:.3%}")
    print(f"Balanced Acc: {bal_acc:.3%}")
    print(f"F1: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}" if np.isfinite(auc) else "AUC-ROC: n/a")
    print(f"MCC: {matthews_corrcoef(y_test, pred) if len(np.unique(pred)) > 1 else 0.0:.4f}")
    print(f"AP: {average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else float('nan'):.4f}")

    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

    model.save("sonnet_v3_final.keras")
    with open("sonnet_v3_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved: sonnet_v3_final.keras and sonnet_v3_scaler.pkl")

    return acc


if __name__ == "__main__":
    final_accuracy = main()
    print(f"FINAL SONNET v3.0 ACCURACY: {final_accuracy:.1%}")
