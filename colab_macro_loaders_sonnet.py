# Colab-ready: MOEX/CBR feature engineering + LSTM classifier (Sonnet ENHANCED v2.0)
# NOTE: This is an experimental, feature-rich variant aimed at improving
# generalization under regime drift. It does NOT guarantee any target accuracy.
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
# ENHANCED CONFIG
# ----------------------------
CFG: Dict[str, Any] = {
    "TICKER": "SBER",
    "START": "2015-01-01",
    "END": None,  # None => today

    # target
    "TARGET_HORIZON_DAYS": 7,
    "TARGET_UP_THRESHOLD": 0.025,

    # windows
    "SEQ_LEN": 40,

    # splits (time-based)
    "TRAIN_SPLIT": 0.75,
    "VAL_SPLIT": 0.15,  # fraction of TRAIN part (time-based)

    # training
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "LR": 5e-4,

    "SEED": 42,
    "CACHE_DIR": "cache",

    # evaluation
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
    """GET url and decode JSON with better error messages."""

    r = requests.get(url, params=params, timeout=timeout)
    ct = (r.headers.get("Content-Type") or "").lower()
    try:
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(
            f"HTTP error for {url} status={r.status_code} ct={ct} preview={preview!r}"
        ) from e

    try:
        return r.json()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(
            f"Non-JSON response for {url} status={r.status_code} ct={ct} preview={preview!r}"
        ) from e


# ----------------------------
# DATA LOADING
# ----------------------------

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
        j = _get_json(url, params=params, timeout=30)
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
    """CBR XML dynamic for USD (R01235)."""
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
# TARGET + FEATURES
# ----------------------------

def make_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    fwd = close.shift(-horizon)
    return (fwd - close) / close


def make_smart_target(close: pd.Series, volume: pd.Series, horizon: int, thr: float) -> pd.Series:
    """Target with a simple volume confirmation and risk-adjusted outlier mode.

    This is still best-effort: it can improve label quality sometimes, but can
    also introduce instability if volume series is noisy.
    """

    fwd_ret = make_forward_return(close, horizon)

    vol_ma = volume.rolling(20).mean()
    future_vol = volume.shift(-horizon)
    vol_confirmation = future_vol > vol_ma

    # risk-adjusted return proxy (divide by rolling realized vol)
    ret_vol = close.pct_change().rolling(20).std()
    future_price_vol = ret_vol.shift(-horizon)
    risk_adj_return = fwd_ret / (future_price_vol + 1e-8)

    strong_move = (fwd_ret >= thr) & vol_confirmation
    exceptional_move = risk_adj_return > risk_adj_return.quantile(0.85)

    return (strong_move | exceptional_move).astype(int)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_enhanced_features(df: pd.DataFrame, usd: pd.Series) -> pd.DataFrame:
    """Expanded feature set (technical + FX context)."""

    out = df.copy()

    # BASIC RETURNS
    out["ret_1d"] = out["CLOSE"].pct_change(1)
    out["ret_3d"] = out["CLOSE"].pct_change(3)
    out["ret_5d"] = out["CLOSE"].pct_change(5)
    out["ret_10d"] = out["CLOSE"].pct_change(10)
    out["ret_20d"] = out["CLOSE"].pct_change(20)
    out["log_ret"] = np.log(out["CLOSE"] / out["CLOSE"].shift(1))

    # MOMENTUM
    out["rsi_14"] = _rsi(out["CLOSE"], 14)
    out["rsi_30"] = _rsi(out["CLOSE"], 30)
    out["rsi_divergence"] = out["rsi_14"] - out["rsi_30"]

    # MACD
    ema12 = out["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = out["CLOSE"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = out["macd"] - out["macd_signal"]
    out["macd_cross"] = (out["macd"] > out["macd_signal"]).astype(int)

    # TREND
    out["sma_5"] = out["CLOSE"].rolling(5).mean()
    out["sma_10"] = out["CLOSE"].rolling(10).mean()
    out["sma_20"] = out["CLOSE"].rolling(20).mean()
    out["sma_50"] = out["CLOSE"].rolling(50).mean()
    out["sma_200"] = out["CLOSE"].rolling(200).mean()

    out["price_vs_sma20"] = out["CLOSE"] / (out["sma_20"] + 1e-12) - 1
    out["price_vs_sma50"] = out["CLOSE"] / (out["sma_50"] + 1e-12) - 1
    out["sma_slope_20"] = (out["sma_20"] - out["sma_20"].shift(5)) / (out["sma_20"].shift(5) + 1e-12)

    out["trend_strength"] = (
        (out["CLOSE"] > out["sma_5"]).astype(int)
        + (out["CLOSE"] > out["sma_10"]).astype(int)
        + (out["CLOSE"] > out["sma_20"]).astype(int)
        + (out["CLOSE"] > out["sma_50"]).astype(int)
    ) / 4.0

    # VOLATILITY
    out["volatility_5"] = out["ret_1d"].rolling(5).std()
    out["volatility_20"] = out["ret_1d"].rolling(20).std()
    out["volatility_60"] = out["ret_1d"].rolling(60).std()
    out["vol_regime"] = (out["volatility_20"] > out["volatility_20"].rolling(60).quantile(0.7)).astype(int)

    # BOLLINGER
    bb_middle = out["CLOSE"].rolling(20).mean()
    bb_std = out["CLOSE"].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    out["bb_position"] = (out["CLOSE"] - bb_lower) / ((bb_upper - bb_lower) + 1e-8)
    out["bb_squeeze"] = (bb_upper - bb_lower) / (bb_middle + 1e-12)

    # VOLUME
    out["volume_sma"] = out["VOLUME"].rolling(20).mean()
    out["volume_ratio"] = out["VOLUME"] / (out["volume_sma"] + 1e-8)
    out["volume_spike"] = (out["volume_ratio"] > 2.0).astype(int)
    out["pv_trend"] = (out["ret_1d"] * np.log(out["volume_ratio"] + 1)).rolling(5).mean()

    # SUPPORT/RESISTANCE proxy
    out["high_20"] = out["HIGH"].rolling(20).max()
    out["low_20"] = out["LOW"].rolling(20).min()
    out["price_position"] = (out["CLOSE"] - out["low_20"]) / ((out["high_20"] - out["low_20"]) + 1e-8)

    # MEAN REVERSION
    out["z_score_20"] = (out["CLOSE"] - out["sma_20"]) / (out["CLOSE"].rolling(20).std() + 1e-8)
    out["mean_reversion_signal"] = (out["z_score_20"].abs() > 2.0).astype(int)

    # MOMENTUM confirmation
    out["momentum_5"] = out["ret_5d"] / (out["volatility_20"] + 1e-8)
    out["momentum_consistency"] = (out["ret_1d"] > 0).rolling(5).mean()

    # CURRENCY effects (CBR USD/RUB)
    if isinstance(usd, pd.Series) and not usd.empty:
        usd_df = usd.reset_index().rename(columns={"index": "date", "CBR_USD_RUB": "usd_rub"})
        usd_df["date"] = pd.to_datetime(usd_df["date"])
        out = pd.merge_asof(out.sort_values("date"), usd_df.sort_values("date"), on="date", direction="backward")
        out["usd_ret_1d"] = out["usd_rub"].pct_change(1)
        out["usd_ret_5d"] = out["usd_rub"].pct_change(5)
        out["currency_pressure"] = -(out["usd_ret_5d"])
    else:
        out["usd_ret_1d"] = 0.0
        out["usd_ret_5d"] = 0.0
        out["currency_pressure"] = 0.0

    # MARKET microstructure proxies
    out["bid_ask_proxy"] = (out["HIGH"] - out["LOW"]) / (out["CLOSE"] + 1e-12)
    true_range = np.maximum(
        out["HIGH"] - out["LOW"],
        np.maximum(
            (out["HIGH"] - out["CLOSE"].shift(1)).abs(),
            (out["LOW"] - out["CLOSE"].shift(1)).abs(),
        ),
    )
    atr = true_range.rolling(14).mean()
    out["atr_ratio"] = true_range / (atr + 1e-8)

    # cleanup intermediates
    out = out.drop(
        columns=[
            "sma_5",
            "sma_10",
            "volume_sma",
            "high_20",
            "low_20",
        ],
        errors="ignore",
    )

    out = out.dropna().reset_index(drop=True)
    return out


# ----------------------------
# SEQUENCES
# ----------------------------

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


# ----------------------------
# MODEL
# ----------------------------

def build_enhanced_model(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(CFG["SEQ_LEN"], n_features))

    x = tf.keras.layers.LSTM(
        48,
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        dropout=0.3,
        recurrent_dropout=0.2,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LSTM(
        24,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        dropout=0.3,
        recurrent_dropout=0.2,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        8,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

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
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_enhanced_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_roc",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_roc",
            mode="max",
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_sonnet_model.keras",
            monitor="val_auc_roc",
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


def load_or_build_enhanced_dataset(secid: str) -> Dataset:
    cache_file = _cache_path(f"enhanced_dataset_{secid}_v2.pkl")
    if os.path.exists(cache_file):
        print("Loading cached dataset...")
        with open(cache_file, "rb") as f:
            df_feat = pickle.load(f)
    else:
        print("Building new enhanced dataset...")
        df_price = fetch_moex_history(secid, CFG["START"], CFG["END"])
        usd = fetch_cbr_usdrub(CFG["START"], CFG["END"])
        df_feat = add_enhanced_features(df_price, usd)
        with open(cache_file, "wb") as f:
            pickle.dump(df_feat, f)

    y = make_smart_target(
        df_feat["CLOSE"],
        df_feat["VOLUME"],
        int(CFG["TARGET_HORIZON_DAYS"]),
        float(CFG["TARGET_UP_THRESHOLD"]),
    )
    fret = make_forward_return(df_feat["CLOSE"], int(CFG["TARGET_HORIZON_DAYS"]))
    return Dataset(df_feat=df_feat, target=y, fwd_ret=fret)


def main() -> float:
    print("ENHANCED SONNET MODEL v2.0")
    print("=" * 60)

    secid = CFG["TICKER"]
    data = load_or_build_enhanced_dataset(secid)
    df_feat = data.df_feat.copy()

    horizon = int(CFG["TARGET_HORIZON_DAYS"])

    # Align: remove last horizon rows (target/returns are NaN there)
    df_feat = df_feat.iloc[:-horizon].reset_index(drop=True)
    y = data.target.iloc[:-horizon].astype(int).values
    fwd_ret = data.fwd_ret.iloc[:-horizon].astype(float).values

    # feature cols: exclude raw price columns and merge helpers
    exclude = {"date", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE", "usd_rub"}
    feature_cols = [c for c in df_feat.columns if c not in exclude]

    print(f"Features used: {len(feature_cols)}")
    print(f"Target balance: pos={y.mean():.3%} neg={(1 - y.mean()):.3%}")

    X = df_feat[feature_cols].values
    dates = pd.to_datetime(df_feat["date"]).values

    split = int(len(X) * float(CFG["TRAIN_SPLIT"]))
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]
    d_train_raw, d_test_raw = dates[:split], dates[split:]
    r_train_raw, r_test_raw = fwd_ret[:split], fwd_ret[split:]

    # Train/Val time split inside train
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

    X_tr, y_tr, d_tr, r_tr = make_sequences_with_dates(
        X_tr_scaled,
        y_tr_raw,
        d_tr_raw,
        r_tr_raw,
        int(CFG["SEQ_LEN"]),
    )
    X_val, y_val, d_val, r_val = make_sequences_with_dates(
        X_val_scaled,
        y_val_raw,
        d_val_raw,
        r_val_raw,
        int(CFG["SEQ_LEN"]),
    )
    X_test, y_test, d_test, r_test = make_sequences_with_dates(
        X_test_scaled,
        y_test_raw,
        d_test_raw,
        r_test_raw,
        int(CFG["SEQ_LEN"]),
    )

    print(f"Sequence shapes: train={X_tr.shape}, val={X_val.shape}, test={X_test.shape}")

    unique_classes = np.unique(y_tr)
    cw_vals = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(unique_classes, cw_vals)}
    print("class_weight:", class_weight)

    model = build_enhanced_model(X_tr.shape[-1])
    model.summary()

    callbacks = get_enhanced_callbacks()

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,  # time-series safety
        verbose=1,
    )

    # load best weights if checkpoint exists
    if os.path.exists("best_sonnet_model.keras"):
        model = tf.keras.models.load_model("best_sonnet_model.keras")

    proba = model.predict(X_test, verbose=0).reshape(-1)
    pred = (proba >= float(CFG["BEST_THR"])).astype(int)

    acc = float(accuracy_score(y_test, pred))
    bal_acc = float(balanced_accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")

    print("\n" + "=" * 60)
    print("ENHANCED SONNET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {acc:.3%}")
    print(f"Balanced Accuracy: {bal_acc:.3%}")
    print(f"F1: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}" if np.isfinite(auc) else "AUC-ROC: n/a")
    print(f"MCC: {matthews_corrcoef(y_test, pred) if len(np.unique(pred)) > 1 else 0.0:.4f}")
    print(f"AP: {average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else float('nan'):.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

    model.save("enhanced_sonnet_final.keras")
    with open("enhanced_sonnet_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nSaved: enhanced_sonnet_final.keras and enhanced_sonnet_scaler.pkl")

    # lightweight train dynamics summary
    if "val_auc_roc" in history.history:
        best_val = max(history.history["val_auc_roc"])
        print(f"Best val AUC-ROC: {best_val:.4f}")

    return acc


if __name__ == "__main__":
    final_accuracy = main()
    print(f"\nFINAL ACCURACY: {final_accuracy:.3%}")
