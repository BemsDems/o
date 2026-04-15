"""SONNET TCN v4.1 — LSTM → TCN (Temporal Convolutional Network)

Преимущества TCN: causal convolutions, dilation, стабильность, меньше переобучения.

Colab deps:
  !pip -q install requests pandas numpy scikit-learn tensorflow lxml html5lib

Notes:
- Датасет/фичи оставлены как в v3 (stable features), модель — TCN.
- Для временных рядов: shuffle=False.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# CONFIG - TCN optimized
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
    return str(CFG["CACHE_DIR"])


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
    except Exception as e:
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(
            f"HTTP error for {url} status={r.status_code} ct={ct} preview={preview!r}"
        ) from e

    try:
        return r.json()
    except Exception as e:
        preview = (r.text or "")[:300].replace("\n", " ")
        raise RuntimeError(
            f"Non-JSON response for {url} status={r.status_code} ct={ct} preview={preview!r}"
        ) from e


# ----------------------------
# DATA LOADING
# ----------------------------
def fetch_moex_history(secid: str, start: str, end: Optional[str]) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/history/engines/stock/markets/shares/"
        f"securities/{secid}.json"
    )
    all_rows: list[list[Any]] = []
    start_pos = 0
    cols: list[str] = []

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
    rows: list[tuple[pd.Timestamp, float]] = []
    for rec in root.findall("Record"):
        d = rec.attrib.get("Date")
        v = rec.findtext("Value")
        if d and v:
            rows.append((pd.to_datetime(d, dayfirst=True), float(v.replace(",", "."))))

    s = pd.Series(dict(rows)).sort_index()
    s.name = "CBR_USD_RUB"
    return s


# ----------------------------
# FEATURES + TARGET
# ----------------------------
def make_simple_target(close: pd.Series, horizon: int, thr: float) -> pd.Series:
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
    out = df.copy()

    out["ret_1d"] = out["CLOSE"].pct_change(1)
    out["ret_5d"] = out["CLOSE"].pct_change(5)
    out["ret_10d"] = out["CLOSE"].pct_change(10)
    out["ret_20d"] = out["CLOSE"].pct_change(20)
    out["log_ret"] = np.log(out["CLOSE"] / out["CLOSE"].shift(1))

    sma20 = out["CLOSE"].rolling(20).mean()
    sma50 = out["CLOSE"].rolling(50).mean()
    sma200 = out["CLOSE"].rolling(200).mean()

    out["price_vs_sma20"] = out["CLOSE"] / (sma20 + 1e-12) - 1
    out["price_vs_sma50"] = out["CLOSE"] / (sma50 + 1e-12) - 1
    out["trend_up"] = (out["CLOSE"] > sma200).astype(int)

    out["rsi_14"] = _rsi(out["CLOSE"], 14)

    ema12 = out["CLOSE"].ewm(span=12, adjust=False).mean()
    ema26 = out["CLOSE"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_histogram"] = macd - macd_signal

    out["volatility_20"] = out["ret_1d"].rolling(20).std()
    out["volume_ratio"] = out["VOLUME"] / (out["VOLUME"].rolling(20).mean() + 1e-8)

    if isinstance(usd, pd.Series) and not usd.empty:
        usd_df = usd.reset_index().rename(columns={"index": "date", "CBR_USD_RUB": "usd_rub"})
        usd_df["date"] = pd.to_datetime(usd_df["date"])
        out = pd.merge_asof(
            out.sort_values("date"), usd_df.sort_values("date"), on="date", direction="backward"
        )
        out["usd_ret_5d"] = out["usd_rub"].pct_change(5)
    else:
        out["usd_ret_5d"] = 0.0

    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_position"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-8)

    bb_mid = out["CLOSE"].rolling(20).mean()
    bb_std = out["CLOSE"].rolling(20).std()
    out["bb_position"] = (out["CLOSE"] - bb_mid) / ((2 * bb_std) + 1e-8)

    out = out.dropna().reset_index(drop=True)
    return out


# ----------------------------
# TCN MODEL
# ----------------------------
def tcn_residual_block(
    inputs: tf.Tensor,
    *,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout: float = 0.25,
    l2: float = 1e-4,
) -> tf.Tensor:
    """TCN residual block: (causal Conv → LN → Dropout) × 2 + residual."""

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
    )(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    res = inputs
    if res.shape[-1] != filters:
        res = tf.keras.layers.Conv1D(filters, 1, padding="same")(res)

    out = tf.keras.layers.Add()([res, x])
    return out


def build_tcn_model(n_features: int, seq_len: int) -> tf.keras.Model:
    """TCN: stack dilated residual blocks + global pooling."""

    inputs = tf.keras.Input(shape=(seq_len, n_features))

    x = inputs
    for dr in (1, 2, 4, 8):
        x = tcn_residual_block(x, filters=32, kernel_size=3, dilation_rate=dr)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(
        16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(CFG["LR"]), clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_tcn_callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_roc",
            mode="max",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_roc",
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="best_tcn_sonnet_v4_1.keras",
            monitor="val_auc_roc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ----------------------------
# MAIN
# ----------------------------
# ----------------------------
# DIAGNOSTICS: PROBS / DECILES / BACKTEST
# ----------------------------

def prob_summary_block(y_true: np.ndarray, prob: np.ndarray, name: str = "TEST") -> None:
    """Краткая сводка по вероятностям."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)

    pos_rate = float(y_true.mean()) if len(y_true) else float("nan")
    print(f"\n=== PROB SUMMARY {name} ===")
    print(f"Size={len(y_true)}, Pos rate={pos_rate:.3%}")

    print(f"Prob mean={prob.mean():.4f}, std={prob.std():.4f}")
    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qv = np.quantile(prob, qs)
    for q, v in zip(qs, qv):
        print(f" q{int(q*100):02d}: {v:.4f}")

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, prob)
        ap = average_precision_score(y_true, prob)
        print(f"AUC-ROC={auc:.4f}, PR-AUC={ap:.4f}")
    else:
        print("AUC-ROC/PR-AUC: n/a (only one class)")


def decile_report(y_true: np.ndarray, prob: np.ndarray, fwd_ret: np.ndarray, name: str = "TEST") -> None:
    """Децильный отчёт: по децилям prob смотрим средний future_ret."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    fwd_ret = np.asarray(fwd_ret).astype(float)

    df = pd.DataFrame({"y": y_true, "p": prob, "ret": fwd_ret}).dropna()
    if df.empty:
        print(f"\n=== DECILE REPORT {name} ===")
        print("No data")
        return

    df = df.sort_values("p")
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")

    print(f"\n=== DECILE REPORT {name} ===")
    print("decile | count | pos_rate | mean_p | mean_ret")
    for d in sorted(df["decile"].unique()):
        sub = df[df["decile"] == d]
        print(
            f"{int(d):2d} | "
            f"{len(sub):5d} | "
            f"{sub['y'].mean():7.3f} | "
            f"{sub['p'].mean():7.4f} | "
            f"{sub['ret'].mean():8.4f}"
        )


def backtest_nonoverlap_long_only(
    prob: np.ndarray,
    dates_signal: np.ndarray,
    close_by_date: pd.Series,
    thr: float,
    horizon: int,
    fee: float = 0.001,
) -> None:
    """Backtest: long-only, non-overlap, close-to-close, fee applied as 2*fee.

    close_by_date: pd.Series indexed by trading dates (same calendar as candles), values = close.
    For each signal date d0 we enter at close[d0] and exit at close[d0+horizon trading days].
    """
    prob = np.asarray(prob).astype(float)
    dates = pd.to_datetime(dates_signal)

    close_by_date = close_by_date.copy()
    close_by_date.index = pd.to_datetime(close_by_date.index)

    # Ensure unique index to avoid get_loc returning slice
    close_by_date = close_by_date[~close_by_date.index.duplicated(keep='last')]

    trades = []
    i = 0
    while i < len(prob):
        if prob[i] >= thr:
            d0 = dates[i]
            if d0 not in close_by_date.index:
                i += 1
                continue
            loc0_arr = close_by_date.index.get_indexer([d0])
            loc0 = int(loc0_arr[0])
            if loc0 < 0:
                i += 1
                continue
            loc1 = loc0 + horizon
            if loc1 >= len(close_by_date.index):
                break

            entry = float(close_by_date.iloc[loc0])
            exitp = float(close_by_date.iloc[loc1])
            gross_ret = exitp / entry - 1.0
            net_ret = gross_ret - 2.0 * fee

            trades.append(
                {
                    "entry_date": close_by_date.index[loc0],
                    "exit_date": close_by_date.index[loc1],
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                }
            )
            i += horizon  # non-overlap
        else:
            i += 1

    print("\n=== BACKTEST non-overlap LONG ONLY ===")
    print(f"Threshold={thr:.3f}, horizon={horizon}, fee(one-way)={fee:.4f}")

    if not trades:
        print("No trades at this threshold.")
        return

    bt = pd.DataFrame(trades)
    cum_net = float((1.0 + bt["net_ret"]).prod() - 1.0)
    winrate = float((bt["net_ret"] > 0).mean())
    avg_tr = float(bt["net_ret"].mean())

    # Buy&Hold on the same segment (from first entry to last exit)
    d_start = pd.to_datetime(bt["entry_date"].iloc[0])
    d_end = pd.to_datetime(bt["exit_date"].iloc[-1])
    loc_s = close_by_date.index.get_loc(d_start)
    loc_e = close_by_date.index.get_loc(d_end)
    bh_ret = float(close_by_date.iloc[loc_e] / close_by_date.iloc[loc_s] - 1.0)

    print(f"trades={len(bt)}")
    print(f"Total net return: {cum_net:.2%}")
    print(f"Avg trade net: {avg_tr:.3%}")
    print(f"Winrate: {winrate:.1%}")
    print(f"Buy&Hold same period: {bh_ret:.2%}")

@dataclass
class Dataset:
    df_feat: pd.DataFrame
    target: pd.Series
    fwd_ret: pd.Series


def load_or_build_v3_dataset(secid: str) -> Dataset:
    """v3 features dataset (cached) used by v4 TCN."""

    cache_file = _cache_path(f"simple_dataset_{secid}_v3.pkl")
    if os.path.exists(cache_file):
        print("Loading cached dataset...")
        with open(cache_file, "rb") as f:
            df_feat = pickle.load(f)
    else:
        print("Building simple stable dataset...")
        df_price = fetch_moex_history(secid, str(CFG["START"]), CFG["END"])
        usd = fetch_cbr_usdrub(str(CFG["START"]), CFG["END"])
        df_feat = add_stable_features(df_price, usd)
        with open(cache_file, "wb") as f:
            pickle.dump(df_feat, f)

    y = make_simple_target(
        df_feat["CLOSE"], int(CFG["TARGET_HORIZON_DAYS"]), float(CFG["TARGET_UP_THRESHOLD"])
    )
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
    print("SONNET TCN v4.1 - Temporal Convolutional Network")
    print("=" * 60)

    secid = str(CFG["TICKER"])
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
    feature_cols = [c for c in feature_cols if c in df_feat.columns and c != "usd_ret_5d"]

    print(f"Features (no usd_ret_5d): {len(feature_cols)}", feature_cols)
    print(f"Target balance: {y.mean():.3%} positive")

    X = df_feat[feature_cols].fillna(0).values
    dates = pd.to_datetime(df_feat["date"]).values

    # Time splits
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

    # RobustScaler
    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Sequences: [samples, timesteps, features]
    seq_len = int(CFG["SEQ_LEN"])
    X_tr, y_tr, d_tr, r_tr = make_sequences_with_dates(
        X_tr_scaled, y_tr_raw, d_tr_raw, r_tr_raw, seq_len
    )
    X_val, y_val, d_val, r_val = make_sequences_with_dates(
        X_val_scaled, y_val_raw, d_val_raw, r_val_raw, seq_len
    )
    X_test, y_test, d_test, r_test = make_sequences_with_dates(
        X_test_scaled, y_test_raw, d_test_raw, r_test_raw, seq_len
    )

    print(f"TCN sequences: Train={X_tr.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Class weights
    unique_classes = np.unique(y_tr)
    cw_vals = compute_class_weight("balanced", classes=unique_classes, y=y_tr)
    class_weight = {int(cls): float(weight) for cls, weight in zip(unique_classes, cw_vals)}
    print("Class weights:", class_weight)

    model = build_tcn_model(X_tr.shape[-1], X_tr.shape[1])
    print(f"TCN params: {model.count_params():,}")

    callbacks = get_tcn_callbacks()

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )

    if os.path.exists("best_tcn_sonnet_v4_1.keras"):
        model = tf.keras.models.load_model("best_tcn_sonnet_v4_1.keras")

    # Threshold tuning on VAL by F1 (AUC does not depend on threshold)
    prob_val = model.predict(X_val, verbose=0).reshape(-1)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.10, 0.91, 0.02):
        pred_val = (prob_val >= thr).astype(int)
        f1v = f1_score(y_val, pred_val, zero_division=0)
        if f1v > best_f1:
            best_f1 = float(f1v)
            best_thr = float(thr)

    auc_val = roc_auc_score(y_val, prob_val) if len(np.unique(y_val)) > 1 else float("nan")
    print(f"Best threshold (val F1={best_f1:.3f}): {best_thr:.2f}")

    # TEST results
    proba = model.predict(X_test, verbose=0).reshape(-1)
    pred = (proba >= best_thr).astype(int)

    acc = accuracy_score(y_test, pred)
    bal_acc = balanced_accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")
    mcc = matthews_corrcoef(y_test, pred) if len(np.unique(pred)) > 1 else 0.0
    ap = average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")

    print("\n" + "=" * 60)
    print("SONNET TCN v4.1 RESULTS")
    print("=" * 60)
    print(f"Accuracy: {acc:.3%}")
    print(f"Balanced Acc: {bal_acc:.3%}")
    print(f"F1: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"PR-AUC: {ap:.4f}")
    print("\nConfusion:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))

    # --- DIAGNOSTICS ---
    prob_train = model.predict(X_tr, verbose=0).reshape(-1)
    prob_val_diag = model.predict(X_val, verbose=0).reshape(-1)
    prob_test = proba

    prob_summary_block(y_tr, prob_train, name="TRAIN")
    prob_summary_block(y_val, prob_val_diag, name="VAL")
    prob_summary_block(y_test, prob_test, name="TEST")

    decile_report(y_test, prob_test, r_test, name="TEST")

    close_by_date = pd.Series(df_feat["CLOSE"].values, index=pd.to_datetime(df_feat["date"]))
    backtest_nonoverlap_long_only(
        prob=prob_test,
        dates_signal=d_test,
        close_by_date=close_by_date,
        thr=float(best_thr),
        horizon=int(CFG["TARGET_HORIZON_DAYS"]),
        fee=float(CFG["FEE"]),
    )


    model.save("tcn_sonnet_v4_1_final.keras")
    with open("tcn_sonnet_v4_1_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved: tcn_sonnet_v4_1_final.keras + tcn_sonnet_v4_1_scaler.pkl")
    return float(acc)


if __name__ == "__main__":
    final_accuracy = main()
    print(f"FINAL TCN v4.1 ACCURACY: {final_accuracy:.1%}")
