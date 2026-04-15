from __future__ import annotations

"""Multi‑ticker SONNET TCN (v0.1)

Один TCN обучается на нескольких акциях сразу (MOEX candles через moexalgo).
Цель: бинарная классификация — вырастет ли цена >= THR_MOVE за HORIZON дней.

Запуск (Colab):
  !pip -q install moexalgo pandas numpy scikit-learn tensorflow
  !python multi_ticker_tcn_sonnet.py

Вывод:
- GLOBAL TEST METRICS (AUC/PR‑AUC/MCC/BalAcc)
- PER‑TICKER METRICS (AUC/PR‑AUC)
- PROB SUMMARY TEST
- PER‑TICKER backtest (non-overlap long-only) по fwd_ret

Примечание:
- Данные разных тикеров не смешиваются внутри окна: окно строится только если весь seq_len
  принадлежит одному secid.
- Разбиение train/val/test делается по глобальной временной оси (по датам), а потом окна
  фильтруются по secid.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from moexalgo import Ticker
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight


# ==============================
# CONFIG
# ==============================
CFG: Dict[str, Any] = {
    "TICKERS": ["SBER", "GAZP", "LKOH", "YNDX"],
    "START": "2015-01-01",
    "END": None,
    "HORIZON": 5,
    "THR_MOVE": 0.02,
    "SEQ_LEN": 30,
    "TRAIN_SPLIT": 0.70,
    "VAL_SPLIT": 0.15,
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "LR": 1e-3,
    "SEED": 42,
    "FEE": 0.001,
}

np.random.seed(int(CFG["SEED"]))
tf.random.set_seed(int(CFG["SEED"]))


# ==============================
# DATA LOADING
# ==============================

def fetch_moex_candles(secid: str, start: str, end: str | None) -> pd.DataFrame:
    raw = Ticker(secid).candles(start=start, end=end, period="1D")
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    # moexalgo returns begin/end columns (datetime strings)
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    df = df.dropna(subset=["begin"]).drop_duplicates(subset=["begin"]).set_index("begin")

    # minimal required fields
    keep = ["close", "high", "low", "volume"]
    df = df[keep].sort_index()

    df = df.rename(
        columns={"close": "CLOSE", "high": "HIGH", "low": "LOW", "volume": "VOLUME"}
    )
    df["secid"] = str(secid)
    return df


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features_one(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Log-returns
    for lag in (1, 2, 3, 5):
        out[f"logret_{lag}"] = np.log(out["CLOSE"] / out["CLOSE"].shift(lag))

    # Trend
    out["sma_200"] = out["CLOSE"].rolling(200).mean()
    out["trend_up_200"] = (out["CLOSE"] > out["sma_200"]).astype(int)

    # Relative volume
    out["vol_rel"] = out["VOLUME"] / (out["VOLUME"].rolling(20).mean() + 1e-9)

    # RSI
    out["rsi_14"] = compute_rsi(out["CLOSE"], 14)

    # 20d price position
    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_pos_20"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-9)

    out = out.dropna().copy()
    return out


def make_target(close: pd.Series, horizon: int, thr: float) -> Tuple[pd.Series, pd.Series]:
    fwd = close.shift(-horizon)
    fwd_ret = (fwd - close) / close
    y = (fwd_ret >= thr).astype(int)
    return y, fwd_ret


@dataclass
class MultiDataset:
    X: np.ndarray
    y: np.ndarray
    fwd_ret: np.ndarray
    dates: np.ndarray
    secids: np.ndarray


def build_multi_ticker_dataset() -> Tuple[MultiDataset, List[str]]:
    rows: List[pd.DataFrame] = []
    for secid in CFG["TICKERS"]:
        print(f"Loading {secid}...")
        df = fetch_moex_candles(secid, str(CFG["START"]), CFG["END"])
        if df.empty:
            print(f"  -> empty, skip")
            continue

        df_feat = build_features_one(df)
        y, fwd_ret = make_target(df_feat["CLOSE"], int(CFG["HORIZON"]), float(CFG["THR_MOVE"]))

        h = int(CFG["HORIZON"])
        df_feat = df_feat.iloc[:-h]
        y = y.iloc[:-h]
        fwd_ret = fwd_ret.iloc[:-h]

        tmp = df_feat.copy()
        tmp["target"] = y.values
        tmp["fwd_ret"] = fwd_ret.values
        tmp["date"] = tmp.index
        rows.append(tmp.reset_index(drop=True))

    if not rows:
        raise RuntimeError("No tickers loaded. Check MOEX availability / tickers list.")

    full = pd.concat(rows, axis=0).sort_values(["date", "secid"]).reset_index(drop=True)

    feature_cols = [
        "trend_up_200",
        "logret_1",
        "logret_2",
        "logret_3",
        "logret_5",
        "vol_rel",
        "rsi_14",
        "price_pos_20",
    ]
    feature_cols = [c for c in feature_cols if c in full.columns]

    X = full[feature_cols].values
    y = full["target"].astype(int).values
    fwd_ret = full["fwd_ret"].astype(float).values
    dates = pd.to_datetime(full["date"]).values
    secids = full["secid"].astype(str).values

    print(
        f"Multi-ticker dataset: X={X.shape}, pos_rate={y.mean():.3%}, tickers={sorted(set(secids))}"
    )
    print(f"Features: {feature_cols}")
    return MultiDataset(X=X, y=y, fwd_ret=fwd_ret, dates=dates, secids=secids), feature_cols


# ==============================
# SEQUENCES
# ==============================

def make_sequences_with_meta(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    fwd_ret: np.ndarray,
    secids: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, ds, rs, ss = [], [], [], [], []
    for i in range(seq_len, len(X)):
        # Window must be within one ticker
        if not (secids[i - seq_len : i] == secids[i]).all():
            continue
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
        ds.append(dates[i])
        rs.append(fwd_ret[i])
        ss.append(secids[i])
    return (
        np.asarray(Xs),
        np.asarray(ys),
        np.asarray(ds),
        np.asarray(rs),
        np.asarray(ss),
    )


# ==============================
# TCN MODEL
# ==============================

def tcn_block(inputs: tf.Tensor, filters: int, kernel_size: int, dilations: List[int]) -> tf.Tensor:
    x = inputs
    for d in dilations:
        pad = (kernel_size - 1) * d
        x_padded = tf.pad(x, [[0, 0], [pad, 0], [0, 0]])
        x_conv = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            dilation_rate=d,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        )(x_padded)
        x_conv = tf.keras.layers.Dropout(0.3)(x_conv)
        x_conv = tf.keras.layers.LayerNormalization()(x_conv)

        if x.shape[-1] != filters:
            x = tf.keras.layers.Conv1D(filters, 1)(x)
        x = tf.keras.layers.Add()([x, x_conv])
    return x


def build_tcn_model(n_features: int, seq_len: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, n_features))
    x = tcn_block(inp, filters=32, kernel_size=3, dilations=[1, 2])
    x = tcn_block(x, filters=32, kernel_size=3, dilations=[4, 8])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(
        16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(CFG["LR"]), clipnorm=1.0),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=[
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )
    return model


# ==============================
# DIAGNOSTICS
# ==============================

def prob_summary(y_true: np.ndarray, prob: np.ndarray, name: str) -> None:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    print(f"\n=== PROB SUMMARY {name} ===")
    print(f"Size={len(y_true)}, Pos rate={y_true.mean():.3%}")
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, prob)
        ap = average_precision_score(y_true, prob)
        print(f"AUC-ROC={auc:.4f}, PR-AUC={ap:.4f}")
    else:
        print("AUC/PR-AUC n/a")


def per_ticker_metrics(secids: np.ndarray, y_true: np.ndarray, prob: np.ndarray, name: str) -> None:
    print(f"\n=== PER-TICKER METRICS {name} ===")
    secids = np.asarray(secids)
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    for sec in np.unique(secids):
        mask = secids == sec
        if int(mask.sum()) < 50:
            continue
        yt = y_true[mask]
        pr = prob[mask]
        auc = roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan
        ap = average_precision_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan
        print(f"{sec}: n={mask.sum()}, pos={yt.mean():.3%}, AUC={auc:.4f}, PR-AUC={ap:.4f}")


def backtest_nonoverlap(prob: np.ndarray, fwd_ret: np.ndarray, thr: float, horizon: int, fee: float, name: str) -> None:
    """Простой backtest по fwd_ret (уже horizon-рет):

    - long-only
    - non-overlap: после сделки пропускаем horizon дней
    - комиссия: 2*fee

    Важно: fwd_ret здесь относится к entry-дню (t -> t+horizon).
    """
    prob = np.asarray(prob).astype(float)
    fwd_ret = np.asarray(fwd_ret).astype(float)

    idx = 0
    trades = []
    n = len(prob)
    while idx < n:
        if prob[idx] >= thr:
            net = float(fwd_ret[idx] - 2.0 * fee)
            trades.append(net)
            idx += int(horizon) + 1
        else:
            idx += 1

    print(f"\n=== BACKTEST {name} thr={thr:.2f} ===")
    if not trades:
        print("No trades.")
        return

    trades = np.asarray(trades, dtype=float)
    total = float((1.0 + trades).prod() - 1.0)
    win = float((trades > 0).mean())
    print(f"Trades={len(trades)}, total_net_ret={total:.2%}, winrate={win:.1%}")


# ==============================
# MAIN
# ==============================

def main() -> None:
    data, _feature_cols = build_multi_ticker_dataset()
    X, y, fwd_ret, dates, secids = data.X, data.y, data.fwd_ret, data.dates, data.secids

    # global chronological order
    order = np.argsort(dates)
    X, y, fwd_ret, dates, secids = X[order], y[order], fwd_ret[order], dates[order], secids[order]

    n = len(X)
    split_train = int(n * float(CFG["TRAIN_SPLIT"]))
    split_val = int(n * (float(CFG["TRAIN_SPLIT"]) + float(CFG["VAL_SPLIT"])))

    X_train_raw, X_val_raw, X_test_raw = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train_raw, y_val_raw, y_test_raw = y[:split_train], y[split_train:split_val], y[split_val:]
    r_train_raw, r_val_raw, r_test_raw = (
        fwd_ret[:split_train],
        fwd_ret[split_train:split_val],
        fwd_ret[split_val:],
    )
    s_train_raw, s_val_raw, s_test_raw = (
        secids[:split_train],
        secids[split_train:split_val],
        secids[split_val:],
    )
    d_train_raw, d_val_raw, d_test_raw = (
        dates[:split_train],
        dates[split_train:split_val],
        dates[split_val:],
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    X_tr, y_tr, d_tr, r_tr, s_tr = make_sequences_with_meta(
        X_train_scaled, y_train_raw, d_train_raw, r_train_raw, s_train_raw, int(CFG["SEQ_LEN"])
    )
    X_va, y_va, d_va, r_va, s_va = make_sequences_with_meta(
        X_val_scaled, y_val_raw, d_val_raw, r_val_raw, s_val_raw, int(CFG["SEQ_LEN"])
    )
    X_te, y_te, d_te, r_te, s_te = make_sequences_with_meta(
        X_test_scaled, y_test_raw, d_test_raw, r_test_raw, s_test_raw, int(CFG["SEQ_LEN"])
    )

    print(f"Seq shapes: Train={X_tr.shape}, Val={X_va.shape}, Test={X_te.shape}")

    # class weights on train windows
    cls = np.unique(y_tr)
    cw_vals = compute_class_weight("balanced", classes=cls, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(cls, cw_vals)}
    print("Class weights:", class_weight)

    model = build_tcn_model(int(X_tr.shape[-1]), int(X_tr.shape[1]))
    print("Params:", model.count_params())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_roc", mode="max", patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_roc", mode="max", factor=0.5, patience=5, min_lr=1e-5
        ),
    ]

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        class_weight=class_weight,
        shuffle=False,
        callbacks=callbacks,
        verbose=2,
    )

    prob_val = model.predict(X_va, verbose=0).reshape(-1)
    prob_test = model.predict(X_te, verbose=0).reshape(-1)

    # threshold sweep on VAL by F1
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.30, 0.81, 0.02):
        pred_val = (prob_val >= thr).astype(int)
        f1v = f1_score(y_va, pred_val, zero_division=0)
        if f1v > best_f1:
            best_f1 = float(f1v)
            best_thr = float(thr)
    print(f"Best threshold on VAL (F1={best_f1:.3f}): {best_thr:.2f}")

    pred_test = (prob_test >= best_thr).astype(int)
    auc_test = roc_auc_score(y_te, prob_test) if len(np.unique(y_te)) > 1 else np.nan
    ap_test = average_precision_score(y_te, prob_test) if len(np.unique(y_te)) > 1 else np.nan
    mcc_test = matthews_corrcoef(y_te, pred_test) if len(np.unique(pred_test)) > 1 else 0.0
    bal_test = balanced_accuracy_score(y_te, pred_test)

    print("\n=== GLOBAL TEST METRICS ===")
    print(f"AUC-ROC={auc_test:.4f}, PR-AUC={ap_test:.4f}")
    print(f"Balanced Acc={bal_test:.3f}, MCC={mcc_test:.4f}")

    prob_summary(y_te, prob_test, name="TEST GLOBAL")
    per_ticker_metrics(s_te, y_te, prob_test, name="TEST")

    # per-ticker backtest on TEST
    for sec in np.unique(s_te):
        mask = s_te == sec
        if int(mask.sum()) < 100:
            continue
        backtest_nonoverlap(
            prob=prob_test[mask],
            fwd_ret=r_te[mask],
            thr=best_thr,
            horizon=int(CFG["HORIZON"]),
            fee=float(CFG["FEE"]),
            name=f"TEST {sec}",
        )


if __name__ == "__main__":
    # Disable TF excessive logs if desired
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
