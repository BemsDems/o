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

FIX (2026‑04):
- moexalgo требует, чтобы оба параметра диапазона дат были заданы.
  Если CFG["END"] = None, подставляем сегодняшнюю дату.
"""

import os
from dataclasses import dataclass
from datetime import datetime
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
    "END": datetime.now().strftime("%Y-%m-%d"),
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

def _resolve_end_date(end: str | None) -> str:
    """moexalgo не принимает None для конца диапазона."""
    if end is None:
        return datetime.now().strftime("%Y-%m-%d")
    return str(end)


def fetch_moex_candles(secid: str, start: str, end: str | None) -> pd.DataFrame:
    end_resolved = _resolve_end_date(end)

    # В разных версиях moexalgo встречаются разные имена параметров,
    # но start/end поддерживаются широко; главное — не передавать None.
    raw = Ticker(secid).candles(start=str(start), end=str(end_resolved), period="1D")
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    # moexalgo returns begin/end columns (datetime strings)
    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    df = df.dropna(subset=["begin"]).drop_duplicates(subset=["begin"]).set_index("begin")

    # minimal required fields
    keep = ["close", "high", "low", "volume"]
    df = df[keep].sort_index()

    df = df.rename(columns={"close": "CLOSE", "high": "HIGH", "low": "LOW", "volume": "VOLUME"})
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

    print(f"Multi-ticker dataset: X={X.shape}, pos_rate={y.mean():.3%}, tickers={sorted(set(secids))}")
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
# MODEL
# ==============================


def build_tcn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    # Simple dilated causal CNN (TCN-like) without external deps.
    x_in = tf.keras.Input(shape=input_shape)

    x = x_in
    for dilation in (1, 2, 4, 8):
        x = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
        )(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    opt = tf.keras.optimizers.Adam(learning_rate=float(CFG["LR"]))
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ==============================
# TRAIN / EVAL
# ==============================


def time_split_masks(dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Split by unique sorted dates globally
    uniq = np.unique(dates)
    uniq = np.sort(uniq)
    n = len(uniq)

    n_train = int(n * float(CFG["TRAIN_SPLIT"]))
    n_val = int(n * float(CFG["VAL_SPLIT"]))

    train_end = uniq[n_train - 1] if n_train > 0 else uniq[0]
    val_end = uniq[n_train + n_val - 1] if (n_train + n_val) > 0 else uniq[-1]

    m_train = dates <= train_end
    m_val = (dates > train_end) & (dates <= val_end)
    m_test = dates > val_end
    return m_train, m_val, m_test


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # roc_auc_score падает если один класс
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def evaluate_global(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    return {
        "AUC": _safe_auc(y_true, y_prob),
        "PR_AUC": _safe_ap(y_true, y_prob),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "BalAcc": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "F1": float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def per_ticker_metrics(y_true: np.ndarray, y_prob: np.ndarray, secids: np.ndarray) -> pd.DataFrame:
    rows = []
    for s in sorted(set(secids)):
        m = secids == s
        yt = y_true[m]
        yp = y_prob[m]
        rows.append({"secid": s, "n": int(m.sum()), "AUC": _safe_auc(yt, yp), "PR_AUC": _safe_ap(yt, yp)})
    return pd.DataFrame(rows).sort_values("secid")


def simple_backtest_nonoverlap_longonly(
    y_prob: np.ndarray,
    fwd_ret: np.ndarray,
    dates: np.ndarray,
    secids: np.ndarray,
    thr: float,
    fee: float,
) -> pd.DataFrame:
    # For each ticker separately: take signal days (prob>=thr), then skip next horizon days (non-overlap)
    # Here we approximate by sorting by date and greedy selection.
    res_rows = []
    horizon = int(CFG["HORIZON"])

    for s in sorted(set(secids)):
        m = secids == s
        if m.sum() == 0:
            continue

        d = pd.to_datetime(dates[m])
        p = y_prob[m]
        r = fwd_ret[m]

        order = np.argsort(d.values)
        d = d.values[order]
        p = p[order]
        r = r[order]

        picks = []
        i = 0
        while i < len(d):
            if p[i] >= thr:
                net = float(r[i]) - float(fee)
                picks.append(net)
                i += horizon
            else:
                i += 1

        if picks:
            res_rows.append(
                {
                    "secid": s,
                    "n_trades": int(len(picks)),
                    "mean_net": float(np.mean(picks)),
                    "sum_net": float(np.sum(picks)),
                }
            )
        else:
            res_rows.append({"secid": s, "n_trades": 0, "mean_net": float("nan"), "sum_net": 0.0})

    return pd.DataFrame(res_rows).sort_values("secid")


def main() -> None:
    ds, feature_cols = build_multi_ticker_dataset()

    # Build sequences
    Xs, ys, ds_dates, ds_ret, ds_secids = make_sequences_with_meta(
        ds.X,
        ds.y,
        ds.dates,
        ds.fwd_ret,
        ds.secids,
        int(CFG["SEQ_LEN"]),
    )

    print(f"Sequences: Xs={Xs.shape}, ys={ys.shape}")

    # Split
    m_train, m_val, m_test = time_split_masks(ds_dates)

    X_train, y_train = Xs[m_train], ys[m_train]
    X_val, y_val = Xs[m_val], ys[m_val]
    X_test, y_test = Xs[m_test], ys[m_test]

    dates_test = ds_dates[m_test]
    secids_test = ds_secids[m_test]
    fwd_test = ds_ret[m_test]

    # Scale features (fit on train only)
    n_steps, n_feat = X_train.shape[1], X_train.shape[2]
    scaler = RobustScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)

    def tr(x: np.ndarray) -> np.ndarray:
        x2 = x.reshape(-1, n_feat)
        x2 = scaler.transform(x2)
        return x2.reshape(-1, n_steps, n_feat)

    X_train = tr(X_train)
    X_val = tr(X_val)
    X_test = tr(X_test)

    # Class weights
    classes = np.array([0, 1])
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {0: float(w[0]), 1: float(w[1])}

    # Model
    model = build_tcn_model((n_steps, n_feat))
    model.summary()

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=15, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=7, mode="max", min_lr=1e-5),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        shuffle=False,
        class_weight=cw,
        callbacks=cb,
        verbose=2,
    )

    y_prob = model.predict(X_test, batch_size=int(CFG["BATCH_SIZE"])).ravel()

    print("\n=== GLOBAL TEST METRICS ===")
    g = evaluate_global(y_test, y_prob, thr=0.5)
    for k, v in g.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== PER-TICKER METRICS (TEST) ===")
    print(per_ticker_metrics(y_test, y_prob, secids_test).to_string(index=False))

    print("\n=== PROB SUMMARY (TEST) ===")
    s = pd.Series(y_prob)
    print(s.describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_string())

    print("\n=== BACKTEST (non-overlap long-only, TEST) ===")
    bt = simple_backtest_nonoverlap_longonly(
        y_prob=y_prob,
        fwd_ret=fwd_test,
        dates=dates_test,
        secids=secids_test,
        thr=0.5,
        fee=float(CFG["FEE"]),
    )
    print(bt.to_string(index=False))


if __name__ == "__main__":
    # Disable TF excessive logs if desired
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
