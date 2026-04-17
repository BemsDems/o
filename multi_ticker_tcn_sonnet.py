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
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin


# ==============================
# CONFIG
# ==============================
CFG: Dict[str, Any] = {
    "TICKERS": ["SBER", "GAZP", "LKOH", "YNDX"],
    "START": "2015-01-01",
    "END": datetime.now().strftime("%Y-%m-%d"),
    "HORIZON": 5,
    "THR_MOVE": 0.03,
    "SEQ_LEN": 30,
    "TRAIN_SPLIT": 0.70,
    "VAL_SPLIT": 0.15,
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "LR": 3e-4,
    "SEED": 42,
    "FEE": 0.001,

    # Extended (diploma) diagnostics — can be slow (plots/permutation).
    "EXTENDED_DIAGNOSTICS": True,
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


def build_features_one(df: pd.DataFrame, *, secid: str = "") -> pd.DataFrame:
    """Feature engineering with minimal data loss.

    Key idea: do NOT drop rows because of long-window indicators (e.g., SMA_200).
    We drop NaNs only for "critical" short-window features needed downstream.
    """
    out = df.copy()

    n0 = len(out)
    if secid:
        print(f"  candles: {n0}")

    # Log-returns (lags kept small to reduce NaN loss)
    for lag in (1, 2, 3, 5, 10):
        price_ratio = out["CLOSE"] / out["CLOSE"].shift(lag)
        price_ratio = price_ratio.clip(0.5, 2.0)
        out[f"logret_{lag}"] = np.log(price_ratio)
        out[f"logret_{lag}"] = out[f"logret_{lag}"].replace([np.inf, -np.inf], np.nan)

    # Trend: short + long
    out["sma_20"] = out["CLOSE"].rolling(20).mean()
    out["sma_50"] = out["CLOSE"].rolling(50).mean()
    out["sma_200"] = out["CLOSE"].rolling(200).mean()
    out["trend_up_20"] = (out["CLOSE"] > out["sma_20"]).astype(int)
    out["trend_up_50"] = (out["CLOSE"] > out["sma_50"]).astype(int)
    out["trend_up_200"] = (out["CLOSE"] > out["sma_200"]).astype(int)

    # Relative volume
    out["vol_ma_20"] = out["VOLUME"].rolling(20).mean()
    out["vol_rel"] = out["VOLUME"] / (out["vol_ma_20"] + 1e-9)
    out["vol_rel"] = out["vol_rel"].clip(0.1, 3.0)
    out["vol_spike"] = (out["vol_rel"] > 2.0).astype(int)

    # RSI
    out["rsi_14"] = compute_rsi(out["CLOSE"], 14)
    out["rsi_14"] = out["rsi_14"].clip(0.0, 100.0)
    out["rsi_oversold"] = (out["rsi_14"] < 30).astype(int)
    out["rsi_overbought"] = (out["rsi_14"] > 70).astype(int)

    # 20d price position
    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_pos_20"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-9)
    out["price_pos_20"] = out["price_pos_20"].clip(0.0, 1.0)

    # Volatility proxy
    out["volatility_20"] = out["logret_1"].rolling(20).std()
    out["volatility_20"] = out["volatility_20"].clip(0.0, 0.1)

    # Diagnostics before drop
    if secid:
        print(f"  after indicators: {len(out)}")
        print(f"  NaN sma_200: {int(out['sma_200'].isna().sum())}")

    # Drop NaNs only for short-window features
    out = out.replace([np.inf, -np.inf], np.nan)

    critical_cols = [
        "logret_1",
        "logret_10",
        "sma_20",
        "vol_ma_20",
        "rsi_14",
        "price_pos_20",
    ]
    out = out.dropna(subset=critical_cols).copy()

    # Fill long-window indicators (past-only)
    out["sma_200"] = out["sma_200"].ffill()
    out["trend_up_200"] = out["trend_up_200"].fillna(0).astype(int)

    if secid:
        n1 = len(out)
        share = (n1 / n0 * 100.0) if n0 else 0.0
        print(f"  after dropna(critical): {n1} ({share:.1f}%)")

    # Final cleanup: remove any remaining NaN/Inf (should be rare)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    if secid:
        try:
            ok = (not out.isnull().any().any())
        except Exception:
            ok = True
        print(f"  final NaN check: {ok}")

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
        print(f"  loaded rows: {len(df)}")
        if df.empty:
            print(f"  -> empty, skip")
            continue

        df_feat = build_features_one(df, secid=secid)

        min_rows = max(250, int(CFG["SEQ_LEN"]) + int(CFG["HORIZON"]) + 50)
        if len(df_feat) < min_rows:
            print(f"  -> too few rows after features ({len(df_feat)} < {min_rows}), skip")
            continue

        y, fwd_ret = make_target(df_feat["CLOSE"], int(CFG["HORIZON"]), float(CFG["THR_MOVE"]))

        h = int(CFG["HORIZON"])
        df_feat = df_feat.iloc[:-h]
        y = y.iloc[:-h]
        fwd_ret = fwd_ret.iloc[:-h]

        print(f"  final rows (after horizon trim): {len(df_feat)}")

        tmp = df_feat.copy()
        tmp["target"] = y.values
        tmp["fwd_ret"] = fwd_ret.values
        tmp["date"] = tmp.index
        rows.append(tmp.reset_index(drop=True))

    if not rows:
        raise RuntimeError("No tickers loaded. Check MOEX availability / tickers list.")

    full = pd.concat(rows, axis=0).sort_values(["secid", "date"]).reset_index(drop=True)

    feature_cols = [
        "logret_1",
        "logret_2",
        "logret_3",
        "logret_5",
        "logret_10",
        "trend_up_20",
        "trend_up_50",
        "trend_up_200",
        "vol_rel",
        "vol_spike",
        "rsi_14",
        "rsi_oversold",
        "rsi_overbought",
        "price_pos_20",
        "volatility_20",
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




def make_sequences_multi_ticker(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    fwd_ret: np.ndarray,
    secids: np.ndarray,
    seq_len: int,
    split_masks: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Create sequences per ticker and per split (more robust).

    We avoid windows that cross split boundaries and never mix tickers inside a window.
    split_masks are row-level masks (same length as X/y/dates/secids).
    """
    train_mask, val_mask, test_mask = split_masks

    def _collect_for_split(mask_sec: np.ndarray, X_sec, y_sec, d_sec, r_sec, secid: str):
        Xs, ys, ds, rs, ss = [], [], [], [], []
        # build window ending at i (exclusive window [i-seq_len, i))
        for i in range(seq_len, len(X_sec)):
            if not mask_sec[i]:
                continue
            # require the entire window to be inside this split
            if not mask_sec[i - seq_len : i].all():
                continue
            Xs.append(X_sec[i - seq_len : i])
            ys.append(y_sec[i])
            ds.append(d_sec[i])
            rs.append(r_sec[i])
            ss.append(secid)
        return (
            np.asarray(Xs),
            np.asarray(ys),
            np.asarray(ds),
            np.asarray(rs),
            np.asarray(ss),
        )

    out = []
    for mask_global in (train_mask, val_mask, test_mask):
        out.append(([], [], [], [], []))

    for secid in np.unique(secids):
        m = secids == secid
        X_sec = X[m]
        y_sec = y[m]
        d_sec = dates[m]
        r_sec = fwd_ret[m]

        tr_sec = train_mask[m]
        va_sec = val_mask[m]
        te_sec = test_mask[m]

        tr = _collect_for_split(tr_sec, X_sec, y_sec, d_sec, r_sec, str(secid))
        va = _collect_for_split(va_sec, X_sec, y_sec, d_sec, r_sec, str(secid))
        te = _collect_for_split(te_sec, X_sec, y_sec, d_sec, r_sec, str(secid))

        # append
        for bucket, part in zip(out, (tr, va, te)):
            for i, arr in enumerate(part):
                bucket[i].append(arr)

    def _cat(parts):
        if not parts:
            return np.asarray([])
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return np.asarray([])
        return np.concatenate(parts, axis=0)

    Xs_tr = _cat(out[0][0]); ys_tr = _cat(out[0][1]); ds_tr = _cat(out[0][2]); rs_tr = _cat(out[0][3]); ss_tr = _cat(out[0][4])
    Xs_va = _cat(out[1][0]); ys_va = _cat(out[1][1]); ds_va = _cat(out[1][2]); rs_va = _cat(out[1][3]); ss_va = _cat(out[1][4])
    Xs_te = _cat(out[2][0]); ys_te = _cat(out[2][1]); ds_te = _cat(out[2][2]); rs_te = _cat(out[2][3]); ss_te = _cat(out[2][4])

    print("\nSequences created:")
    print(f" Train: {len(Xs_tr)} sequences")
    print(f" Val:   {len(Xs_va)} sequences")
    print(f" Test:  {len(Xs_te)} sequences")

    return (
        Xs_tr, ys_tr, ds_tr, rs_tr, ss_tr,
        Xs_va, ys_va, ds_va, rs_va, ss_va,
        Xs_te, ys_te, ds_te, rs_te, ss_te,
    )


# ==============================
# MODEL
# ==============================


def build_tcn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.LayerNormalization()(x_in)

    # TCN blocks with MODERATE regularization
    for filters, dilation in [(64, 1), (64, 2), (64, 4), (32, 8)]:
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(CFG["LR"]),
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model



# ==============================
# TRAIN / EVAL
# ==============================


def time_split_masks(dates: np.ndarray, secids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-ticker time split (prevents one ticker dominating / YNDX issues).

    For each ticker independently: sort by date, then split by fractions.
    Returns row-level boolean masks aligned with input arrays.
    """

    n = len(dates)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for secid in np.unique(secids):
        sec_mask = secids == secid
        sec_dates = dates[sec_mask]
        sec_indices = np.where(sec_mask)[0]
        if len(sec_dates) == 0:
            continue

        order = np.argsort(sec_dates)
        sorted_indices = sec_indices[order]

        n_sec = int(len(sorted_indices))
        n_train = int(n_sec * float(CFG["TRAIN_SPLIT"]))
        n_val = int(n_sec * float(CFG["VAL_SPLIT"]))

        # guardrails
        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_train + n_val >= n_sec:
            n_train = max(1, n_sec - 2)
            n_val = 1

        train_end = n_train
        val_end = n_train + n_val

        train_mask[sorted_indices[:train_end]] = True
        val_mask[sorted_indices[train_end:val_end]] = True
        test_mask[sorted_indices[val_end:]] = True

    print("\n=== SPLIT DISTRIBUTION BY TICKER ===")
    for secid in np.unique(secids):
        sec_mask = secids == secid
        n_tr = int((train_mask & sec_mask).sum())
        n_va = int((val_mask & sec_mask).sum())
        n_te = int((test_mask & sec_mask).sum())
        print(f"{str(secid):4s}: Train={n_tr:4d}, Val={n_va:4d}, Test={n_te:4d}")

    return train_mask, val_mask, test_mask



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


def improved_backtest_per_ticker(
    y_prob: np.ndarray,
    fwd_ret: np.ndarray,
    dates: np.ndarray,
    secids: np.ndarray,
    threshold: float,
    fee: float,
) -> pd.DataFrame:
    """Backtest per ticker with 2 policies:

    1) Fixed threshold
    2) Top-20% confidence (dynamic threshold)

    Uses non-overlapping trades with horizon=CFG["HORIZON"].
    Fee is applied as 2*fee (entry+exit) to be more conservative.
    """

    horizon = int(CFG["HORIZON"])
    results = []

    for secid in sorted(set(secids)):
        mask = secids == secid
        if int(mask.sum()) < 50:
            continue

        prob_sec = y_prob[mask]
        ret_sec = fwd_ret[mask]
        dates_sec = dates[mask]

        order = np.argsort(dates_sec)
        prob_sec = prob_sec[order]
        ret_sec = ret_sec[order]

        # fixed threshold trades
        trades_fixed = []
        i = 0
        while i < len(prob_sec):
            if prob_sec[i] >= threshold:
                net = float(ret_sec[i]) - 2.0 * float(fee)
                trades_fixed.append(net)
                i += horizon
            else:
                i += 1

        # top-20% confidence trades
        top20_thr = float(np.percentile(prob_sec, 80))
        trades_top20 = []
        i = 0
        while i < len(prob_sec):
            if prob_sec[i] >= top20_thr:
                net = float(ret_sec[i]) - 2.0 * float(fee)
                trades_top20.append(net)
                i += horizon
            else:
                i += 1

        results.append(
            {
                "secid": str(secid),
                "n_signals": int((prob_sec >= threshold).sum()),
                "n_trades_fixed": int(len(trades_fixed)),
                "mean_ret_fixed": float(np.mean(trades_fixed)) if trades_fixed else 0.0,
                "total_ret_fixed": float(np.sum(trades_fixed)) if trades_fixed else 0.0,
                "n_trades_top20": int(len(trades_top20)),
                "mean_ret_top20": float(np.mean(trades_top20)) if trades_top20 else 0.0,
                "total_ret_top20": float(np.sum(trades_top20)) if trades_top20 else 0.0,
                "sharpe_fixed": float(np.mean(trades_fixed) / (np.std(trades_fixed) + 1e-9)) if len(trades_fixed) > 1 else 0.0,
            }
        )

    return pd.DataFrame(results)



# ==============================
# EXTENDED DIAGNOSTICS (for diploma)
# ==============================


def plot_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, name: str = ""):
    """Histogram of probabilities for each class (best-effort; skips if matplotlib missing)."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot_probability_distribution] matplotlib not available: {e}")
        return

    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)

    prob_class0 = y_prob[y_true == 0]
    prob_class1 = y_prob[y_true == 1]

    print(f"\n=== PROBABILITY DISTRIBUTION {name} ===")
    if len(prob_class0):
        print(f"Class 0 (no growth): mean={prob_class0.mean():.3f}, std={prob_class0.std():.3f}")
    if len(prob_class1):
        print(f"Class 1 (growth):    mean={prob_class1.mean():.3f}, std={prob_class1.std():.3f}")
    if len(prob_class0) and len(prob_class1):
        print(f"Separation (mean diff): {abs(prob_class1.mean() - prob_class0.mean()):.3f}")

    plt.figure(figsize=(10, 5))
    plt.hist(prob_class0, bins=50, alpha=0.6, label="No Growth (y=0)", color="red")
    plt.hist(prob_class1, bins=50, alpha=0.6, label="Growth (y=1)", color="green")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title(f"Probability Distribution {name}")
    plt.axvline(0.5, color="black", linestyle="--")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def calibration_curve_analysis(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Calibration curve report (text)."""
    from sklearn.calibration import calibration_curve

    y_true = y_true.astype(int)
    y_prob = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    print("\n=== CALIBRATION ANALYSIS ===")
    print("Predicted Prob | True Frequency | Difference")
    print("-" * 50)
    for pred, true in zip(prob_pred, prob_true):
        diff = abs(true - pred)
        status = "✓" if diff < 0.1 else "✗"
        print(f"{pred:14.3f} | {true:14.3f} | {diff:10.3f} {status}")

    ece = float(np.mean(np.abs(prob_true - prob_pred))) if len(prob_true) else float("nan")
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")


def analyze_predictions_by_confidence(y_true: np.ndarray, y_prob: np.ndarray, fwd_ret: np.ndarray):
    """Quality by confidence deciles."""
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    fwd_ret = fwd_ret.astype(float)

    print("\n=== PREDICTIONS BY CONFIDENCE LEVEL ===")
    deciles = np.percentile(y_prob, np.arange(0, 101, 10))

    print("Decile | Prob Range | N | Accuracy | Avg Future Ret | Precision")
    print("-" * 80)

    for i in range(len(deciles) - 1):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(deciles) - 2 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue

        n = int(mask.sum())
        y_t = y_true[mask]
        y_p = y_prob[mask]
        ret = fwd_ret[mask]

        accuracy = float((y_t == (y_p >= 0.5)).mean())
        avg_ret = float(ret.mean())
        precision = float(y_t.mean())

        print(f"D{i+1:2d} | {lo:.3f} - {hi:.3f} | {n:4d} | {accuracy:8.3f} | {avg_ret:14.4f} | {precision:9.3f}")

    top_10pct = y_prob >= np.percentile(y_prob, 90)
    bot_10pct = y_prob <= np.percentile(y_prob, 10)

    print("\n=== TOP 10% MOST CONFIDENT ===")
    print(f"N: {int(top_10pct.sum())}")
    print(f"Avg prob: {float(y_prob[top_10pct].mean()):.3f}")
    print(f"Precision: {float(y_true[top_10pct].mean()):.3f}")
    print(f"Avg future return: {float(fwd_ret[top_10pct].mean()):.4f}")

    print("\n=== BOTTOM 10% LEAST CONFIDENT ===")
    print(f"N: {int(bot_10pct.sum())}")
    print(f"Avg prob: {float(y_prob[bot_10pct].mean()):.3f}")
    print(f"Precision: {float(y_true[bot_10pct].mean()):.3f}")
    print(f"Avg future return: {float(fwd_ret[bot_10pct].mean()):.4f}")


def confusion_matrix_analysis(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Detailed confusion matrix."""
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== CONFUSION MATRIX (threshold={threshold:.2f}) ===")
    print(" Predicted")
    print(" NO   YES")
    print(f"Actual NO {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"Actual YES {cm[1,0]:5d} {cm[1,1]:5d}")

    tn, fp, fn, tp = cm.ravel()
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, target_names=["No Growth", "Growth"], digits=3, zero_division=0))

    if (tp + fp) > 0:
        prec = tp / (tp + fp)
    else:
        prec = float('nan')
    if (tp + fn) > 0:
        rec = tp / (tp + fn)
    else:
        rec = float('nan')
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f}")


def feature_importance_proxy(model, X_test: np.ndarray, y_test: np.ndarray, feature_names: list, max_features: int = 20):
    """Permutation importance proxy via AUC drop. Can be slow."""
    print("\n=== FEATURE IMPORTANCE (Permutation, AUC drop) ===")
    y_test = y_test.astype(int)

    base_pred = model.predict(X_test, verbose=0).ravel()
    if len(np.unique(y_test)) < 2:
        print("AUC undefined (single class in y_test)")
        return []
    base_auc = float(roc_auc_score(y_test, base_pred))

    n_feat = X_test.shape[2]
    max_features = min(max_features, n_feat)

    importances = []
    for i in range(max_features):
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        X_perm = X_test.copy()
        for t in range(X_perm.shape[1]):
            np.random.shuffle(X_perm[:, t, i])
        perm_pred = model.predict(X_perm, verbose=0).ravel()
        perm_auc = float(roc_auc_score(y_test, perm_pred))
        importances.append((fname, base_auc - perm_auc))

    importances.sort(key=lambda x: x[1], reverse=True)
    print(f"Base AUC: {base_auc:.4f}")
    print("Feature | Importance")
    print("-" * 40)
    for fname, imp in importances:
        print(f"{fname:20s} | {imp:10.4f}")
    return importances


def temporal_performance_analysis(y_true: np.ndarray, y_prob: np.ndarray, dates: np.ndarray):
    """Monthly AUC / stats."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "y_true": y_true.astype(int),
        "y_prob": y_prob.astype(float),
    })
    df["month"] = df["date"].dt.to_period("M")

    print("\n=== TEMPORAL PERFORMANCE (Monthly) ===")
    print("Month | N | Pos Rate | Avg Prob | AUC")
    print("-" * 60)

    for month, g in df.groupby("month"):
        if len(g) < 10:
            continue
        pos_rate = float(g["y_true"].mean())
        avg_prob = float(g["y_prob"].mean())
        if len(np.unique(g["y_true"].values)) > 1:
            auc = float(roc_auc_score(g["y_true"].values, g["y_prob"].values))
        else:
            auc = float('nan')
        print(f"{str(month):10s} | {len(g):4d} | {pos_rate:8.3f} | {avg_prob:8.3f} | {auc:.3f}")


def check_random_baseline(y_true: np.ndarray, n_iterations: int = 100):
    """Compare to random predictor AUC distribution."""
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        print("Random baseline AUC undefined (single class)")
        return

    aucs = []
    for _ in range(int(n_iterations)):
        rp = np.random.uniform(0.0, 1.0, len(y_true))
        aucs.append(float(roc_auc_score(y_true, rp)))
    mu = float(np.mean(aucs))
    sd = float(np.std(aucs))
    print("\n=== RANDOM BASELINE COMPARISON ===")
    print(f"Random model AUC: {mu:.4f} ± {sd:.4f}")
    print(f"2-sigma threshold: {mu + 2*sd:.4f}")


def main() -> None:
    ds, feature_cols = build_multi_ticker_dataset()

    print("\n=== NaN DIAGNOSTICS (RAW FEATURES) ===")
    print(f"X_raw NaN: {np.isnan(ds.X).any()}, Inf: {np.isinf(ds.X).any()}")
    for i, col in enumerate(feature_cols):
        n_nan = int(np.isnan(ds.X[:, i]).sum())
        n_inf = int(np.isinf(ds.X[:, i]).sum())
        if n_nan > 0 or n_inf > 0:
            print(f"  WARNING {col}: {n_nan} NaN, {n_inf} Inf")

    # Fix raw X to prevent scaler from propagating NaN/Inf
    ds = MultiDataset(
        X=np.nan_to_num(ds.X, nan=0.0, posinf=0.0, neginf=0.0),
        y=ds.y,
        fwd_ret=ds.fwd_ret,
        dates=ds.dates,
        secids=ds.secids,
    )


    # Split by dates on ROWS (not windows), then build sequences per ticker within each split
    m_train_rows, m_val_rows, m_test_rows = time_split_masks(ds.dates, ds.secids)

    (
        X_train, y_train, _, _, _,
        X_val, y_val, _, _, _,
        X_test, y_test, dates_test, fwd_test, secids_test,
    ) = make_sequences_multi_ticker(
        ds.X,
        ds.y,
        ds.dates,
        ds.fwd_ret,
        ds.secids,
        int(CFG["SEQ_LEN"]),
        split_masks=(m_train_rows, m_val_rows, m_test_rows),
    )

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError(
            "Not enough sequences in one of splits (train/val/test). "
            "Try: reduce SEQ_LEN, move START forward, increase history, or reduce tickers."
        )

    print(f"Seq shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

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

    # ------------------------------
    # DATA SANITY CHECK (to prevent NaN loss from epoch 1)
    # ------------------------------
    print("\n=== DATA SANITY CHECK ===")
    print(f"Train X: min={np.nanmin(X_train):.4f}, max={np.nanmax(X_train):.4f}")
    print(f"Train X has NaN: {np.isnan(X_train).any()}")
    print(f"Train X has Inf: {np.isinf(X_train).any()}")
    print(f"Train y: unique={np.unique(y_train)}, distribution={np.bincount(y_train.astype(int))}")

    extreme_mask = np.abs(X_train) > 10
    if extreme_mask.any():
        print(f"WARNING: {int(extreme_mask.sum())} values with |x| > 10")
        print(f"Max absolute value: {float(np.abs(X_train).max()):.2f}")

    # Clip outliers AFTER scaling (helps prevent collapse to ~0.5 probabilities)
    X_train = np.clip(X_train, -5.0, 5.0)
    X_val = np.clip(X_val, -5.0, 5.0)
    X_test = np.clip(X_test, -5.0, 5.0)
    print(f"✅ Clipped to [-5, 5]: train max={X_train.max():.2f}")

    # Replace NaN/Inf with zeros (keep pipeline running; diagnostics above should highlight sources)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print("✅ NaN/Inf replaced with 0 after scaling")

    # Data augmentation (small Gaussian noise) for train only
    noise_std = 0.05
    X_train_aug = X_train + np.random.normal(0.0, noise_std, X_train.shape)
    X_train_aug = np.clip(X_train_aug, -5.0, 5.0)



    # Class weights
    classes = np.array([0, 1])
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = {0: float(w[0]), 1: float(w[1])}

    # Model
    model = build_tcn_model((n_steps, n_feat))
    model.summary()

    class NanCheck(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is None:
                return
            loss = logs.get("loss")
            if loss is not None and (np.isnan(loss) or np.isinf(loss)):
                print(f"\nNaN/Inf loss detected at batch={batch}, stopping training")
                self.model.stop_training = True


    cb = [
        NanCheck(),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=10, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=7, mode="max", min_lr=1e-5),
    ]

    model.fit(
        X_train_aug,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        shuffle=False,
        class_weight=cw,
        callbacks=cb,
        verbose=2,
    )



    # ==============================
    # CALIBRATION (Platt scaling) on VAL
    # ==============================
    class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, model):
            self.model = model

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            prob = self.model.predict(X, verbose=0).ravel()
            prob = np.clip(prob, 1e-6, 1 - 1e-6)
            return np.column_stack([1 - prob, prob])

    print("\n=== CALIBRATING MODEL ===")
    wrapped_model = KerasClassifierWrapper(model)
    calibrator = CalibratedClassifierCV(wrapped_model, method="sigmoid", cv="prefit")
    calibrator.fit(X_val, y_val)

    # ==============================
    # THRESHOLD OPTIMIZATION (precision target) on VAL
    # ==============================
    print("\n=== THRESHOLD OPTIMIZATION ===")
    val_prob_cal = calibrator.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, val_prob_cal)

    target_precision = 0.30
    mask = precision >= target_precision
    if mask.any() and len(thresholds) > 0:
        idx = np.where(mask)[0][-1]
        idx = min(idx, len(thresholds) - 1)
        best_thr = float(thresholds[idx])
        best_precision = float(precision[idx])
        best_recall = float(recall[idx])
    else:
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        idx = int(np.argmax(f1_scores))
        best_thr = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        best_precision = float(precision[idx])
        best_recall = float(recall[idx])

    best_f1 = float(2 * best_precision * best_recall / (best_precision + best_recall + 1e-12))
    print(f"Optimized threshold: {best_thr:.3f}")
    print(f" Precision: {best_precision:.3f}")
    print(f" Recall: {best_recall:.3f}")
    print(f" F1: {best_f1:.3f}")

    # Predict (raw + calibrated)
    y_prob_raw = model.predict(X_test, verbose=0).ravel()
    y_prob = calibrator.predict_proba(X_test)[:, 1]

    print(f"Before calibration: mean={float(y_prob_raw.mean()):.3f}")
    print(f"After calibration:  mean={float(y_prob.mean()):.3f}")
    print(f"True positive rate: {float(y_test.mean()):.3f}")


    # Prediction sanity check
    if np.isnan(y_prob).any():
        print("\nWARNING: NaN in predictions — model likely failed to train")
        print(f"Train X NaN: {np.isnan(X_train).any()}, Inf: {np.isinf(X_train).any()}")
        return


    print("\n" + "="*80)
    print("EXTENDED MODEL DIAGNOSTICS")
    print("="*80)

    plot_probability_distribution(y_test, y_prob, name="TEST")
    calibration_curve_analysis(y_test, y_prob, n_bins=10)
    analyze_predictions_by_confidence(y_test, y_prob, fwd_test)
    confusion_matrix_analysis(y_test, y_prob, threshold=0.5)
    importances = feature_importance_proxy(model, X_test, y_test, feature_cols)
    temporal_performance_analysis(y_test, y_prob, dates_test)
    check_random_baseline(y_test, n_iterations=100)

    print("\n" + "="*80)






    print("\n=== GLOBAL TEST METRICS ===")
    g = evaluate_global(y_test, y_prob, thr=best_thr)
    for k, v in g.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n=== PER-TICKER METRICS (TEST) ===")
    print(per_ticker_metrics(y_test, y_prob, secids_test).to_string(index=False))

    print("\n=== PROB SUMMARY (TEST) ===")
    s = pd.Series(y_prob)
    print(s.describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_string())

    print("\n=== IMPROVED BACKTEST (TEST) ===")
    bt = improved_backtest_per_ticker(
        y_prob=y_prob,
        fwd_ret=fwd_test,
        dates=dates_test,
        secids=secids_test,
        threshold=best_thr,
        fee=float(CFG["FEE"]),
    )
    print(bt.to_string(index=False))


if __name__ == "__main__":
    # Disable TF excessive logs if desired
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
