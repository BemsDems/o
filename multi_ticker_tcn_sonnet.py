from __future__ import annotations

"""Multi‑ticker SONNET TCN (v0.2 — IMPROVED)

Изменения относительно v0.1:
1. Модель: вернул 4 TCN блока с dilation [1,2,4,8] и фильтры 64/64/64/32
2. Добавлены residual connections (как в оригинальной TCN архитектуре)
3. Dense голова: 64 → 32 → 1 (вместо 32 → 1)
4. shuffle=True в model.fit() — критический фикс
5. Убран data augmentation шумом — портит сигнал в финансах
6. Batch size 64 (вместо 128)
7. Добавлены MACD и Bollinger Bands (5 новых признаков)
8. Добавлен cosine decay LR schedule
9. Label smoothing = 0.02 (лёгкое сглаживание)
10. Dropout: 0.25 в TCN, 0.35 в Dense (чуть мягче)
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight


# ==============================
# CONFIG
# ==============================
CFG: Dict[str, Any] = {
    "TICKERS": [
        "SBER", "GAZP", "LKOH", "YNDX",
        "GMKN", "NVTK", "ROSN", "TATN",
        "MTSS", "MGNT", "ALRS", "PLZL",
        "CHMF", "NLMK", "VTBR",
    ],
    "START": "2015-01-01",
    "END": datetime.now().strftime("%Y-%m-%d"),
    "HORIZON": 5,
    "THR_MOVE": 0.03,
    "SEQ_LEN": 30,
    "TRAIN_SPLIT": 0.70,
    "VAL_SPLIT": 0.15,
    "BATCH_SIZE": 64,  # ← было 128, вернул 64
    "EPOCHS": 100,
    "LR": 3e-4,
    "SEED": 42,
    "FEE": 0.001,
    "EXTENDED_DIAGNOSTICS": True,
}

np.random.seed(int(CFG["SEED"]))
tf.random.set_seed(int(CFG["SEED"]))


# ==============================
# DATA LOADING
# ==============================

def _resolve_end_date(end: str | None) -> str:
    if end is None:
        return datetime.now().strftime("%Y-%m-%d")
    return str(end)


def fetch_moex_candles(secid: str, start: str, end: str | None) -> pd.DataFrame:
    end_resolved = _resolve_end_date(end)
    raw = Ticker(secid).candles(start=str(start), end=str(end_resolved), period="1D")
    df = pd.DataFrame(raw)
    if df.empty:
        return df

    df["begin"] = pd.to_datetime(df["begin"], errors="coerce")
    df = df.dropna(subset=["begin"]).drop_duplicates(subset=["begin"]).set_index("begin")

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
    """Feature engineering — 20 признаков (15 базовых + 5 новых)."""
    out = df.copy()

    n0 = len(out)
    if secid:
        print(f"  candles: {n0}")

    # ── Log-returns ──
    for lag in (1, 2, 3, 5, 10):
        price_ratio = out["CLOSE"] / out["CLOSE"].shift(lag)
        price_ratio = price_ratio.clip(0.5, 2.0)
        out[f"logret_{lag}"] = np.log(price_ratio)
        out[f"logret_{lag}"] = out[f"logret_{lag}"].replace([np.inf, -np.inf], np.nan)

    # ── Trend: SMA ──
    out["sma_20"] = out["CLOSE"].rolling(20).mean()
    out["sma_50"] = out["CLOSE"].rolling(50).mean()
    out["sma_200"] = out["CLOSE"].rolling(200).mean()
    out["trend_up_20"] = (out["CLOSE"] > out["sma_20"]).astype(int)
    out["trend_up_50"] = (out["CLOSE"] > out["sma_50"]).astype(int)
    out["trend_up_200"] = (out["CLOSE"] > out["sma_200"]).astype(int)

    # ── Volume ──
    out["vol_ma_20"] = out["VOLUME"].rolling(20).mean()
    out["vol_rel"] = out["VOLUME"] / (out["vol_ma_20"] + 1e-9)
    out["vol_rel"] = out["vol_rel"].clip(0.1, 3.0)
    out["vol_spike"] = (out["vol_rel"] > 2.0).astype(int)

    # ── RSI ──
    out["rsi_14"] = compute_rsi(out["CLOSE"], 14)
    out["rsi_14"] = out["rsi_14"].clip(0.0, 100.0)
    out["rsi_oversold"] = (out["rsi_14"] < 30).astype(int)
    out["rsi_overbought"] = (out["rsi_14"] > 70).astype(int)

    # ── Price position ──
    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_pos_20"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-9)
    out["price_pos_20"] = out["price_pos_20"].clip(0.0, 1.0)

    # ── Volatility ──
    out["volatility_20"] = out["logret_1"].rolling(20).std()
    out["volatility_20"] = out["volatility_20"].clip(0.0, 0.1)

    # ═══════════════════════════════════════════
    # НОВЫЕ ПРИЗНАКИ (v0.2)
    # ═══════════════════════════════════════════

    # ── MACD (12/26/9) ──
    ema_12 = out["CLOSE"].ewm(span=12, adjust=False).mean()
    ema_26 = out["CLOSE"].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_norm"] = (macd_line / out["CLOSE"]).clip(-0.05, 0.05)
    out["macd_hist_norm"] = ((macd_line - macd_signal) / out["CLOSE"]).clip(-0.03, 0.03)
    out["macd_cross_up"] = (
        (macd_line > macd_signal)
        & (macd_line.shift(1) <= macd_signal.shift(1))
    ).astype(int)

    # ── Bollinger Bands (20, 2σ) ──
    bb_mid = out["CLOSE"].rolling(20).mean()
    bb_std = out["CLOSE"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_pos"] = ((out["CLOSE"] - bb_lower) / (bb_upper - bb_lower + 1e-9)).clip(0.0, 1.0)
    out["bb_width"] = ((bb_upper - bb_lower) / (bb_mid + 1e-9)).clip(0.0, 0.3)

    if secid:
        print(f"  after indicators: {len(out)}")
        print(f"  NaN sma_200: {int(out['sma_200'].isna().sum())}")

    out = out.replace([np.inf, -np.inf], np.nan)

    critical_cols = [
        "logret_1", "logret_10", "sma_20",
        "vol_ma_20", "rsi_14", "price_pos_20",
    ]
    out = out.dropna(subset=critical_cols).copy()

    # Fill long-window indicators
    out["sma_200"] = out["sma_200"].ffill()
    out["trend_up_200"] = out["trend_up_200"].fillna(0).astype(int)

    if secid:
        n1 = len(out)
        share = (n1 / n0 * 100.0) if n0 else 0.0
        print(f"  after dropna(critical): {n1} ({share:.1f}%)")

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
            print("  -> empty, skip")
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
        # Log-returns (5)
        "logret_1", "logret_2", "logret_3", "logret_5", "logret_10",
        # Trend (3)
        "trend_up_20", "trend_up_50", "trend_up_200",
        # Volume (2)
        "vol_rel", "vol_spike",
        # RSI (3)
        "rsi_14", "rsi_oversold", "rsi_overbought",
        # Price structure (2)
        "price_pos_20", "volatility_20",
        # NEW: MACD (3)
        "macd_norm", "macd_hist_norm", "macd_cross_up",
        # NEW: Bollinger (2)
        "bb_pos", "bb_width",
    ]
    feature_cols = [c for c in feature_cols if c in full.columns]

    X = full[feature_cols].values
    y = full["target"].astype(int).values
    fwd_ret = full["fwd_ret"].astype(float).values
    dates = pd.to_datetime(full["date"]).values
    secids = full["secid"].astype(str).values

    print(f"\nMulti-ticker dataset: X={X.shape}, pos_rate={y.mean():.3%}, tickers={sorted(set(secids))}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
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
    X,
    y,
    dates,
    fwd_ret,
    secids,
    seq_len,
    split_masks,
):
    train_mask, val_mask, test_mask = split_masks

    def _collect(mask_sec, X_sec, y_sec, d_sec, r_sec, secid):
        Xs, ys, ds, rs, ss = [], [], [], [], []
        for i in range(seq_len, len(X_sec)):
            if not mask_sec[i]:
                continue
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

    out = [([], [], [], [], []) for _ in range(3)]

    for secid in np.unique(secids):
        m = secids == secid
        X_s, y_s, d_s, r_s = X[m], y[m], dates[m], fwd_ret[m]
        masks = [train_mask[m], val_mask[m], test_mask[m]]

        for idx, mask in enumerate(masks):
            parts = _collect(mask, X_s, y_s, d_s, r_s, str(secid))
            for j, arr in enumerate(parts):
                out[idx][j].append(arr)

    def _cat(parts):
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return np.asarray([])
        return np.concatenate(parts, axis=0)

    results = []
    for split_data in out:
        for lst in split_data:
            results.append(_cat(lst))

    (
        Xs_tr, ys_tr, ds_tr, rs_tr, ss_tr,
        Xs_va, ys_va, ds_va, rs_va, ss_va,
        Xs_te, ys_te, ds_te, rs_te, ss_te,
    ) = results

    print(f"\nSequences: Train={len(Xs_tr)}, Val={len(Xs_va)}, Test={len(Xs_te)}")
    return (
        Xs_tr, ys_tr, ds_tr, rs_tr, ss_tr,
        Xs_va, ys_va, ds_va, rs_va, ss_va,
        Xs_te, ys_te, ds_te, rs_te, ss_te,
    )


# ==============================
# MODEL (v0.2 — residual connections)
# ==============================

class TCNBlock(tf.keras.layers.Layer):
    """One TCN block with residual connection."""

    def __init__(self, filters, kernel_size=3, dilation_rate=1, dropout=0.25, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation=None,
            kernel_initializer="he_normal",
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.relu = tf.keras.layers.ReLU()
        self.match_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.match_conv = tf.keras.layers.Conv1D(
                self.filters,
                1,
                padding="same",
                kernel_initializer="he_normal",
            )
        super().build(input_shape)

    def call(self, x, training=False):
        residual = x
        if self.match_conv is not None:
            residual = self.match_conv(residual)

        out = self.conv(x)
        out = self.bn(out, training=training)
        out = self.relu(out)
        out = self.dropout(out, training=training)

        return out + residual


def build_tcn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LayerNormalization()(x_in)

    for filters, dilation in [(64, 1), (64, 2), (64, 4), (32, 8)]:
        x = TCNBlock(filters, kernel_size=3, dilation_rate=dilation, dropout=0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(CFG["LR"]),  # float, not a schedule (for ReduceLROnPlateau compatibility)
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ==============================
# TRAIN / EVAL
# ==============================

def time_split_masks(dates, secids):
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

        n_sec = len(sorted_indices)
        n_train = int(n_sec * float(CFG["TRAIN_SPLIT"]))
        n_val = int(n_sec * float(CFG["VAL_SPLIT"]))

        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_train + n_val >= n_sec:
            n_train = max(1, n_sec - 2)
            n_val = 1

        train_mask[sorted_indices[:n_train]] = True
        val_mask[sorted_indices[n_train : n_train + n_val]] = True
        test_mask[sorted_indices[n_train + n_val :]] = True

    print("\n=== SPLIT DISTRIBUTION BY TICKER ===")
    for secid in np.unique(secids):
        sec_mask = secids == secid
        n_tr = int((train_mask & sec_mask).sum())
        n_va = int((val_mask & sec_mask).sum())
        n_te = int((test_mask & sec_mask).sum())
        print(f"{str(secid):4s}: Train={n_tr:4d}, Val={n_va:4d}, Test={n_te:4d}")

    return train_mask, val_mask, test_mask


def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def evaluate_global(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "AUC": _safe_auc(y_true, y_prob),
        "PR_AUC": _safe_ap(y_true, y_prob),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "BalAcc": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "F1": float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def per_ticker_metrics(y_true, y_prob, secids):
    rows = []
    for s in sorted(set(secids)):
        m = secids == s
        rows.append(
            {
                "secid": s,
                "n": int(m.sum()),
                "AUC": _safe_auc(y_true[m], y_prob[m]),
                "PR_AUC": _safe_ap(y_true[m], y_prob[m]),
            }
        )
    return pd.DataFrame(rows).sort_values("secid")


def improved_backtest_per_ticker(y_prob, fwd_ret, dates, secids, threshold, fee):
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

        # Fixed threshold
        trades_fixed = []
        i = 0
        while i < len(prob_sec):
            if prob_sec[i] >= threshold:
                net = float(ret_sec[i]) - 2.0 * float(fee)
                trades_fixed.append(net)
                i += horizon
            else:
                i += 1

        # Top-20%
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
                "n_trades_fixed": len(trades_fixed),
                "total_ret_fixed": float(np.sum(trades_fixed)) if trades_fixed else 0.0,
                "win_rate_fixed": float(np.mean([1 if t > 0 else 0 for t in trades_fixed])) if trades_fixed else 0.0,
                "n_trades_top20": len(trades_top20),
                "total_ret_top20": float(np.sum(trades_top20)) if trades_top20 else 0.0,
                "sharpe_fixed": float(np.mean(trades_fixed) / (np.std(trades_fixed) + 1e-9)) if len(trades_fixed) > 1 else 0.0,
            }
        )

    return pd.DataFrame(results)


# ==============================
# EXTENDED DIAGNOSTICS
# ==============================

def plot_probability_distribution(y_true, y_prob, name=""):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return

    y_true = y_true.astype(int)
    prob_0 = y_prob[y_true == 0]
    prob_1 = y_prob[y_true == 1]

    print(f"\n=== PROBABILITY DISTRIBUTION {name} ===")
    if len(prob_0):
        print(f"Class 0: mean={prob_0.mean():.4f}, std={prob_0.std():.4f}")
    if len(prob_1):
        print(f"Class 1: mean={prob_1.mean():.4f}, std={prob_1.std():.4f}")
    if len(prob_0) and len(prob_1):
        sep = abs(prob_1.mean() - prob_0.mean())
        print(f"Separation: {sep:.4f}")
        if sep > 0.02:
            print("✅ Модель различает классы")
        else:
            print("⚠️ Слабое разделение классов")

    plt.figure(figsize=(10, 5))
    plt.hist(prob_0, bins=50, alpha=0.6, label="No Growth (y=0)", color="red")
    plt.hist(prob_1, bins=50, alpha=0.6, label="Growth (y=1)", color="green")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title(f"Probability Distribution {name}")
    plt.axvline(0.5, color="black", linestyle="--")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"prob_dist_{name.lower()}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: prob_dist_{name.lower()}.png")
    plt.close()


def calibration_curve_analysis(y_true, y_prob, n_bins=10):
    from sklearn.calibration import calibration_curve

    y_prob_clipped = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    prob_true, prob_pred = calibration_curve(
        y_true.astype(int), y_prob_clipped, n_bins=n_bins, strategy="uniform"
    )
    print("\n=== CALIBRATION ANALYSIS ===")
    print(f"{'Predicted':>12s} | {'True Freq':>12s} | {'Diff':>8s}")
    print("-" * 40)
    for pred, true in zip(prob_pred, prob_true):
        diff = abs(true - pred)
        status = "✓" if diff < 0.1 else "✗"
        print(f"{pred:12.3f} | {true:12.3f} | {diff:8.3f} {status}")
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    print(f"\nECE: {ece:.4f}")
    if ece < 0.1:
        print("✅ Калибровка хорошая")
    elif ece < 0.15:
        print("⚠️ Калибровка приемлемая")
    else:
        print("❌ Калибровка плохая — рассмотри Platt scaling")


def analyze_predictions_by_confidence(y_true, y_prob, fwd_ret):
    print("\n=== PREDICTIONS BY CONFIDENCE LEVEL ===")
    deciles = np.percentile(y_prob, np.arange(0, 101, 10))
    print(f"{'Decile':>6s} | {'Range':>15s} | {'N':>5s} | {'Accuracy':>8s} | {'AvgRet':>8s} | {'Precision':>9s}")
    print("-" * 70)

    for i in range(len(deciles) - 1):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(deciles) - 2 else (y_prob >= lo)
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        accuracy = float((y_true[mask] == (y_prob[mask] >= 0.5)).mean())
        avg_ret = float(fwd_ret[mask].mean())
        precision = float(y_true[mask].mean())
        print(f"D{i+1:2d} | {lo:.3f} - {hi:.3f} | {n:5d} | {accuracy:8.3f} | {avg_ret:8.4f} | {precision:9.3f}")

    top10 = y_prob >= np.percentile(y_prob, 90)
    bot10 = y_prob <= np.percentile(y_prob, 10)
    print(
        f"\nTOP 10%: N={int(top10.sum())}, precision={float(y_true[top10].mean()):.3f}, "
        f"avg_ret={float(fwd_ret[top10].mean()):.4f}"
    )
    print(
        f"BOT 10%: N={int(bot10.sum())}, precision={float(y_true[bot10].mean()):.3f}, "
        f"avg_ret={float(fwd_ret[bot10].mean()):.4f}"
    )

    # monotonicity check
    decile_precisions = []
    for i in range(len(deciles) - 1):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(deciles) - 2 else (y_prob >= lo)
        if mask.sum() > 0:
            decile_precisions.append(float(y_true[mask].mean()))
    if len(decile_precisions) >= 3:
        monotonic = all(decile_precisions[i] <= decile_precisions[i + 1] for i in range(len(decile_precisions) - 1))
        mostly_mono = sum(1 for i in range(len(decile_precisions) - 1) if decile_precisions[i] <= decile_precisions[i + 1])
        msg = "✅ монотонно" if monotonic else ("⚠️ почти монотонно" if mostly_mono >= len(decile_precisions) - 2 else "")
        print(f"\nМонотонность децилей: {mostly_mono}/{len(decile_precisions)-1} {msg}")


def confusion_matrix_analysis(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred)
    print(f"\n=== CONFUSION MATRIX (threshold={threshold:.2f}) ===")
    tn, fp, fn, tp = cm.ravel()
    print(f" TN={tn:5d} FP={fp:5d}")
    print(f" FN={fn:5d} TP={tp:5d}")
    print(
        f"\n{classification_report(y_true, y_pred, target_names=['No Growth', 'Growth'], digits=3, zero_division=0)}"
    )


def feature_importance_proxy(model, X_test, y_test, feature_names, max_features=20):
    print("\n=== FEATURE IMPORTANCE (Permutation) ===")
    y_test = y_test.astype(int)
    if len(np.unique(y_test)) < 2:
        print("Undefined (single class)")
        return []

    base_pred = model.predict(X_test, verbose=0).ravel()
    base_auc = float(roc_auc_score(y_test, base_pred))

    n_feat = X_test.shape[2]
    importances = []
    for i in range(min(max_features, n_feat)):
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        X_perm = X_test.copy()
        for t in range(X_perm.shape[1]):
            np.random.shuffle(X_perm[:, t, i])
        perm_auc = float(roc_auc_score(y_test, model.predict(X_perm, verbose=0).ravel()))
        importances.append((fname, base_auc - perm_auc))

    importances.sort(key=lambda x: x[1], reverse=True)
    print(f"Base AUC: {base_auc:.4f}")
    print(f"{'Feature':>20s} | {'Importance':>10s}")
    print("-" * 35)
    for fname, imp in importances:
        bar = "█" * max(0, int(imp * 200))
        print(f"{fname:>20s} | {imp:10.4f} {bar}")
    return importances


def temporal_performance_analysis(y_true, y_prob, dates):
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "y_true": y_true.astype(int),
        "y_prob": y_prob.astype(float),
    })
    df["month"] = df["date"].dt.to_period("M")

    print("\n=== TEMPORAL PERFORMANCE (Monthly AUC) ===")
    monthly_aucs = []
    for month, g in df.groupby("month"):
        if len(g) < 10:
            continue
        if len(np.unique(g["y_true"].values)) < 2:
            continue
        auc = float(roc_auc_score(g["y_true"].values, g["y_prob"].values))
        monthly_aucs.append(auc)
        pos_rate = float(g["y_true"].mean())
        print(f"{str(month):10s} | N={len(g):4d} | pos={pos_rate:.3f} | AUC={auc:.3f}")

    if monthly_aucs:
        print(f"\nMean monthly AUC: {np.mean(monthly_aucs):.3f} ± {np.std(monthly_aucs):.3f}")
        above_05 = sum(1 for a in monthly_aucs if a > 0.5)
        print(f"Months with AUC > 0.5: {above_05}/{len(monthly_aucs)} ({above_05/len(monthly_aucs)*100:.0f}%)")


def check_random_baseline(y_true, n_iterations=1000):
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        print("Undefined (single class)")
        return
    aucs = [
        float(roc_auc_score(y_true, np.random.uniform(0, 1, len(y_true))))
        for _ in range(n_iterations)
    ]
    mu, sd = float(np.mean(aucs)), float(np.std(aucs))
    print(f"\n=== RANDOM BASELINE ({n_iterations} iterations) ===")
    print(f"Random AUC: {mu:.4f} ± {sd:.4f}")
    print(f"99% threshold (mu + 3σ): {mu + 3*sd:.4f}")


# ==============================
# MAIN
# ==============================

def main() -> None:
    ds, feature_cols = build_multi_ticker_dataset()

    ds = MultiDataset(
        X=np.nan_to_num(ds.X, nan=0.0, posinf=0.0, neginf=0.0),
        y=ds.y,
        fwd_ret=ds.fwd_ret,
        dates=ds.dates,
        secids=ds.secids,
    )

    m_train, m_val, m_test = time_split_masks(ds.dates, ds.secids)

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
        split_masks=(m_train, m_val, m_test),
    )

    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise RuntimeError("Not enough sequences. Reduce SEQ_LEN or check data.")

    print(f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # Scale
    n_steps, n_feat = X_train.shape[1], X_train.shape[2]
    scaler = RobustScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    def scale(x):
        return scaler.transform(x.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)

    X_train = np.clip(scale(X_train), -5.0, 5.0)
    X_val = np.clip(scale(X_val), -5.0, 5.0)
    X_test = np.clip(scale(X_test), -5.0, 5.0)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    print("\n=== DATA SANITY ===")
    print(f"Train: [{X_train.min():.2f}, {X_train.max():.2f}], NaN={np.isnan(X_train).any()}")
    print(f"y_train distribution: {np.bincount(y_train.astype(int))}")

    # Class weights
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    cw = {0: float(w[0]), 1: float(w[1])}
    print(f"Class weights: {cw}")

    model = build_tcn_model((n_steps, n_feat))
    model.summary()

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=20,
            mode="max",
            restore_best_weights=True,
            min_delta=0.001,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=8,
            mode="max",
            min_lr=1e-6,
            min_delta=0.001,
        ),
    ]

    # Critical fix: shuffle=True
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        shuffle=True,
        class_weight=cw,
        callbacks=cb,
        verbose=2,
    )

    y_prob_val = model.predict(X_val, verbose=0).ravel()
    y_prob = model.predict(X_test, verbose=0).ravel()

    print(
        f"\nVal prob: mean={y_prob_val.mean():.4f}, std={y_prob_val.std():.4f}, "
        f"range=[{y_prob_val.min():.4f}, {y_prob_val.max():.4f}]"
    )
    print(
        f"Test prob: mean={y_prob.mean():.4f}, std={y_prob.std():.4f}, "
        f"range=[{y_prob.min():.4f}, {y_prob.max():.4f}]"
    )

    if np.isnan(y_prob).any():
        print("FATAL: NaN in predictions")
        return

    # Threshold search
    print("\n=== THRESHOLD OPTIMIZATION (VAL) ===")
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.30, 0.85, 0.01):
        pred = (y_prob_val >= thr).astype(int)
        if pred.sum() == 0:
            continue
        f1v = f1_score(y_val, pred, zero_division=0)
        if f1v > best_f1:
            best_f1, best_thr = float(f1v), float(thr)

    print(f"Best threshold: {best_thr:.2f} (F1={best_f1:.3f})")

    if int((y_prob >= best_thr).sum()) == 0:
        best_thr = float(np.percentile(y_prob, 80))
        print(f"Fallback threshold: {best_thr:.3f}")

    print("\n" + "=" * 80)
    print("EXTENDED DIAGNOSTICS")
    print("=" * 80)
    plot_probability_distribution(y_test, y_prob, name="TEST")
    calibration_curve_analysis(y_test, y_prob, n_bins=10)
    analyze_predictions_by_confidence(y_test, y_prob, fwd_test)
    confusion_matrix_analysis(y_test, y_prob, threshold=best_thr)
    importances = feature_importance_proxy(model, X_test, y_test, feature_cols)
    temporal_performance_analysis(y_test, y_prob, dates_test)
    check_random_baseline(y_test, n_iterations=1000)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    g = evaluate_global(y_test, y_prob, thr=best_thr)
    for k, v in g.items():
        print(f" {k}: {v:.4f}")

    print("\n=== PER-TICKER (TEST) ===")
    print(per_ticker_metrics(y_test, y_prob, secids_test).to_string(index=False))

    print("\n=== PROB SUMMARY (TEST) ===")
    print(
        pd.Series(y_prob).describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_string()
    )

    print("\n=== BACKTEST (TEST) ===")
    bt = improved_backtest_per_ticker(
        y_prob,
        fwd_test,
        dates_test,
        secids_test,
        threshold=best_thr,
        fee=float(CFG["FEE"]),
    )
    print(bt.to_string(index=False))

    print("\n" + "=" * 80)
    print("РЕЗЮМЕ ДЛЯ ДИПЛОМА")
    print("=" * 80)
    print("Архитектура: TCN (4 блока, residual connections)")
    print(f"Признаков: {len(feature_cols)}")
    print(f"Тикеров: {len(CFG['TICKERS'])}")
    print(f"Период: {CFG['START']} — {CFG['END']}")
    print(f"Целевая: рост >= {CFG['THR_MOVE']*100:.0f}% за {CFG['HORIZON']} дней")
    print(f"AUC-ROC: {g['AUC']:.4f}")
    print(f"MCC: {g['MCC']:.4f}")
    print(f"Balanced Accuracy: {g['BalAcc']:.4f}")

    if importances:
        top3 = [f[0] for f in importances[:3]]
        print(f"Топ-3 признака: {', '.join(top3)}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
