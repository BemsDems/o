#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-ticker RNN (LSTM + GRU) для классификации временных рядов MOEX.

Полный код проекта в одном файле для курсовой работы.
Архитектура: Bidirectional LSTM + GRU
Задача: прогнозирование роста цены акций на 5 дней вперёд

Использование:
    python multi_ticker_rnn_full.py

Зависимости:
- numpy, pandas
- tensorflow
- moexalgo
- scikit-learn

Примечание:
- Это учебный код. Для реального трейдинга требуется строгая валидация,
  учёт проскальзывания/ликвидности и проверка отсутствия leakage.
"""

from __future__ import annotations

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


# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

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
    "N_RUNS": 3,
    "FEE": 0.001,
    "EXTENDED_DIAGNOSTICS": True,
}


def seed_everything(seed: int | None = None) -> None:
    s = int(CFG["SEED"]) if seed is None else int(seed)
    np.random.seed(s)
    tf.random.set_seed(s)


# =============================================================================
# ЗАГРУЗКА ДАННЫХ И ФОРМИРОВАНИЕ ПРИЗНАКОВ
# =============================================================================


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


def build_features_one(df: pd.DataFrame, *, secid: str = "") -> pd.DataFrame:
    """Feature engineering — 15 признаков."""

    out = df.copy()

    n0 = len(out)
    if secid:
        print(f" candles: {n0}")

    # Log-returns
    for lag in (1, 2, 3, 5, 10):
        price_ratio = out["CLOSE"] / out["CLOSE"].shift(lag)
        price_ratio = price_ratio.clip(0.5, 2.0)
        out[f"logret_{lag}"] = np.log(price_ratio)
        out[f"logret_{lag}"] = out[f"logret_{lag}"].replace([np.inf, -np.inf], np.nan)

    # Trend: SMA
    out["sma_20"] = out["CLOSE"].rolling(20).mean()
    out["sma_50"] = out["CLOSE"].rolling(50).mean()
    out["sma_200"] = out["CLOSE"].rolling(200).mean()
    out["trend_up_20"] = (out["CLOSE"] > out["sma_20"]).astype(int)
    out["trend_up_50"] = (out["CLOSE"] > out["sma_50"]).astype(int)
    out["trend_up_200"] = (out["CLOSE"] > out["sma_200"]).astype(int)

    # Volume
    out["vol_ma_20"] = out["VOLUME"].rolling(20).mean()
    out["vol_rel"] = out["VOLUME"] / (out["vol_ma_20"] + 1e-9)
    out["vol_rel"] = out["vol_rel"].clip(0.1, 3.0)
    out["vol_spike"] = (out["vol_rel"] > 2.0).astype(int)

    # RSI
    out["rsi_14"] = compute_rsi(out["CLOSE"], 14)
    out["rsi_14"] = out["rsi_14"].clip(0.0, 100.0)
    out["rsi_oversold"] = (out["rsi_14"] < 30).astype(int)
    out["rsi_overbought"] = (out["rsi_14"] > 70).astype(int)

    # Price position
    high_20 = out["HIGH"].rolling(20).max()
    low_20 = out["LOW"].rolling(20).min()
    out["price_pos_20"] = (out["CLOSE"] - low_20) / ((high_20 - low_20) + 1e-9)
    out["price_pos_20"] = out["price_pos_20"].clip(0.0, 1.0)

    # Volatility
    out["volatility_20"] = out["logret_1"].rolling(20).std()
    out["volatility_20"] = out["volatility_20"].clip(0.0, 0.1)

    if secid:
        print(f" after indicators: {len(out)}")

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

    # Fill long-window indicators
    out["sma_200"] = out["sma_200"].ffill()
    out["trend_up_200"] = out["trend_up_200"].fillna(0).astype(int)

    if secid:
        n1 = len(out)
        share = (n1 / n0 * 100.0) if n0 else 0.0
        print(f" after dropna(critical): {n1} ({share:.1f}%)")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)

    if secid:
        ok = not out.isnull().any().any()
        print(f" final NaN check: {ok}")

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
        print(f" loaded rows: {len(df)}")
        if df.empty:
            print(" -> empty, skip")
            continue

        df_feat = build_features_one(df, secid=secid)

        min_rows = max(250, int(CFG["SEQ_LEN"]) + int(CFG["HORIZON"]) + 50)
        if len(df_feat) < min_rows:
            print(f" -> too few rows after features ({len(df_feat)} < {min_rows}), skip")
            continue

        y, fwd_ret = make_target(
            df_feat["CLOSE"], int(CFG["HORIZON"]), float(CFG["THR_MOVE"])
        )

        h = int(CFG["HORIZON"])
        df_feat = df_feat.iloc[:-h]
        y = y.iloc[:-h]
        fwd_ret = fwd_ret.iloc[:-h]

        print(f" final rows (after horizon trim): {len(df_feat)}")

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

    missing = [c for c in feature_cols if c not in full.columns]
    if missing:
        raise RuntimeError(
            "Missing expected feature columns. "
            f"Expected={len(feature_cols)}, missing={missing}. "
            "Check build_features_one() and upstream data columns."
        )

    X = full[feature_cols].values
    y = full["target"].astype(int).values
    fwd_ret = full["fwd_ret"].astype(float).values
    dates = pd.to_datetime(full["date"]).values
    secids = full["secid"].astype(str).values

    print(
        f"\nMulti-ticker dataset: X={X.shape}, pos_rate={y.mean():.3%}, "
        f"tickers={sorted(set(secids))}"
    )
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    return (
        MultiDataset(X=X, y=y, fwd_ret=fwd_ret, dates=dates, secids=secids),
        feature_cols,
    )


# =============================================================================
# ФОРМИРОВАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ
# =============================================================================


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
        Xs_tr,
        ys_tr,
        ds_tr,
        rs_tr,
        ss_tr,
        Xs_va,
        ys_va,
        ds_va,
        rs_va,
        ss_va,
        Xs_te,
        ys_te,
        ds_te,
        rs_te,
        ss_te,
    ) = results

    print(f"\nSequences: Train={len(Xs_tr)}, Val={len(Xs_va)}, Test={len(Xs_te)}")
    return (
        Xs_tr,
        ys_tr,
        ds_tr,
        rs_tr,
        ss_tr,
        Xs_va,
        ys_va,
        ds_va,
        rs_va,
        ss_va,
        Xs_te,
        ys_te,
        ds_te,
        rs_te,
        ss_te,
    )


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


# =============================================================================
# RNN МОДЕЛЬ (LSTM + GRU)
# =============================================================================


def build_rnn_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """RNN model with stacked LSTM + GRU layers for time-series classification."""

    x_in = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LayerNormalization()(x_in)

    # LSTM block 1 — bidirectional, 64 units
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_initializer="glorot_uniform",
        )
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # LSTM block 2 — bidirectional, 32 units
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
        )
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # GRU block — final recurrent layer, return last hidden state
    x = tf.keras.layers.GRU(
        32,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.1,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Dense head
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(x_in, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(CFG["LR"]),
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


# =============================================================================
# МЕТРИКИ КАЧЕСТВА
# =============================================================================


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
        "MCC": float(matthews_corrcoef(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan"),
        "BalAcc": float(balanced_accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1
        else float("nan"),
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
                "win_rate_fixed": float(np.mean([1 if t > 0 else 0 for t in trades_fixed]))
                if trades_fixed
                else 0.0,
                "n_trades_top20": len(trades_top20),
                "total_ret_top20": float(np.sum(trades_top20)) if trades_top20 else 0.0,
                "sharpe_fixed": float(np.mean(trades_fixed) / (np.std(trades_fixed) + 1e-9))
                if len(trades_fixed) > 1
                else 0.0,
            }
        )

    return pd.DataFrame(results)


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


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================


def main() -> None:
    # Build dataset ONCE (deterministic), then run multiple training seeds.
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
        X_train,
        y_train,
        _,
        _,
        _,
        X_val,
        y_val,
        _,
        _,
        _,
        X_test,
        y_test,
        dates_test,
        fwd_test,
        secids_test,
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

    base_seed = int(CFG.get("SEED", 42))
    n_runs = int(CFG.get("N_RUNS", 1))

    for run_idx in range(n_runs):
        run_seed = base_seed + run_idx
        print(
            f"\n\n================ RUN {run_idx + 1}/{n_runs} | SEED={run_seed} ================"
        )

        seed_everything(run_seed)
        tf.keras.backend.clear_session()

        model = build_rnn_model((n_steps, n_feat))
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
                patience=15,
                mode="max",
                min_lr=1e-6,
            ),
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
            continue

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

        # Diagnostics
        print("\n=== FINAL RESULTS ===")
        g = evaluate_global(y_test, y_prob, thr=best_thr)
        for k, v in g.items():
            print(f" {k}: {v:.4f}")

        print("\n=== PROB SUMMARY ===")
        print(
            f" min={y_prob.min():.4f}, max={y_prob.max():.4f}, std={y_prob.std():.4f}, "
            f"mean={y_prob.mean():.4f}"
        )

        if bool(CFG.get("EXTENDED_DIAGNOSTICS", True)):
            print("\n=== FEATURE IMPORTANCE (top-5 + negative) ===")
            importances = feature_importance_proxy(model, X_test, y_test, feature_cols)
            if importances:
                for fname, imp in importances[:5]:
                    print(f" {fname:>20s} | {imp:+.4f}")
                negs = [(f, i) for f, i in importances if i < 0]
                if negs:
                    print(" --- negative ---")
                    for fname, imp in negs:
                        print(f" {fname:>20s} | {imp:+.4f}")

            print("\n=== PER-TICKER AUC ===")
            print(per_ticker_metrics(y_test, y_prob, secids_test).to_string(index=False))

            print("\n=== BACKTEST ===")
            bt = improved_backtest_per_ticker(
                y_prob,
                fwd_test,
                dates_test,
                secids_test,
                threshold=best_thr,
                fee=float(CFG["FEE"]),
            )
            if not bt.empty:
                print(bt.to_string(index=False))
            else:
                print("No backtest rows")


if __name__ == "__main__":
    main()
