
# chat gpt
# Colab-ready: MOEX/CBR feature engineering + LSTM classifier (ChatGPT variant)
# Goal: run end-to-end in one notebook/script.
#
# In Colab:
# !pip -q install moexalgo requests pandas numpy scikit-learn tensorflow lxml html5lib keras-tcn

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from tcn import TCN
from moexalgo import Ticker

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
    average_precision_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.utils.class_weight import compute_class_weight

# ----------------------------
# CONFIG
# ----------------------------
CFG = {
    "TICKER": "SBER",
    "START": "2015-01-01",
    "END": "2025-12-31",

    "HORIZON": 5,        # target horizon in trading days
    "THR_MOVE": 0.015,   # target: future_return > 1.5%

    "WINDOW": 30,

    # time split (by dates order)
    "TRAIN_FRAC": 0.70,
    "VAL_FRAC": 0.15,  # rest goes to test

    # training
    "EPOCHS": 200,
    "BATCH": 32,
    "LR": 1e-3,
    "PATIENCE": 15,

    # backtest assumptions (simple)
    "FEE": 0.001,        # 0.1% per trade (simplified)
    "NON_OVERLAP": True, # skip next HORIZON days after entry
}

np.random.seed(42)
tf.random.set_seed(42)

# ----------------------------
# 1) DATA LOADERS (Russian sources)
# ----------------------------
def load_candles_moexalgo(ticker: str, start: str, end: str, period: str = "1D") -> pd.DataFrame:
    df = pd.DataFrame(Ticker(ticker).candles(start=start, end=end, period=period))
    if df.empty:
        return df
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.drop_duplicates(subset=["begin"]).set_index("begin").sort_index()
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return df[["Open", "High", "Low", "Close", "Volume"]]


def cbr_key_rate_range(start: str, end: str) -> pd.DataFrame:
    """CBR key rate history within a range. Returns daily table (date, key_rate)."""
    url = "https://www.cbr.ru/hd_base/KeyRate/"
    params = {
        "UniDbQuery.Posted": "True",
        "UniDbQuery.From": pd.to_datetime(start).strftime("%d.%m.%Y"),
        "UniDbQuery.To": pd.to_datetime(end).strftime("%d.%m.%Y"),
    }

    html = requests.get(url, params=params, timeout=30).text
    tables = pd.read_html(html)
    if not tables:
        return pd.DataFrame(columns=["date", "key_rate"])

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    # identify date/rate columns (ru)
    date_col = None
    rate_col = None
    for c in df.columns:
        lc = c.lower()
        if date_col is None and "дата" in lc:
            date_col = c
        if rate_col is None and ("став" in lc or "ключ" in lc):
            rate_col = c

    if date_col is None or rate_col is None:
        return pd.DataFrame(columns=["date", "key_rate"])

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[rate_col] = (
        df[rate_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna().sort_values(date_col).reset_index(drop=True)
    df = df.rename(columns={date_col: "date", rate_col: "key_rate"})

    # fix if parsed as 1500 instead of 15.00
    if not df.empty and df["key_rate"].max() > 100:
        df["key_rate"] = df["key_rate"] / 100.0

    return df


def moex_iss_dividends(ticker: str) -> pd.DataFrame:
    """Best-effort dividends via MOEX ISS."""
    url = f"https://iss.moex.com/iss/securities/{ticker}/dividends.json"
    j = requests.get(url, params={"iss.meta": "off"}, timeout=30).json()

    div = j.get("dividends", {})
    if not div or not div.get("data"):
        return pd.DataFrame(columns=["date", "dividend_rub", "currency"])

    df = pd.DataFrame(div["data"], columns=div["columns"])
    if "registryclosedate" not in df.columns or "value" not in df.columns:
        return pd.DataFrame(columns=["date", "dividend_rub", "currency"])

    out = (
        pd.DataFrame(
            {
                "date": pd.to_datetime(df["registryclosedate"], errors="coerce"),
                "dividend_rub": pd.to_numeric(df["value"], errors="coerce"),
                "currency": df.get("currencyid", "RUB"),
            }
        )
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


# ----------------------------
# 2) FEATURE ENGINEERING
# ----------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def add_dividend_past_only_features(df: pd.DataFrame, div: pd.DataFrame) -> pd.DataFrame:
    """Non-leaky: only last known dividend (as-of merge backward)."""
    out = df.copy().sort_index()

    tmp = out.reset_index().rename(columns={out.index.name or out.reset_index().columns[0]: "date"})
    tmp["date"] = pd.to_datetime(tmp["date"])

    tmp["last_dividend"] = 0.0
    tmp["days_since_last_dividend"] = 9999
    tmp["last_div_yield_approx"] = 0.0

    if div is None or div.empty:
        out = tmp.set_index("date")
        return out

    div2 = div.copy()
    div2["date"] = pd.to_datetime(div2["date"], errors="coerce")
    div2 = div2.dropna().sort_values("date")

    # past-only merge: last dividend by date
    tmp = pd.merge_asof(tmp.sort_values("date"), div2[["date", "dividend_rub"]].sort_values("date"), on="date", direction="backward")
    tmp["dividend_rub"] = tmp["dividend_rub"].fillna(0.0)

    tmp["last_dividend"] = tmp["dividend_rub"]
    # approximate yield
    tmp["last_div_yield_approx"] = tmp["last_dividend"] / (tmp["Close"].replace(0, np.nan))
    tmp["last_div_yield_approx"] = tmp["last_div_yield_approx"].fillna(0.0)

    # days since last dividend
    last_div_date = tmp["date"].where(tmp["dividend_rub"] > 0).ffill()
    tmp["days_since_last_dividend"] = (tmp["date"] - last_div_date).dt.days.fillna(9999).astype(int)

    out = tmp.set_index("date")
    return out


def build_features(
    sber: pd.DataFrame,
    usd: pd.DataFrame,
    imo: pd.DataFrame,
    key_rate: pd.DataFrame,
    divs: pd.DataFrame,
) -> pd.DataFrame:
    df = sber.copy()

    # returns
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_2"] = df["Close"].pct_change(2)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)
    df["ret_20"] = df["Close"].pct_change(20)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # SMA distances
    sma20 = df["Close"].rolling(20).mean()
    sma50 = df["Close"].rolling(50).mean()
    sma200 = df["Close"].rolling(200).mean()
    df["dist_sma20"] = (df["Close"] - sma20) / (sma20 + 1e-12)
    df["dist_sma50"] = (df["Close"] - sma50) / (sma50 + 1e-12)
    df["trend_up_200"] = (df["Close"] > sma200).astype(int)

    # RSI
    df["rsi_14"] = rsi(df["Close"], 14)

    # Volatility
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()
    df["vol_rel"] = df["vol_20"] / (df["vol_60"] + 1e-12)

    # ATR-like relative range
    df["hl_range"] = (df["High"] - df["Low"]) / (df["Close"].shift(1) + 1e-12)
    df["atr_rel"] = df["hl_range"].rolling(14).mean()

    # Bollinger band diagnostics
    mid = df["Close"].rolling(20).mean()
    sd = df["Close"].rolling(20).std()
    upper = mid + 2 * sd
    lower = mid - 2 * sd
    df["bb_width"] = (upper - lower) / (mid + 1e-12)
    df["bb_pos"] = (df["Close"] - lower) / ((upper - lower) + 1e-12)

    # Volume features
    v20 = df["Volume"].rolling(20).mean()
    v60 = df["Volume"].rolling(60).mean()
    df["vol_ratio_5_20"] = df["Volume"].rolling(5).mean() / (v20 + 1e-12)
    df["vol_spike"] = (df["Volume"] > (v20 + 2 * df["Volume"].rolling(20).std())).astype(int)
    # Market / FX context (align to SBER calendar first)
    usd_close = usd["Close"].reindex(df.index).ffill()
    imo_close = imo["Close"].reindex(df.index).ffill()

    df["usd_ret_1"] = usd_close.pct_change(1)
    df["usd_ret_5"] = usd_close.pct_change(5)

    df["imoex_ret_1"] = imo_close.pct_change(1)
    df["imoex_ret_5"] = imo_close.pct_change(5)
    df["imoex_ret_20"] = imo_close.pct_change(20)

    df["sber_vs_imoex_5"] = df["ret_5"] - df["imoex_ret_5"]

    # Macro: key rate derivatives only
    if key_rate is None or key_rate.empty:
        df["key_rate"] = np.nan
    else:
        kr = key_rate.copy()
        kr["date"] = pd.to_datetime(kr["date"], errors="coerce")
        kr = kr.dropna().drop_duplicates(subset=["date"]).sort_values("date")
        kr = kr.set_index("date").sort_index()
        kr_daily = kr.reindex(df.index, method="ffill")
        df["key_rate"] = kr_daily["key_rate"]

    df["key_rate_chg"] = df["key_rate"].diff()
    df["rate_rising"] = (df["key_rate_chg"] > 0).astype(int)

    # Dividends (optional features; may be excluded from FEATURES)
    if divs is not None and not divs.empty:
        df = add_dividend_past_only_features(df, divs)
    else:
        df["last_dividend"] = 0.0
        df["days_since_last_dividend"] = 9999
        df["last_div_yield_approx"] = 0.0

    df = df.dropna().copy()
    return df


def add_target(df: pd.DataFrame, horizon: int, thr: float) -> pd.DataFrame:
    out = df.copy()
    out["future_close"] = out["Close"].shift(-horizon)
    out["future_ret"] = (out["future_close"] - out["Close"]) / (out["Close"] + 1e-12)
    out["Target"] = (out["future_ret"] >= thr).astype(int)
    out = out.dropna().copy()
    return out


def time_splits(df: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = df.index[:n_train]
    val_idx = df.index[n_train : n_train + n_val]
    test_idx = df.index[n_train + n_val :]
    return train_idx, val_idx, test_idx


def make_windows_aligned(X_2d: np.ndarray, y_1d: np.ndarray, dates_1d: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make windows and align y/dates to the LAST index of each window."""
    Xw, yw, dw = [], [], []
    for i in range(window - 1, len(X_2d)):
        Xw.append(X_2d[i - window + 1 : i + 1])
        yw.append(y_1d[i])
        dw.append(dates_1d[i])
    return np.asarray(Xw), np.asarray(yw), np.asarray(dw)


def build_tcn_model(window: int, n_features: int, lr: float) -> tf.keras.Model:
    """
    TCN‑архитектура как замена LSTM для задачи классификации.
    """
    inp = tf.keras.Input(shape=(window, n_features))

    # TCN блок с несколькими уровнями расширенной (dilated) свертки
    x = TCN(
        nb_filters=64,  # число фильтров; можно снизить до 32 для уменьшения переобучения
        kernel_size=3,  # размер окна свертки
        nb_stacks=1,  # 1 стек => 1 “слой” TCN
        dilations=(1, 2, 4, 8),  # расширенная (dilated) свертка
        padding="causal",  # только прошлые данные, без lookahead
        use_skip_connections=True,  # residual‑skip, как в оригинальной TCN
        dropout_rate=0.3,  # dropout внутри TCN‑блока
        return_sequences=False,  # нужен только последний шаг (для классификации)
        activation="relu",
        kernel_initializer="he_normal",
    )(inp)

    # Post‑TCN блок (аналог Dense‑слоя после LSTM)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )
    return model


def non_overlap_pnl(pred: np.ndarray, future_ret: np.ndarray, horizon: int, fee: float) -> Tuple[float, int]:
    """Average return per trade, non-overlapping trades (simple)."""
    i = 0
    trades = []
    while i < len(pred):
        if pred[i] == 1:
            trades.append(float(future_ret[i] - fee))
            i += horizon
        else:
            i += 1
    if not trades:
        return 0.0, 0
    return float(np.mean(trades)), int(len(trades))


def pick_threshold_on_val(
    y_true_val: np.ndarray,
    prob_val: np.ndarray,
    future_ret_val: np.ndarray,
    horizon: int,
    fee: float,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)

    rows = []
    for thr in thresholds:
        pred = (prob_val >= thr).astype(int)

        f1_1 = f1_score(y_true_val, pred, pos_label=1, zero_division=0)
        bal = balanced_accuracy_score(y_true_val, pred)
        mcc = matthews_corrcoef(y_true_val, pred) if len(np.unique(pred)) > 1 else 0.0

        avg_pnl, n_tr = non_overlap_pnl(pred, future_ret_val, horizon, fee)

        MIN_TRADES = 15
        MIN_BUY_RATE = 0.05
        MAX_BUY_RATE = 0.35
        feasible = (n_tr >= MIN_TRADES) and (MIN_BUY_RATE <= pred.mean() <= MAX_BUY_RATE)
        if not feasible:
            avg_pnl = -1e9

        rows.append(
            {
                "thr": float(thr),
                "f1_class1": float(f1_1),
                "balanced_acc": float(bal),
                "mcc": float(mcc),
                "share_buy": float(pred.mean()),
                "avg_trade_ret_nonoverlap": float(avg_pnl),
                "n_trades_nonoverlap": int(n_tr),
            }
        )

    tab = pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)

    # feasible = not those where we forced -1e9
    tab_feas = tab[tab["avg_trade_ret_nonoverlap"] > -1e8].copy()

    if tab_feas.empty:
        # if no threshold passed constraints, fall back to 0.50
        thr_f1 = 0.50
        thr_pnl = 0.50
    else:
        thr_f1 = float(tab_feas.iloc[tab_feas["f1_class1"].values.argmax()]["thr"])
        thr_pnl = float(tab_feas.iloc[tab_feas["avg_trade_ret_nonoverlap"].values.argmax()]["thr"])

    return float(thr_f1), float(thr_pnl), tab


def eval_block(name: str, y_true: np.ndarray, prob: np.ndarray, thr: float):
    pred = (prob >= thr).astype(int)
    print("\n" + "=" * 70)
    print(f"{name} | threshold={thr:.2f}")
    print("=" * 70)

    print(f"Accuracy: {accuracy_score(y_true, pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, pred):.3f}")
    print(f"F1 macro: {f1_score(y_true, pred, average='macro', zero_division=0):.3f}")
    print(f"F1 (BUY=1): {f1_score(y_true, pred, pos_label=1, zero_division=0):.3f}")
    print(f"MCC: {(matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0):.3f}")

    if len(np.unique(y_true)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_true, prob):.3f}")
        print(f"PR-AUC: {average_precision_score(y_true, prob):.3f}")

    cm = confusion_matrix(y_true, pred)
    print("\nConfusion matrix:")
    print(f"TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"FN={cm[1,0]}, TP={cm[1,1]}")

    print("\nClassification report:")
    print(classification_report(y_true, pred, zero_division=0, target_names=["No Growth", "Growth>thr"]))

    print("\nPred distribution:")
    print(f"Pred HOLD=0: {(pred==0).mean():.1%} | Pred BUY=1: {(pred==1).mean():.1%}")

    base0 = np.zeros_like(y_true)
    base1 = np.ones_like(y_true)
    print("\nBaselines:")
    print(f"Always HOLD (0) accuracy: {accuracy_score(y_true, base0):.3f}")
    print(f"Always BUY  (1) accuracy: {accuracy_score(y_true, base1):.3f}")


# ----------------------------
# EXTRA DIAGNOSTICS (probabilities, calibration, drift)
# ----------------------------

def ece_score(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    y_true = y_true.astype(int)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(prob[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def prob_summary(name: str, y_true: np.ndarray, prob: np.ndarray) -> None:
    y_true = y_true.astype(int)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    pos = float(y_true.mean())
    print("\n" + "=" * 70)
    print(f"PROB SUMMARY: {name}")
    print("=" * 70)
    print(f"Samples: {len(y_true)} | Pos rate (BUY=1): {pos:.3f}")
    print(
        f"Prob mean={prob.mean():.3f} std={prob.std():.3f} "
        f"p05={np.quantile(prob,0.05):.3f} p50={np.quantile(prob,0.50):.3f} p95={np.quantile(prob,0.95):.3f}"
    )

    # ranking
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, prob))
        auc_inv = float(roc_auc_score(y_true, 1 - prob))
        ap = float(average_precision_score(y_true, prob))
        print(f"ROC-AUC(prob): {auc:.3f}")
        print(f"ROC-AUC(1-prob): {auc_inv:.3f} (если > ROC-AUC(prob), сигнал мог 'перевернуться')")
        print(f"PR-AUC(AP): {ap:.3f}")

    # probability quality
    ll = float(log_loss(y_true, prob))
    bs = float(brier_score_loss(y_true, prob))
    ece = float(ece_score(y_true, prob, n_bins=10))
    print(f"LogLoss: {ll:.3f} | Brier: {bs:.3f} | ECE(10 bins): {ece:.3f}")

    # baseline logloss from constant probability = pos_rate
    base_prob = np.full_like(prob, fill_value=pos, dtype=float)
    base_ll = float(log_loss(y_true, np.clip(base_prob, 1e-6, 1 - 1e-6)))
    print(f"Baseline LogLoss (const p=pos_rate): {base_ll:.3f}")


def threshold_sweep(name: str, y_true: np.ndarray, prob: np.ndarray, thresholds=None, top_k: int = 10) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)

    rows = []
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        rows.append({
            "thr": float(thr),
            "buy_rate": float(pred.mean()),
            "acc": float(accuracy_score(y_true, pred)),
            "bal_acc": float(balanced_accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "f1_buy": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
            "prec_buy": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
            "rec_buy": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0),
        })

    tab = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"THRESHOLD SWEEP: {name} (top by F1-macro)")
    print("=" * 70)
    print(tab.sort_values("f1_macro", ascending=False).head(top_k).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"THRESHOLD SWEEP: {name} (top by MCC)")
    print("=" * 70)
    print(tab.sort_values("mcc", ascending=False).head(top_k).to_string(index=False))

    return tab


def decile_report(name: str, y_true: np.ndarray, prob: np.ndarray, future_ret: Optional[np.ndarray] = None) -> None:
    """Check: higher prob => higher BUY rate and (optionally) higher future return."""
    df = pd.DataFrame({"y": y_true.astype(int), "p": prob.astype(float)})
    if future_ret is not None:
        df["fret"] = future_ret.astype(float)

    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        buy_rate=("y", "mean"),
    )
    if future_ret is not None:
        g["avg_future_ret"] = df.groupby("decile")["fret"].mean()

    print("\n" + "=" * 70)
    print(f"DECILE REPORT: {name}")
    print("=" * 70)
    print(g.reset_index().to_string(index=False))


def psi_1d(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index for one feature (train defines bins)."""
    train = train[~np.isnan(train)]
    test = test[~np.isnan(test)]
    if len(train) < 50 or len(test) < 50:
        return float('nan')

    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(train, qs))
    if len(bins) <= 2:
        return float('nan')

    tr_hist, _ = np.histogram(train, bins=bins)
    te_hist, _ = np.histogram(test, bins=bins)

    tr = tr_hist / max(tr_hist.sum(), 1)
    te = te_hist / max(te_hist.sum(), 1)
    tr = np.clip(tr, 1e-6, None)
    te = np.clip(te, 1e-6, None)

    return float(np.sum((te - tr) * np.log(te / tr)))


def drift_report_features(X_train_2d: np.ndarray, X_test_2d: np.ndarray, feature_names: list, top_k: int = 10) -> None:
    """Feature drift report using PSI(train->test)."""
    psis = []
    for j, name in enumerate(feature_names):
        p = psi_1d(X_train_2d[:, j], X_test_2d[:, j], n_bins=10)
        if p is None or np.isnan(p):
            continue
        psis.append((str(name), float(p)))
    psis.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("FEATURE DRIFT (PSI train→test): top")
    print("=" * 70)
    for n, p in psis[:top_k]:
        flag = " !!!" if p >= 0.25 else (" !!" if p >= 0.10 else "")
        print(f"{n:25s} PSI={p:.3f}{flag}")

    # PSI interpretation:
    # <0.10 — ok
    # 0.10–0.25 — moderate drift
    # >0.25 — strong drift


# ----------------------------
# 5) PIPELINE
# ----------------------------
print("Loading market series from MOEX...")
sber = load_candles_moexalgo(CFG["TICKER"], CFG["START"], CFG["END"])
usd = load_candles_moexalgo("USD000UTSTOM", CFG["START"], CFG["END"])
imo = load_candles_moexalgo("IMOEX", CFG["START"], CFG["END"])
print("SBER:", sber.shape, "USD:", usd.shape, "IMOEX:", imo.shape)

print("\nLoading CBR key rate...")
key_rate = cbr_key_rate_range(CFG["START"], CFG["END"])
print("Key rate rows:", len(key_rate))

print("\nLoading MOEX dividends...")
divs = moex_iss_dividends(CFG["TICKER"])
print("Div rows:", len(divs))

print("\nBuilding features (Russian sources only)...")
feat = build_features(sber, usd, imo, key_rate, divs)
feat = add_target(feat, CFG["HORIZON"], CFG["THR_MOVE"])

# Optional (drift reduction): train/val/test on a fresher regime only
# feat = feat.loc["2019-01-01":].copy()
print("Final dataset:", feat.shape)
print("Class share (BUY=1):", float(feat["Target"].mean().round(3)))


def augment_with_smartlab(df: pd.DataFrame, smartlab_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Reserved hook for future Smart-Lab integration (no-op for now)."""
    return df


feat = augment_with_smartlab(feat, smartlab_df=None)

# "Diploma" stable feature set (more robust / lower drift):
# - removed sparse/regime factors: dividends
# - removed raw key_rate and usd_ret_1
# - added broader context (5d/20d) and derived macro only
FEATURES = [
    # SBER returns/trend
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "log_ret",
    "dist_sma20", "dist_sma50", "trend_up_200",
    "rsi_14",

    # Volatility/volume
    "vol_20", "vol_60", "vol_rel",
    "atr_rel", "bb_width", "bb_pos",
    "vol_ratio_5_20", "vol_spike",

    # Market / FX context
    "imoex_ret_1", "imoex_ret_5", "imoex_ret_20",
    "usd_ret_5",
    "sber_vs_imoex_5",

    # Macro: only derivatives
    "key_rate_chg", "rate_rising",
]
FEATURES = [c for c in FEATURES if c in feat.columns]
print(f"Признаков используется: {len(FEATURES)}")
print(FEATURES)

train_idx, val_idx, test_idx = time_splits(feat, CFG["TRAIN_FRAC"], CFG["VAL_FRAC"])

scaler = RobustScaler()
X_all_2d = feat[FEATURES].values
y_all = feat["Target"].values.astype(int)
dates_all = feat.index.values
future_ret_all = feat["future_ret"].values.astype(float)

X_train_raw_2d = feat.loc[train_idx, FEATURES].values

CLIP_Q = 0.005  # 0.5% / 99.5%
lo = np.nanquantile(X_train_raw_2d, CLIP_Q, axis=0)
hi = np.nanquantile(X_train_raw_2d, 1 - CLIP_Q, axis=0)

X_all_2d = np.clip(X_all_2d, lo, hi)

scaler.fit(np.clip(X_train_raw_2d, lo, hi))
X_all_scaled = scaler.transform(X_all_2d)

Xw, yw, dw = make_windows_aligned(X_all_scaled, y_all, dates_all, CFG["WINDOW"])
future_ret_w = future_ret_all[CFG["WINDOW"] - 1 :]

train_mask = np.isin(dw, train_idx.values)
val_mask = np.isin(dw, val_idx.values)
test_mask = np.isin(dw, test_idx.values)

X_train, y_train = Xw[train_mask], yw[train_mask]
X_val, y_val = Xw[val_mask], yw[val_mask]
X_test, y_test = Xw[test_mask], yw[test_mask]

fret_val = future_ret_w[val_mask]

print("\nWindows shapes:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

classes = np.array([0, 1])
cw = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {0: float(cw[0]), 1: float(cw[1])}
print("class_weight:", class_weight)

model = build_tcn_model(CFG["WINDOW"], X_train.shape[2], CFG["LR"])
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_pr",
        mode="max",
        patience=CFG["PATIENCE"],
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc_pr",
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    ),
]

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=CFG["EPOCHS"],
    batch_size=CFG["BATCH"],
    class_weight=class_weight,
    shuffle=False,
    callbacks=callbacks,
    verbose=1,
)

prob_val = model.predict(X_val, verbose=0).reshape(-1)
prob_test = model.predict(X_test, verbose=0).reshape(-1)
prob_train = model.predict(X_train, verbose=0).reshape(-1)

# Orientation calibration on the last part of VAL (no TEST leakage)
N_CAL = min(120, len(y_val))
auc_tail = roc_auc_score(y_val[-N_CAL:], prob_val[-N_CAL:]) if len(np.unique(y_val[-N_CAL:])) > 1 else 0.5
auc_tail_inv = roc_auc_score(y_val[-N_CAL:], 1 - prob_val[-N_CAL:]) if len(np.unique(y_val[-N_CAL:])) > 1 else 0.5

FLIP = auc_tail_inv > auc_tail
print("\n" + "=" * 70)
print(f"VAL tail AUC={auc_tail:.3f} | AUC(1-p)={auc_tail_inv:.3f} | FLIP={FLIP}")
print("=" * 70)

if FLIP:
    prob_train = 1 - prob_train
    prob_val = 1 - prob_val
    prob_test = 1 - prob_test

# ---- Probabilities + probability quality ----
prob_summary("TRAIN", y_train, prob_train)
prob_summary("VAL", y_val, prob_val)
prob_summary("TEST", y_test, prob_test)

# ---- Threshold sweeps ----
sweep_val = threshold_sweep("VAL", y_val, prob_val)
sweep_test = threshold_sweep("TEST", y_test, prob_test)

# ---- Decile analysis (ranking) ----
fret_train = future_ret_w[train_mask]
fret_val = future_ret_w[val_mask]
fret_test = future_ret_w[test_mask]

decile_report("TRAIN", y_train, prob_train, fret_train)
decile_report("VAL", y_val, prob_val, fret_val)
decile_report("TEST", y_test, prob_test, fret_test)

# ---- Feature drift (PSI) ----
# Quick indicator: use the last timestep of each window
X_train_last = X_train[:, -1, :]
X_test_last = X_test[:, -1, :]
drift_report_features(X_train_last, X_test_last, FEATURES, top_k=12)

# ---- Probability drift ----
p_psi = psi_1d(prob_train, prob_test, n_bins=10)
print("\n" + "=" * 70)
print(f"PROB DRIFT PSI(train→test): {p_psi:.3f} (if >0.25 — strong shift)")
print("=" * 70)

thr_f1, thr_pnl, thr_table = pick_threshold_on_val(y_val, prob_val, fret_val, CFG["HORIZON"], CFG["FEE"])

print("\nTop thresholds by F1(BUY):")
print(thr_table.sort_values("f1_class1", ascending=False).head(8).to_string(index=False))
print("\nTop thresholds by avg non-overlap trade return:")
print(thr_table.sort_values("avg_trade_ret_nonoverlap", ascending=False).head(8).to_string(index=False))

print(f"\nChosen threshold (VAL max F1(BUY)): {thr_f1:.2f}")
print(f"Chosen threshold (VAL max avg PnL): {thr_pnl:.2f}")

eval_block("TEST (thr_f1)", y_test, prob_test, thr_f1)
eval_block("TEST (thr_pnl)", y_test, prob_test, thr_pnl)



def backtest_nonoverlap_long_only(prob, dates_signal, close_full, thr, horizon, fee):
    dates = pd.to_datetime(dates_signal)
    close_full = close_full.copy()
    close_full.index = pd.to_datetime(close_full.index)

    # Buy&Hold on the same segment
    start_date = dates[0]
    last_date = dates[-1]
    try:
        last_loc = close_full.index.get_loc(last_date)
        end_loc = min(last_loc + horizon, len(close_full.index) - 1)
        end_date = close_full.index[end_loc]
    except KeyError:
        end_date = close_full.index[-1]

    bh_ret = float(close_full.loc[end_date] / close_full.loc[start_date] - 1.0)

    eq = 1.0
    trades = []
    i = 0
    while i < len(prob):
        if prob[i] >= thr:
            d0 = dates[i]
            try:
                loc0 = close_full.index.get_loc(d0)
            except KeyError:
                i += 1
                continue

            loc1 = loc0 + horizon
            if loc1 >= len(close_full.index):
                break

            entry = float(close_full.iloc[loc0])
            exitp = float(close_full.iloc[loc1])
            ret = exitp / entry - 1.0 - fee

            eq *= (1.0 + ret)
            trades.append(ret)
            i += horizon
        else:
            i += 1

    strat_ret = float(eq - 1.0)
    n_trades = int(len(trades))
    winrate = float(np.mean(np.array(trades) > 0)) if n_trades else 0.0

    print("\n" + "=" * 70)
    print(f"BACKTEST (Close, non-overlap) | thr={thr:.2f} horizon={horizon} fee={fee:.2%}")
    print("=" * 70)
    print(f"Strategy return: {strat_ret:+.2%}")
    print(f"Buy&Hold return: {bh_ret:+.2%}")
    print(f"Alpha: {(strat_ret - bh_ret):+.2%}")
    print(f"Trades: {n_trades} | WinRate: {winrate:.1%}")

    return strat_ret, bh_ret, n_trades, winrate


# test window end dates
_dates_test = dw[test_mask]
_close_full = sber["Close"]

backtest_nonoverlap_long_only(prob_test, _dates_test, _close_full, thr_f1, CFG["HORIZON"], CFG["FEE"])
backtest_nonoverlap_long_only(prob_test, _dates_test, _close_full, thr_pnl, CFG["HORIZON"], CFG["FEE"])

model.save("tcn_ru_model.keras")
with open("tcn_ru_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved: lstm_ru_model.keras and lstm_ru_scaler.pkl")
