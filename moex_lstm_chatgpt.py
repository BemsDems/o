# Colab-ready: MOEX/CBR feature engineering + LSTM classifier (ChatGPT variant)
# Goal: run end-to-end in one notebook/script.
#
# In Colab:
# !pip -q install moexalgo requests pandas numpy scikit-learn tensorflow lxml html5lib

from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

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
    tables = pd.read_html(html, decimal=",", thousands=" ")
    if not tables:
        return pd.DataFrame(columns=["date", "key_rate"])

    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col = next((c for c in df.columns if "дата" in c.lower()), None)
    rate_col = next((c for c in df.columns if "став" in c.lower()), None)
    if date_col is None or rate_col is None:
        return pd.DataFrame(columns=["date", "key_rate"])

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df[rate_col] = (
        df[rate_col]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace("\xa0", "", regex=False)
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
    tmp["days_since_last_dividend"] = 9999.0
    tmp["last_div_yield_approx"] = 0.0

    if div is None or div.empty:
        return tmp.set_index("date")

    div2 = div[["date", "dividend_rub"]].dropna().copy()
    div2["date"] = pd.to_datetime(div2["date"])
    div2 = div2.sort_values("date").drop_duplicates(subset=["date"])
    div2["last_div_date"] = div2["date"]

    tmp = tmp.sort_values("date")
    merged = pd.merge_asof(
        tmp,
        div2[["date", "dividend_rub", "last_div_date"]],
        on="date",
        direction="backward",
    )

    merged["last_dividend"] = pd.to_numeric(merged["dividend_rub"], errors="coerce").fillna(0.0)
    merged["days_since_last_dividend"] = (merged["date"] - merged["last_div_date"]).dt.days
    merged["days_since_last_dividend"] = merged["days_since_last_dividend"].fillna(9999.0)

    merged["last_div_yield_approx"] = merged["last_dividend"] / merged["Close"].replace(0, np.nan)
    merged["last_div_yield_approx"] = merged["last_div_yield_approx"].fillna(0.0)

    merged = merged.drop(columns=["dividend_rub", "last_div_date"], errors="ignore")
    merged = merged.set_index("date").sort_index()
    return merged


def build_features(
    base: pd.DataFrame,
    usd: pd.DataFrame,
    imoex: pd.DataFrame,
    key_rate_df: pd.DataFrame,
    div_df: pd.DataFrame,
) -> pd.DataFrame:
    df = base.copy()

    # join exogenous close series (aligned on trading days)
    df = df.join(usd[["Close"]].rename(columns={"Close": "USD"}), how="left")
    df["USD"] = df["USD"].ffill()
    df = df.join(imoex[["Close"]].rename(columns={"Close": "IMOEX"}), how="left")
    df["IMOEX"] = df["IMOEX"].ffill()

    # returns
    df["ret_1"] = df["Close"].pct_change()
    df["ret_2"] = df["Close"].pct_change(2)
    df["ret_5"] = df["Close"].pct_change(5)

    df["usd_ret_1"] = df["USD"].pct_change()
    df["imoex_ret_1"] = df["IMOEX"].pct_change()

    # volatility & volume
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()
    df["vol_rel"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-12)

    # trend features
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()

    df["dist_sma20"] = df["Close"] / df["sma_20"] - 1.0
    df["dist_sma50"] = df["Close"] / df["sma_50"] - 1.0
    df["trend_up_200"] = (df["Close"] > df["sma_200"]).astype(int)

    # RSI
    df["rsi_14"] = rsi(df["Close"], 14)

    df["ret_10"] = df["Close"].pct_change(10)
    df["ret_20"] = df["Close"].pct_change(20)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # extra index/currency context
    df["usd_ret_5"] = df["USD"].pct_change(5)
    df["imoex_ret_5"] = df["IMOEX"].pct_change(5)
    df["imoex_ret_20"] = df["IMOEX"].pct_change(20)
    df["sber_vs_imoex_5"] = df["ret_5"] - df["imoex_ret_5"]

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)

    # Bollinger Bands (20)
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_up - bb_low) / bb_mid
    df["bb_pos"] = (df["Close"] - bb_low) / (bb_up - bb_low).replace(0, np.nan)

    # ATR (14)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]),
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_rel"] = df["atr_14"] / df["Close"]

    # volume spike / vol ratio
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_ratio_5_20"] = df["vol_5"] / (df["vol_20"] + 1e-12)
    df["vol_spike"] = (df["vol_rel"] > 2.0).astype(int)

    # add CBR key rate (as-of merge backward)
    df2 = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})
    df2["date"] = pd.to_datetime(df2["date"])

    if key_rate_df is not None and not key_rate_df.empty:
        kr = key_rate_df[["date", "key_rate"]].copy()
        kr["date"] = pd.to_datetime(kr["date"])
        df2 = df2.sort_values("date")
        kr = kr.sort_values("date")
        df2 = pd.merge_asof(df2, kr, on="date", direction="backward")
    else:
        df2["key_rate"] = np.nan

    df2 = df2.set_index("date")
    df2["key_rate"] = df2["key_rate"].ffill()
    df2["key_rate_chg"] = df2["key_rate"].diff()
    df2["rate_rising"] = (df2["key_rate_chg"] > 0).astype(int)

    # dividends (past-only)
    df2 = add_dividend_past_only_features(df2, div_df)

    # clean
    df2 = df2.dropna()
    return df2


# ----------------------------
# 3) TARGET + SPLITS
# ----------------------------
def add_target(df: pd.DataFrame, horizon: int, thr_move: float) -> pd.DataFrame:
    out = df.copy()
    out["future_ret"] = out["Close"].shift(-horizon) / out["Close"] - 1.0
    out["Target"] = (out["future_ret"] > thr_move).astype(int)
    out = out.dropna()
    return out


def time_splits(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_idx = df.index[:train_end]
    val_idx = df.index[train_end:val_end]
    test_idx = df.index[val_end:]
    return train_idx, val_idx, test_idx


def make_windows_aligned(X_2d: np.ndarray, y_1d: np.ndarray, end_dates: np.ndarray, window: int):
    """Windows aligned: include day t in window, y corresponds to t."""
    X, y, d = [], [], []
    for t in range(window - 1, len(X_2d)):
        X.append(X_2d[t - window + 1 : t + 1])
        y.append(y_1d[t])
        d.append(end_dates[t])
    return np.array(X), np.array(y), np.array(d)


# ----------------------------
# 4) MODEL
# ----------------------------
def build_lstm_model(window: int, n_features: int, lr: float):
    inp = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(32, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def pick_threshold_on_val(y_true_val, prob_val, future_ret_val, horizon: int, fee: float):
    """Picks threshold using validation only."""
    rows = []
    thresholds = np.arange(0.30, 0.71, 0.02)

    def non_overlap_pnl(signal: np.ndarray, future_ret: np.ndarray, horizon: int, fee: float):
        pnl = []
        i = 0
        while i < len(signal):
            if signal[i] == 1:
                pnl.append(float(future_ret[i]) - fee)
                i += horizon
            else:
                i += 1
        if len(pnl) == 0:
            return 0.0, 0
        return float(np.mean(pnl)), len(pnl)

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
    thr_f1 = float(tab.iloc[tab["f1_class1"].values.argmax()]["thr"])
    thr_pnl = float(tab.iloc[tab["avg_trade_ret_nonoverlap"].values.argmax()]["thr"])
    return thr_f1, thr_pnl, tab


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
print("Final dataset:", feat.shape)
print("Class share (BUY=1):", float(feat["Target"].mean().round(3)))


def augment_with_smartlab(df: pd.DataFrame, smartlab_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Reserved hook for future Smart-Lab integration (no-op for now)."""
    return df


feat = augment_with_smartlab(feat, smartlab_df=None)

FEATURES = [
    "ret_1",
    "ret_2",
    "ret_5",
    "rsi_14",
    "dist_sma20",
    "dist_sma50",
    "trend_up_200",
    "vol_20",
    "vol_60",
    "vol_rel",
    "usd_ret_1",
    "imoex_ret_1",
    "key_rate",
    "key_rate_chg",
    "rate_rising",
    "last_dividend",
    "days_since_last_dividend",
    "last_div_yield_approx",
]

FEATURES = [c for c in FEATURES if c in feat.columns]
print(f"Признаков используется: {len(FEATURES)}")
print(FEATURES)

train_idx, val_idx, test_idx = time_splits(feat, CFG["TRAIN_FRAC"], CFG["VAL_FRAC"])

scaler = StandardScaler()
X_all_2d = feat[FEATURES].values
y_all = feat["Target"].values.astype(int)
dates_all = feat.index.values
future_ret_all = feat["future_ret"].values.astype(float)

scaler.fit(feat.loc[train_idx, FEATURES].values)
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

model = build_lstm_model(CFG["WINDOW"], X_train.shape[2], CFG["LR"])
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

thr_f1, thr_pnl, thr_table = pick_threshold_on_val(y_val, prob_val, fret_val, CFG["HORIZON"], CFG["FEE"])

print("\nTop thresholds by F1(BUY):")
print(thr_table.sort_values("f1_class1", ascending=False).head(8).to_string(index=False))
print("\nTop thresholds by avg non-overlap trade return:")
print(thr_table.sort_values("avg_trade_ret_nonoverlap", ascending=False).head(8).to_string(index=False))

print(f"\nChosen threshold (VAL max F1(BUY)): {thr_f1:.2f}")
print(f"Chosen threshold (VAL max avg PnL): {thr_pnl:.2f}")

eval_block("TEST (thr_f1)", y_test, prob_test, thr_f1)
eval_block("TEST (thr_pnl)", y_test, prob_test, thr_pnl)

model.save("lstm_ru_model.keras")
with open("lstm_ru_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved: lstm_ru_model.keras and lstm_ru_scaler.pkl")
