# Colab-ready: MOEX/CBR feature engineering + LSTM classifier (Sonnet fixed + full diagnostics)
# Fixes requested:
# 1) First LSTM uses return_sequences=True (explicit)
# 2) Dividends are merged via merge_asof in past-only manner
#
# In Colab:
# !pip -q install moexalgo requests pandas numpy scikit-learn tensorflow lxml html5lib

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# HTTP helpers
# ----------------------------

def _get_json(url: str, *, params: dict | None = None, timeout: int = 30) -> dict:
    """GET url and decode JSON with better error messages.

    Fixes common issue: server returns HTML/text (502/403/etc) and .json() crashes
    with JSONDecodeError: Expecting value...

    Raises: RuntimeError with status, content-type and a response preview.
    """
    r = requests.get(url, params=params, timeout=timeout)
    ct = (r.headers.get('Content-Type') or '').lower()
    try:
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or '')[:300].replace('\n', ' ')
        raise RuntimeError(f'HTTP error for {url} status={r.status_code} ct={ct} preview={preview!r}') from e
    try:
        return r.json()
    except Exception as e:  # noqa: BLE001
        preview = (r.text or '')[:300].replace('\n', ' ')
        raise RuntimeError(f'Non-JSON response for {url} status={r.status_code} ct={ct} preview={preview!r}') from e
from sklearn.preprocessing import StandardScaler
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
# CONFIG
# ----------------------------
CFG: Dict[str, Any] = {
    "TICKER": "SBER",
    "START": "2015-01-01",
    "END": None,  # None => today
    "TARGET_HORIZON_DAYS": 5,
    "TARGET_UP_THRESHOLD": 0.01,
    "SEQ_LEN": 60,
    "TRAIN_SPLIT": 0.8,
    "VAL_SPLIT": 0.1,  # fraction of TRAIN part (time-based)
    "BATCH_SIZE": 64,
    "EPOCHS": 20,
    "LR": 1e-3,
    "SEED": 42,
    "CACHE_DIR": "cache",
    "DIAGNOSTICS": True,
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
# DATA: fetch + feature engineering
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
        # MOEX typically paginates by 100
        if len(data) < 100:
            break

    df = pd.DataFrame(all_rows, columns=cols)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce")
    df = df.dropna(subset=["TRADEDATE"]).sort_values("TRADEDATE").reset_index(drop=True)
    df = df.rename(columns={"TRADEDATE": "date"})
    for c in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_moex_dividends(secid: str) -> pd.DataFrame:
    url = (
        "https://iss.moex.com/iss/statistics/engines/stock/markets/shares/"
        f"securities/{secid}/dividends.json"
    )
    params = {"iss.meta": "off", "iss.only": "dividends"}
    j = _get_json(url, params=params, timeout=30)
    block = j.get("dividends", {})
    df = pd.DataFrame(block.get("data", []), columns=block.get("columns", []))
    if df.empty:
        return df

    # columns vary; normalize a couple of useful ones
    for col in ["registryclosedate", "registryclose", "close_date", "registry_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    date_col = None
    for cand in ["registryclosedate", "registryclose", "close_date", "registry_date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        return pd.DataFrame()

    df = df.rename(columns={date_col: "date"})
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    elif "dividendvalue" in df.columns:
        df["dividendvalue"] = pd.to_numeric(df["dividendvalue"], errors="coerce")
        df = df.rename(columns={"dividendvalue": "value"})
    else:
        df["value"] = np.nan

    df = df[["date", "value"]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
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


def make_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    fwd = close.shift(-horizon)
    return (fwd - close) / close


def make_target(close: pd.Series, horizon: int, thr: float) -> pd.Series:
    fwd_ret = make_forward_return(close, horizon)
    return (fwd_ret >= thr).astype(int)


def add_features(df: pd.DataFrame, usd: pd.Series, divs: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # basic returns
    out["ret_1d"] = out["CLOSE"].pct_change()
    out["ret_5d"] = out["CLOSE"].pct_change(5)

    # merge USD/RUB (past-only via asof)
    usd_df = usd.reset_index().rename(columns={"index": "date", "CBR_USD_RUB": "usd_rub"})
    usd_df["date"] = pd.to_datetime(usd_df["date"])
    out = pd.merge_asof(out.sort_values("date"), usd_df.sort_values("date"), on="date", direction="backward")

    # merge dividends (past-only)
    if isinstance(divs, pd.DataFrame) and not divs.empty:
        dd = divs.copy()
        dd["date"] = pd.to_datetime(dd["date"], errors="coerce")
        dd = dd.dropna(subset=["date"]).sort_values("date")
        out = pd.merge_asof(out.sort_values("date"), dd[["date", "value"]], on="date", direction="backward")
        out = out.rename(columns={"value": "last_dividend"})
    else:
        out["last_dividend"] = np.nan

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

def build_model(n_features: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(CFG["SEQ_LEN"], n_features))

    # explicitly return_sequences=True on the first LSTM
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CFG["LR"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc_roc"),
            tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
        ],
    )
    return model


# ----------------------------
# FULL DIAGNOSTICS
# ----------------------------

def _banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def diagnose_data(df: pd.DataFrame, feature_cols: list[str], target_col: str = "Target") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full data diagnostics (temporal balance, correlations, multicollinearity, target quality)."""

    _banner("ПОЛНАЯ ДИАГНОСТИКА МОДЕЛИ — ДАННЫЕ")

    print("1) РАСПРЕДЕЛЕНИЕ КЛАССОВ ПО ВРЕМЕНИ")
    print("-" * 40)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must be indexed by dates (DatetimeIndex)")

    tmp = df.copy()
    tmp["year"] = tmp.index.year
    yearly_balance = tmp.groupby("year")[target_col].agg(["count", "mean", "std"])
    yearly_balance.columns = ["samples", "pos_rate", "volatility"]
    print(yearly_balance.round(3))

    pos_rates = yearly_balance["pos_rate"].values
    drift_score = float(np.std(pos_rates) / np.mean(pos_rates)) if np.mean(pos_rates) > 0 else 0.0
    print(f"\nConcept Drift Score: {drift_score:.3f}")
    if drift_score > 0.30:
        print("WARNING: высокий concept drift — структура данных сильно меняется")
    elif drift_score > 0.15:
        print("WARNING: умеренный concept drift")
    else:
        print("OK: стабильное распределение классов")

    print("\n2) КОРРЕЛЯЦИЯ ПРИЗНАКОВ С TARGET")
    print("-" * 40)
    correlations = []
    for col in feature_cols:
        if col in tmp.columns:
            corr = tmp[col].corr(tmp[target_col])
            if pd.notna(corr):
                correlations.append({"feature": col, "correlation": float(corr), "abs_corr": float(abs(corr))})

    corr_df = pd.DataFrame(correlations).sort_values("abs_corr", ascending=False)
    if not corr_df.empty:
        print(corr_df.head(10).round(4))
        weak_features = corr_df[corr_df["abs_corr"] < 0.05]["feature"].tolist()
        if weak_features:
            print(f"\nWARNING: слабые признаки (|corr| < 0.05): {weak_features}")
    else:
        print("WARNING: не удалось посчитать корреляции (возможно константные признаки)")

    print("\n3) МУЛЬТИКОЛЛИНЕАРНОСТЬ")
    print("-" * 40)
    if len(feature_cols) >= 2:
        feature_corr = tmp[feature_cols].corr().abs()
        high_corr_pairs = []
        cols = list(feature_corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = float(feature_corr.iloc[i, j])
                if v > 0.8:
                    high_corr_pairs.append({"feature1": cols[i], "feature2": cols[j], "correlation": v})

        if high_corr_pairs:
            print("WARNING: высокая корреляция между признаками:")
            for pair in sorted(high_corr_pairs, key=lambda x: x["correlation"], reverse=True)[:30]:
                print(f"  {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print("OK: мультиколлинеарность в норме (порог 0.8)")
    else:
        print("SKIP: мало признаков для проверки")

    print("\n4) КАЧЕСТВО ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("-" * 40)
    print(tmp[target_col].describe())

    # run-length analysis
    run_ids = (tmp[target_col].diff() != 0).cumsum()
    max_run = int(run_ids.value_counts().max()) if not tmp.empty else 0
    print(f"\nМакс. последовательность одинаковых значений: {max_run}")
    if max_run > 20:
        print("WARNING: слишком длинные последовательности — возможна проблема с разметкой/таргетом")

    return corr_df, yearly_balance


class DiagnosticCallback(tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.train_aucs: list[float] = []
        self.val_aucs: list[float] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.lrs: list[float] = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        self.train_aucs.append(float(logs.get("auc_roc", 0.0) or 0.0))
        self.val_aucs.append(float(logs.get("val_auc_roc", 0.0) or 0.0))
        self.train_losses.append(float(logs.get("loss", 0.0) or 0.0))
        self.val_losses.append(float(logs.get("val_loss", 0.0) or 0.0))
        try:
            self.lrs.append(float(self.model.optimizer.learning_rate.numpy()))
        except Exception:
            pass

        # Diagnostics every 10 epochs
        if (epoch + 1) % 10 == 0 and self.val_aucs:
            overfitting = self.train_aucs[-1] - self.val_aucs[-1]
            print(f"\nEpoch {epoch + 1}: overfitting(train_auc - val_auc) = {overfitting:.3f}")
            if overfitting > 0.15:
                print("WARNING: сильное переобучение")
            elif overfitting > 0.08:
                print("WARNING: умеренное переобучение")

            if epoch > 20 and len(self.val_aucs) >= 10:
                recent = self.val_aucs[-10:]
                if max(recent) - min(recent) < 0.01:
                    print("WARNING: стагнация validation AUC")


def analyze_training(history: tf.keras.callbacks.History, diagnostic_cb: DiagnosticCallback) -> None:
    _banner("ПОЛНАЯ ДИАГНОСТИКА МОДЕЛИ — ОБУЧЕНИЕ")

    h = history.history
    if "auc_roc" not in h or "val_auc_roc" not in h:
        print("SKIP: auc_roc / val_auc_roc отсутствуют в history")
        return

    final_train_auc = float(h["auc_roc"][-1])
    final_val_auc = float(h["val_auc_roc"][-1])
    best_val_auc = float(max(h["val_auc_roc"]))

    print("1) ДИНАМИКА МЕТРИК")
    print("-" * 30)
    print(f"Final Train AUC: {final_train_auc:.4f}")
    print(f"Final Val   AUC: {final_val_auc:.4f}")
    print(f"Best  Val   AUC: {best_val_auc:.4f}")
    print(f"Overfitting (train - val): {final_train_auc - final_val_auc:+.4f}")

    gap = final_train_auc - final_val_auc
    if gap > 0.15:
        print("CRITICAL: переобучение")
        print("  Fix ideas: уменьшить модель, увеличить dropout, добавить L2")
    elif gap > 0.08:
        print("WARNING: умеренное переобучение")
        print("  Fix ideas: увеличить dropout/регуляризацию")
    else:
        print("OK: переобучение в норме")

    print("\n2) АНАЛИЗ СХОДИМОСТИ")
    print("-" * 30)
    val_aucs = list(map(float, h["val_auc_roc"]))
    if len(val_aucs) > 10:
        early_auc = float(np.mean(val_aucs[:10]))
        late_auc = float(np.mean(val_aucs[-10:]))
        improvement = late_auc - early_auc
        print(f"AUC improvement (late - early): {improvement:+.4f}")
        if improvement < 0.01:
            print("WARNING: модель почти не обучается")
            print("  Возможные причины: слишком маленький LR / слишком много регуляризации / плохие данные")

    if diagnostic_cb.lrs:
        min_lr = min(diagnostic_cb.lrs)
        max_lr = max(diagnostic_cb.lrs)
        print(f"\nLearning Rate: {max_lr:.2e} -> {min_lr:.2e}")
        if min_lr < 1e-6:
            print("WARNING: LR упал слишком низко — возможное затухание градиентов")


def analyze_predictions(y_true: np.ndarray, y_prob: np.ndarray, dates: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    _banner("ПОЛНАЯ ДИАГНОСТИКА МОДЕЛИ — ПРЕДСКАЗАНИЯ")

    y_pred = (y_prob >= threshold).astype(int)

    print("1) БАЗОВЫЕ МЕТРИКИ")
    print("-" * 30)
    accuracy = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc_roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Balanced Acc:  {bal_acc:.4f}")
    print(f"Precision(BUY):{precision:.4f}")
    print(f"Recall(BUY):   {recall:.4f}")
    print(f"F1(BUY):       {f1:.4f}")
    print(f"AUC-ROC:       {auc_roc:.4f}" if np.isfinite(auc_roc) else "AUC-ROC: n/a")

    if np.isfinite(auc_roc):
        if auc_roc < 0.45:
            print("CRITICAL: AUC < 0.45 — антиобучение (проверь инверсию таргета/сдвиги)")
        elif auc_roc < 0.52:
            print("WARNING: AUC ~ 0.5 — модель не лучше случайной")
        elif auc_roc > 0.60:
            print("OK: AUC > 0.6 — хороший результат")

    print("\n2) CONFUSION MATRIX")
    print("-" * 30)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\n3) РАСПРЕДЕЛЕНИЕ ВЕРОЯТНОСТЕЙ")
    print("-" * 30)
    prob_stats = {
        "min": float(np.min(y_prob)),
        "q25": float(np.percentile(y_prob, 25)),
        "median": float(np.median(y_prob)),
        "q75": float(np.percentile(y_prob, 75)),
        "max": float(np.max(y_prob)),
        "mean": float(np.mean(y_prob)),
        "std": float(np.std(y_prob)),
    }
    for k, v in prob_stats.items():
        print(f"{k:8}: {v:.4f}")

    prob_0 = float(np.mean(y_prob[y_true == 0])) if np.any(y_true == 0) else float("nan")
    prob_1 = float(np.mean(y_prob[y_true == 1])) if np.any(y_true == 1) else float("nan")
    if np.isfinite(prob_0) and np.isfinite(prob_1):
        gap = abs(prob_1 - prob_0)
        print(f"\nmean prob(class 0): {prob_0:.4f}")
        print(f"mean prob(class 1): {prob_1:.4f}")
        print(f"calibration gap:    {gap:.4f}")
        if gap < 0.1:
            print("WARNING: слабая разделимость классов")
        elif prob_1 < prob_0:
            print("CRITICAL: инвертированные предсказания (prob_1 < prob_0)")
        else:
            print("OK: калибровка выглядит адекватно")

    print("\n4) ОШИБКИ ПО ВРЕМЕНИ")
    print("-" * 30)
    df_analysis = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "error": (y_true != y_pred),
    })
    df_analysis["year"] = df_analysis["date"].dt.year
    yearly_errors = df_analysis.groupby("year").agg({
        "error": ["mean", "sum"],
        "y_prob": "mean",
        "y_true": "mean",
    }).round(4)
    print(yearly_errors)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
    }


def analyze_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    _banner("ПОЛНАЯ ДИАГНОСТИКА МОДЕЛИ — ПРИЗНАКИ")

    print("1) FEATURE IMPORTANCE (RandomForest baseline)")
    print("-" * 40)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    flat_feature_names = []
    for t in range(X_train.shape[1]):
        lag = X_train.shape[1] - 1 - t
        for fname in feature_names:
            flat_feature_names.append(f"{fname}_t-{lag}")

    rf = RandomForestClassifier(n_estimators=300, random_state=CFG["SEED"], n_jobs=-1)
    rf.fit(X_train_flat, y_train)
    rf_proba = rf.predict_proba(X_test_flat)[:, 1]
    rf_auc = float(roc_auc_score(y_test, rf_proba)) if len(np.unique(y_test)) > 1 else float("nan")

    print(f"RandomForest AUC: {rf_auc:.4f}" if np.isfinite(rf_auc) else "RandomForest AUC: n/a")
    if np.isfinite(rf_auc):
        if rf_auc > 0.58:
            print("OK: данные содержат сигнал")
        elif rf_auc > 0.52:
            print("WARNING: слабый сигнал")
        else:
            print("CRITICAL: RF тоже не работает — вероятно проблема в данных/таргете")

    feat_imp = pd.DataFrame({
        "feature": flat_feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop-20 важных признаков:")
    print(feat_imp.head(20).round(5))

    print("\n2) АНАЛИЗ ЛАГОВ")
    print("-" * 30)
    lag_importance: Dict[str, Dict[str, float]] = {}
    for fname in feature_names:
        scores = feat_imp[feat_imp["feature"].str.contains(f"{fname}_t-")]["importance"].values
        if len(scores) > 0:
            lag_importance[fname] = {
                "total": float(scores.sum()),
                "max": float(scores.max()),
                "recent_vs_old": float(np.mean(scores[:5]) / (np.mean(scores[5:]) + 1e-8)) if len(scores) > 5 else float("nan"),
            }

    lag_df = pd.DataFrame.from_dict(lag_importance, orient="index").sort_values("total", ascending=False)
    if not lag_df.empty:
        print(lag_df.round(4))

    return rf_auc, feat_imp, lag_df


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 1.0
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def advanced_backtest(
    prob: np.ndarray,
    actual_fwd_ret: np.ndarray,
    dates: np.ndarray,
    threshold: float = 0.5,
    fee: float = 0.001,
    horizon: int = 5,
) -> tuple[pd.DataFrame, list[float]]:
    _banner("ПОЛНАЯ ДИАГНОСТИКА МОДЕЛИ — БЭКТЕСТ")

    signals = (prob >= threshold).astype(int)

    trades = []
    equity_curve = [1.0]
    current_equity = 1.0

    i = 0
    while i < len(signals):
        if signals[i] == 1:
            entry_date = pd.to_datetime(dates[i])
            gross = float(actual_fwd_ret[i])
            net = gross - fee
            trades.append({
                "entry_date": entry_date,
                "return": net,
                "gross_return": gross,
                "prob": float(prob[i]),
            })
            current_equity *= (1.0 + net)
            # keep equity flat during holding horizon (simple approximation)
            for _ in range(max(horizon, 1)):
                equity_curve.append(current_equity)
            i += max(horizon, 1)
        else:
            equity_curve.append(current_equity)
            i += 1

    if not trades:
        print("CRITICAL: нет торговых сигналов")
        return pd.DataFrame(), equity_curve

    tdf = pd.DataFrame(trades)

    total_return = float(current_equity - 1.0)
    win_rate = float((tdf["return"] > 0).mean())
    avg_return = float(tdf["return"].mean())
    avg_win = float(tdf.loc[tdf["return"] > 0, "return"].mean()) if (tdf["return"] > 0).any() else float("nan")
    avg_loss = float(tdf.loc[tdf["return"] < 0, "return"].mean()) if (tdf["return"] < 0).any() else float("nan")

    print("1) ТОРГОВАЯ СТАТИСТИКА")
    print("-" * 30)
    print(f"Total return: {total_return:+.1%}")
    print(f"Trades:       {len(tdf)}")
    print(f"Win rate:     {win_rate:.1%}")
    print(f"Avg return:   {avg_return:+.2%}")
    print(f"Avg win:      {avg_win:+.2%}" if np.isfinite(avg_win) else "Avg win:      n/a")
    print(f"Avg loss:     {avg_loss:+.2%}" if np.isfinite(avg_loss) else "Avg loss:     n/a")

    if len(tdf) > 1:
        std = float(tdf["return"].std())
        sharpe = float(tdf["return"].mean() / std * np.sqrt(252 / max(horizon, 1))) if std > 0 else 0.0
        max_dd = calculate_max_drawdown(equity_curve)
        print(f"Sharpe:       {sharpe:.2f}")
        print(f"Max drawdown: {max_dd:.1%}")

    print("\n2) КАЧЕСТВО СИГНАЛОВ")
    print("-" * 30)
    try:
        tdf["prob_bucket"] = pd.cut(tdf["prob"], bins=5, labels=["Low", "Med-Low", "Medium", "Med-High", "High"])
        bucket = tdf.groupby("prob_bucket")["return"].agg(["count", "mean", "std"]).round(4)
        print(bucket)
    except Exception:
        pass

    corr = float(tdf["prob"].corr(tdf["return"])) if len(tdf) > 2 else float("nan")
    if np.isfinite(corr):
        print(f"\nCorr(prob, return): {corr:.3f}")
        if corr > 0.1:
            print("OK: выше вероятность -> выше доход")
        elif corr < -0.1:
            print("CRITICAL: инвертированная калибровка")
        else:
            print("WARNING: слабая связь вероятности и доходности")

    print("\n3) ВРЕМЕННОЙ АНАЛИЗ")
    print("-" * 30)
    tdf["year"] = pd.to_datetime(tdf["entry_date"]).dt.year
    yearly = tdf.groupby("year")["return"].agg(["count", "mean", "sum"]).round(4)
    yearly.columns = ["trades", "avg_return", "total_return"]
    print(yearly)

    return tdf, equity_curve


def final_diagnosis(
    *,
    auc_roc: float,
    history: tf.keras.callbacks.History,
    rf_auc: float,
    temporal_balance: pd.DataFrame,
) -> None:
    _banner("ПОЛНАЯ ДИАГНОСТИКА МОДЕЛИ — ИТОГ")

    problems: list[str] = []
    recs: list[str] = []

    if np.isfinite(auc_roc):
        if auc_roc < 0.45:
            problems.append("CRITICAL: AUC < 0.45 — антиобучение")
            recs.append("Проверь сдвиги таргета/фич, попробуй инвертировать предсказания")
        elif auc_roc < 0.52:
            problems.append("WARNING: AUC ~ 0.5 — модель не лучше случайной")
            recs.append("Улучшить признаки/таргет, попробовать другие модели")

    h = history.history
    if "auc_roc" in h and "val_auc_roc" in h:
        final_train_auc = float(h["auc_roc"][-1])
        final_val_auc = float(h["val_auc_roc"][-1])
        overfit = final_train_auc - final_val_auc
        if overfit > 0.15:
            problems.append(f"CRITICAL: переобучение {overfit:.3f}")
            recs.append("Уменьшить архитектуру, увеличить dropout/L2, больше данных")
        elif overfit > 0.08:
            problems.append(f"WARNING: умеренное переобучение {overfit:.3f}")
            recs.append("Увеличить регуляризацию")

    if np.isfinite(rf_auc) and rf_auc < 0.52:
        problems.append("CRITICAL: RandomForest baseline не работает — вероятно проблема в данных")
        recs.append("Пересмотреть признаки, горизонт, целевую переменную")

    if temporal_balance is not None and not temporal_balance.empty:
        pos_rates = temporal_balance["pos_rate"].values
        drift_score = float(np.std(pos_rates) / np.mean(pos_rates)) if np.mean(pos_rates) > 0 else 0.0
        if drift_score > 0.30:
            problems.append(f"WARNING: высокий concept drift {drift_score:.3f}")
            recs.append("Использовать скользящее окно обучения или online/rolling retrain")

    print("ПРОБЛЕМЫ:")
    print("-" * 40)
    if problems:
        for i, p in enumerate(problems, 1):
            print(f"{i}. {p}")
    else:
        print("OK: критичных проблем не найдено")

    print("\nРЕКОМЕНДАЦИИ:")
    print("-" * 40)
    if recs:
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r}")
    else:
        print("OK: продолжаем тюнинг и расширение признаков")


# ----------------------------
# MAIN
# ----------------------------

@dataclass
class Dataset:
    df_feat: pd.DataFrame
    target: pd.Series
    fwd_ret: pd.Series


def load_or_build_dataset(secid: str) -> Dataset:
    cache_file = _cache_path(f"dataset_{secid}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            df_feat = pickle.load(f)
    else:
        df_price = fetch_moex_history(secid, CFG["START"], CFG["END"])
        divs = fetch_moex_dividends(secid)
        usd = fetch_cbr_usdrub(CFG["START"], CFG["END"])
        df_feat = add_features(df_price, usd, divs)
        with open(cache_file, "wb") as f:
            pickle.dump(df_feat, f)

    y = make_target(df_feat["CLOSE"], CFG["TARGET_HORIZON_DAYS"], CFG["TARGET_UP_THRESHOLD"])
    fret = make_forward_return(df_feat["CLOSE"], CFG["TARGET_HORIZON_DAYS"])
    return Dataset(df_feat=df_feat, target=y, fwd_ret=fret)


def main() -> None:
    secid = CFG["TICKER"]

    data = load_or_build_dataset(secid)
    df_feat = data.df_feat.copy()

    # Align: remove last horizon rows (target/returns are NaN there)
    horizon = int(CFG["TARGET_HORIZON_DAYS"])
    df_feat = df_feat.iloc[:-horizon].reset_index(drop=True)
    y = data.target.iloc[:-horizon].astype(int).values
    fwd_ret = data.fwd_ret.iloc[:-horizon].astype(float).values

    feature_cols = [c for c in df_feat.columns if c not in {"date"}]

    # Optional full data diagnostics on a date-indexed frame
    corr_df = pd.DataFrame()
    temporal_balance = pd.DataFrame()
    if CFG.get("DIAGNOSTICS", False):
        df_diag = df_feat.set_index(pd.to_datetime(df_feat["date"]))
        df_diag["Target"] = y
        corr_df, temporal_balance = diagnose_data(df_diag, feature_cols, target_col="Target")

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

    scaler = StandardScaler()
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

    # class weights (train only)
    classes = np.unique(y_tr)
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_vals)}

    model = build_model(X_tr.shape[-1])
    model.summary()

    diagnostic_cb = DiagnosticCallback() if CFG.get("DIAGNOSTICS", False) else None
    callbacks: list[tf.keras.callbacks.Callback] = []
    if diagnostic_cb is not None:
        callbacks.append(diagnostic_cb)

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=int(CFG["EPOCHS"]),
        batch_size=int(CFG["BATCH_SIZE"]),
        class_weight=class_weight,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )

    if CFG.get("DIAGNOSTICS", False) and diagnostic_cb is not None:
        analyze_training(history, diagnostic_cb)

    proba = model.predict(X_test, verbose=0).reshape(-1)
    pred = (proba >= float(CFG["BEST_THR"])).astype(int)

    # Base metrics
    print("\n" + "=" * 70)
    print("TEST METRICS")
    print("=" * 70)
    print("balanced_accuracy:", balanced_accuracy_score(y_test, pred))
    print("f1:", f1_score(y_test, pred, zero_division=0))
    print("roc_auc:", roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None)
    print("mcc:", matthews_corrcoef(y_test, pred) if len(np.unique(y_test)) > 1 else None)
    print("confusion_matrix:\n", confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, zero_division=0))
    print("average_precision:", average_precision_score(y_test, proba) if len(np.unique(y_test)) > 1 else None)

    metrics = {}
    if CFG.get("DIAGNOSTICS", False):
        metrics = analyze_predictions(y_test, proba, d_test, threshold=float(CFG["BEST_THR"]))

        rf_auc, _, _ = analyze_features(X_tr, y_tr, X_test, y_test, feature_cols)

        # Backtest on test set using forward returns aligned to each sample
        advanced_backtest(
            proba,
            r_test,
            d_test,
            threshold=float(CFG["BEST_THR"]),
            fee=float(CFG["FEE"]),
            horizon=int(CFG["TARGET_HORIZON_DAYS"]),
        )

        final_diagnosis(
            auc_roc=float(metrics.get("auc_roc", float("nan"))),
            history=history,
            rf_auc=float(rf_auc),
            temporal_balance=temporal_balance,
        )


if __name__ == "__main__":
    main()
