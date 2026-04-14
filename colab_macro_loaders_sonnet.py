# @title
# ============================================================
# LSTM МОДЕЛЬ ДЛЯ ПРОГНОЗИРОВАНИЯ АКЦИЙ СБЕРБАНКА
# ============================================================

!pip install moexalgo requests tensorflow scikit-learn pandas numpy -q

import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import pickle
import warnings
from xml.etree import ElementTree as ET
from datetime import datetime

from moexalgo import Ticker
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, classification_report,
    precision_score, recall_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

print("✅ Зависимости установлены")
print("=" * 70)

# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================
CONFIG = {
    "TICKER":      "SBER",
    "START":       "2015-01-01",
    "END":         "2025-12-31",
    "WINDOW_SIZE": 30,
    "HORIZON":     5,
    "THR_MOVE":    0.015,

    # ── ИСПРАВЛЕНИЕ 1: уменьшена архитектура (было 128/64 → 64/32) ──
    "LSTM_UNITS_1": 64,
    "LSTM_UNITS_2": 32,
    "DENSE_UNITS":  16,
    "DROPOUT":      0.3,

    "EPOCHS":       200,
    "BATCH_SIZE":   32,
    "LR":           0.001,
    "PATIENCE":     25,      # увеличено — дать больше шансов
    "LR_PATIENCE":  10,

    "TRAIN_RATIO":  0.65,
    "VAL_RATIO":    0.15,
    "THRESHOLD":    0.5,

    "MODEL_FILE":  "lstm_sber_v2.keras",
    "SCALER_FILE": "lstm_scaler_v2.pkl",
}

print("⚙️  Конфигурация загружена")

# ============================================================
# БЛОК 1: ЗАГРУЗКА ДАННЫХ (без изменений)
# ============================================================
print("=" * 70)
print("БЛОК 1: ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})


def load_moex_candles(ticker, start, end):
    df = pd.DataFrame(
        Ticker(ticker).candles(start=start, end=end, period='1D')
    )
    df["begin"] = pd.to_datetime(df["begin"]).dt.tz_localize(None)
    df = (df.drop_duplicates(subset=["begin"])
            .set_index("begin")
            .sort_index())
    return df


def load_cbr_usdrub(start="2015-01-01"):
    try:
        end_str   = datetime.today().strftime("%d/%m/%Y")
        start_str = pd.to_datetime(start).strftime("%d/%m/%Y")
        xml = SESSION.get(
            "https://www.cbr.ru/scripts/XML_dynamic.asp",
            params={"date_req1": start_str, "date_req2": end_str,
                    "VAL_NM_RQ": "R01235"},
            timeout=20
        ).text
        root = ET.fromstring(xml)
        rows = []
        for rec in root.findall("Record"):
            d, v = rec.attrib.get("Date"), rec.findtext("Value")
            if d and v:
                rows.append({"date": pd.to_datetime(d, dayfirst=True),
                              "usd_rub": float(v.replace(",", "."))})
        s = pd.DataFrame(rows).set_index("date")["usd_rub"].sort_index()
        s.name = "USD_RUB_CBR"
        print(f"  ✅ USD/RUB ЦБ: {len(s)} точек")
        return s
    except Exception as e:
        print(f"  ⚠️  USD/RUB ЦБ: {e}")
        return pd.Series(dtype=float)


def load_cbr_key_rate(start="2015-01-01"):
    try:
        end_str   = datetime.today().strftime("%d.%m.%Y")
        start_str = pd.to_datetime(start).strftime("%d.%m.%Y")
        r = SESSION.get(
            "https://www.cbr.ru/hd_base/KeyRate/",
            params={"UniDbQuery.Posted": "True",
                    "UniDbQuery.From": start_str,
                    "UniDbQuery.To": end_str},
            timeout=20
        )
        tables = pd.read_html(r.text, thousands='\xa0', decimal=',')
        df = tables[0].copy()
        df.columns = ['date', 'rate']
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df['rate'] = (df['rate'].astype(str)
                      .str.replace(',', '.').str.replace('%', '')
                      .str.replace('\xa0', '').str.strip())
        df['rate'] = pd.to_numeric(df['rate'], errors='coerce')
        if df['rate'].dropna().max() > 100:
            df['rate'] = df['rate'] / 100
        df = df.dropna().sort_values('date').reset_index(drop=True)
        s = df.set_index('date')['rate']
        s.name = "KEY_RATE"
        print(f"  ✅ Ключевая ставка: {len(s)} точек")
        return s
    except Exception as e:
        print(f"  ⚠️  Ключевая ставка: {e}")
        return pd.Series(dtype=float)


print("\n📡 Загрузка котировок MOEX...")
sber_df   = load_moex_candles("SBER",         CONFIG["START"], CONFIG["END"])
imoex_df  = load_moex_candles("IMOEX",        CONFIG["START"], CONFIG["END"])
usdrub_df = load_moex_candles("USD000UTSTOM", CONFIG["START"], CONFIG["END"])
print(f"  ✅ SBER: {len(sber_df)} | IMOEX: {len(imoex_df)} | USD: {len(usdrub_df)}")

print("\n📡 Загрузка макроданных ЦБ РФ...")
usd_cbr  = load_cbr_usdrub(CONFIG["START"])
key_rate = load_cbr_key_rate(CONFIG["START"])


# ============================================================
# БЛОК 2: ПРИЗНАКИ
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 2: ИНЖИНИРИНГ ПРИЗНАКОВ")
print("=" * 70)


def merge_asof_series(base_df, series, col_name):
    if col_name in base_df.columns:
        base_df = base_df.drop(columns=[col_name])
    if series.empty:
        base_df[col_name] = np.nan
        return base_df
    left_idx  = base_df.index.tz_localize(None).normalize()
    right_idx = (series.index.tz_localize(None)
                 if series.index.tz is not None
                 else series.index).normalize()
    s_clean = series.copy()
    s_clean.index = right_idx
    left  = pd.DataFrame({"date": left_idx})
    right = s_clean.reset_index()
    right.columns = ["date", col_name]
    right = right.drop_duplicates("date").sort_values("date")
    merged = pd.merge_asof(left.sort_values("date"), right,
                           on="date", direction="backward")
    base_df[col_name] = merged[col_name].values
    pct = base_df[col_name].notna().mean() * 100
    print(f"  ✅ {col_name}: {pct:.1f}% заполнено")
    return base_df


def build_features(sber_df, imoex_df, usdrub_df, usd_cbr, key_rate):
    df = pd.DataFrame(index=sber_df.index)
    df["SBER"]    = sber_df["close"]
    df["IMOEX"]   = imoex_df["close"].reindex(df.index).ffill()
    df["USD_RUB"] = usdrub_df["close"].reindex(df.index).ffill()

    # ── Доходности ──
    df["ret_1d"]  = df["SBER"].pct_change(1)
    df["ret_3d"]  = df["SBER"].pct_change(3)
    df["ret_5d"]  = df["SBER"].pct_change(5)
    df["ret_10d"] = df["SBER"].pct_change(10)
    df["ret_20d"] = df["SBER"].pct_change(20)
    df["log_ret"] = np.log(df["SBER"] / df["SBER"].shift(1))

    # ── Скользящие средние ──
    sma20  = df["SBER"].rolling(20).mean()
    sma50  = df["SBER"].rolling(50).mean()
    sma200 = df["SBER"].rolling(200).mean()

    df["Dist_SMA20"]  = df["SBER"] / sma20  - 1
    df["Dist_SMA200"] = df["SBER"] / sma200 - 1
    df["Trend_Up"]    = (df["SBER"] > sma200).astype(int)
    df["SMA_cross"]   = (sma20 > sma50).astype(int)

    # ── RSI ──
    delta = df["SBER"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"]          = 100 - (100 / (1 + gain / (loss + 1e-12)))
    df["RSI_oversold"]  = (df["RSI"] < 30).astype(int)
    df["RSI_overbought"]= (df["RSI"] > 70).astype(int)

    # ── MACD ──
    ema12 = df["SBER"].ewm(span=12, adjust=False).mean()
    ema26 = df["SBER"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]
    df["MACD_cross"]  = (df["MACD"] > df["MACD_signal"]).astype(int)

    # ── Bollinger Bands ──
    bb_mid = df["SBER"].rolling(20).mean()
    bb_std = df["SBER"].rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_lo  = bb_mid - 2 * bb_std
    df["BB_width"] = (bb_up - bb_lo) / (bb_mid + 1e-12)
    df["BB_pos"]   = (df["SBER"] - bb_lo) / (bb_up - bb_lo + 1e-12)

    # ── ATR ──
    high = sber_df["high"].reindex(df.index)
    low  = sber_df["low"].reindex(df.index)
    tr   = pd.concat([
        high - low,
        (high - df["SBER"].shift()).abs(),
        (low  - df["SBER"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"]     = tr.rolling(14).mean()
    df["ATR_rel"] = df["ATR"] / (df["SBER"] + 1e-12)

    # ── Волатильность ──
    df["Vol_5"]     = df["ret_1d"].rolling(5).std()
    df["Vol_20"]    = df["ret_1d"].rolling(20).std()
    df["Vol_ratio"] = df["Vol_5"] / (df["Vol_20"] + 1e-12)

    # ── Объём ──
    vol = sber_df["volume"].reindex(df.index)
    df["Vol_Rel"]   = vol / (vol.rolling(20).mean() + 1e-12)
    df["Vol_spike"] = (df["Vol_Rel"] > 2.0).astype(int)

    # ── Рыночный контекст ──
    df["IMOEX_ret_1d"]  = df["IMOEX"].pct_change(1)
    df["IMOEX_ret_5d"]  = df["IMOEX"].pct_change(5)
    df["IMOEX_ret_20d"] = df["IMOEX"].pct_change(20)
    di = df["IMOEX"].diff()
    gi = di.where(di > 0, 0).rolling(14).mean()
    li = (-di.where(di < 0, 0)).rolling(14).mean()
    df["IMOEX_RSI"]     = 100 - (100 / (1 + gi / (li + 1e-12)))
    df["SBER_vs_IMOEX"] = df["ret_5d"] - df["IMOEX_ret_5d"]

    # ── Валюта ──
    df["USD_ret_1d"] = df["USD_RUB"].pct_change(1)
    df["USD_ret_5d"] = df["USD_RUB"].pct_change(5)
    df["USD_Vol_20"] = df["USD_ret_1d"].rolling(20).std()

    # ── Макро ЦБ РФ ──
    df = merge_asof_series(df, usd_cbr,  "USD_RUB_CBR")
    df["USD_CBR_ret"] = df["USD_RUB_CBR"].pct_change()
    df["USD_CBR_Vol"] = df["USD_CBR_ret"].rolling(20).std()

    df = merge_asof_series(df, key_rate, "KEY_RATE")
    df["RATE_change"] = df["KEY_RATE"].diff()
    df["RATE_rising"] = (df["RATE_change"] > 0).astype(int)
    df["RATE_high"]   = (df["KEY_RATE"] >= 15).astype(int)
    rate_min, rate_max = df["KEY_RATE"].min(), df["KEY_RATE"].max()
    df["RATE_norm"]   = (df["KEY_RATE"] - rate_min) / (rate_max - rate_min + 1e-8)

    # ── Дивиденды (без look-ahead) ──
    try:
        div_url  = f"https://iss.moex.com/iss/securities/{CONFIG['TICKER']}/dividends.json"
        div_j    = SESSION.get(div_url, timeout=15).json()
        div_data = div_j.get("dividends", {})
        if div_data.get("data"):
            div_df = pd.DataFrame(div_data["data"], columns=div_data["columns"])
            div_df["registryclosedate"] = pd.to_datetime(
                div_df["registryclosedate"], errors="coerce")
            div_df = (div_df.dropna(subset=["registryclosedate"])
                            .sort_values("registryclosedate"))
            div_s = div_df.set_index("registryclosedate")["value"]
            div_s.index = div_s.index.tz_localize(None).normalize()
            div_s = div_s[~div_s.index.duplicated(keep="last")].sort_index()

            df["LAST_DIV"]       = np.nan
            df["DAYS_SINCE_DIV"] = np.nan
            df["DIV_YIELD"]      = np.nan

            df_idx = df.index.tz_localize(None).normalize()
            for i, date in enumerate(df_idx):
                past = div_s[div_s.index <= date]
                if not past.empty:
                    df.iloc[i, df.columns.get_loc("LAST_DIV")]       = past.iloc[-1]
                    df.iloc[i, df.columns.get_loc("DAYS_SINCE_DIV")] = (date - past.index[-1]).days
                    df.iloc[i, df.columns.get_loc("DIV_YIELD")]      = past.iloc[-1] / (df["SBER"].iloc[i] + 1e-12)

            print(f"  ✅ Дивиденды: {df['LAST_DIV'].notna().sum()}/{len(df)} (без look-ahead)")
    except Exception as e:
        print(f"  ⚠️  Дивиденды: {e}")

    # Удаляем вспомогательные колонки
    df = df.drop(columns=["SBER", "IMOEX", "USD_RUB"], errors="ignore")
    return df


print("\n🔧 Построение признаков...")
dataset = build_features(sber_df, imoex_df, usdrub_df, usd_cbr, key_rate)
print(f"\n📊 Признаков до фильтра: {len(dataset.columns)}")


# ============================================================
# БЛОК 3: ЦЕЛЕВАЯ ПЕРЕМЕННАЯ
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 3: ЦЕЛЕВАЯ ПЕРЕМЕННАЯ")
print("=" * 70)

sber_close            = sber_df["close"].reindex(dataset.index)
future_ret            = sber_close.shift(-CONFIG["HORIZON"]) / sber_close - 1.0
dataset["future_ret"] = future_ret
dataset["Target"]     = (future_ret > CONFIG["THR_MOVE"]).astype(int)
dataset               = dataset.dropna(subset=["Target"])

pos_rate = dataset["Target"].mean()
print(f"  Горизонт: {CONFIG['HORIZON']} дней, порог: {CONFIG['THR_MOVE']:.1%}")
print(f"  Класс 1 (рост): {pos_rate:.1%} | Класс 0: {1-pos_rate:.1%}")
print(f"  Строк: {len(dataset)}")


# ============================================================
# БЛОК 4: ОТБОР ПРИЗНАКОВ
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 4: ОТБОР ПРИЗНАКОВ")
print("=" * 70)

EXCLUDE = {"Target", "future_ret"}

fill_stats = {
    col: dataset[col].notna().mean()
    for col in dataset.columns if col not in EXCLUDE
}

# ── ИСПРАВЛЕНИЕ 2: жёсткий список вместо фильтра по заполненности ──
# 42 признака при 1678 обучающих примерах → переобучение
# Оставляем 25 наиболее информативных
PREFERRED = [
    # Доходности
    "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # Тренд
    "Dist_SMA20", "Dist_SMA200", "Trend_Up", "SMA_cross",
    # Моментум
    "RSI", "MACD_hist", "MACD_cross",
    # Волатильность
    "BB_width", "BB_pos", "ATR_rel",
    "Vol_ratio", "Vol_spike",
    # Рыночный контекст
    "IMOEX_ret_1d", "IMOEX_ret_5d", "SBER_vs_IMOEX",
    # Валюта
    "USD_ret_1d", "USD_ret_5d",
    # Макро
    "KEY_RATE", "RATE_rising",
    # Дивиденды
    "DIV_YIELD",
]

# Берём только те что реально есть и заполнены > 85%
FEATURE_COLS = [
    col for col in PREFERRED
    if col in fill_stats and fill_stats[col] > 0.85
]

print(f"\n  Целевых признаков: {len(PREFERRED)}")
print(f"  Доступных (>85%):  {len(FEATURE_COLS)}")
print(f"  Список: {FEATURE_COLS}")

df_clean = dataset[FEATURE_COLS + ["Target", "future_ret"]].dropna()
print(f"\n  Итоговый датасет: {df_clean.shape}")


# ============================================================
# БЛОК 5: РАЗБИВКА И МАСШТАБИРОВАНИЕ
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 5: РАЗБИВКА ДАННЫХ")
print("=" * 70)

n         = len(df_clean)
train_end = int(n * CONFIG["TRAIN_RATIO"])
val_end   = int(n * (CONFIG["TRAIN_RATIO"] + CONFIG["VAL_RATIO"]))

train_df = df_clean.iloc[:train_end]
val_df   = df_clean.iloc[train_end:val_end]
test_df  = df_clean.iloc[val_end:]

print(f"  Train: {len(train_df)} ({train_df.index[0].date()} — {train_df.index[-1].date()})")
print(f"  Val:   {len(val_df)} ({val_df.index[0].date()} — {val_df.index[-1].date()})")
print(f"  Test:  {len(test_df)} ({test_df.index[0].date()} — {test_df.index[-1].date()})")

scaler     = RobustScaler()
X_train_2d = scaler.fit_transform(train_df[FEATURE_COLS])
X_val_2d   = scaler.transform(val_df[FEATURE_COLS])
X_test_2d  = scaler.transform(test_df[FEATURE_COLS])

y_train    = train_df["Target"].values
y_val      = val_df["Target"].values
y_test     = test_df["Target"].values
fret_test  = test_df["future_ret"].values

print(f"\n  Train баланс: {y_train.mean():.1%} роста / {1-y_train.mean():.1%} нет роста")


# ============================================================
# БЛОК 6: СКОЛЬЗЯЩИЕ ОКНА
# ============================================================
WINDOW = CONFIG["WINDOW_SIZE"]


def make_windows(X_2d, y_1d, window):
    X, y = [], []
    for i in range(window, len(X_2d)):
        X.append(X_2d[i-window:i])
        y.append(y_1d[i])
    return np.array(X), np.array(y)


X_train, y_train_w = make_windows(X_train_2d, y_train, WINDOW)
X_val,   y_val_w   = make_windows(X_val_2d,   y_val,   WINDOW)
X_test,  y_test_w  = make_windows(X_test_2d,  y_test,  WINDOW)

# future_ret для бэктеста (выравниваем с окнами)
fret_test_w = fret_test[WINDOW:]

print(f"\n  Windows: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
print(f"  Признаков в окне: {X_train.shape[2]}")

classes      = np.unique(y_train_w)
weights      = compute_class_weight("balanced", classes=classes, y=y_train_w)
class_weight = dict(zip(classes.astype(int), weights))
print(f"  Class weights: {class_weight}")


# ============================================================
# БЛОК 7: АРХИТЕКТУРА
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 7: АРХИТЕКТУРА МОДЕЛИ")
print("=" * 70)

n_features = X_train.shape[2]


# ── ИСПРАВЛЕНИЕ 3: передаём CONFIG["LR"] — не весь словарь ──
def build_model(window, n_feat, lr):
    inp = tf.keras.Input(shape=(window, n_feat))

    x = tf.keras.layers.LSTM(
        CONFIG["LSTM_UNITS_1"],
        return_sequences=True,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inp)
    x = tf.keras.layers.Dropout(CONFIG["DROPOUT"])(x)

    x = tf.keras.layers.LSTM(
        CONFIG["LSTM_UNITS_2"],
        return_sequences=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(CONFIG["DROPOUT"])(x)

    x   = tf.keras.layers.Dense(CONFIG["DENSE_UNITS"], activation="relu")(x)
    x   = tf.keras.layers.Dropout(CONFIG["DROPOUT"] * 0.5)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(lr),   # ← явный float
            clipnorm=1.0
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
            tf.keras.metrics.AUC(curve="PR",  name="auc_pr"),
        ]
    )
    return model


# ── ИСПРАВЛЕНИЕ 3: передаём CONFIG["LR"], не CONFIG ──
model = build_model(WINDOW, n_features, CONFIG["LR"])
model.summary()
print(f"\n  Параметров: {model.count_params():,}")


# ============================================================
# БЛОК 8: ОБУЧЕНИЕ
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 8: ОБУЧЕНИЕ")
print("=" * 70)

# ── ИСПРАВЛЕНИЕ 4: monitor="val_auc_roc" (не "val_auc") ──
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CONFIG["MODEL_FILE"],
        monitor="val_auc_roc",     # ← исправлено
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_roc",     # ← исправлено
        mode="max",
        patience=CONFIG["PATIENCE"],
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc_roc",     # ← исправлено
        mode="max",
        factor=0.5,
        patience=CONFIG["LR_PATIENCE"],
        min_lr=1e-6,
        verbose=1
    ),
]

history = model.fit(
    X_train, y_train_w,
    validation_data=(X_val, y_val_w),
    epochs=CONFIG["EPOCHS"],
    batch_size=CONFIG["BATCH_SIZE"],
    class_weight=class_weight,
    callbacks=callbacks,
    shuffle=False,
    verbose=1,
)

model = tf.keras.models.load_model(CONFIG["MODEL_FILE"])
with open(CONFIG["SCALER_FILE"], "wb") as f:
    pickle.dump(scaler, f)
print(f"\n✅ Модель и scaler сохранены")


# ============================================================
# БЛОК 9: ОПТИМИЗАЦИЯ ПОРОГА
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 9: ОПТИМИЗАЦИЯ ПОРОГА")
print("=" * 70)

pred_prob_val  = model.predict(X_val,  verbose=0).reshape(-1)
pred_prob_test = model.predict(X_test, verbose=0).reshape(-1)

print(f"  Val  prob: [{pred_prob_val.min():.3f}, {pred_prob_val.max():.3f}]"
      f" mean={pred_prob_val.mean():.3f}")
print(f"  Test prob: [{pred_prob_test.min():.3f}, {pred_prob_test.max():.3f}]"
      f" mean={pred_prob_test.mean():.3f}")

rows = []
for thr in np.arange(0.30, 0.71, 0.02):
    pred = (pred_prob_val > thr).astype(int)
    f1   = f1_score(y_val_w, pred, average="macro",  zero_division=0)
    f1_1 = f1_score(y_val_w, pred, pos_label=1, zero_division=0)
    rows.append({
        "threshold": round(thr, 2),
        "f1_macro":  round(f1,  4),
        "f1_buy":    round(f1_1, 4),
        "precision": round(precision_score(y_val_w, pred, pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y_val_w, pred, pos_label=1, zero_division=0), 4),
        "buy_rate":  round(pred.mean(), 3),
    })

thr_df   = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
BEST_THR = float(thr_df.iloc[0]["threshold"])

print(f"\n  Топ-10 порогов по F1-macro:")
print(thr_df.head(10).to_string(index=False))
print(f"\n  ✅ Оптимальный порог: {BEST_THR}")


# ============================================================
# БЛОК 10: ОЦЕНКА НА ТЕСТЕ
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 10: ФИНАЛЬНАЯ ОЦЕНКА (TEST SET)")
print("=" * 70)

pred_cls_test = (pred_prob_test > BEST_THR).astype(int)

acc     = accuracy_score(y_test_w,     pred_cls_test)
bal_acc = balanced_accuracy_score(y_test_w, pred_cls_test)
f1_mac  = f1_score(y_test_w, pred_cls_test, average="macro", zero_division=0)
f1_buy  = f1_score(y_test_w, pred_cls_test, pos_label=1, zero_division=0)
prec    = precision_score(y_test_w, pred_cls_test, pos_label=1, zero_division=0)
rec     = recall_score(y_test_w, pred_cls_test, pos_label=1, zero_division=0)
auc_roc = roc_auc_score(y_test_w, pred_prob_test)
auc_pr  = average_precision_score(y_test_w, pred_prob_test)
cm      = confusion_matrix(y_test_w, pred_cls_test)

print(f"\n{'─'*50}")
print(f"  Accuracy:            {acc:.4f}")
print(f"  Balanced Accuracy:   {bal_acc:.4f}")
print(f"  F1-macro:            {f1_mac:.4f}")
print(f"  F1 (BUY=1):          {f1_buy:.4f}")
print(f"  Precision (BUY):     {prec:.4f}")
print(f"  Recall (BUY):        {rec:.4f}")
print(f"  AUC-ROC:             {auc_roc:.4f}")
print(f"  AUC-PR:              {auc_pr:.4f}")
print(f"  Порог:               {BEST_THR}")
print(f"{'─'*50}")
print(f"\n  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n{classification_report(y_test_w, pred_cls_test, zero_division=0)}")


# ============================================================
# БЛОК 11: БЭКТЕСТ
# ============================================================
print("=" * 70)
print("БЛОК 11: БЭКТЕСТ ТОРГОВОЙ СТРАТЕГИИ")
print("=" * 70)


def backtest(prob, actual_ret, thr, fee=0.001, horizon=5):
    equity, bh = [1.0], [1.0]
    trades = []
    i = 0
    while i < len(prob):
        ret = float(actual_ret[i])
        bh.append(bh[-1] * (1 + ret))
        if prob[i] >= thr:
            trades.append(ret - fee)
            equity.append(equity[-1] * (1 + ret - fee))
            i += horizon
        else:
            equity.append(equity[-1])
            i += 1

    total  = equity[-1] - 1
    bh_ret = bh[-1] - 1
    wr     = sum(t > 0 for t in trades) / max(len(trades), 1)

    print(f"\n  Порог: {thr:.2f} | Комиссия: {fee:.1%}")
    print(f"  {'─'*40}")
    print(f"  Доходность стратегии: {total:+.1%}")
    print(f"  Доходность Buy&Hold:  {bh_ret:+.1%}")
    print(f"  Альфа:                {total - bh_ret:+.1%}")
    print(f"  Кол-во сделок:        {len(trades)}")
    print(f"  Win rate:             {wr:.1%}")
    return equity, bh, trades


backtest(pred_prob_test, fret_test_w, BEST_THR,
         fee=0.001, horizon=CONFIG["HORIZON"])


# ============================================================
# БЛОК 12: LIVE СИГНАЛ
# ============================================================
print("\n" + "=" * 70)
print("БЛОК 12: LIVE СИГНАЛ")
print("=" * 70)

last_window = df_clean[FEATURE_COLS].iloc[-WINDOW:]
last_scaled  = scaler.transform(last_window)
last_input   = last_scaled.reshape(1, WINDOW, n_features)

prob_live  = float(model.predict(last_input, verbose=0)[0, 0])
pred_live  = int(prob_live > BEST_THR)
conf_live  = abs(prob_live - 0.5) * 2
last_date  = df_clean.index[-1]
last_price = float(sber_df["close"].iloc[-1])

print(f"\n  Дата:         {last_date.date()}")
print(f"  Цена SBER:    {last_price:.2f} руб")
print(f"  P(рост):      {prob_live:.4f}")
print(f"  Порог:        {BEST_THR}")
print(f"  Уверенность:  {conf_live:.1%}")
print(f"  СИГНАЛ:       {'🔴 BUY' if pred_live == 1 else '🟢 HOLD'}")


# ============================================================
# ФИНАЛЬНЫЙ ОТЧЁТ
# ============================================================
print("\n" + "=" * 70)
print("ФИНАЛЬНЫЙ ОТЧЁТ ДЛЯ ДИПЛОМА")
print("=" * 70)

print(f"""
  Тикер:              {CONFIG['TICKER']}
  Период:             {CONFIG['START']} — {CONFIG['END']}
  Обучающих дней:     {len(X_train)}
  Тестовых дней:      {len(X_test)}
  Признаков:          {n_features}  (было 42, стало {n_features})
  Горизонт:           {CONFIG['HORIZON']} дней
  Порог Target:       {CONFIG['THR_MOVE']:.1%}
  ─────────────────────────────────
  Accuracy:           {acc:.4f}
  Balanced Accuracy:  {bal_acc:.4f}
  F1-macro:           {f1_mac:.4f}
  AUC-ROC:            {auc_roc:.4f}
  AUC-PR:             {auc_pr:.4f}
  Precision (BUY):    {prec:.4f}
  Recall (BUY):       {rec:.4f}
  Оптим. порог:       {BEST_THR}
  ─────────────────────────────────
  Архитектура:        LSTM({CONFIG['LSTM_UNITS_1']}→{CONFIG['LSTM_UNITS_2']}) + L2
  Параметров:         {model.count_params():,}
  Scaler:             RobustScaler
""")

print("✅ ВСЕ БЛОКИ ЗАВЕРШЕНЫ")
