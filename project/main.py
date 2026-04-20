from __future__ import annotations

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

from project.config import CFG, seed_everything
from project.data_loader import MultiDataset, build_multi_ticker_dataset
from project.diagnostics import (
    analyze_predictions_by_confidence,
    calibration_curve_analysis,
    check_random_baseline,
    confusion_matrix_analysis,
    feature_importance_proxy,
    plot_probability_distribution,
    temporal_performance_analysis,
)
from project.metrics import evaluate_global, improved_backtest_per_ticker, per_ticker_metrics
from project.model import build_tcn_model
from project.sequences import make_sequences_multi_ticker, time_split_masks


def main() -> None:
    seed_everything()
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
    from sklearn.metrics import f1_score
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
    print(pd.Series(y_prob).describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]).to_string())

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
