from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import average_precision_score, roc_auc_score

from src.config.settings import CFG, FIT_VERBOSE, SHOW_MODEL_SUMMARY, SHOW_TRAIN_VAL_DIAG
from src.evaluation.backtest import backtest_nonoverlap_long_only_stats
from src.evaluation.diagnostics import decile_report, drift_report_features, prob_summary, psi_1d
from src.evaluation.metrics import (
    apply_platt_calibrator,
    compact_prob_metrics,
    fit_platt_calibrator,
    history_summary,
    make_decile_table,
    pick_threshold_on_val,
)
from src.models.tcn_model import build_tcn_model
from src.training.callbacks import ShortMetrics, set_global_seed


def run_once(
    run_seed: int,
    prepared: Dict[str, Any],
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one full train/eval cycle for a given seed (data is already prepared)."""
    tf.keras.backend.clear_session()
    gc.collect()
    set_global_seed(int(run_seed))

    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_weight = prepared["class_weight"]

    dw = prepared["dw"]
    test_mask = prepared["test_mask"]
    future_ret_w = prepared["future_ret_w"]
    px = prepared["px"]

    ckpt_path = f"tcn_best_seed_{int(run_seed)}.weights.h5"

    model = build_tcn_model(CFG["WINDOW"], X_train.shape[2], CFG["LR"])
    if SHOW_MODEL_SUMMARY:
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        ShortMetrics(every=5),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CFG["EPOCHS"],
        batch_size=CFG["BATCH"],
        class_weight=class_weight,
        shuffle=False,
        callbacks=callbacks,
        verbose=FIT_VERBOSE,
    )

    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    prob_train = model.predict(X_train, verbose=0).reshape(-1)
    prob_val = model.predict(X_val, verbose=0).reshape(-1)
    prob_test = model.predict(X_test, verbose=0).reshape(-1)

    # Orientation calibration on the last part of VAL (no TEST leakage)
    N_CAL = min(120, len(y_val))
    auc_tail = roc_auc_score(y_val[-N_CAL:], prob_val[-N_CAL:]) if len(np.unique(y_val[-N_CAL:])) > 1 else 0.5
    auc_tail_inv = roc_auc_score(y_val[-N_CAL:], 1 - prob_val[-N_CAL:]) if len(np.unique(y_val[-N_CAL:])) > 1 else 0.5
    FLIP = auc_tail_inv > auc_tail

    if SHOW_TRAIN_VAL_DIAG:
        print("\n" + "=" * 70)
        print(f"VAL tail AUC={auc_tail:.3f} | AUC(1-p)={auc_tail_inv:.3f} | FLIP={FLIP}")
        print("=" * 70)

    if FLIP:
        prob_train = 1 - prob_train
        prob_val = 1 - prob_val
        prob_test = 1 - prob_test

    # Optional Platt scaling on the tail of VAL (no TEST leakage).
    if bool(CFG.get("USE_PLATT_CALIBRATION", True)):
        n_cal = max(
            int(len(y_val) * float(CFG.get("CALIB_VAL_TAIL_FRAC", 0.50))),
            int(CFG.get("CALIB_VAL_MIN_SAMPLES", 120)),
        )
        n_cal = min(n_cal, len(y_val))
        if n_cal >= 10 and len(np.unique(y_val[-n_cal:])) > 1:
            calibrator = fit_platt_calibrator(
                y_val[-n_cal:],
                prob_val[-n_cal:],
            )

            prob_train = apply_platt_calibrator(calibrator, prob_train)
            prob_val = apply_platt_calibrator(calibrator, prob_val)
            prob_test = apply_platt_calibrator(calibrator, prob_test)

            print(f"Applied Platt calibration on last {n_cal} VAL samples")
        else:
            print(f"Skip Platt calibration: n_cal={n_cal}, classes={np.unique(y_val[-n_cal:])}")

    if SHOW_TRAIN_VAL_DIAG:
        prob_summary("TRAIN", y_train, prob_train)
        prob_summary("VAL", y_val, prob_val)
    prob_summary("TEST", y_test, prob_test)

    fret_train = future_ret_w[prepared["train_mask"]]
    fret_val = future_ret_w[prepared["val_mask"]]
    fret_test = future_ret_w[prepared["test_mask"]]

    if SHOW_TRAIN_VAL_DIAG:
        decile_report("TRAIN", y_train, prob_train, fret_train)
        decile_report("VAL", y_val, prob_val, fret_val)
    decile_report("TEST", y_test, prob_test, fret_test)

    X_train_last = X_train[:, -1, :]
    X_test_last = X_test[:, -1, :]
    drift_report_features(X_train_last, X_test_last, prepared["FEATURES"], top_k=12)

    p_psi = psi_1d(prob_train, prob_test, n_bins=10)
    print("\n" + "=" * 70)
    print(f"PROB DRIFT PSI(train→test): {p_psi:.3f} (if >0.25 — strong shift)")
    print("=" * 70)

    thr_f1, thr_pnl, thr_tab = pick_threshold_on_val(
        y_val,
        prob_val,
        fret_val,
        CFG["HORIZON"],
        CFG["FEE"],
    )

    hist_info = history_summary(history)
    test_metrics = compact_prob_metrics(y_test, prob_test)
    test_metrics["pr_auc"] = float(average_precision_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else float("nan")
    test_metrics["pr_auc_lift"] = float(test_metrics["pr_auc"] - test_metrics["pos_rate"]) if np.isfinite(test_metrics["pr_auc"]) else float("nan")

    dec_test = make_decile_table(y_test, prob_test, future_ret=fret_test, n_bins=10)
    dec_spread = float(dec_test.iloc[-1]["buy_rate"] - dec_test.iloc[0]["buy_rate"]) if len(dec_test) >= 2 else float("nan")

    _dates_test = dw[test_mask]
    _close_full = px["Close"]

    bt_f1 = backtest_nonoverlap_long_only_stats(prob_test, _dates_test, _close_full, thr_f1, CFG["HORIZON"], CFG["FEE"])
    bt_pnl = backtest_nonoverlap_long_only_stats(prob_test, _dates_test, _close_full, thr_pnl, CFG["HORIZON"], CFG["FEE"])

    print("\n" + "=" * 70)
    print("=== COMPACT DASHBOARD ===")
    print("=" * 70)
    print(
        f"HISTORY | epochs_run={hist_info['epochs_run']} "
        f"best_epoch={hist_info['best_epoch']} "
        f"best_val_auc_pr={hist_info['best_val_auc_pr']:.4f} "
        f"best_val_auc_roc={hist_info['best_val_auc_roc']:.4f} "
        f"best_val_loss={hist_info['best_val_loss']:.4f}"
    )
    print(
        f"TEST | roc_auc={test_metrics['roc_auc']:.4f} "
        f"pr_auc={test_metrics['pr_auc']:.4f} "
        f"pr_lift={test_metrics['pr_auc_lift']:+.4f} | "
        f"logloss_gain={test_metrics['logloss_gain_vs_baseline']:+.4f} | "
        f"ece10={test_metrics['ece10']:.4f} | "
        f"prob_mean={test_metrics['prob_mean']:.4f}±{test_metrics['prob_std']:.4f} | "
        f"prob_psi(train→test)={p_psi:.3f}"
    )
    print(f"inverted_signal={test_metrics['roc_auc_inv'] > test_metrics['roc_auc']} (roc_auc(1-p)={test_metrics['roc_auc_inv']:.4f})")
    print(f"decile_spread(buy_rate top-bottom)={dec_spread:+.4f}")
    print("deciles (TEST):")
    print(dec_test.to_string(index=False))

    print("\nBACKTEST TEST (non-overlap, Close):")
    print(
        f"thr_f1={thr_f1:.2f} | strat={bt_f1['strategy_return']:+.2%} "
        f"bh={bt_f1['buyhold_return']:+.2%} alpha={bt_f1['alpha']:+.2%} "
        f"trades={bt_f1['n_trades']} win={bt_f1['winrate']:.1%}"
    )
    print(
        f"thr_pnl={thr_pnl:.2f} | strat={bt_pnl['strategy_return']:+.2%} "
        f"bh={bt_pnl['buyhold_return']:+.2%} alpha={bt_pnl['alpha']:+.2%} "
        f"trades={bt_pnl['n_trades']} win={bt_pnl['winrate']:.1%}"
    )
    print("=" * 70)

    alpha_thr_pnl = float(bt_pnl["alpha"])
    n_trades_thr_pnl = int(bt_pnl["n_trades"])

    if bool(CFG.get("SAVE_PER_SEED_TABLES", True)) and run_dir is not None:
        seed_dir = Path(run_dir) / f"seed_{int(run_seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        thr_tab.to_csv(seed_dir / "thresholds_val.csv", index=False)
        dec_test.to_csv(seed_dir / "decile_test.csv", index=False)

        pd.DataFrame(
            [
                {
                    "seed": int(run_seed),
                    "best_epoch": int(hist_info["best_epoch"]) if pd.notna(hist_info["best_epoch"]) else np.nan,
                    "thr_f1": float(thr_f1),
                    "thr_pnl": float(thr_pnl),
                    "roc_auc": float(test_metrics["roc_auc"]),
                    "pr_auc": float(test_metrics["pr_auc"]),
                    "logloss_gain_vs_baseline": float(test_metrics["logloss_gain_vs_baseline"]),
                    "prob_psi": float(p_psi),
                    "alpha_thr_pnl": float(alpha_thr_pnl),
                    "n_trades_thr_pnl": int(n_trades_thr_pnl),
                }
            ]
        ).to_csv(seed_dir / "seed_summary.csv", index=False)

    # For summary table
    return {
        "seed": int(run_seed),
        "roc_auc": float(test_metrics["roc_auc"]),
        "pr_auc": float(test_metrics["pr_auc"]),
        "logloss_gain_vs_baseline": float(test_metrics["logloss_gain_vs_baseline"]),
        "prob_psi": float(p_psi),
        "alpha_thr_pnl": float(alpha_thr_pnl),
        "n_trades_thr_pnl": int(n_trades_thr_pnl),
        "best_epoch": int(hist_info["best_epoch"]) if np.isfinite(hist_info["best_epoch"]) else -1,
        "thr_f1": float(thr_f1),
        "thr_pnl": float(thr_pnl),
    }
