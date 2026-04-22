from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from project.config import CFG, _set_seed
from project.data_loader import MultiDataset, build_multi_ticker_dataset
from project.diagnostics import feature_importance_proxy
from project.metrics import evaluate_global, improved_backtest_per_ticker, per_ticker_metrics
from project.model import build_tcn_model
from project.sequences import make_sequences_multi_ticker, time_split_masks


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

    use_ensemble = bool(CFG.get("USE_ENSEMBLE", False))
    if use_ensemble:
        seeds = [int(s) for s in (CFG.get("ENSEMBLE_SEEDS") or [])]
        if not seeds:
            seeds = [42, 43, 44, 45, 46]
        print(f"\nUsing ensemble seeds: {seeds}")
    else:
        base_seed = int(CFG.get("SEED", 42))
        n_runs = int(CFG.get("N_RUNS", 1))
        seeds = [base_seed + i for i in range(n_runs)]

    prob_val_list = []
    prob_test_list = []

    for run_idx, run_seed in enumerate(seeds):
        print(f"\n\n================ RUN {run_idx + 1}/{len(seeds)} | SEED={run_seed} ================")

        _set_seed(run_seed)
        tf.keras.backend.clear_session()

        model = build_tcn_model((n_steps, n_feat))
        model.summary()

        cb = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                patience=int(CFG.get("ES_PATIENCE", 20)),
                mode="max",
                restore_best_weights=True,
                min_delta=float(CFG.get("ES_MIN_DELTA", 0.001)),
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc",
                factor=0.5,
                patience=max(2, int(CFG.get("ES_PATIENCE", 20)) // 2),
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
            shuffle=True,
            class_weight=None,
            callbacks=cb,
            verbose=2,
        )

        y_prob_val = model.predict(X_val, verbose=0).ravel()
        y_prob = model.predict(X_test, verbose=0).ravel()

        prob_val_list.append(y_prob_val)
        prob_test_list.append(y_prob)

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

        # Threshold search (per-run)
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

        # ── Diagnostics (compact) ──
        print("\n=== FINAL RESULTS ===")
        g = evaluate_global(y_test, y_prob, thr=best_thr)
        for k, v in g.items():
            print(f" {k}: {v:.4f}")

        print("\n=== PROB SUMMARY ===")
        print(
            f" min={y_prob.min():.4f}, max={y_prob.max():.4f}, std={y_prob.std():.4f}, mean={y_prob.mean():.4f}"
        )

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
        print(bt.to_string(index=False))

    # Ensemble summary (average probabilities)
    if use_ensemble and prob_test_list:
        prob_val_ens = np.mean(np.vstack(prob_val_list), axis=0)
        prob_test_ens = np.mean(np.vstack(prob_test_list), axis=0)

        print("\n\n================ ENSEMBLE (mean probs) ================")
        print(
            f"Val prob: mean={prob_val_ens.mean():.4f}, std={prob_val_ens.std():.4f}, "
            f"range=[{prob_val_ens.min():.4f}, {prob_val_ens.max():.4f}]"
        )
        print(
            f"Test prob: mean={prob_test_ens.mean():.4f}, std={prob_test_ens.std():.4f}, "
            f"range=[{prob_test_ens.min():.4f}, {prob_test_ens.max():.4f}]"
        )

        # Threshold search on ensemble VAL
        print("\n=== THRESHOLD OPTIMIZATION (VAL, ENSEMBLE) ===")
        best_thr, best_f1 = 0.5, 0.0
        from sklearn.metrics import f1_score

        for thr in np.arange(0.30, 0.85, 0.01):
            pred = (prob_val_ens >= thr).astype(int)
            if pred.sum() == 0:
                continue
            f1v = f1_score(y_val, pred, zero_division=0)
            if f1v > best_f1:
                best_f1, best_thr = float(f1v), float(thr)

        print(f"Best threshold (ensemble): {best_thr:.2f} (F1={best_f1:.3f})")

        print("\n=== FINAL RESULTS (ENSEMBLE) ===")
        g = evaluate_global(y_test, prob_test_ens, thr=best_thr)
        for k, v in g.items():
            print(f" {k}: {v:.4f}")

        print("\n=== PER-TICKER AUC (ENSEMBLE) ===")
        print(per_ticker_metrics(y_test, prob_test_ens, secids_test).to_string(index=False))

        print("\n=== BACKTEST (ENSEMBLE) ===")
        bt = improved_backtest_per_ticker(
            prob_test_ens,
            fwd_test,
            dates_test,
            secids_test,
            threshold=best_thr,
            fee=float(CFG["FEE"]),
        )
        print(bt.to_string(index=False))


if __name__ == "__main__":
    main()
