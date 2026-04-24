from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

from project.config import CFG, CODE_FINGERPRINT, _set_seed
from project.data_loader import (
    add_dividend_features_past_only,
    build_features_one,
    fetch_dividends_moex,
    fetch_macro_data,
    fetch_moex_candles,
    make_target,
)
from project.metrics import evaluate_global, improved_backtest_per_ticker, per_ticker_metrics
from project.model import build_tcn_model
from project.sequences import make_sequences_multi_ticker, time_split_masks


def _banner() -> None:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_sha = "unknown"

    print("\n" + "=" * 72)
    print("SONNET TCN RUN (multi-model)")
    print(f"CODE: {__file__}")
    print(f"GIT_SHA: {git_sha}")
    print(f"FINGERPRINT: {CODE_FINGERPRINT}")
    print(
        "CFG: "
        f"TICKERS={len(CFG.get('TICKERS', []))} "
        f"SEQ_LEN={CFG.get('SEQ_LEN')} "
        f"EPOCHS={CFG.get('EPOCHS')} BATCH_SIZE={CFG.get('BATCH_SIZE')} "
        f"ENSEMBLE_SEEDS={CFG.get('ENSEMBLE_SEEDS')} "
        f"CACHE_DIR={CFG.get('CACHE_DIR')} CACHE_ENABLED={CFG.get('CACHE_ENABLED')}"
    )
    print("=" * 72 + "\n")


def main() -> None:
    _banner()

    save_dir = Path(CFG.get("SAVE_DIR", "artifacts"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── 1) Load raw data ONCE ──
    macro_df = fetch_macro_data(str(CFG["START"]), str(CFG["END"]))
    macro_cols = list(macro_df.columns) if not macro_df.empty else []

    use_div = bool(CFG.get("USE_DIVIDEND_FEATURES", True))
    div_map: dict[str, pd.DataFrame] = {}
    if use_div:
        print("\n=== LOADING DIVIDEND HISTORY ===")
        for secid in CFG["TICKERS"]:
            div_map[str(secid)] = fetch_dividends_moex(str(secid))

    # Build features per ticker ONCE (features don't depend on horizon)
    ticker_features: dict[str, pd.DataFrame] = {}
    for secid in CFG["TICKERS"]:
        secid = str(secid)
        print(f"Loading {secid}...")
        df = fetch_moex_candles(secid, str(CFG["START"]), CFG["END"])  # END may be None
        print(f"  loaded rows: {len(df)}")
        if df.empty:
            print("  -> empty, skip")
            continue

        df_feat = build_features_one(df, secid=secid)

        if use_div:
            df_feat = add_dividend_features_past_only(
                df_feat,
                div_map.get(secid, pd.DataFrame()),
                lag_days=int(CFG.get("DIV_LAG_DAYS", 1)),
            )

        if not macro_df.empty:
            df_feat = df_feat.join(macro_df, how="left")
            for col in macro_cols:
                df_feat[col] = df_feat[col].ffill().bfill().fillna(0.0)
            print(f"  after macro merge: {len(df_feat)} rows, macro cols={len(macro_cols)}")

        ticker_features[secid] = df_feat

    print(f"\nLoaded {len(ticker_features)} tickers")

    # ── 2) Train one model per horizon ──
    multi_models = CFG.get("MULTI_HORIZON_MODELS") or [
        {"name": "short", "HORIZON": 5, "THR_MOVE": 0.03},
        {"name": "medium", "HORIZON": 30, "THR_MOVE": 0.05},
        {"name": "long", "HORIZON": 120, "THR_MOVE": 0.12},
    ]

    for model_cfg in multi_models:
        model_name = str(model_cfg["name"])
        horizon = int(model_cfg["HORIZON"])
        thr_move = float(model_cfg["THR_MOVE"])

        print(f"\n{'='*72}")
        print(f"TRAINING MODEL: {model_name} (horizon={horizon}d, thr={thr_move})")
        print(f"{'='*72}")

        rows: list[pd.DataFrame] = []
        for secid, df_feat in ticker_features.items():
            y, fwd_ret = make_target(df_feat["CLOSE"], horizon, thr_move)

            df_h = df_feat.iloc[:-horizon].copy()
            y = y.iloc[:-horizon]
            fwd_ret = fwd_ret.iloc[:-horizon]

            if len(df_h) < 250:
                print(f"  {secid} h={horizon}d: too few rows ({len(df_h)}), skip")
                continue

            tmp = df_h.copy()
            tmp["target"] = y.values
            tmp["fwd_ret"] = fwd_ret.values
            tmp["date"] = tmp.index
            tmp["secid"] = secid
            rows.append(tmp.reset_index(drop=True))

            print(f"  {secid} h={horizon}d: {len(df_h)} rows, pos={float(y.mean()):.1%}")

        if not rows:
            print(f"SKIP {model_name}: no data")
            continue

        full = pd.concat(rows).sort_values(["secid", "date"]).reset_index(drop=True)

        # Feature columns (NO horizon_norm; horizon is fixed per model)
        technical_cols = [
            "logret_1",
            "logret_2",
            "logret_3",
            "logret_5",
            "logret_10",
            "trend_up_20",
            "trend_up_200",
            "vol_rel",
            "vol_spike",
            "rsi_14",
            "rsi_oversold",
            "rsi_overbought",
            "price_pos_20",
            "volatility_20",
        ]
        fundamental_cols = (
            ["div_yield_ttm", "days_since_last_div", "div_yield_is_missing"] if use_div else []
        )

        feature_cols = [c for c in (technical_cols + fundamental_cols + macro_cols) if c in full.columns]

        print(f"Features: {len(feature_cols)}")
        print(f"Dataset rows: {len(full)} | pos_rate={float(full['target'].mean()):.1%}")

        X = full[feature_cols].values
        y_all = full["target"].astype(int).values
        dates_all = pd.to_datetime(full["date"]).values
        secids_all = full["secid"].astype(str).values
        fwd_ret_all = full["fwd_ret"].astype(float).values

        # Split per ticker by time.
        m_train, m_val, m_test = time_split_masks(dates_all, secids_all)

        scaler = RobustScaler()
        scaler.fit(X[m_train])
        X_scaled = scaler.transform(X)

        (
            X_tr,
            y_tr,
            _,
            _,
            _,
            X_va,
            y_va,
            _,
            _,
            _,
            X_te,
            y_te,
            d_te,
            r_te,
            s_te,
        ) = make_sequences_multi_ticker(
            X_scaled,
            y_all,
            dates_all,
            fwd_ret_all,
            secids_all,
            int(CFG["SEQ_LEN"]),
            split_masks=(m_train, m_val, m_test),
        )

        if X_tr.size == 0 or X_va.size == 0 or X_te.size == 0:
            print(f"SKIP {model_name}: not enough sequences")
            continue

        # GPU-friendly
        X_tr = np.clip(X_tr, -5, 5).astype("float32")
        X_va = np.clip(X_va, -5, 5).astype("float32")
        X_te = np.clip(X_te, -5, 5).astype("float32")

        print(f"Train={X_tr.shape}, Val={X_va.shape}, Test={X_te.shape}")

        n_steps, n_feat = X_tr.shape[1], X_tr.shape[2]

        seeds = [int(s) for s in CFG.get("ENSEMBLE_SEEDS", [42, 43, 44, 45, 46])]
        prob_test_list: list[np.ndarray] = []
        best_val_auc = -1.0
        best_model: tf.keras.Model | None = None

        from sklearn.metrics import roc_auc_score

        for seed in seeds:
            _set_seed(seed)
            tf.keras.backend.clear_session()

            model = build_tcn_model((n_steps, n_feat))

            cb = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    patience=int(CFG.get("ES_PATIENCE", 5)),
                    mode="max",
                    restore_best_weights=True,
                    min_delta=float(CFG.get("ES_MIN_DELTA", 0.005)),
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_auc",
                    factor=0.5,
                    patience=max(2, int(CFG.get("ES_PATIENCE", 5)) // 2),
                    mode="max",
                    min_lr=1e-6,
                ),
            ]

            model.fit(
                X_tr,
                y_tr,
                validation_data=(X_va, y_va),
                epochs=int(CFG["EPOCHS"]),
                batch_size=int(CFG["BATCH_SIZE"]),
                shuffle=True,
                callbacks=cb,
                verbose=2,
            )

            prob_va = model.predict(X_va, verbose=0).ravel()
            prob_te = model.predict(X_te, verbose=0).ravel()
            prob_test_list.append(prob_te)

            va_auc = float(roc_auc_score(y_va, prob_va)) if len(np.unique(y_va)) > 1 else 0.0
            print(f"Seed {seed}: val_auc={va_auc:.4f}")

            if va_auc > best_val_auc:
                best_val_auc = va_auc
                best_model = model

        prob_ens = np.mean(np.vstack(prob_test_list), axis=0)
        test_auc = float(roc_auc_score(y_te, prob_ens)) if len(np.unique(y_te)) > 1 else 0.0

        print(f"\n=== {model_name.upper()} RESULT ===")
        print(f"Ensemble AUC (test): {test_auc:.4f}")
        print(f"Best single val_auc: {best_val_auc:.4f}")

        # Optional: show per-ticker and backtest for ensemble
        try:
            print("\n=== PER-TICKER AUC (ENSEMBLE) ===")
            print(per_ticker_metrics(y_te, prob_ens, s_te).to_string(index=False))

            print("\n=== BACKTEST (ENSEMBLE) ===")
            bt = improved_backtest_per_ticker(
                prob_ens,
                r_te,
                d_te,
                s_te,
                threshold=0.5,
                fee=float(CFG.get("FEE", 0.001)),
            )
            print(bt.to_string(index=False))
        except Exception as e:
            print(f"[warn] diagnostics failed: {type(e).__name__}: {e}")

        # Save best single model for this horizon + scaler + meta
        if best_model is None:
            print(f"SKIP save {model_name}: best_model is None")
            continue

        model_path = save_dir / f"model_{model_name}.keras"
        scaler_path = save_dir / f"scaler_{model_name}.pkl"
        meta_path = save_dir / f"meta_{model_name}.json"

        best_model.save(str(model_path))
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        meta = {
            "name": model_name,
            "horizon": horizon,
            "thr_move": thr_move,
            "test_auc_ensemble": test_auc,
            "best_val_auc_single": best_val_auc,
            "feature_cols": feature_cols,
            "n_tickers": int(len(ticker_features)),
            "cfg": dict(CFG),
            "fingerprint": CODE_FINGERPRINT,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Saved: {model_path}")
        print(f"Saved: {scaler_path}")
        print(f"Saved: {meta_path}")

    print("\n=== ALL MODELS TRAINED ===")
    for m in multi_models:
        p = save_dir / f"model_{m['name']}.keras"
        print(f" {m['name']}: {p} {'✓' if p.exists() else '✗'}")


if __name__ == "__main__":
    main()
