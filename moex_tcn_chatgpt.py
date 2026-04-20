"""Colab entrypoint for the MOEX TCN baseline.

Usage (Colab):
  %run '/content/o/moex_tcn_chatgpt.py'

The implementation lives in src/*, but this entrypoint also handles
experiment packaging (artifacts, logs, snapshots).
"""

from __future__ import annotations

import contextlib
import traceback
from pathlib import Path

import pandas as pd

from src.config.settings import (
    BASE_FEATURES,
    CFG,
    FIT_VERBOSE,
    SHOW_MODEL_SUMMARY,
    SHOW_TRAIN_VAL_DIAG,
)
from src.training.artifacts import (
    TeeStream,
    build_chat_report,
    download_artifacts_if_needed,
    make_run_dir,
    save_json,
    save_prepared_snapshot,
    write_text,
)
from src.training.dataset import prepare_dataset_once
from src.training.train import run_once


if __name__ == "__main__":
    run_dir = make_run_dir(CFG) if CFG.get("SAVE_RUN_ARTIFACTS", True) else Path(".")
    log_path = run_dir / "run_log.txt"

    save_json(
        run_dir / "config_snapshot.json",
        {
            **CFG,
            "BASE_FEATURES": BASE_FEATURES,
            "SHOW_MODEL_SUMMARY": SHOW_MODEL_SUMMARY,
            "SHOW_TRAIN_VAL_DIAG": SHOW_TRAIN_VAL_DIAG,
            "FIT_VERBOSE": FIT_VERBOSE,
        },
    )

    import sys

    orig_stdout, orig_stderr = sys.stdout, sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_f:
        tee_out = TeeStream(orig_stdout, log_f)
        tee_err = TeeStream(orig_stderr, log_f)

        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                print(f"RUN_DIR: {run_dir.resolve()}")
                print(f"LOG_PATH: {log_path.resolve()}")

                prepared = prepare_dataset_once()
                save_prepared_snapshot(run_dir, prepared, CFG)

                results = []
                for sd in CFG.get("RUN_SEEDS", [42]):
                    try:
                        res = run_once(int(sd), prepared, run_dir=run_dir)
                    except TypeError:
                        # Backward compatibility if older train.py is cached in the runtime
                        res = run_once(int(sd), prepared)
                    results.append(res)
                    print(
                        f"Seed {sd}: "
                        f"roc_auc={res['roc_auc']:.3f} "
                        f"pr_auc={res['pr_auc']:.3f} "
                        f"ll_gain={res['logloss_gain_vs_baseline']:+.4f} "
                        f"psi={res['prob_psi']:.3f} "
                        f"alpha@thr_pnl={res['alpha_thr_pnl']:+.2%} "
                        f"trades={res['n_trades_thr_pnl']}"
                    )

                df = pd.DataFrame(results)

                print("\n=== MULTI-SEED SUMMARY ===")
                with pd.option_context("display.max_columns", 50):
                    print(df.to_string(index=False))
                    print("\nDescribe:")
                    print(df.describe(include="all").to_string())

                num_cols = [
                    "roc_auc",
                    "pr_auc",
                    "logloss_gain_vs_baseline",
                    "prob_psi",
                    "alpha_thr_pnl",
                    "n_trades_thr_pnl",
                ]
                summary = pd.DataFrame(
                    {
                        "metric": num_cols,
                        "mean": [df[c].mean() for c in num_cols],
                        "std": [df[c].std(ddof=1) for c in num_cols],
                        "min": [df[c].min() for c in num_cols],
                        "max": [df[c].max() for c in num_cols],
                    }
                )

                print("\n=== AGGREGATED SUMMARY ===")
                print(summary.to_string(index=False))

                df.to_csv(run_dir / "multi_seed_summary.csv", index=False)
                summary.to_csv(run_dir / "multi_seed_aggregated.csv", index=False)

                write_text(run_dir / "for_chat.txt", build_chat_report(run_dir, CFG, df, summary))

                print("\nSaved artifacts:")
                print(run_dir / "run_log.txt")
                print(run_dir / "config_snapshot.json")
                print(run_dir / "prepared_dataset_info.json")
                print(run_dir / "selected_features.txt")
                print(run_dir / "multi_seed_summary.csv")
                print(run_dir / "multi_seed_aggregated.csv")
                print(run_dir / "for_chat.txt")

                downloaded_path = download_artifacts_if_needed(run_dir, CFG)
                if downloaded_path is not None:
                    print(f"Auto-downloaded: {downloaded_path}")

        except Exception:
            err = traceback.format_exc()
            write_text(run_dir / "error_log.txt", err)
            raise
