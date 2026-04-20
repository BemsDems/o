from __future__ import annotations

import pandas as pd

from src.config.settings import CFG
from src.training.dataset import prepare_dataset_once
from src.training.train import run_once


def main() -> None:
    prepared = prepare_dataset_once()

    results = []
    for sd in CFG.get("RUN_SEEDS", [42]):
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

    df.to_csv("multi_seed_summary.csv", index=False)

