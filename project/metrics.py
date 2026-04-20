from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from project.config import CFG


def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def evaluate_global(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "AUC": _safe_auc(y_true, y_prob),
        "PR_AUC": _safe_ap(y_true, y_prob),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "BalAcc": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "F1": float(f1_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def per_ticker_metrics(y_true, y_prob, secids):
    rows = []
    for s in sorted(set(secids)):
        m = secids == s
        rows.append(
            {
                "secid": s,
                "n": int(m.sum()),
                "AUC": _safe_auc(y_true[m], y_prob[m]),
                "PR_AUC": _safe_ap(y_true[m], y_prob[m]),
            }
        )
    return pd.DataFrame(rows).sort_values("secid")


def improved_backtest_per_ticker(y_prob, fwd_ret, dates, secids, threshold, fee):
    horizon = int(CFG["HORIZON"])
    results = []

    for secid in sorted(set(secids)):
        mask = secids == secid
        if int(mask.sum()) < 50:
            continue

        prob_sec = y_prob[mask]
        ret_sec = fwd_ret[mask]
        dates_sec = dates[mask]

        order = np.argsort(dates_sec)
        prob_sec = prob_sec[order]
        ret_sec = ret_sec[order]

        # Fixed threshold
        trades_fixed = []
        i = 0
        while i < len(prob_sec):
            if prob_sec[i] >= threshold:
                net = float(ret_sec[i]) - 2.0 * float(fee)
                trades_fixed.append(net)
                i += horizon
            else:
                i += 1

        # Top-20%
        top20_thr = float(np.percentile(prob_sec, 80))
        trades_top20 = []
        i = 0
        while i < len(prob_sec):
            if prob_sec[i] >= top20_thr:
                net = float(ret_sec[i]) - 2.0 * float(fee)
                trades_top20.append(net)
                i += horizon
            else:
                i += 1

        results.append(
            {
                "secid": str(secid),
                "n_trades_fixed": len(trades_fixed),
                "total_ret_fixed": float(np.sum(trades_fixed)) if trades_fixed else 0.0,
                "win_rate_fixed": float(np.mean([1 if t > 0 else 0 for t in trades_fixed])) if trades_fixed else 0.0,
                "n_trades_top20": len(trades_top20),
                "total_ret_top20": float(np.sum(trades_top20)) if trades_top20 else 0.0,
                "sharpe_fixed": float(np.mean(trades_fixed) / (np.std(trades_fixed) + 1e-9)) if len(trades_fixed) > 1 else 0.0,
            }
        )

    return pd.DataFrame(results)

