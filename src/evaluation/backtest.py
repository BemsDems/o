from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def non_overlap_pnl(pred: np.ndarray, future_ret: np.ndarray, horizon: int, fee: float) -> Tuple[float, int]:
    """Average return per trade, non-overlapping trades (simple)."""
    i = 0
    trades = []
    while i < len(pred):
        if pred[i] == 1:
            trades.append(float(future_ret[i] - fee))
            i += horizon
        else:
            i += 1
    if not trades:
        return 0.0, 0
    return float(np.mean(trades)), int(len(trades))


def backtest_nonoverlap_long_only_stats(prob, dates_signal, close_full, thr, horizon, fee) -> dict:
    dates = pd.to_datetime(dates_signal)
    close_full = close_full.copy()
    close_full.index = pd.to_datetime(close_full.index)

    start_date = dates[0]
    last_date = dates[-1]
    try:
        last_loc = close_full.index.get_loc(last_date)
        end_loc = min(last_loc + horizon, len(close_full.index) - 1)
        end_date = close_full.index[end_loc]
    except KeyError:
        end_date = close_full.index[-1]

    bh_ret = float(close_full.loc[end_date] / close_full.loc[start_date] - 1.0)

    eq = 1.0
    trades = []
    i = 0
    while i < len(prob):
        if prob[i] >= thr:
            d0 = dates[i]
            try:
                loc0 = close_full.index.get_loc(d0)
            except KeyError:
                i += 1
                continue

            loc1 = loc0 + horizon
            if loc1 >= len(close_full.index):
                break

            entry = float(close_full.iloc[loc0])
            exitp = float(close_full.iloc[loc1])
            ret = exitp / entry - 1.0 - fee

            eq *= (1.0 + ret)
            trades.append(ret)
            i += horizon
        else:
            i += 1

    strat = float(eq - 1.0)
    alpha = float(strat - bh_ret)

    return {
        "strategy_return": strat,
        "buyhold_return": bh_ret,
        "alpha": alpha,
        "n_trades": int(len(trades)),
        "winrate": float((sum(1 for x in trades if x > 0) / len(trades)) if trades else 0.0),
        "avg_trade_ret": float((sum(trades) / len(trades)) if trades else 0.0),
        "median_trade_ret": float(np.median(trades) if trades else 0.0),
    }


def alpha_nonoverlap_stats(prob, dates_signal, close_full, thr, horizon, fee):
    out = backtest_nonoverlap_long_only_stats(prob, dates_signal, close_full, thr, horizon, fee)
    return {"alpha": float(out["alpha"]), "n_trades": int(out["n_trades"])}


def alpha_nonoverlap(prob: np.ndarray, dates_signal: np.ndarray, close_full: pd.Series, thr: float, horizon: int, fee: float) -> float:
    return float(alpha_nonoverlap_stats(prob, dates_signal, close_full, thr, horizon, fee)["alpha"])

