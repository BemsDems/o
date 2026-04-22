from __future__ import annotations

from typing import Dict, Tuple

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


def non_overlap_pnl_panel(
    pred: np.ndarray,
    future_ret: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
    horizon: int,
    fee: float,
) -> Tuple[float, int]:
    """Non-overlap PnL computed per ticker, then trades are pooled.

    This avoids cross-ticker overlap artifacts and matches a panel setup where
    windows/signals are per-asset.
    """
    tmp = pd.DataFrame(
        {
            "pred": pred.astype(int),
            "future_ret": future_ret.astype(float),
            "date": pd.to_datetime(dates),
            "ticker": tickers.astype(str),
        }
    ).sort_values(["ticker", "date"]).reset_index(drop=True)

    trades = []
    for _, g in tmp.groupby("ticker", sort=False):
        p = g["pred"].values
        fr = g["future_ret"].values

        i = 0
        while i < len(g):
            if p[i] == 1:
                trades.append(float(fr[i] - fee))
                i += horizon
            else:
                i += 1

    if not trades:
        return 0.0, 0

    return float(np.mean(trades)), int(len(trades))


def panel_backtest_by_ticker(
    prob: np.ndarray,
    dates_signal: np.ndarray,
    tickers_signal: np.ndarray,
    close_map: Dict[str, pd.DataFrame],
    thr: float,
    horizon: int,
    fee: float,
) -> pd.DataFrame:
    rows = []

    tickers_signal = np.asarray(tickers_signal).astype(str)
    dates_signal = pd.to_datetime(dates_signal)

    for ticker in sorted(np.unique(tickers_signal)):
        mask = tickers_signal == ticker
        if int(mask.sum()) == 0:
            continue
        if ticker not in close_map:
            continue

        bt = backtest_nonoverlap_long_only_stats(
            prob[mask],
            dates_signal[mask],
            close_map[ticker]["Close"],
            thr,
            horizon,
            fee,
        )

        rows.append(
            {
                "ticker": ticker,
                "strategy_return": float(bt["strategy_return"]),
                "buyhold_return": float(bt["buyhold_return"]),
                "alpha": float(bt["alpha"]),
                "n_trades": int(bt["n_trades"]),
                "winrate": float(bt["winrate"]),
                "avg_trade_ret": float(bt["avg_trade_ret"]),
                "median_trade_ret": float(bt["median_trade_ret"]),
            }
        )

    return pd.DataFrame(rows)


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
