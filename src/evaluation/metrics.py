from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from src.evaluation.backtest import non_overlap_pnl, non_overlap_pnl_panel, non_overlap_pnl_panel_by_ticker


def ece_score(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    y_true = y_true.astype(int)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(prob[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def fit_platt_calibrator(y_true: np.ndarray, prob: np.ndarray) -> LogisticRegression:
    """Fit Platt scaling calibrator on logits.

    Uses logistic regression on the logit(p) so we calibrate probabilities
    without changing rank ordering too aggressively.
    """
    p = np.clip(prob.astype(float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)

    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(x, y_true.astype(int))
    return clf


def apply_platt_calibrator(calibrator: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob.astype(float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    return calibrator.predict_proba(x)[:, 1]


def history_summary(history) -> dict:
    h = pd.DataFrame(getattr(history, "history", {}) or {})
    out = {
        "epochs_run": int(len(h)),
        "best_epoch": np.nan,
        "best_val_auc_pr": np.nan,
        "best_val_auc_roc": np.nan,
        "best_val_loss": np.nan,
    }
    if h.empty:
        return out

    if "val_auc_pr" in h.columns:
        out["best_epoch"] = int(h["val_auc_pr"].idxmax()) + 1
        out["best_val_auc_pr"] = float(h["val_auc_pr"].max())
    elif "val_loss" in h.columns:
        out["best_epoch"] = int(h["val_loss"].idxmin()) + 1

    if "val_auc_roc" in h.columns:
        out["best_val_auc_roc"] = float(h["val_auc_roc"].max())
    if "val_loss" in h.columns:
        out["best_val_loss"] = float(h["val_loss"].min())

    return out


def compact_prob_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    prob = np.clip(prob.astype(float), 1e-6, 1 - 1e-6)

    pos_rate = float(y_true.mean())
    out = {
        "pos_rate": pos_rate,
        "prob_mean": float(prob.mean()),
        "prob_std": float(prob.std()),
        "logloss": float(log_loss(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece10": float(ece_score(y_true, prob, n_bins=10)),
    }

    base_prob = np.full_like(prob, fill_value=pos_rate, dtype=float)
    out["baseline_logloss"] = float(log_loss(y_true, np.clip(base_prob, 1e-6, 1 - 1e-6)))
    out["logloss_gain_vs_baseline"] = float(out["baseline_logloss"] - out["logloss"])

    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["roc_auc_inv"] = float(roc_auc_score(y_true, 1 - prob))
        # PR-AUC is computed in diagnostics module (needs average_precision_score); keep minimal here.
    else:
        out["roc_auc"] = float("nan")
        out["roc_auc_inv"] = float("nan")

    return out


def make_decile_table(
    y_true: np.ndarray,
    prob: np.ndarray,
    future_ret: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true.astype(int), "p": prob.astype(float)})
    if future_ret is not None:
        df["fret"] = future_ret.astype(float)

    df["decile"] = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        buy_rate=("y", "mean"),
    )
    if future_ret is not None:
        g["avg_future_ret"] = df.groupby("decile")["fret"].mean()

    return g.reset_index()


def pick_threshold_on_val(
    y_true_val: np.ndarray,
    prob_val: np.ndarray,
    future_ret_val: np.ndarray,
    horizon: int,
    fee: float,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)

    rows = []
    for thr in thresholds:
        pred = (prob_val >= thr).astype(int)

        f1_1 = f1_score(y_true_val, pred, pos_label=1, zero_division=0)
        bal = balanced_accuracy_score(y_true_val, pred)
        mcc = matthews_corrcoef(y_true_val, pred) if len(np.unique(pred)) > 1 else 0.0

        avg_pnl, n_tr = non_overlap_pnl(pred, future_ret_val, horizon, fee)

        MIN_TRADES = 15
        MIN_BUY_RATE = 0.05
        MAX_BUY_RATE = 0.35
        feasible = (n_tr >= MIN_TRADES) and (MIN_BUY_RATE <= pred.mean() <= MAX_BUY_RATE)
        if not feasible:
            avg_pnl = -1e9

        rows.append(
            {
                "thr": float(thr),
                "f1_class1": float(f1_1),
                "balanced_acc": float(bal),
                "mcc": float(mcc),
                "share_buy": float(pred.mean()),
                "avg_trade_ret_nonoverlap": float(avg_pnl),
                "n_trades_nonoverlap": int(n_tr),
            }
        )

    tab = pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)
    tab_feas = tab[tab["avg_trade_ret_nonoverlap"] > -1e8].copy()

    if tab_feas.empty:
        thr_f1 = 0.50
        thr_pnl = 0.50
    else:
        thr_f1 = float(tab_feas.iloc[tab_feas["f1_class1"].values.argmax()]["thr"])
        thr_pnl = float(tab_feas.iloc[tab_feas["avg_trade_ret_nonoverlap"].values.argmax()]["thr"])

    return float(thr_f1), float(thr_pnl), tab


def pick_threshold_on_val_panel(
    y_true_val: np.ndarray,
    prob_val: np.ndarray,
    future_ret_val: np.ndarray,
    dates_val: np.ndarray,
    tickers_val: np.ndarray,
    horizon: int,
    fee: float,
    thresholds=None,
):
    """Threshold search for a panel dataset.

    Uses non_overlap_pnl_panel() so trades are evaluated independently per ticker.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)

    rows = []
    for thr in thresholds:
        pred = (prob_val >= thr).astype(int)

        f1_1 = f1_score(y_true_val, pred, pos_label=1, zero_division=0)
        bal = balanced_accuracy_score(y_true_val, pred)
        mcc = matthews_corrcoef(y_true_val, pred) if len(np.unique(pred)) > 1 else 0.0

        pnl_tbl = non_overlap_pnl_panel_by_ticker(
            pred,
            future_ret_val,
            dates_val,
            tickers_val,
            horizon,
            fee,
        )

        avg_pnl = float(pnl_tbl["avg_trade_ret"].mean()) if not pnl_tbl.empty else -1e9
        n_tr = int(pnl_tbl["n_trades"].sum()) if not pnl_tbl.empty else 0

        MIN_TRADES = 20
        MIN_TRADES_PER_TICKER = 3
        MIN_BUY_RATE = 0.05
        MAX_BUY_RATE = 0.35

        feasible_tickers = int((pnl_tbl["n_trades"] >= MIN_TRADES_PER_TICKER).sum()) if not pnl_tbl.empty else 0
        n_unique_tickers = int(len(np.unique(np.asarray(tickers_val).astype(str))))

        feasible = (
            (n_tr >= MIN_TRADES)
            and (MIN_BUY_RATE <= pred.mean() <= MAX_BUY_RATE)
            and (feasible_tickers >= max(2, n_unique_tickers - 1))
        )
        if not feasible:
            avg_pnl = -1e9

        rows.append(
            {
                "thr": float(thr),
                "f1_class1": float(f1_1),
                "balanced_acc": float(bal),
                "mcc": float(mcc),
                "share_buy": float(pred.mean()),
                "avg_trade_ret_nonoverlap": float(avg_pnl),
                "n_trades_nonoverlap": int(n_tr),
            }
        )

    tab = pd.DataFrame(rows).sort_values("thr").reset_index(drop=True)
    tab_feas = tab[tab["avg_trade_ret_nonoverlap"] > -1e8].copy()

    if tab_feas.empty:
        thr_f1 = 0.50
        thr_pnl = 0.50
    else:
        thr_f1 = float(tab_feas.iloc[tab_feas["f1_class1"].values.argmax()]["thr"])
        thr_pnl = float(tab_feas.iloc[tab_feas["avg_trade_ret_nonoverlap"].values.argmax()]["thr"])

    return float(thr_f1), float(thr_pnl), tab
