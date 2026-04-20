from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from src.evaluation.metrics import ece_score


def prob_summary(name: str, y_true: np.ndarray, prob: np.ndarray) -> None:
    y_true = y_true.astype(int)
    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    pos = float(y_true.mean())
    print("\n" + "=" * 70)
    print(f"PROB SUMMARY: {name}")
    print("=" * 70)
    print(f"Samples: {len(y_true)} | Pos rate (BUY=1): {pos:.3f}")
    print(
        f"Prob mean={prob.mean():.3f} std={prob.std():.3f} "
        f"p05={np.quantile(prob,0.05):.3f} p50={np.quantile(prob,0.50):.3f} p95={np.quantile(prob,0.95):.3f}"
    )

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, prob))
        auc_inv = float(roc_auc_score(y_true, 1 - prob))
        ap = float(average_precision_score(y_true, prob))
        print(f"ROC-AUC(prob): {auc:.3f}")
        print(f"ROC-AUC(1-prob): {auc_inv:.3f} (если > ROC-AUC(prob), сигнал мог 'перевернуться')")
        print(f"PR-AUC(AP): {ap:.3f}")

    # probability quality
    from sklearn.metrics import log_loss, brier_score_loss

    ll = float(log_loss(y_true, prob))
    bs = float(brier_score_loss(y_true, prob))
    ece = float(ece_score(y_true, prob, n_bins=10))
    print(f"LogLoss: {ll:.3f} | Brier: {bs:.3f} | ECE(10 bins): {ece:.3f}")

    base_prob = np.full_like(prob, fill_value=pos, dtype=float)
    base_ll = float(log_loss(y_true, np.clip(base_prob, 1e-6, 1 - 1e-6)))
    print(f"Baseline LogLoss (const p=pos_rate): {base_ll:.3f}")


def decile_report(name: str, y_true: np.ndarray, prob: np.ndarray, future_ret=None) -> None:
    df = pd.DataFrame({"y": y_true.astype(int), "p": prob.astype(float)})
    if future_ret is not None:
        df["fret"] = future_ret.astype(float)

    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    g = df.groupby("decile").agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        buy_rate=("y", "mean"),
    )
    if future_ret is not None:
        g["avg_future_ret"] = df.groupby("decile")["fret"].mean()

    print("\n" + "=" * 70)
    print(f"DECILE REPORT: {name}")
    print("=" * 70)
    print(g.reset_index().to_string(index=False))


def psi_1d(train: np.ndarray, test: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index for one feature (train defines bins)."""
    train = train[~np.isnan(train)]
    test = test[~np.isnan(test)]
    if len(train) < 50 or len(test) < 50:
        return float("nan")

    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(train, qs))
    if len(bins) <= 2:
        return float("nan")

    tr_hist, _ = np.histogram(train, bins=bins)
    te_hist, _ = np.histogram(test, bins=bins)

    tr = tr_hist / max(tr_hist.sum(), 1)
    te = te_hist / max(te_hist.sum(), 1)
    tr = np.clip(tr, 1e-6, None)
    te = np.clip(te, 1e-6, None)

    return float(np.sum((te - tr) * np.log(te / tr)))


def drift_report_features(X_train_2d: np.ndarray, X_test_2d: np.ndarray, feature_names: list, top_k: int = 10) -> None:
    """Feature drift report using PSI(train->test)."""
    psis = []
    for j, name in enumerate(feature_names):
        p = psi_1d(X_train_2d[:, j], X_test_2d[:, j], n_bins=10)
        if p is None or np.isnan(p):
            continue
        psis.append((str(name), float(p)))
    psis.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("FEATURE DRIFT (PSI train→test): top")
    print("=" * 70)
    for n, p in psis[:top_k]:
        flag = " !!!" if p >= 0.25 else (" !!" if p >= 0.10 else "")
        print(f"{n:25s} PSI={p:.3f}{flag}")


def threshold_sweep(name: str, y_true: np.ndarray, prob: np.ndarray, thresholds=None, top_k: int = 10) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.02)

    rows = []
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        rows.append({
            "thr": float(thr),
            "buy_rate": float(pred.mean()),
            "acc": float(accuracy_score(y_true, pred)),
            "bal_acc": float(balanced_accuracy_score(y_true, pred)),
            "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "f1_buy": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0),
        })

    tab = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(f"THRESHOLD SWEEP: {name} (top by F1-macro)")
    print("=" * 70)
    print(tab.sort_values("f1_macro", ascending=False).head(top_k).to_string(index=False))
    return tab


def eval_block(name: str, y_true: np.ndarray, prob: np.ndarray, thr: float):
    pred = (prob >= thr).astype(int)
    print("\n" + "=" * 70)
    print(f"{name} | threshold={thr:.2f}")
    print("=" * 70)

    print(f"Accuracy: {accuracy_score(y_true, pred):.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, pred):.3f}")
    print(f"F1 macro: {f1_score(y_true, pred, average='macro', zero_division=0):.3f}")
    print(f"F1 (BUY=1): {f1_score(y_true, pred, pos_label=1, zero_division=0):.3f}")
    print(f"MCC: {(matthews_corrcoef(y_true, pred) if len(np.unique(pred)) > 1 else 0.0):.3f}")

    if len(np.unique(y_true)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_true, prob):.3f}")
        print(f"PR-AUC: {average_precision_score(y_true, prob):.3f}")

    cm = confusion_matrix(y_true, pred)
    print("\nConfusion matrix:")
    print(f"TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"FN={cm[1,0]}, TP={cm[1,1]}")

    print("\nClassification report:")
    print(classification_report(y_true, pred, zero_division=0, target_names=["No Growth", "Growth>thr"]))

