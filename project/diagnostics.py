from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def plot_probability_distribution(y_true, y_prob, name=""):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return

    y_true = y_true.astype(int)
    prob_0 = y_prob[y_true == 0]
    prob_1 = y_prob[y_true == 1]

    print(f"\n=== PROBABILITY DISTRIBUTION {name} ===")
    if len(prob_0):
        print(f"Class 0: mean={prob_0.mean():.4f}, std={prob_0.std():.4f}")
    if len(prob_1):
        print(f"Class 1: mean={prob_1.mean():.4f}, std={prob_1.std():.4f}")
    if len(prob_0) and len(prob_1):
        sep = abs(prob_1.mean() - prob_0.mean())
        print(f"Separation: {sep:.4f}")
        if sep > 0.02:
            print("✅ Модель различает классы")
        else:
            print("⚠️ Слабое разделение классов")

    plt.figure(figsize=(10, 5))
    plt.hist(prob_0, bins=50, alpha=0.6, label="No Growth (y=0)", color="red")
    plt.hist(prob_1, bins=50, alpha=0.6, label="Growth (y=1)", color="green")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title(f"Probability Distribution {name}")
    plt.axvline(0.5, color="black", linestyle="--")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(f"prob_dist_{name.lower()}.png", dpi=150, bbox_inches="tight")
    print(f"Saved: prob_dist_{name.lower()}.png")
    plt.close()


def calibration_curve_analysis(y_true, y_prob, n_bins=10):
    from sklearn.calibration import calibration_curve

    y_prob_clipped = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    prob_true, prob_pred = calibration_curve(
        y_true.astype(int), y_prob_clipped, n_bins=n_bins, strategy="uniform"
    )
    print("\n=== CALIBRATION ANALYSIS ===")
    print(f"{'Predicted':>12s} | {'True Freq':>12s} | {'Diff':>8s}")
    print("-" * 40)
    for pred, true in zip(prob_pred, prob_true):
        diff = abs(true - pred)
        status = "✓" if diff < 0.1 else "✗"
        print(f"{pred:12.3f} | {true:12.3f} | {diff:8.3f} {status}")
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    print(f"\nECE: {ece:.4f}")


def analyze_predictions_by_confidence(y_true, y_prob, fwd_ret):
    print("\n=== PREDICTIONS BY CONFIDENCE LEVEL ===")
    deciles = np.percentile(y_prob, np.arange(0, 101, 10))
    print(
        f"{'Decile':>6s} | {'Range':>15s} | {'N':>5s} | {'Accuracy':>8s} | {'AvgRet':>8s} | {'Precision':>9s}"
    )
    print("-" * 70)

    for i in range(len(deciles) - 1):
        lo, hi = deciles[i], deciles[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < len(deciles) - 2 else (y_prob >= lo)
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        accuracy = float((y_true[mask] == (y_prob[mask] >= 0.5)).mean())
        avg_ret = float(fwd_ret[mask].mean())
        precision = float(y_true[mask].mean())
        print(f"D{i+1:2d} | {lo:.3f} - {hi:.3f} | {n:5d} | {accuracy:8.3f} | {avg_ret:8.4f} | {precision:9.3f}")

    top10 = y_prob >= np.percentile(y_prob, 90)
    bot10 = y_prob <= np.percentile(y_prob, 10)
    print(
        f"\nTOP 10%: N={int(top10.sum())}, precision={float(y_true[top10].mean()):.3f}, "
        f"avg_ret={float(fwd_ret[top10].mean()):.4f}"
    )
    print(
        f"BOT 10%: N={int(bot10.sum())}, precision={float(y_true[bot10].mean()):.3f}, "
        f"avg_ret={float(fwd_ret[bot10].mean()):.4f}"
    )


def confusion_matrix_analysis(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred)
    print(f"\n=== CONFUSION MATRIX (threshold={threshold:.2f}) ===")
    tn, fp, fn, tp = cm.ravel()
    print(f" TN={tn:5d} FP={fp:5d}")
    print(f" FN={fn:5d} TP={tp:5d}")
    print(
        f"\n{classification_report(y_true, y_pred, target_names=['No Growth', 'Growth'], digits=3, zero_division=0)}"
    )


def feature_importance_proxy(model, X_test, y_test, feature_names, max_features=20):
    print("\n=== FEATURE IMPORTANCE (Permutation) ===")
    y_test = y_test.astype(int)
    if len(np.unique(y_test)) < 2:
        print("Undefined (single class)")
        return []

    base_pred = model.predict(X_test, verbose=0).ravel()
    base_auc = float(roc_auc_score(y_test, base_pred))

    n_feat = X_test.shape[2]
    importances = []
    for i in range(min(max_features, n_feat)):
        fname = feature_names[i] if i < len(feature_names) else f"f{i}"
        X_perm = X_test.copy()
        for t in range(X_perm.shape[1]):
            np.random.shuffle(X_perm[:, t, i])
        perm_auc = float(roc_auc_score(y_test, model.predict(X_perm, verbose=0).ravel()))
        importances.append((fname, base_auc - perm_auc))

    importances.sort(key=lambda x: x[1], reverse=True)
    print(f"Base AUC: {base_auc:.4f}")
    print(f"{'Feature':>20s} | {'Importance':>10s}")
    print("-" * 35)
    for fname, imp in importances:
        bar = "█" * max(0, int(imp * 200))
        print(f"{fname:>20s} | {imp:10.4f} {bar}")
    return importances


def temporal_performance_analysis(y_true, y_prob, dates):
    df = pd.DataFrame(
        {"date": pd.to_datetime(dates), "y_true": y_true.astype(int), "y_prob": y_prob.astype(float)}
    )
    df["month"] = df["date"].dt.to_period("M")

    print("\n=== TEMPORAL PERFORMANCE (Monthly AUC) ===")
    monthly_aucs = []
    for month, g in df.groupby("month"):
        if len(g) < 10:
            continue
        if len(np.unique(g["y_true"].values)) < 2:
            continue
        auc = float(roc_auc_score(g["y_true"].values, g["y_prob"].values))
        monthly_aucs.append(auc)
        pos_rate = float(g["y_true"].mean())
        print(f"{str(month):10s} | N={len(g):4d} | pos={pos_rate:.3f} | AUC={auc:.3f}")

    if monthly_aucs:
        print(f"\nMean monthly AUC: {np.mean(monthly_aucs):.3f} ± {np.std(monthly_aucs):.3f}")


def check_random_baseline(y_true, n_iterations=1000):
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2:
        print("Undefined (single class)")
        return
    aucs = [float(roc_auc_score(y_true, np.random.uniform(0, 1, len(y_true)))) for _ in range(n_iterations)]
    mu, sd = float(np.mean(aucs)), float(np.std(aucs))
    print(f"\n=== RANDOM BASELINE ({n_iterations} iterations) ===")
    print(f"Random AUC: {mu:.4f} ± {sd:.4f}")

