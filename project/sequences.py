from __future__ import annotations

import numpy as np

from project.config import CFG


def make_sequences_with_meta(X, y, dates, fwd_ret, secids, seq_len):
    Xs, ys, ds, rs, ss = [], [], [], [], []
    for i in range(seq_len, len(X)):
        if not (secids[i - seq_len : i] == secids[i]).all():
            continue
        Xs.append(X[i - seq_len : i])
        ys.append(y[i])
        ds.append(dates[i])
        rs.append(fwd_ret[i])
        ss.append(secids[i])
    return (
        np.asarray(Xs),
        np.asarray(ys),
        np.asarray(ds),
        np.asarray(rs),
        np.asarray(ss),
    )


def make_sequences_multi_ticker(X, y, dates, fwd_ret, secids, seq_len, split_masks):
    train_mask, val_mask, test_mask = split_masks

    def _collect(mask_sec, X_sec, y_sec, d_sec, r_sec, secid):
        Xs, ys, ds, rs, ss = [], [], [], [], []
        for i in range(seq_len, len(X_sec)):
            if not mask_sec[i]:
                continue
            if not mask_sec[i - seq_len : i].all():
                continue
            Xs.append(X_sec[i - seq_len : i])
            ys.append(y_sec[i])
            ds.append(d_sec[i])
            rs.append(r_sec[i])
            ss.append(secid)
        return (
            np.asarray(Xs),
            np.asarray(ys),
            np.asarray(ds),
            np.asarray(rs),
            np.asarray(ss),
        )

    out = [([], [], [], [], []) for _ in range(3)]

    for secid in np.unique(secids):
        m = secids == secid
        X_s, y_s, d_s, r_s = X[m], y[m], dates[m], fwd_ret[m]
        masks = [train_mask[m], val_mask[m], test_mask[m]]

        for idx, mask in enumerate(masks):
            parts = _collect(mask, X_s, y_s, d_s, r_s, str(secid))
            for j, arr in enumerate(parts):
                out[idx][j].append(arr)

    def _cat(parts):
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return np.asarray([])
        return np.concatenate(parts, axis=0)

    results = []
    for split_data in out:
        for lst in split_data:
            results.append(_cat(lst))

    (
        Xs_tr,
        ys_tr,
        ds_tr,
        rs_tr,
        ss_tr,
        Xs_va,
        ys_va,
        ds_va,
        rs_va,
        ss_va,
        Xs_te,
        ys_te,
        ds_te,
        rs_te,
        ss_te,
    ) = results

    print(f"\nSequences: Train={len(Xs_tr)}, Val={len(Xs_va)}, Test={len(Xs_te)}")
    return (
        Xs_tr,
        ys_tr,
        ds_tr,
        rs_tr,
        ss_tr,
        Xs_va,
        ys_va,
        ds_va,
        rs_va,
        ss_va,
        Xs_te,
        ys_te,
        ds_te,
        rs_te,
        ss_te,
    )


def time_split_masks(dates, secids):
    n = len(dates)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    for secid in np.unique(secids):
        sec_mask = secids == secid
        sec_dates = dates[sec_mask]
        sec_indices = np.where(sec_mask)[0]
        if len(sec_dates) == 0:
            continue

        order = np.argsort(sec_dates)
        sorted_indices = sec_indices[order]

        n_sec = len(sorted_indices)
        n_train = int(n_sec * float(CFG["TRAIN_SPLIT"]))
        n_val = int(n_sec * float(CFG["VAL_SPLIT"]))

        if n_train <= 0:
            n_train = 1
        if n_val <= 0:
            n_val = 1
        if n_train + n_val >= n_sec:
            n_train = max(1, n_sec - 2)
            n_val = 1

        train_mask[sorted_indices[:n_train]] = True
        val_mask[sorted_indices[n_train : n_train + n_val]] = True
        test_mask[sorted_indices[n_train + n_val :]] = True

    print("\n=== SPLIT DISTRIBUTION BY TICKER ===")
    for secid in np.unique(secids):
        sec_mask = secids == secid
        n_tr = int((train_mask & sec_mask).sum())
        n_va = int((val_mask & sec_mask).sum())
        n_te = int((test_mask & sec_mask).sum())
        print(f"{str(secid):4s}: Train={n_tr:4d}, Val={n_va:4d}, Test={n_te:4d}")

    return train_mask, val_mask, test_mask

