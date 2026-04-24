from __future__ import annotations

from datetime import datetime
import os
import random
from typing import Any, Dict

import numpy as np
import tensorflow as tf


CFG: Dict[str, Any] = {
    "TICKERS": [
        "SBER", "GAZP", "LKOH", "ROSN", "GMKN", "NVTK", "YNDX", "MTSS",
        "MGNT", "FIVE", "PLZL", "POLY", "ALRS", "CHMF", "NLMK", "MAGN",
        "VTBR", "MOEX", "PHOR", "RUAL", "OZON", "VKCO", "IRAO", "FEES",
        "HYDR", "RTKM", "AFLT", "PIKK", "SMLT", "SGZH", "MTLR", "AFKS",
        "CBOM", "TATN", "SNGS", "BANEP", "TRNFP", "NMTP", "FLOT",
        "TCSG", "FIXP", "ENPG", "LENT", "RASP", "SELG",
        "BSPB", "AQUA", "RNFT", "MSNG", "LSRG", "RENI",
    ],
    "START": "2015-01-01",
    # None = up to today (inclusive). Data loader resolves it to date.today().
    "END": None,

    # Cache (Colab-friendly)
    # NOTE: In Colab, /content is the default working dir.
    # We keep cache under /content/cache to avoid re-downloading market data.
    "CACHE_DIR": "/content/cache",
    "CACHE_ENABLED": True,
    # Multi-horizon training: each horizon becomes a separate "panel" via horizon_norm feature.
    "HORIZONS": [5, 10, 30, 60, 120, 240, 360],
    "THR_MAP": {
        5: 0.03,
        10: 0.04,
        30: 0.05,
        60: 0.08,
        120: 0.12,
        240: 0.18,
        360: 0.25,
    },

    # Backward compatibility (single-horizon consumers may still use these keys).
    "HORIZON": 5,
    "THR_MOVE": 0.03,
    "SEQ_LEN": 30,
    "TRAIN_SPLIT": 0.70,
    "VAL_SPLIT": 0.15,
    # Training speed defaults (CPU-friendly). On Colab GPU you can lower batch and/or increase seeds.
    "BATCH_SIZE": 256,
    "EPOCHS": 50,
    "LR": 3e-4,
    "SEED": 42,
    "N_RUNS": 3,

    # Ensemble is a defensible way to improve stability without "picking the best seed".
    "USE_ENSEMBLE": True,
    # Start with 1 seed to verify training end-to-end; scale up later.
    "ENSEMBLE_SEEDS": [42],

    # Early stopping tuned for fast overfitting regimes.
    "ES_PATIENCE": 5,
    "ES_MIN_DELTA": 0.005,
    "FEE": 0.001,
    "EXTENDED_DIAGNOSTICS": True,
}


# Human-readable fingerprint printed at runtime to detect stale Colab imports.
CODE_FINGERPRINT = (
    "sonnet-v3: ensemble(42..46)+smallTCN+dropout=0.5+ES(p5,delta0.005)+shuffle=True+noCW"
)


def seed_everything(seed: int | None = None) -> None:
    """Backward-compatible seed setter.

    Accepts an explicit seed so each run can use different initialization.
    """
    s = int(CFG["SEED"]) if seed is None else int(seed)
    _set_seed(s)


def _set_seed(seed: int) -> None:
    """Set all relevant RNG seeds for reproducible (but controllable) runs."""
    s = int(seed)
    os.environ["PYTHONHASHSEED"] = str(s)
    random.seed(s)
    np.random.seed(s)
    tf.random.set_seed(s)
