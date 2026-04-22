from __future__ import annotations

from datetime import datetime
import os
import random
from typing import Any, Dict

import numpy as np
import tensorflow as tf


CFG: Dict[str, Any] = {
    "TICKERS": ["SBER", "GAZP", "LKOH", "YNDX"],
    "START": "2015-01-01",
    "END": datetime.now().strftime("%Y-%m-%d"),
    "HORIZON": 5,
    "THR_MOVE": 0.03,
    "SEQ_LEN": 30,
    "TRAIN_SPLIT": 0.70,
    "VAL_SPLIT": 0.15,
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "LR": 3e-4,
    "SEED": 42,
    "N_RUNS": 3,

    # Ensemble is a defensible way to improve stability without "picking the best seed".
    "USE_ENSEMBLE": True,
    "ENSEMBLE_SEEDS": [42, 43, 44, 45, 46],

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
