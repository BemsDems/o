from __future__ import annotations

from datetime import datetime
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
    "FEE": 0.001,
    "EXTENDED_DIAGNOSTICS": True,
}


def seed_everything(seed: int | None = None) -> None:
    s = int(CFG["SEED"]) if seed is None else int(seed)
    np.random.seed(s)
    tf.random.set_seed(s)
