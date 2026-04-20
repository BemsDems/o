from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
import tensorflow as tf


CFG: Dict[str, Any] = {
    "TICKERS": [
        "SBER", "GAZP", "LKOH", "YNDX",
        "GMKN", "NVTK", "ROSN", "TATN",
        "MTSS", "MGNT", "ALRS", "PLZL",
        "CHMF", "NLMK", "VTBR",
    ],
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
    "FEE": 0.001,
    "EXTENDED_DIAGNOSTICS": True,
}


def seed_everything() -> None:
    np.random.seed(int(CFG["SEED"]))
    tf.random.set_seed(int(CFG["SEED"]))

