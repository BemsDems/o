from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int):
    """Best-effort reproducibility across Python/NumPy/TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


class ShortMetrics(tf.keras.callbacks.Callback):
    def __init__(self, every: int = 5):
        super().__init__()
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep = epoch + 1
        if ep == 1 or ep % self.every == 0:
            print(
                f"ep={ep:03d} "
                f"loss={logs.get('loss', np.nan):.4f} "
                f"val_loss={logs.get('val_loss', np.nan):.4f} "
                f"val_auc_pr={logs.get('val_auc_pr', np.nan):.4f} "
                f"val_auc_roc={logs.get('val_auc_roc', np.nan):.4f}"
            )

