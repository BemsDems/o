from __future__ import annotations

"""Central settings for the MOEX TCN baseline.

Keep this file small and stable: it defines experiment knobs and the baseline feature list.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional


CFG: Dict[str, Any] = {
    "TICKER": "GAZP",
    "START": "2015-01-01",
    "END": "2025-12-31",

    "HORIZON": 5,        # target horizon in trading days
    "THR_MOVE": 0.015,   # target: future_return > 1.5%

    "WINDOW": 30,

    # time split (by dates order)
    "TRAIN_FRAC": 0.70,
    "VAL_FRAC": 0.15,  # rest goes to test

    # training
    "EPOCHS": 200,
    "BATCH": 32,
    "LR": 1e-3,
    "PATIENCE": 15,

    # backtest assumptions (simple)
    "FEE": 0.001,        # 0.1% per trade (simplified)
    "NON_OVERLAP": True, # skip next HORIZON days after entry

    # Multi-seed evaluation
    "RUN_SEEDS": [11, 21, 31, 41, 51],

    # Feature clipping (robustness against outliers)
    "CLIP_Q": 0.005,

    # Current practical baseline: no dividends
    "USE_DIVIDENDS": False,

    # Save model/scaler only for single-seed runs
    "SAVE_SINGLE_RUN_ARTIFACTS": False,
}


# Baseline features (single source of truth)
BASE_FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "log_ret",
    "dist_sma20", "dist_sma50", "trend_up_200", "rsi_14",
    "vol_rel", "bb_width", "bb_pos", "vol_ratio_5_20", "vol_spike",
    "imoex_ret_1", "imoex_ret_5", "imoex_ret_20",
    "stock_vs_imoex_5",
    "key_rate_chg", "rate_rising",
]


# Run flags (to reduce output)
SHOW_MODEL_SUMMARY = False
SHOW_TRAIN_VAL_DIAG = False
FIT_VERBOSE = 0


@dataclass
class FundamentalConfig:
    edisclosure_token: Optional[str] = os.getenv("EDISCLOSURE_TOKEN")
    edisclosure_base: str = "https://gateway.e-disclosure.ru/api"
    request_timeout: int = 30
    sleep_sec: float = 0.4
    user_agent: str = "Mozilla/5.0 (compatible; DiplomaResearchBot/1.0)"


CFG_FUND = FundamentalConfig()

