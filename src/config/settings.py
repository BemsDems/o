from __future__ import annotations

"""Central settings for the MOEX TCN baseline.

Keep this file small and stable: it defines experiment knobs and the baseline feature list.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional


CFG: Dict[str, Any] = {
    "TICKER": "GAZP",
    # Multi-ticker (panel) mode: used by moex_tcn_chatgpt_panel.py.
    # Keep TICKER as the primary ticker for run naming / artifacts.
    "TICKERS": ["GAZP", "SBER", "LKOH", "ROSN"],
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

    # Probability calibration (post-processing)
    "USE_PLATT_CALIBRATION": True,
    "CALIB_VAL_TAIL_FRAC": 0.50,
    "CALIB_VAL_MIN_SAMPLES": 120,

    # Feature clipping (robustness against outliers)
    "CLIP_Q": 0.005,

    # Current practical baseline: no dividends
    "USE_DIVIDENDS": False,

    # Fundamentals (optional; fetched from Smart-Lab and attached past-only)
    "USE_FUNDAMENTALS": True,
    "FUND_LAG_DAYS": 1,

    # Save model/scaler only for single-seed runs
    "SAVE_SINGLE_RUN_ARTIFACTS": False,

    # Artifacts / experiment packaging
    "SAVE_RUN_ARTIFACTS": True,
    "SAVE_PER_SEED_TABLES": True,
    "ARTIFACTS_DIR": "/content/o/artifacts",
    "RUN_TAG": None,  # None => timestamp

    "AUTO_DOWNLOAD_ARTIFACTS": True,
    "DOWNLOAD_ARTIFACTS_AS_ZIP": True,
}


# Baseline features (single source of truth)
BASE_FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20", "log_ret",
    "dist_sma20", "dist_sma50", "trend_up_200", "rsi_14",
    "vol_rel", "atr_rel", "bb_width", "bb_pos",
    "vol_ratio_5_20", "vol_spike",
    "usd_ret_1", "usd_ret_5",
    "imoex_ret_1", "imoex_ret_5", "imoex_ret_20",
    "stock_vs_imoex_5",
    "key_rate_chg", "rate_rising",
]


# Fundamental features (single source of truth)
FUND_FEATURES = [
    "roe",
    "pb_ratio",
    "value_quality",
    "eps",
    "fund_age_days",

    "roe_is_missing",
    "pb_ratio_is_missing",
    "value_quality_is_missing",
    "eps_is_missing",
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
