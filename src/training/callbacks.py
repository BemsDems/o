from __future__ import annotations

import json
import shutil
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
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


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def _jsonable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def make_run_dir(cfg: dict) -> Path:
    base_dir = Path(cfg.get("ARTIFACTS_DIR", "artifacts"))
    tag = cfg.get("RUN_TAG")
    if not tag:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{cfg['TICKER']}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_prepared_snapshot(run_dir: Path, prepared: Dict[str, Any], cfg: dict) -> None:
    feat = prepared["feat"]
    info = {
        "ticker": cfg["TICKER"],
        "start": cfg["START"],
        "end": cfg["END"],
        "window": cfg["WINDOW"],
        "horizon": cfg["HORIZON"],
        "thr_move": cfg["THR_MOVE"],
        "dataset_rows": int(len(feat)),
        "n_features": int(len(prepared["FEATURES"])),
        "features": list(prepared["FEATURES"]),
        "train_shape": list(prepared["X_train"].shape),
        "val_shape": list(prepared["X_val"].shape),
        "test_shape": list(prepared["X_test"].shape),
        "class_weight": prepared["class_weight"],
        "target_share_full": float(feat["Target"].mean()),
        "target_share_train": float(prepared["y_train"].mean()),
        "target_share_val": float(prepared["y_val"].mean()),
        "target_share_test": float(prepared["y_test"].mean()),
    }
    save_json(run_dir / "prepared_dataset_info.json", info)
    write_text(run_dir / "selected_features.txt", "\n".join(prepared["FEATURES"]))


def build_chat_report(run_dir: Path, cfg: dict, df: pd.DataFrame, summary: pd.DataFrame) -> str:
    metric_mean = summary.set_index("metric")["mean"].to_dict()

    def _fmt(x):
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)

    lines = [
        "ЗАДАЧА:",
        "оценить результат эксперимента",
        "",
        "ФАЙЛ / ФУНКЦИЯ:",
        "__main__ / run_once()",
        "",
        "ЧТО ИЗМЕНИЛ:",
        "- заполнить вручную",
        "",
        "ЧТО НЕ МЕНЯЛ:",
        f"- TICKER = {cfg['TICKER']}",
        f"- HORIZON = {cfg['HORIZON']}",
        f"- THR_MOVE = {cfg['THR_MOVE']}",
        f"- WINDOW = {cfg['WINDOW']}",
        f"- RUN_SEEDS = {cfg.get('RUN_SEEDS', [42])}",
        "",
        "ЧТО ПОЛУЧИЛ:",
        f"- roc_auc mean = {_fmt(metric_mean.get('roc_auc'))}",
        f"- pr_auc mean = {_fmt(metric_mean.get('pr_auc'))}",
        f"- logloss_gain_vs_baseline mean = {_fmt(metric_mean.get('logloss_gain_vs_baseline'))}",
        f"- prob_psi mean = {_fmt(metric_mean.get('prob_psi'))}",
        f"- alpha_thr_pnl mean = {_fmt(metric_mean.get('alpha_thr_pnl'))}",
        f"- n_trades_thr_pnl mean = {_fmt(metric_mean.get('n_trades_thr_pnl'))}",
        "",
        "ЧТО НУЖНО ОТ ТЕБЯ:",
        "- сказать, это улучшение или шум",
        "- указать, что в правке выглядит слабо",
        "- дать точечную правку",
        "",
        "ФАЙЛЫ:",
        f"- {run_dir / 'run_log.txt'}",
        f"- {run_dir / 'config_snapshot.json'}",
        f"- {run_dir / 'prepared_dataset_info.json'}",
        f"- {run_dir / 'selected_features.txt'}",
        f"- {run_dir / 'multi_seed_summary.csv'}",
        f"- {run_dir / 'multi_seed_aggregated.csv'}",
        "",
        "КРАТКАЯ ТАБЛИЦА ПО SEED:",
        df.to_string(index=False),
    ]
    return "\n".join(lines)


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return False


def download_artifacts_if_needed(run_dir: Path, cfg: dict) -> Optional[Path]:
    if not bool(cfg.get("AUTO_DOWNLOAD_ARTIFACTS", False)):
        return None

    if not is_colab():
        print("AUTO_DOWNLOAD_ARTIFACTS=True, но среда не Colab — скачивание пропущено.")
        return None

    try:
        from google.colab import files
    except Exception:
        print("Не удалось импортировать google.colab.files — скачивание пропущено.")
        return None

    run_dir = Path(run_dir)

    if bool(cfg.get("DOWNLOAD_ARTIFACTS_AS_ZIP", True)):
        zip_base = str(run_dir)
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
        print(f"Downloading ZIP: {zip_path}")
        files.download(zip_path)
        return Path(zip_path)

    # fallback: download for_chat.txt only
    for_chat = run_dir / "for_chat.txt"
    if for_chat.exists():
        print(f"Downloading file: {for_chat}")
        files.download(str(for_chat))
        return for_chat

    print("Нет файлов для скачивания.")
    return None


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
