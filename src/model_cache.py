"""
Simple file-based model cache.

Cache key  : symbol + model_name
Validity   : train_end date + val_end date must match
Storage    : models/{symbol}_{model_name}.pkl  (joblib)
             models/{symbol}_{model_name}_lstm.pt  (torch state_dict + scaler)
             models/{symbol}_{model_name}_meta.json
"""

import json
import joblib
import torch
from pathlib import Path

CACHE_DIR = Path("models")


def _meta_path(symbol: str, model_name: str) -> Path:
    return CACHE_DIR / f"{symbol}_{model_name}_meta.json"


def _pkl_path(symbol: str, model_name: str) -> Path:
    return CACHE_DIR / f"{symbol}_{model_name}.pkl"


def _pt_path(symbol: str, model_name: str) -> Path:
    return CACHE_DIR / f"{symbol}_{model_name}.pt"


def is_valid(symbol: str, model_name: str, train_end: str, val_end: str) -> bool:
    meta_p = _meta_path(symbol, model_name)
    if not meta_p.exists():
        return False
    meta = json.loads(meta_p.read_text())
    return meta.get("train_end") == train_end and meta.get("val_end") == val_end


def _write_meta(symbol: str, model_name: str, train_end: str, val_end: str):
    CACHE_DIR.mkdir(exist_ok=True)
    _meta_path(symbol, model_name).write_text(
        json.dumps({"train_end": train_end, "val_end": val_end})
    )


# ── joblib-based (LR, ARIMA, Prophet) ───────────────────────────────────────

def save_pkl(symbol: str, model_name: str, obj, train_end: str, val_end: str):
    CACHE_DIR.mkdir(exist_ok=True)
    joblib.dump(obj, _pkl_path(symbol, model_name))
    _write_meta(symbol, model_name, train_end, val_end)


def load_pkl(symbol: str, model_name: str):
    return joblib.load(_pkl_path(symbol, model_name))


# ── torch-based (LSTM) ───────────────────────────────────────────────────────

def save_lstm(symbol: str, state_dict: dict, scaler, train_end: str, val_end: str):
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": state_dict, "scaler": scaler}, _pt_path(symbol, "LSTM"))
    _write_meta(symbol, "LSTM", train_end, val_end)


def load_lstm(symbol: str) -> tuple:
    """Returns (state_dict, scaler)."""
    data = torch.load(_pt_path(symbol, "LSTM"), weights_only=False)
    return data["state_dict"], data["scaler"]
