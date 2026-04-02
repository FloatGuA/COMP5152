"""
Re-run only LSTM and patch the existing metrics.csv.
Run after clearing LSTM cache when other model results are still valid.
"""

import time
import csv
import traceback
from pathlib import Path

from src.data_loader import SELECTED, load_close, train_val_test_split
from src import lstm_model
from src.evaluator import compute_all

RESULTS_DIR = Path("results")
PREDS_DIR   = RESULTS_DIR / "preds"
OUTPUT_FILE = RESULTS_DIR / "metrics.csv"
FIELDNAMES  = ["category", "symbol", "model",
               "mape", "rmse", "mae", "da", "train_time_sec", "test_rows"]


def run_lstm_only():
    # Load existing rows (non-LSTM)
    existing = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, newline="") as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if r["model"] != "LSTM"]

    lstm_rows = []
    for category, symbol in SELECTED:
        series = load_close(category, symbol)
        train, val, test = train_val_test_split(series)

        print(f"\n{'='*60}")
        print(f"  [{category}] {symbol}  train={len(train)} val={len(val)} test={len(test)}")

        t0 = time.time()
        try:
            preds, actuals = lstm_model.predict(train, val, test)
            elapsed = time.time() - t0
            metrics = compute_all(actuals, preds)
            print(f"  LSTM  {elapsed:.1f}s  MAPE={metrics['mape']:.2f}%  DA={metrics['da']:.1f}%")
            import numpy as np
            PREDS_DIR.mkdir(exist_ok=True)
            np.savez(PREDS_DIR / f"{symbol}_LSTM.npz", preds=preds, actuals=actuals)
            lstm_rows.append({
                "category":       category,
                "symbol":         symbol,
                "model":          "LSTM",
                "mape":           round(metrics["mape"], 4),
                "rmse":           round(metrics["rmse"], 4),
                "mae":            round(metrics["mae"], 4),
                "da":             round(metrics["da"], 2),
                "train_time_sec": round(elapsed, 2),
                "test_rows":      len(actuals),
            })
        except Exception:
            elapsed = time.time() - t0
            print(f"  LSTM FAILED ({elapsed:.1f}s)")
            traceback.print_exc()
            lstm_rows.append({
                "category": category, "symbol": symbol, "model": "LSTM",
                "mape": None, "rmse": None, "mae": None, "da": None,
                "train_time_sec": round(elapsed, 2), "test_rows": 0,
            })

    # Merge: preserve original order (LR/ARIMA/Prophet/LSTM per symbol)
    all_rows = existing + lstm_rows
    order = {(r["category"], r["symbol"], r["model"]): r for r in all_rows}
    from src.data_loader import SELECTED as SEL
    models_order = ["LinearRegression", "ARIMA", "Prophet", "LSTM"]
    final = []
    for cat, sym in SEL:
        for m in models_order:
            key = (cat, sym, m)
            if key in order:
                final.append(order[key])

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(final)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print("\n=== LSTM MAPE per symbol ===")
    for r in lstm_rows:
        print(f"  {r['symbol']:<8} {r['mape']}%  ({r['train_time_sec']}s)")
    scores = [r["mape"] for r in lstm_rows if r["mape"] is not None]
    print(f"  Mean: {sum(scores)/len(scores):.2f}%")


if __name__ == "__main__":
    run_lstm_only()
