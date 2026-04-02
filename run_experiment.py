"""
Run all 4 models on selected stocks/ETFs.
Supports resume: if results/preds/{symbol}_{model}.npz already exists,
skips training and loads predictions from disk.
Writes metrics incrementally (after each symbol) so progress survives crashes.
"""

import time
import csv
import traceback
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.seed import set_seed
set_seed()

from src.data_loader import SELECTED, load_close, train_val_test_split
from src import linear_model, arima_model, prophet_model, lstm_model
from src.evaluator import compute_all

RESULTS_DIR = Path("results")
PREDS_DIR   = RESULTS_DIR / "preds"
RESULTS_DIR.mkdir(exist_ok=True)
PREDS_DIR.mkdir(exist_ok=True)

MODELS = {
    "LinearRegression": linear_model,
    "ARIMA":            arima_model,
    "Prophet":          prophet_model,
    "LSTM":             lstm_model,
}

OUTPUT_FILE = RESULTS_DIR / "metrics.csv"
FIELDNAMES  = ["category", "symbol", "model",
               "mape", "rmse", "mae", "da", "train_time_sec", "test_rows"]


def _load_existing_train_times() -> dict:
    """Load train_time_sec from existing metrics.csv to preserve timing for resumed tasks."""
    times = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, newline="") as f:
            for row in csv.DictReader(f):
                times[(row["symbol"], row["model"])] = row["train_time_sec"]
    return times


def run_all():
    existing_times = _load_existing_train_times()
    rows = []

    # Count tasks that still need training for accurate progress bar
    pending = sum(
        1 for cat, sym in SELECTED for mname in MODELS
        if not (PREDS_DIR / f"{sym}_{mname}.npz").exists()
    )
    resumed = len(SELECTED) * len(MODELS) - pending
    if resumed:
        print(f"Resume mode: {resumed} task(s) already have saved predictions — loading from disk.")

    pbar = tqdm(total=len(SELECTED) * len(MODELS), ncols=80, desc="Experiment", unit="task")

    # Write header once (overwrite file so we start clean)
    with open(OUTPUT_FILE, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    for category, symbol in SELECTED:
        series = load_close(category, symbol)
        train, val, test = train_val_test_split(series)

        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"  Asset : [{category}] {symbol}")
        tqdm.write(f"  Rows  : total={len(series)}  train={len(train)}  val={len(val)}  test={len(test)}")

        symbol_rows = []

        for model_name, module in MODELS.items():
            pbar.set_description(f"{symbol} / {model_name}")
            npz_path = PREDS_DIR / f"{symbol}_{model_name}.npz"

            if npz_path.exists():
                # --- RESUME: load saved predictions ---
                tqdm.write(f"\n  > {model_name}  [resumed from disk]")
                data    = np.load(npz_path)
                preds   = data["preds"]
                actuals = data["actuals"]
                elapsed = float(existing_times.get((symbol, model_name), 0))
                metrics = compute_all(actuals, preds)
                tqdm.write(
                    f"    Loaded: MAPE={metrics['mape']:.2f}%  DA={metrics['da']:.1f}%"
                )
                symbol_rows.append({
                    "category":       category,
                    "symbol":         symbol,
                    "model":          model_name,
                    "mape":           round(metrics["mape"], 4),
                    "rmse":           round(metrics["rmse"], 4),
                    "mae":            round(metrics["mae"], 4),
                    "da":             round(metrics["da"], 2),
                    "train_time_sec": round(elapsed, 2),
                    "test_rows":      len(actuals),
                })
            else:
                # --- RUN: train and predict ---
                tqdm.write(f"\n  > {model_name}")
                tqdm.write(f"    Stage: fitting + predicting ...")
                t0 = time.time()
                try:
                    preds, actuals = module.predict(train, val, test)
                    elapsed = time.time() - t0

                    tqdm.write(f"    Stage: evaluating ...")
                    metrics = compute_all(actuals, preds)
                    tqdm.write(
                        f"    Done  : {elapsed:.1f}s  |  "
                        f"MAPE={metrics['mape']:.2f}%  "
                        f"RMSE={metrics['rmse']:.4f}  "
                        f"DA={metrics['da']:.1f}%"
                    )
                    np.savez(npz_path, preds=preds, actuals=actuals)
                    symbol_rows.append({
                        "category":       category,
                        "symbol":         symbol,
                        "model":          model_name,
                        "mape":           round(metrics["mape"], 4),
                        "rmse":           round(metrics["rmse"], 4),
                        "mae":            round(metrics["mae"], 4),
                        "da":             round(metrics["da"], 2),
                        "train_time_sec": round(elapsed, 2),
                        "test_rows":      len(actuals),
                    })
                except Exception:
                    elapsed = time.time() - t0
                    tqdm.write(f"    FAILED ({elapsed:.1f}s)")
                    traceback.print_exc()
                    symbol_rows.append({
                        "category": category, "symbol": symbol, "model": model_name,
                        "mape": None, "rmse": None, "mae": None, "da": None,
                        "train_time_sec": round(elapsed, 2), "test_rows": 0,
                    })

            pbar.update(1)

        # Flush this symbol's rows to CSV immediately
        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerows(symbol_rows)
        rows.extend(symbol_rows)

    pbar.close()
    print(f"\nResults saved to {OUTPUT_FILE}")

    print("\n=== Aggregate MAPE (1/n mean across symbols) ===")
    for model_name in MODELS:
        scores = [r["mape"] for r in rows if r["model"] == model_name and r["mape"] is not None]
        if scores:
            print(f"  {model_name:<20} {sum(scores)/len(scores):.2f}%  (n={len(scores)})")
        else:
            print(f"  {model_name:<20} N/A (all failed)")

    print("\n=== Mean training time per model ===")
    for model_name in MODELS:
        times = [r["train_time_sec"] for r in rows if r["model"] == model_name and r["train_time_sec"]]
        if times:
            print(f"  {model_name:<20} {sum(times)/len(times):.1f}s avg")


if __name__ == "__main__":
    run_all()
