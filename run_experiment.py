"""
Run all 4 models on 6 selected stocks/ETFs.
Records MAPE, RMSE, MAE and training time per (symbol, model).
Outputs results/metrics.csv and prints aggregate MAPE per model.
"""

import time
import csv
import traceback
from pathlib import Path

from tqdm import tqdm

from src.data_loader import SELECTED, load_close, train_val_test_split
from src import linear_model, arima_model, prophet_model, lstm_model
from src.evaluator import compute_all

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "LinearRegression": linear_model,
    "ARIMA":            arima_model,
    "Prophet":          prophet_model,
    "LSTM":             lstm_model,
}

OUTPUT_FILE = RESULTS_DIR / "metrics.csv"
FIELDNAMES = ["category", "symbol", "model", "mape", "rmse", "mae", "train_time_sec", "test_rows"]

TOTAL = len(SELECTED) * len(MODELS)


def run_all():
    rows = []

    pbar = tqdm(total=TOTAL, ncols=80, desc="Experiment", unit="task")

    for category, symbol in SELECTED:
        series = load_close(category, symbol)
        train, val, test = train_val_test_split(series)

        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"  Asset : [{category}] {symbol}")
        tqdm.write(f"  Rows  : total={len(series)}  train={len(train)}  val={len(val)}  test={len(test)}")

        for model_name, module in MODELS.items():
            pbar.set_description(f"{symbol} / {model_name}")
            tqdm.write(f"\n  ▶ {model_name}")
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
                    f"MAE={metrics['mae']:.4f}"
                )
                rows.append({
                    "category":       category,
                    "symbol":         symbol,
                    "model":          model_name,
                    "mape":           round(metrics["mape"], 4),
                    "rmse":           round(metrics["rmse"], 4),
                    "mae":            round(metrics["mae"], 4),
                    "train_time_sec": round(elapsed, 2),
                    "test_rows":      len(actuals),
                })
            except Exception:
                elapsed = time.time() - t0
                tqdm.write(f"    FAILED ({elapsed:.1f}s)")
                traceback.print_exc()
                rows.append({
                    "category": category, "symbol": symbol, "model": model_name,
                    "mape": None, "rmse": None, "mae": None,
                    "train_time_sec": round(elapsed, 2), "test_rows": 0,
                })

            pbar.update(1)

    pbar.close()

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUTPUT_FILE}")

    # Aggregate: mean MAPE per model
    print("\n=== Aggregate MAPE (1/n mean across symbols) ===")
    for model_name in MODELS:
        scores = [r["mape"] for r in rows if r["model"] == model_name and r["mape"] is not None]
        if scores:
            print(f"  {model_name:<20} {sum(scores)/len(scores):.2f}%  (n={len(scores)})")
        else:
            print(f"  {model_name:<20} N/A (all failed)")

    print("\n=== Mean training time per model ===")
    for model_name in MODELS:
        times = [r["train_time_sec"] for r in rows if r["model"] == model_name]
        print(f"  {model_name:<20} {sum(times)/len(times):.1f}s avg")


if __name__ == "__main__":
    run_all()
