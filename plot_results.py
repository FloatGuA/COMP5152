"""
Visualize experiment results from results/metrics.csv.
Run after run_experiment.py completes.

Produces:
  results/fig_mape_bar.png       -- mean MAPE per model (bar)
  results/fig_mape_heatmap.png   -- MAPE heatmap (symbol x model)
  results/fig_time_bar.png       -- mean training time per model (bar)
  results/fig_lstm_curves.png    -- LSTM train/val loss curves (one panel per symbol)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")

RESULTS = Path("results")
MODELS  = ["LinearRegression", "ARIMA", "Prophet", "LSTM"]


def load_metrics() -> pd.DataFrame:
    path = RESULTS / "metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run run_experiment.py first")
    df = pd.read_csv(path)
    df["mape"] = pd.to_numeric(df["mape"], errors="coerce")
    df["rmse"] = pd.to_numeric(df["rmse"], errors="coerce")
    df["mae"]  = pd.to_numeric(df["mae"],  errors="coerce")
    return df


# ── 1. Mean MAPE bar chart ──────────────────────────────────────────────────
def plot_mape_bar(df: pd.DataFrame):
    means = df.groupby("model")["mape"].mean().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.index, means.values, color=sns.color_palette("muted", len(MODELS)))
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_title("Mean MAPE per Model (equal weight across symbols)")
    ax.set_ylim(0, means.max() * 1.25)
    plt.tight_layout()
    out = RESULTS / "fig_mape_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. MAPE heatmap ─────────────────────────────────────────────────────────
def plot_mape_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="symbol", columns="model", values="mape")[MODELS]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "MAPE (%)"}
    )
    ax.set_title("MAPE Heatmap (symbol × model)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    out = RESULTS / "fig_mape_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. Training time bar chart ───────────────────────────────────────────────
def plot_time_bar(df: pd.DataFrame):
    means = df.groupby("model")["train_time_sec"].mean().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.index, means.values, color=sns.color_palette("pastel", len(MODELS)))
    ax.bar_label(bars, fmt="%.1fs", padding=3)
    ax.set_ylabel("Mean training time (s)")
    ax.set_title("Mean Training + Inference Time per Model")
    ax.set_ylim(0, means.max() * 1.25)
    plt.tight_layout()
    out = RESULTS / "fig_time_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 4. LSTM training curves ──────────────────────────────────────────────────
def plot_lstm_curves():
    curve_files = sorted(RESULTS.glob("lstm_curve_*.csv"))
    if not curve_files:
        print("No LSTM curve files found, skipping.")
        return

    n = len(curve_files)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), squeeze=False)

    for ax, path in zip(axes[0], curve_files):
        symbol = path.stem.replace("lstm_curve_", "")
        curve  = pd.read_csv(path)
        ax.plot(curve["epoch"], curve["train_loss"], label="train")
        ax.plot(curve["epoch"], curve["val_loss"],   label="val", linestyle="--")
        ax.set_title(f"LSTM — {symbol}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8)

    plt.suptitle("LSTM Training Curves", y=1.02)
    plt.tight_layout()
    out = RESULTS / "fig_lstm_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── 5. Per-symbol MAPE grouped bar ──────────────────────────────────────────
def plot_mape_grouped(df: pd.DataFrame):
    pivot  = df.pivot_table(index="symbol", columns="model", values="mape")[MODELS]
    colors = sns.color_palette("muted", len(MODELS))
    x      = range(len(pivot))
    width  = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (model, color) in enumerate(zip(MODELS, colors)):
        offset = (j - 1.5) * width
        vals   = pivot[model].values
        bars   = ax.bar([i + offset for i in x], vals, width, label=model, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE per Symbol × Model")
    ax.legend()
    plt.tight_layout()
    out = RESULTS / "fig_mape_grouped.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    df = load_metrics()
    print(df.to_string(index=False))
    plot_mape_bar(df)
    plot_mape_heatmap(df)
    plot_time_bar(df)
    plot_lstm_curves()
    plot_mape_grouped(df)
    print("\nAll figures saved to results/")
