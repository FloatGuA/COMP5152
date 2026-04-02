"""
Visualize experiment results from results/metrics.csv.
Run after run_experiment.py completes.

Produces (all saved to results/):
  fig_mape_bar.png          -- mean MAPE per model (full group)
  fig_mape_heatmap.png      -- MAPE heatmap: symbol x model (full group)
  fig_mape_grouped.png      -- per-symbol grouped bar (full group)
  fig_time_bar.png          -- mean training time per model (full group)
  fig_lstm_curves.png       -- LSTM train/val loss curves
"""

from pathlib import Path
import pandas as pd
import numpy as np
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
    for col in ("mape", "rmse", "mae"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Add min-max normalised composite score column."""
    out = df.copy()
    for metric in ("mape", "rmse", "mae"):
        mn, mx = out[metric].min(), out[metric].max()
        out[f"norm_{metric}"] = (out[metric] - mn) / (mx - mn) if mx > mn else 0.0
    out["composite"] = (
        0.5 * out["norm_mape"] +
        0.3 * out["norm_rmse"] +
        0.2 * out["norm_mae"]
    )
    return out


# ── helpers ─────────────────────────────────────────────────────────────────

def _save(fig, name):
    out = RESULTS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _full(df):
    return df[df["data_group"] == "full"] if "data_group" in df.columns else df


# ── 1. Mean MAPE bar (full group) ────────────────────────────────────────────

def plot_mape_bar(df: pd.DataFrame):
    means = _full(df).groupby("model")["mape"].mean().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.index, means.values, color=sns.color_palette("muted", len(MODELS)))
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_ylabel("Mean MAPE (%)")
    ax.set_title("Mean MAPE per Model — full dataset")
    ax.set_ylim(0, means.max() * 1.25)
    plt.tight_layout()
    _save(fig, "fig_mape_bar.png")


# ── 2. MAPE heatmap (full group) ─────────────────────────────────────────────

def plot_mape_heatmap(df: pd.DataFrame):
    pivot = _full(df).pivot_table(index="symbol", columns="model", values="mape")[MODELS]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "MAPE (%)"})
    ax.set_title("MAPE Heatmap (symbol × model) — full dataset")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, "fig_mape_heatmap.png")


# ── 3. Per-symbol grouped bar (full group) ───────────────────────────────────

def plot_mape_grouped(df: pd.DataFrame):
    pivot  = _full(df).pivot_table(index="symbol", columns="model", values="mape")[MODELS]
    colors = sns.color_palette("muted", len(MODELS))
    x, width = range(len(pivot)), 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (model, color) in enumerate(zip(MODELS, colors)):
        offset = (j - 1.5) * width
        vals   = pivot[model].values
        bars   = ax.bar([i + offset for i in x], vals, width, label=model, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE per Symbol × Model — full dataset")
    ax.legend()
    plt.tight_layout()
    _save(fig, "fig_mape_grouped.png")


# ── 4. Training time bar (full group) ────────────────────────────────────────

def plot_time_bar(df: pd.DataFrame):
    means = _full(df).groupby("model")["train_time_sec"].mean().reindex(MODELS)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.index, means.values, color=sns.color_palette("pastel", len(MODELS)))
    ax.bar_label(bars, fmt="%.1fs", padding=3)
    ax.set_ylabel("Mean time (s)")
    ax.set_title("Mean Training + Inference Time per Model — full dataset")
    ax.set_ylim(0, means.max() * 1.25)
    plt.tight_layout()
    _save(fig, "fig_time_bar.png")


# ── 5. LSTM training curves ──────────────────────────────────────────────────

def plot_lstm_curves():
    curve_files = sorted(RESULTS.glob("lstm_curve_*.csv"))
    if not curve_files:
        print("No LSTM curve files found, skipping.")
        return

    n   = len(curve_files)
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
    _save(fig, "fig_lstm_curves.png")


# ── 6. Composite score heatmap (model × symbol) ─────────────────────────────

def plot_composite_heatmap(df: pd.DataFrame):
    df_c  = add_composite(df.dropna(subset=["mape", "rmse", "mae"]))
    pivot = df_c.pivot_table(index="model", columns="symbol", values="composite").reindex(MODELS)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Composite Score (lower = better)"})
    ax.set_title("Composite Score: 0.5×MAPE + 0.3×RMSE + 0.2×MAE (min-max norm)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, "fig_composite_heatmap.png")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_metrics()
    print(df.to_string(index=False))

    plot_mape_bar(df)
    plot_mape_heatmap(df)
    plot_mape_grouped(df)
    plot_time_bar(df)
    plot_lstm_curves()
    plot_composite_heatmap(df)

    print("\nAll figures saved to results/")
