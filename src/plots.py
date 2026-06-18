"""
Plotting helpers. Every figure is saved to reports/figures/ at 130 dpi so the
GitHub repo renders them inline and the demo video can walk through them.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True,
})


def save(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_target_series(df):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(df.index, df[C.TARGET], lw=0.5, color="#1f6feb")
    ax.set_title(f"Benzene concentration {C.TARGET} - full hourly series")
    ax.set_ylabel("micro-g/m^3")
    return save(fig, "01_target_series.png")


def plot_missingness(raw_na):
    fig, ax = plt.subplots(figsize=(7, 4))
    raw_na.sort_values().plot.barh(ax=ax, color="#d29922")
    ax.set_title("Missing readings per channel (raw, -200 sentinel)")
    ax.set_xlabel("% missing")
    return save(fig, "02_missingness.png")


def plot_correlation(df):
    cols = [c for c in C.SENSOR_COLS + [C.TARGET] if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Channel correlation")
    return save(fig, "03_correlation.png")


def plot_cleaning_window(raw, cleaned, start, hours=240):
    """Before/after view of one window, showing dropouts + spikes repaired."""
    import numpy as np
    end = start + pd.Timedelta(hours=hours)
    r = raw[C.TARGET].replace(C.MISSING_SENTINEL, np.nan).loc[start:end]
    c = cleaned[C.TARGET].loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(c.index, c.values, color="#2da44e", lw=1.0, label="cleaned")
    ax.plot(r.index, r.values, color="#cf222e", lw=0.0, marker=".", ms=3, label="raw (observed)")
    ax.set_title("Cleaning: capped interpolation + Hampel spike removal (sample window)")
    ax.set_ylabel(C.TARGET); ax.legend(loc="upper right")
    return save(fig, "04_cleaning_window.png")


def plot_predictions(idx, y_true, preds: dict, name="05_predictions.png", last=400):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(idx[-last:], np.asarray(y_true)[-last:], color="black", lw=1.4, label="ground truth")
    colors = ["#1f6feb", "#cf222e", "#8250df"]
    for (label, yp), col in zip(preds.items(), colors):
        ax.plot(idx[-last:], np.asarray(yp)[-last:], lw=1.0, alpha=0.85, color=col, label=label)
    ax.set_title("Test set: predicted vs ground-truth benzene (last window)")
    ax.set_ylabel("micro-g/m^3"); ax.legend(loc="upper right", ncol=2, fontsize=8)
    return save(fig, name)


def plot_residuals(y_true, y_pred, model_name):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].scatter(y_pred, resid, s=5, alpha=0.4, color="#1f6feb")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("residual")
    axes[0].set_title(f"Residuals vs predicted - {model_name}")
    axes[1].hist(resid, bins=40, color="#8250df", alpha=0.8)
    axes[1].set_title("Residual distribution"); axes[1].set_xlabel("residual")
    return save(fig, "06_residuals.png")


def plot_model_comparison(results: dict):
    names = list(results.keys())
    rmses = [results[n]["RMSE"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, rmses, color="#1f6feb")
    bars[int(np.argmin(rmses))].set_color("#2da44e")
    ax.set_ylabel("Test RMSE (micro-g/m^3)")
    ax.set_title("Model comparison (lower is better; green = best)")
    ax.set_xticklabels(names, rotation=20, ha="right")
    for b, v in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    return save(fig, "07_model_comparison.png")


def plot_feature_importance(model, feature_names, top=20):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return None
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in order][::-1], imp[order][::-1], color="#2da44e")
    ax.set_title(f"Top {top} feature importances")
    return save(fig, "08_feature_importance.png")
