"""
Phase 5 - Evaluation metrics.

We report RMSE and MAE (same units as benzene, micro-g/m^3), R^2, and -- most
importantly -- the *skill score*: the percentage RMSE improvement over the
persistence baseline. A model that cannot beat persistence has negative skill
and is not worth deploying, however good its raw RMSE looks.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "n": int(mask.sum())}


def skill_vs_baseline(rmse_model: float, rmse_baseline: float) -> float:
    """Percentage RMSE reduction relative to a baseline (higher is better)."""
    return float((1 - rmse_model / rmse_baseline) * 100)
