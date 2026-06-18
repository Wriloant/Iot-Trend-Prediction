"""
run_pipeline.py - one command runs the whole Assessment 2 pipeline end to end:

    raw CSV  ->  clean  ->  features  ->  chronological split  ->  train 6 models
             ->  evaluate on held-out test  ->  save metrics.json + all figures

Usage:
    python run_pipeline.py
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from src import config as C
from src import data_cleaning as dc
from src import features as fe
from src import models as M
from src import plots as P
from src.evaluate import regression_metrics, skill_vs_baseline

warnings.filterwarnings("ignore")


def main():
    t0 = time.time()
    print("=" * 70)
    print("IoT SENSOR TREND PREDICTION  |  target:", C.TARGET, f"(+{C.HORIZON}h)")
    print("=" * 70)

    # ---- Phase 1-2: load + clean ------------------------------------------
    raw = dc.load_raw()
    cleaned = dc.clean(raw)
    cleaned.to_csv(C.DATA_PROCESSED)
    raw_na = (raw.replace(C.MISSING_SENTINEL, np.nan).isna().mean() * 100)
    print(f"[clean] rows={len(cleaned)}  residual-NaN(target)="
          f"{cleaned[C.TARGET].isna().mean()*100:.1f}%")

    # ---- Phase 3: features + split ----------------------------------------
    feat = fe.build_features(cleaned)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = fe.time_split(feat)
    feat_names = list(Xtr.columns)
    print(f"[features] {len(feat_names)} features | "
          f"train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    # Full target series (for naive baselines, indexed by feature-row time t).
    target_series = cleaned[C.TARGET]

    results, preds_test = {}, {}

    # ---- Phase 4: baselines -----------------------------------------------
    yhat_persist = M.persistence_forecast(yte, target_series)
    results["Persistence"] = regression_metrics(yte, yhat_persist)
    preds_test["Persistence"] = yhat_persist

    yhat_seasonal = M.seasonal_naive_forecast(yte, target_series)
    results["SeasonalNaive"] = regression_metrics(yte, yhat_seasonal)

    rmse_persist = results["Persistence"]["RMSE"]

    # ---- Ridge ------------------------------------------------------------
    ridge = M.fit_ridge(Xtr, ytr)
    yhat_ridge = ridge.predict(Xte)
    results["Ridge"] = regression_metrics(yte, yhat_ridge)

    # ---- LightGBM ---------------------------------------------------------
    lgbm = M.fit_lightgbm(Xtr, ytr, Xva, yva)
    yhat_lgbm = lgbm.predict(Xte)
    results["LightGBM"] = regression_metrics(yte, yhat_lgbm)
    preds_test["LightGBM"] = yhat_lgbm
    print(f"[lightgbm] best_iteration={getattr(lgbm,'best_iteration_',None)}")

    # ---- XGBoost ----------------------------------------------------------
    xgbm = M.fit_xgboost(Xtr, ytr, Xva, yva)
    yhat_xgb = xgbm.predict(Xte)
    results["XGBoost"] = regression_metrics(yte, yhat_xgb)

    # ---- LSTM -------------------------------------------------------------
    try:
        bundle = M.fit_lstm(Xtr, ytr, Xva, yva)
        yhat_lstm = M.lstm_predict(bundle, Xte)
        results["LSTM"] = regression_metrics(yte, yhat_lstm)
        preds_test["LSTM"] = yhat_lstm
    except Exception as e:  # keep the pipeline robust if torch is unavailable
        print("[lstm] skipped:", repr(e))

    # ---- skill scores -----------------------------------------------------
    for name, m in results.items():
        m["skill_vs_persistence_%"] = round(skill_vs_baseline(m["RMSE"], rmse_persist), 2)

    # ---- Phase 5: figures -------------------------------------------------
    P.plot_target_series(cleaned)
    P.plot_missingness(raw_na)
    P.plot_correlation(cleaned)
    spike_start = cleaned.index[len(cleaned) // 3]
    P.plot_cleaning_window(raw, cleaned, spike_start)
    best_name = min(
        [k for k in results if k not in ("Persistence", "SeasonalNaive")],
        key=lambda k: results[k]["RMSE"],
    )
    comp = {k: preds_test[k] for k in ("Persistence", "LightGBM", "LSTM") if k in preds_test}
    P.plot_predictions(Xte.index, yte.to_numpy(), comp)
    P.plot_residuals(yte.to_numpy(), yhat_lgbm, "LightGBM")
    P.plot_model_comparison(results)
    P.plot_feature_importance(lgbm, feat_names)

    # ---- persist results --------------------------------------------------
    summary = {
        "target": C.TARGET,
        "horizon_hours": C.HORIZON,
        "n_features": len(feat_names),
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "best_model": best_name,
        "results": results,
    }
    with open(C.RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- console table ----------------------------------------------------
    print("\n" + "-" * 70)
    print(f"{'model':<16}{'RMSE':>9}{'MAE':>9}{'R2':>8}{'skill% vs persist':>20}")
    print("-" * 70)
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["RMSE"]):
        print(f"{name:<16}{m['RMSE']:>9.3f}{m['MAE']:>9.3f}{m['R2']:>8.3f}"
              f"{m['skill_vs_persistence_%']:>19.1f}%")
    print("-" * 70)
    print(f"BEST: {best_name}   |   runtime {time.time()-t0:.1f}s")
    print("figures -> reports/figures/   metrics -> results/metrics.json")


if __name__ == "__main__":
    main()
