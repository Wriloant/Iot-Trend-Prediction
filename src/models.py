"""
Phase 4 - Models.

Five families, chosen to show range and to make the metrics interpretable:

  * Persistence baseline     y_hat(t+1) = y(t)
  * Seasonal-naive baseline  y_hat(t+1) = y(t+1-24)        (same hour, prev day)
        -> These two exist so that RMSE/MAE have *meaning*. Any learned model
           must beat persistence to justify its existence.
  * Ridge regression         a transparent linear trend-regression reference.
  * LightGBM / XGBoost       gradient-boosted trees on the engineered features,
                             with early stopping on a time-ordered validation
                             fold + regularisation to fight overfitting.
  * LSTM (PyTorch)           a sequential neural net fed 24 h windows, with
                             dropout + early stopping.

Overfitting defence is built into the training calls (early stopping, depth /
L1-L2 regularisation, dropout) and into the data layer (chronological split,
leak-safe features, scalers fit on train only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

from . import config as C


# ---------------------------------------------------------------------------
# Naive baselines (operate on the raw target series, not the feature matrix)
# ---------------------------------------------------------------------------
def persistence_forecast(y_index_target: pd.Series, target_series: pd.Series) -> np.ndarray:
    """y_hat(t+1) = y(t). The feature-row index is time t; target is t+1."""
    return target_series.reindex(y_index_target.index).to_numpy()


def seasonal_naive_forecast(y_index_target: pd.Series, target_series: pd.Series) -> np.ndarray:
    """y_hat(t+1) = y(t+1-24): the observed value 24 h before the target hour."""
    shifted = target_series.shift(23)  # value at t-23 aligns to target at t+1
    return shifted.reindex(y_index_target.index).to_numpy()


# ---------------------------------------------------------------------------
# Ridge (trend-regression reference)
# ---------------------------------------------------------------------------
def fit_ridge(Xtr, ytr):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=C.SEED))
    model.fit(Xtr, ytr)
    return model


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------
def fit_lightgbm(Xtr, ytr, Xva, yva):
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,      # regularisation: no leaf on a handful of rows
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=C.SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def fit_xgboost(Xtr, ytr, Xva, yva):
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=C.SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------
def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Turn a (rows, features) matrix into overlapping (seq_len, features) windows."""
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _lstm_base_columns(columns):
    """
    Select the *raw* channels for the recurrent net: current sensor readings,
    the current target value, and the cyclical time encodings. We deliberately
    drop the pre-computed lag/rolling features here -- the LSTM is meant to learn
    that temporal structure itself from the 24 h window, and feeding it 100+
    redundant engineered columns only adds noise and overfitting surface.
    """
    keep = []
    for c in columns:
        if "_lag" in c or "_rmean" in c or "_rstd" in c:
            continue
        keep.append(c)
    return keep


def fit_lstm(Xtr, ytr, Xva, yva):
    """Train a small LSTM. Returns a bundle with the model + scalers + columns."""
    import torch
    import torch.nn as nn

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    base_cols = _lstm_base_columns(list(Xtr.columns))
    Xtr = Xtr[base_cols]
    Xva = Xva[base_cols]

    # Scale on TRAIN ONLY, then apply to val (no leakage).
    x_scaler = StandardScaler().fit(Xtr)
    y_mean, y_std = float(ytr.mean()), float(ytr.std() + 1e-8)

    Xtr_s = x_scaler.transform(Xtr)
    Xva_s = x_scaler.transform(Xva)
    ytr_s = (ytr.to_numpy() - y_mean) / y_std
    yva_s = (yva.to_numpy() - y_mean) / y_std

    seq = C.SEQ_LEN
    Xtr_seq, ytr_seq = _make_sequences(Xtr_s, ytr_s, seq)
    Xva_seq, yva_seq = _make_sequences(Xva_s, yva_s, seq)

    n_features = Xtr_seq.shape[2]

    class LSTMRegressor(nn.Module):
        def __init__(self, n_in, hidden, layers):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=layers,
                                batch_first=True,
                                dropout=0.2 if layers > 1 else 0.0)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    device = torch.device("cpu")
    model = LSTMRegressor(n_features, C.LSTM_HIDDEN, C.LSTM_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=C.LSTM_LR, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    tr_x = torch.tensor(Xtr_seq); tr_y = torch.tensor(ytr_seq)
    va_x = torch.tensor(Xva_seq); va_y = torch.tensor(yva_seq)
    ds = torch.utils.data.TensorDataset(tr_x, tr_y)
    dl = torch.utils.data.DataLoader(ds, batch_size=C.LSTM_BATCH, shuffle=True)

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(C.LSTM_EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(va_x), va_y).item()
        if vloss < best_val - 1e-5:
            best_val, best_state, patience = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= C.LSTM_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    bundle = dict(model=model, x_scaler=x_scaler, y_mean=y_mean, y_std=y_std,
                  seq=seq, base_cols=base_cols)
    return bundle


def lstm_predict(bundle, X) -> np.ndarray:
    """Predict on a feature matrix; first `seq` rows have no full window -> NaN."""
    import torch
    X = X[bundle["base_cols"]]
    Xs = bundle["x_scaler"].transform(X)
    seq = bundle["seq"]
    if len(Xs) <= seq:
        return np.full(len(Xs), np.nan)
    windows = np.stack([Xs[i - seq:i] for i in range(seq, len(Xs))]).astype(np.float32)
    bundle["model"].eval()
    with torch.no_grad():
        pred_s = bundle["model"](torch.tensor(windows)).numpy()
    pred = pred_s * bundle["y_std"] + bundle["y_mean"]
    out = np.full(len(Xs), np.nan)
    out[seq:] = pred
    return out
