"""
Phase 3 - Feature engineering.

Design rule that protects against leakage: every engineered feature at time *t*
uses information available strictly at or before *t*, and the target is the
benzene value at *t + HORIZON*. Concretely, every rolling statistic is computed
on a series that has been ``shift(1)``-ed first, so the value at time t can never
see its own reading -- only the past.

Feature families
----------------
* Lags (1, 2, 3, 24 h): short-term momentum plus the 24 h daily cycle.
* Rolling mean / std over 3 h, 6 h, 24 h: local level and local volatility
  (the std captures hardware noise / instability the raw value hides).
* Cyclical calendar encodings (hour, day-of-week, month) via sin/cos so the
  model sees that hour 23 and hour 0 are adjacent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _cyclical(values: pd.Series, period: int, name: str) -> pd.DataFrame:
    rad = 2 * np.pi * values / period
    return pd.DataFrame(
        {f"{name}_sin": np.sin(rad), f"{name}_cos": np.cos(rad)},
        index=values.index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the full leak-safe feature matrix + supervised target column."""
    feat = pd.DataFrame(index=df.index)

    base_cols = [c for c in C.SENSOR_COLS if c in df.columns] + [C.TARGET]

    # Current-hour sensor readings (available at decision time t).
    for col in C.SENSOR_COLS:
        if col in df.columns:
            feat[col] = df[col]

    # Current-hour target reading. At decision time t the latest benzene value
    # IS known -- this is exactly the information the persistence baseline uses,
    # so the learned models must be allowed to see it for a fair comparison.
    feat[f"{C.TARGET}_now"] = df[C.TARGET]

    # Lag features.
    for col in base_cols:
        for lag in C.LAGS:
            feat[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Leak-safe rolling statistics (shift(1) BEFORE rolling).
    for col in base_cols:
        shifted = df[col].shift(1)
        for w in C.ROLL_WINDOWS:
            feat[f"{col}_rmean{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).mean()
            feat[f"{col}_rstd{w}"] = shifted.rolling(w, min_periods=max(2, w // 2)).std()

    # Cyclical time encodings.
    idx = df.index
    feat = pd.concat(
        [
            feat,
            _cyclical(pd.Series(idx.hour, index=idx), 24, "hour"),
            _cyclical(pd.Series(idx.dayofweek, index=idx), 7, "dow"),
            _cyclical(pd.Series(idx.month, index=idx), 12, "month"),
        ],
        axis=1,
    )

    # Supervised target: benzene HORIZON hours into the future.
    feat["target"] = df[C.TARGET].shift(-C.HORIZON)

    # Drop rows with any NaN (created by lags/rolling edges or long outages).
    feat = feat.dropna()
    return feat


def time_split(feat: pd.DataFrame):
    """Chronological train / val / test split -- never shuffled."""
    n = len(feat)
    n_test = int(n * C.TEST_FRAC)
    n_val = int(n * C.VAL_FRAC)
    n_train = n - n_val - n_test

    train = feat.iloc[:n_train]
    val = feat.iloc[n_train:n_train + n_val]
    test = feat.iloc[n_train + n_val:]

    def xy(d):
        return d.drop(columns=["target"]), d["target"]

    return xy(train), xy(val), xy(test)


if __name__ == "__main__":
    cleaned = pd.read_csv(C.DATA_PROCESSED, index_col=0, parse_dates=True)
    feat = build_features(cleaned)
    print("feature matrix:", feat.shape)
    print("n features:", feat.shape[1] - 1)
    (tr, _), (va, _), (te, _) = time_split(feat)
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}")
