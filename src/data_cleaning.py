"""
Phase 2 - Data cleaning.

Real field IoT data has four characteristic pathologies; this module addresses
each one explicitly and records *why* each choice was made:

  1. Missing timestamps  -> reindex onto a complete hourly grid so gaps become
                            explicit NaNs instead of silent jumps in time.
  2. Sensor dropouts     -> the device writes the sentinel -200; convert to NaN,
                            then time-interpolate ONLY across short gaps (<= 6h).
                            Longer outages are left as NaN and dropped from the
                            supervised set -- imputing a full day of physics we
                            never observed would be fiction, not data.
  3. Out-of-bounds spikes-> a Hampel filter (rolling median + MAD) flags points
                            that deviate too far from their local neighbourhood.
                            MAD is robust to the very outliers we are hunting,
                            unlike a mean/standard-deviation z-score.
  4. Hardware noise      -> handled downstream by rolling-window features rather
                            than by destroying the raw signal here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_raw(path=C.DATA_RAW) -> pd.DataFrame:
    """Load the raw UCI Air Quality CSV and build a clean hourly DatetimeIndex."""
    df = pd.read_csv(path)

    # Drop the two trailing empty columns the file ships with.
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(how="all")

    # Build a single datetime column. Source format is dd-mm-yyyy + HH:MM:SS.
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])
    df.index = dt
    df.index.name = "datetime"
    df = df[df.index.notna()].sort_index()

    # Some numeric columns arrive as strings with comma decimals on certain
    # mirrors; coerce everything to float defensively.
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str).str.replace(",", ".", regex=False)
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Force a gap-free hourly index so missing timestamps are made explicit."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    return df.reindex(full)


def sentinel_to_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the device's -200 'missing' sentinel with real NaN."""
    return df.replace(C.MISSING_SENTINEL, np.nan)


def hampel_filter(s: pd.Series, window: int = C.HAMPEL_WINDOW,
                  n_sigma: float = C.HAMPEL_NSIGMA) -> pd.Series:
    """
    Replace local outliers with NaN using a robust median/MAD test.

    A point is an outlier if it sits more than ``n_sigma`` scaled-MADs away from
    the rolling median of its neighbourhood. We blank the spike (-> NaN) and let
    the same capped interpolation that handles dropouts fill it back in, so
    spikes and dropouts are repaired by one consistent mechanism.
    """
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    scaled_mad = 1.4826 * mad                      # MAD -> std-equivalent
    diff = (s - med).abs()
    mask = (scaled_mad > 0) & (diff > n_sigma * scaled_mad)
    out = s.copy()
    out[mask] = np.nan
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the analysis-ready frame."""
    df = df.drop(columns=[c for c in C.DROP_COLS if c in df.columns])
    df = reindex_hourly(df)
    df = sentinel_to_nan(df)

    # Spike removal on the predictor channels + target.
    for col in C.SENSOR_COLS + [C.TARGET]:
        if col in df.columns:
            df[col] = hampel_filter(df[col])

    # Capped, time-aware interpolation of short gaps; long gaps stay NaN.
    df = df.interpolate(method="time", limit=C.INTERP_LIMIT, limit_area="inside")

    return df


def cleaning_report(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Small before/after table used in the notebook and README."""
    raw_na = (raw.replace(C.MISSING_SENTINEL, np.nan).isna().mean() * 100).round(2)
    clean_na = (cleaned.isna().mean() * 100).round(2)
    rep = pd.DataFrame({"missing_%_raw": raw_na, "missing_%_clean": clean_na})
    return rep.dropna(how="all")


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    cleaned.to_csv(C.DATA_PROCESSED)
    print(f"Saved cleaned data -> {C.DATA_PROCESSED}  shape={cleaned.shape}")
    print(cleaning_report(raw, cleaned).to_string())
