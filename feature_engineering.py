"""
Advanced feature engineering for retail demand forecasting.

Mathematical notes (for viva):
------------------------------
1) Rolling statistics
   For window size w, the rolling mean at time t uses observations in [t-w+1, t]:
   \\bar{x}_t^{(w)} = (1/w) * sum_{i=0}^{w-1} x_{t-i}
   Rolling std captures local volatility (useful when demand variance is non-stationary).

2) Lag features
   L_k(t) = y_{t-k} injects autoregressive structure into ML models without assuming
   a specific parametric form (unlike ARIMA, which ties coefficients across lags).

3) Harmonic (sin/cos) seasonality encoding
   A naive integer month (1..12) implies that December is "far" from January in Euclidean
   distance, which is wrong for seasonality. Map cyclic time t with period T to the unit circle:
   \\theta_t = 2\\pi * (t mod T) / T
   Features: sin(\\theta_t), cos(\\theta_t)
   This preserves continuity: January is adjacent to December in the embedding.
   Multiple harmonics (k=1,2,...) can capture non-sinusoidal seasonality:
   sin(2\\pi k t / T), cos(2\\pi k t / T)
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep first occurrence when CSV uploads accidentally repeat column names."""
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated(keep="first")].copy()
    return df


def _target_series(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Single Series for the target. If duplicate column labels exist, pandas returns a
    DataFrame — that breaks lag/shift assignments (ValueError on lag_1).
    """
    y = df[target_col]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    name = target_col if isinstance(target_col, str) else str(target_col)
    return pd.Series(pd.to_numeric(y, errors="coerce"), index=df.index, name=name)


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str,
) -> pd.DataFrame:
    """Extract calendar fields from a datetime column."""
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day_of_month"] = dt.dt.day
    out["day_of_week"] = dt.dt.dayofweek  # Mon=0
    out["day_of_year"] = dt.dt.dayofyear
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    return out


def harmonic_encoding(
    values: pd.Series,
    period: float,
    orders: Sequence[int] = (1,),
) -> pd.DataFrame:
    """
    Encode a cyclic scalar (e.g., day-of-year 1..365) using sin/cos at multiple harmonics.

    For harmonic order k and period T:
        \\theta = 2\\pi * v / T
        feature_k = (sin(k*\\theta), cos(k*\\theta))

    Parameters
    ----------
    values : Series of positions on the cycle (e.g., day_of_year).
    period : Cycle length T (e.g., 365.25 for annual).
    orders : Harmonic indices k (1 = annual fundamental; 2 = first overtone, etc.).
    """
    theta = 2.0 * np.pi * (values.astype(float) / period)
    frames = []
    for k in orders:
        frames.append(
            pd.DataFrame(
                {
                    f"harm_{int(period)}_sin_k{k}": np.sin(k * theta),
                    f"harm_{int(period)}_cos_k{k}": np.cos(k * theta),
                }
            )
        )
    return pd.concat(frames, axis=1)


def add_seasonality_harmonics(
    df: pd.DataFrame,
    date_col: str,
    annual_period: float = 365.25,
    weekly_period: float = 7.0,
    annual_orders: Tuple[int, ...] = (1, 2),
    weekly_orders: Tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """
    Add annual and weekly harmonic features based on the date column.

    Annual: uses day_of_year (1..366) — handles leap years via max day in year if needed.
    Weekly: uses day_of_week as 0..6 mapped to a 7-day cycle.
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    day_of_year = dt.dt.dayofyear.astype(float)

    ann = harmonic_encoding(pd.Series(day_of_year.values, index=out.index), annual_period, annual_orders)
    # Day of week: 0..6 -> map to 1..7 for cleaner period
    dow = dt.dt.dayofweek.astype(float) + 1.0
    wk = harmonic_encoding(pd.Series(dow.values, index=out.index), weekly_period, weekly_orders)
    return pd.concat([out, ann, wk], axis=1)


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: Iterable[int] = (7, 14, 30),
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling mean and std for demand. Expanding window optional pattern: use min_periods
    to stabilize early rows (or drop NaNs downstream).

    Rolling mean \\bar{y}_t^{(w)} and rolling std s_t^{(w)} are computed on past values
    including t (pandas default). For strict "past-only" features for forecasting at t+1,
    shift by 1 after rolling (see `add_lag_and_roll_for_forecast`).
    """
    out = df.copy()
    s = _target_series(out, target_col)
    for w in windows:
        mp = min_periods if min_periods is not None else max(1, w // 2)
        out[f"roll_mean_{w}"] = s.rolling(window=w, min_periods=mp).mean()
        out[f"roll_std_{w}"] = s.rolling(window=w, min_periods=mp).std()
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Iterable[int] = (1, 7, 14, 30),
) -> pd.DataFrame:
    """Lag-k: y_{t-k}."""
    out = df.copy()
    y = _target_series(out, target_col)
    for k in lags:
        out[f"lag_{k}"] = y.shift(k)
    return out


def add_lag_and_roll_for_forecast(
    df: pd.DataFrame,
    target_col: str,
    windows: Iterable[int] = (7, 14, 30),
    lags: Iterable[int] = (1, 7, 14, 30),
) -> pd.DataFrame:
    """
    Build features safe for one-step-ahead training: rolling stats and lags use only past.

    After rolling, shift by 1 so that row t features depend on data up to t-1 only,
    aligning with predicting y_t from information available at start of day t.
    """
    out = add_lag_features(df, target_col, lags)
    s = _target_series(df, target_col)
    for w in windows:
        rm = s.rolling(window=w, min_periods=1).mean().shift(1)
        rs = s.rolling(window=w, min_periods=1).std().shift(1)
        out[f"roll_mean_{w}_past"] = rm
        out[f"roll_std_{w}_past"] = rs
    return out


def _lags_and_windows_for_series_length(n: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Shorter histories cannot support lag-30 / roll-30 without discarding too many rows.
    After dropna(), we need enough rows left for train/test splits.
    """
    if n < 35:
        return (1,), (7,)
    if n < 70:
        return (1, 7), (7,)
    if n < 120:
        return (1, 7, 14), (7, 14)
    return (1, 7, 14, 30), (7, 14, 30)


def build_full_feature_table(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    id_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline: calendar + harmonics + lags + past-only rolling.
    Expects rows sorted by date ascending within each series if multi-SKU.

    Lag / rolling windows scale down automatically when the series is short, so more rows
    survive ``dropna()`` than with a fixed lag-30 design.
    """
    df = _dedupe_columns(df)
    out = df.sort_values(by=[*(id_cols or []), date_col]).reset_index(drop=True)
    n = len(out)
    lags, windows = _lags_and_windows_for_series_length(n)
    out = add_calendar_features(out, date_col)
    out = add_seasonality_harmonics(out, date_col)
    out = add_lag_and_roll_for_forecast(out, target_col, windows=windows, lags=lags)
    return out
