"""Orchestration: pick best model from compare_models, export tables, demand stats for inventory."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from forecasting import fit_predict_xgboost


def select_best_model(compare_result: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, float]]]:
    """Choose model with lowest WMAPE among entries that have metrics."""
    best_name: Optional[str] = None
    best_w = float("inf")
    best_m: Optional[Dict[str, float]] = None
    for name, payload in compare_result.get("models", {}).items():
        m = payload.get("metrics")
        if not m:
            continue
        w = m.get("WMAPE", float("inf"))
        if w < best_w:
            best_w = w
            best_name = name
            best_m = m
    return best_name, best_m


def forecast_mean_daily_from_holdout(compare_result: Dict[str, Any], best_name: Optional[str]) -> float:
    """
    Use mean of hold-out predictions from the best model as expected daily demand (single-level proxy).
    """
    if not best_name:
        return float("nan")
    pred = compare_result["models"][best_name].get("pred")
    if pred is None:
        return float("nan")
    return float(np.mean(np.asarray(pred, dtype=float)))


def historical_mu_sigma(y: np.ndarray, window: int = 30) -> Tuple[float, float]:
    """Recent demand mean and std (ddof=1)."""
    y = np.asarray(y, dtype=float)
    tail = y[-window:] if len(y) >= window else y
    mu = float(np.mean(tail))
    sigma = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
    return mu, max(sigma, 1e-6)


def naive_rolling_forecast_series(y: np.ndarray, window: int = 7) -> np.ndarray:
    """Baseline: rolling mean of past demand (shifted). Same length as y."""
    s = pd.Series(y, dtype=float)
    return np.maximum(0.0, s.rolling(window, min_periods=1).mean().shift(1).bfill().values)


def xgboost_forecast_tail(
    feat_clean: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    tail: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train XGBoost on the prefix; predict the last `tail` rows (aligned with viva backtest story).
    Returns (y_actual_tail, y_pred_tail).
    """
    n = len(feat_clean)
    if n <= tail + 25:
        raise ValueError("Not enough history for tail forecast — reduce tail or add rows.")
    tr = feat_clean.iloc[: n - tail].copy()
    te = feat_clean.iloc[n - tail :].copy()
    y_act = te[target_col].values.astype(float)
    pred = fit_predict_xgboost(tr, te, feature_cols, target_col)
    return y_act, np.asarray(pred, dtype=float)


def comparison_to_dataframe(compare_result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, payload in compare_result.get("models", {}).items():
        if payload.get("metrics"):
            rows.append({"Model": name, **payload["metrics"]})
        else:
            rows.append({"Model": name, "MAE": None, "RMSE": None, "WMAPE": None, "error": payload.get("error")})
    return pd.DataFrame(rows)
