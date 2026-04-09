"""
Forecast accuracy metrics (for viva):

- MAE:  mean absolute error — same units as demand; robust to outliers vs squared errors.
        MAE = (1/n) * sum |y_i - \\hat{y}_i|

- RMSE: root mean squared error — penalizes large errors more heavily.
        RMSE = sqrt( (1/n) * sum (y_i - \\hat{y}_i)^2 )

- WMAPE: weighted MAPE — stable when some periods have near-zero demand (classic sMAPE/WAPE issues).
  Common definition used in retail forecasting:
        WMAPE = ( sum_i |y_i - \\hat{y}_i| / sum_i |y_i| ) * 100
  If sum |y_i| == 0, return nan or 0 by convention.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom < eps:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "WMAPE": wmape(y_true, y_pred)}
