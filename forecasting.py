"""
Hybrid forecasting engine: SARIMA vs tree boosting vs LSTM.

Design (viva talking points):
- SARIMAX/SARIMA: classical structure for trend + seasonality + autocorrelation; strong baseline.
- Gradient boosting (XGBoost/LightGBM): learns nonlinear interactions between lags, harmonics, promos.
- LSTM: sequence model for long-range dependence; needs more data and tuning.

Cross-validation: TimeSeriesSplit avoids lookahead — folds respect temporal order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from metrics import all_metrics


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------
def fit_predict_sarima(
    y: pd.Series,
    horizon: int,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
) -> np.ndarray:
    """
    Fit SARIMA on univariate series and forecast `horizon` steps.

    Notes:
    - `seasonal_order` last element is seasonal period m (e.g., 7 for weekly seasonality on daily data).
    - For short series, simpler orders reduce convergence failures.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = y.astype(float).asfreq("D") if hasattr(y.index, "freq") else y.astype(float)
    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False, maxiter=100)
    fc = res.get_forecast(steps=horizon)
    return fc.predicted_mean.values


# ---------------------------------------------------------------------------
# XGBoost / LightGBM helpers
# ---------------------------------------------------------------------------
def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


def fit_predict_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    **kwargs: Any,
) -> np.ndarray:
    import xgboost as xgb

    X_tr, y_tr = _prepare_xy(train_df, feature_cols, target_col)
    X_te, _ = _prepare_xy(test_df, feature_cols, target_col)
    # Align test rows that may have NaNs in features
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        **kwargs,
    )
    model.fit(X_tr, y_tr)
    X_te_full = test_df[feature_cols].copy()
    med = X_tr.median(numeric_only=True)
    X_te_full = X_te_full.fillna(med)
    pred = model.predict(X_te_full)
    return pred


def fit_predict_lightgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    **kwargs: Any,
) -> np.ndarray:
    import lightgbm as lgb

    X_tr, y_tr = _prepare_xy(train_df, feature_cols, target_col)
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        **kwargs,
    )
    model.fit(X_tr, y_tr)
    X_te_full = test_df[feature_cols].copy()
    med = X_tr.median(numeric_only=True)
    X_te_full = X_te_full.fillna(med)
    pred = model.predict(X_te_full)
    return pred


# ---------------------------------------------------------------------------
# LSTM (Keras)
# ---------------------------------------------------------------------------
def _make_sequences(arr: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq_len, len(arr)):
        xs.append(arr[i - seq_len : i])
        ys.append(arr[i])
    return np.array(xs), np.array(ys)


def fit_predict_lstm(
    y: np.ndarray,
    horizon: int,
    seq_len: int = 14,
    epochs: int = 40,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Univariate LSTM: scale series, create sliding windows, train, recursive multi-step forecast.

    For viva: LSTM learns latent state over seq_len steps; multi-step uses previous predictions
    (error accumulation — common limitation; direct multi-horizon heads or seq2seq are alternatives).
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as e:
        raise ImportError("TensorFlow is required for LSTM. pip install tensorflow") from e

    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if len(y) < seq_len + horizon + 5:
        raise ValueError("Series too short for LSTM; increase history or reduce seq_len.")

    scaler = MinMaxScaler()
    ys = scaler.fit_transform(y)

    X_seq, Y_seq = _make_sequences(ys.flatten(), seq_len)
    X_seq = X_seq.reshape(X_seq.shape[0], seq_len, 1)

    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, 1)),
            layers.LSTM(32, return_sequences=False),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")

    model.fit(X_seq, Y_seq, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

    # Recursive forecast
    window = ys[-seq_len:].flatten().tolist()
    preds = []
    for _ in range(horizon):
        x = np.array(window[-seq_len:]).reshape(1, seq_len, 1)
        p = float(model.predict(x, verbose=0)[0, 0])
        preds.append(p)
        window.append(p)

    inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return inv


# ---------------------------------------------------------------------------
# Time-series CV + model comparison
# ---------------------------------------------------------------------------
@dataclass
class CVResult:
    model_name: str
    fold_metrics: List[Dict[str, float]]
    mean_metrics: Dict[str, float]


def time_series_cv_boosting(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    n_splits: int = 5,
    model: str = "xgboost",
) -> CVResult:
    """
    Expanding-window CV with TimeSeriesSplit. Trains boosting on train fold, evaluates on next block.
    """
    df = df.dropna(subset=list(feature_cols) + [target_col]).reset_index(drop=True)
    X = df[feature_cols].values
    y = df[target_col].values
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        tr = df.iloc[train_idx]
        te = df.iloc[test_idx]
        if model == "xgboost":
            pred = fit_predict_xgboost(tr, te, feature_cols, target_col)
        else:
            pred = fit_predict_lightgbm(tr, te, feature_cols, target_col)
        yt = te[target_col].values
        mask = np.isfinite(pred) & np.isfinite(yt)
        fold_metrics.append(all_metrics(yt[mask], pred[mask]))

    mean_metrics = {
        "MAE": float(np.mean([m["MAE"] for m in fold_metrics])),
        "RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
        "WMAPE": float(np.nanmean([m["WMAPE"] for m in fold_metrics])),
    }
    name = "XGBoost" if model == "xgboost" else "LightGBM"
    return CVResult(model_name=name, fold_metrics=fold_metrics, mean_metrics=mean_metrics)


def compare_models(
    y_series: pd.Series,
    df_features: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    test_size: int = 30,
    sarima_order: Tuple[int, int, int] = (1, 1, 1),
    sarima_seasonal: Tuple[int, int, int, int] = (1, 1, 1, 7),
    lstm_epochs: int = 25,
) -> Dict[str, Any]:
    """
    Hold-out last `test_size` points: compare SARIMA, XGBoost, LightGBM, LSTM on identical test slice.

    Returns dict with predictions and metrics per model.
    """
    y_series = y_series.reset_index(drop=True)
    df_features = df_features.reset_index(drop=True)
    n = len(y_series)
    min_train = 35
    if n <= test_size + min_train:
        raise ValueError(
            f"Need more clean rows: have n={n}, hold-out={test_size}, need at least "
            f"{test_size + min_train + 1}. Shorten hold-out or add history / fix gaps."
        )

    train_slice = slice(0, n - test_size)
    test_slice = slice(n - test_size, n)
    horizon = test_size

    y_train = y_series.iloc[train_slice]
    y_test = y_series.iloc[test_slice].values

    results: Dict[str, Any] = {}

    # SARIMA on raw train
    try:
        sarima_fc = fit_predict_sarima(y_train, horizon, order=sarima_order, seasonal_order=sarima_seasonal)
        results["SARIMA"] = {
            "pred": sarima_fc,
            "metrics": all_metrics(y_test, sarima_fc),
        }
    except Exception as ex:  # noqa: BLE001
        results["SARIMA"] = {"error": str(ex), "pred": None, "metrics": None}

    train_df = df_features.iloc[train_slice]
    test_df = df_features.iloc[test_slice]

    try:
        pred_xgb = fit_predict_xgboost(train_df, test_df, feature_cols, target_col)
        results["XGBoost"] = {"pred": pred_xgb, "metrics": all_metrics(y_test, pred_xgb)}
    except Exception as ex:  # noqa: BLE001
        results["XGBoost"] = {"error": str(ex), "pred": None, "metrics": None}

    try:
        pred_lgb = fit_predict_lightgbm(train_df, test_df, feature_cols, target_col)
        results["LightGBM"] = {"pred": pred_lgb, "metrics": all_metrics(y_test, pred_lgb)}
    except Exception as ex:  # noqa: BLE001
        results["LightGBM"] = {"error": str(ex), "pred": None, "metrics": None}

    try:
        lstm_fc = fit_predict_lstm(y_train.values, horizon=horizon, epochs=lstm_epochs)
        results["LSTM"] = {"pred": lstm_fc, "metrics": all_metrics(y_test, lstm_fc)}
    except Exception as ex:  # noqa: BLE001
        results["LSTM"] = {"error": str(ex), "pred": None, "metrics": None}

    return {
        "y_test": y_test,
        "test_index": np.arange(n - test_size, n),
        "models": results,
    }
