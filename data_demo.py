"""Synthetic daily demand for demos (trend + weekly seasonality + noise)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_sales(
    n_days: int = 400,
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = np.arange(n_days)
    t = pd.date_range(start=start_date, periods=n_days, freq="D")
    trend = 5.0 + 0.01 * days
    weekly = 3.0 * np.sin(2 * np.pi * (days % 7) / 7.0)
    annual = 2.0 * np.sin(2 * np.pi * days / 365.25)
    noise = rng.normal(0, 2.0, size=n_days)
    demand = np.maximum(0.0, trend + weekly + annual + noise)
    return pd.DataFrame({"date": t, "demand": demand})


def make_multi_sku_demo(
    n_days: int = 400,
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Long-format sample: date, sku, demand, on_hand (latest snapshot repeated for convenience)."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start=start_date, periods=n_days, freq="D")
    days = np.arange(n_days)
    rows = []
    sku_scales = {"SKU-A": 1.0, "SKU-B": 0.65, "SKU-C": 1.35}
    for sku, scale in sku_scales.items():
        trend = (5.0 + 0.008 * days) * scale
        weekly = 2.5 * scale * np.sin(2 * np.pi * (days % 7) / 7.0 + (abs(hash(sku)) % 7))
        annual = 1.5 * scale * np.sin(2 * np.pi * days / 365.25)
        noise = rng.normal(0, 1.8 * scale, size=n_days)
        d = np.maximum(0.0, trend + weekly + annual + noise)
        oh = float(max(0.0, 15.0 * scale + rng.normal(0, 3)))
        for i in range(n_days):
            rows.append({"date": t[i], "sku": sku, "demand": d[i], "on_hand": oh})
    out = pd.DataFrame(rows)
    return out.sort_values(["sku", "date"]).reset_index(drop=True)
