"""Load and normalize retail CSV uploads (date, demand, optional SKU, optional on_hand)."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


def _coerce_date(s: pd.Series) -> pd.Series:
    """Try ISO/US first, then day-first (common in EU CSVs)."""
    v = pd.to_datetime(s, errors="coerce", utc=False)
    if v.notna().any():
        return v
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def normalize_retail_dataframe(
    df: pd.DataFrame,
    date_col: str,
    demand_col: str,
    sku_col: Optional[str] = None,
    on_hand_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse dates, coerce demand to float, drop invalid rows, sort by date (and SKU).

    If ``date_col`` and ``demand_col`` are the same column (common mis-click), the second
    coercion destroys parsed dates — callers should validate distinct columns first.
    """
    if date_col == demand_col:
        raise ValueError(
            "Date column and demand column must be different. "
            "If your file has only one column, add a quantity column or use a wider export."
        )
    out = df.copy()
    out[date_col] = _coerce_date(out[date_col])
    out[demand_col] = pd.to_numeric(out[demand_col], errors="coerce")
    if sku_col and sku_col in out.columns:
        out[sku_col] = out[sku_col].astype(str)
    if on_hand_col and on_hand_col in out.columns:
        out[on_hand_col] = pd.to_numeric(out[on_hand_col], errors="coerce")
    out = out.dropna(subset=[date_col, demand_col])
    sort_cols: List[str] = []
    if sku_col and sku_col in out.columns:
        sort_cols.append(sku_col)
    sort_cols.append(date_col)
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def aggregate_skus_to_total(
    df: pd.DataFrame,
    date_col: str,
    demand_col: str,
    sku_col: str,
) -> pd.DataFrame:
    """Sum demand across SKUs by day for a single national series."""
    g = df.groupby(date_col, as_index=False)[demand_col].sum()
    g = g.rename(columns={demand_col: "demand"})
    return g.sort_values(date_col).reset_index(drop=True)


def latest_on_hand_by_sku(
    df: pd.DataFrame,
    sku_col: str,
    on_hand_col: str,
) -> dict[str, float]:
    """Last non-null on-hand per SKU."""
    d = df.dropna(subset=[on_hand_col]).sort_values(sku_col)
    last = d.groupby(sku_col)[on_hand_col].last()
    return {str(k): float(v) for k, v in last.items()}
