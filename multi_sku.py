"""Per-SKU feature tables for multi-product retail panels."""

from __future__ import annotations

from typing import List

import pandas as pd

from feature_engineering import build_full_feature_table


def build_features_per_sku(
    df: pd.DataFrame,
    date_col: str,
    sku_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Build calendar + harmonic + lag + rolling features within each SKU time series.
    """
    parts: List[pd.DataFrame] = []
    for sku, g in df.groupby(sku_col, sort=False):
        g = g[[date_col, sku_col, target_col]].copy()
        if len(g) < 40:
            continue
        ft = build_full_feature_table(g, date_col, target_col, id_cols=[sku_col])
        parts.append(ft)
    if not parts:
        raise ValueError("No SKU has enough rows (need ~40+) after cleaning.")
    return pd.concat(parts, ignore_index=True)
