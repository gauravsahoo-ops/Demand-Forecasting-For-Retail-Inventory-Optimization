"""
Financial impact: cost of stockouts vs cost of overstock (holding + obsolescence).

Use in ROI narrative:
- Stockout cost per unit ≈ lost margin (price - variable cost) * units short,
  optionally add goodwill penalty.
- Overstock cost per unit-year ≈ holding rate * unit cost (capital + warehousing + spoilage).

This module compares scenario totals to show value of better forecasting + inventory policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FinancialParams:
    """Unit economics for a SKU or category."""

    unit_cost: float
    selling_price: float
    holding_cost_rate_per_year: float  # e.g. 0.25 => 25% of unit cost per year held
    goodwill_stockout_penalty_per_unit: float = 0.0  # optional add-on beyond lost margin


def lost_margin_per_unit(params: FinancialParams) -> float:
    return max(0.0, float(params.selling_price - params.unit_cost))


def holding_cost_per_unit_per_year(params: FinancialParams) -> float:
    return float(params.unit_cost * params.holding_cost_rate_per_year)


def stockout_cost(
    unmet_demand_units: float,
    params: FinancialParams,
) -> float:
    """
    Cost of stockouts for the period: unmet demand * (lost margin + optional goodwill).
    """
    u = max(0.0, float(unmet_demand_units))
    return u * (lost_margin_per_unit(params) + params.goodwill_stockout_penalty_per_unit)


def overstock_cost(
    excess_units: float,
    params: FinancialParams,
    fraction_of_year_held: float = 1.0,
) -> float:
    """
    Cost of carrying excess inventory: excess_units * holding_cost_per_unit_year * time held.

    For a monthly review, fraction_of_year_held might be 1/12.
    """
    e = max(0.0, float(excess_units))
    return e * holding_cost_per_unit_per_year(params) * max(0.0, fraction_of_year_held)


def compare_scenarios(
    actual_demand: np.ndarray,
    baseline_forecast: np.ndarray,
    improved_forecast: np.ndarray,
    starting_inventory: float,
    params: FinancialParams,
    fraction_of_year_per_step: float,
) -> pd.DataFrame:
    """
    Simple two-period-style ledger: for each day/step, assume you stock to `forecast`
    at start (toy model for demo). Compare total stockout + holding.

    This is illustrative for viva — real systems use stochastic simulation or rolling MILP.

    Mechanics (transparent assumptions):
    - Ending inventory evolves: I_{t+1} = max(0, I_t + order_t - actual_t)
    - order_t chosen to raise inventory up to forecast (capped at non-negative order).
    """
    n = len(actual_demand)
    a = np.asarray(actual_demand, dtype=float)
    fb = np.asarray(baseline_forecast, dtype=float)
    fi = np.asarray(improved_forecast, dtype=float)

    def simulate(f: np.ndarray) -> tuple[float, float]:
        inv = float(starting_inventory)
        total_so = 0.0
        total_ov = 0.0
        for t in range(n):
            # Order up to forecast target (same-period delivery for coursework clarity)
            order_qty = max(0.0, float(f[t]) - inv)
            inv_after_receipt = inv + order_qty
            shipped = min(inv_after_receipt, a[t])
            short = max(0.0, a[t] - inv_after_receipt)
            inv = inv_after_receipt - shipped
            total_so += stockout_cost(short, params)
            excess = max(0.0, inv - float(f[t]))
            total_ov += overstock_cost(excess, params, fraction_of_year_per_step)
        return total_so, total_ov

    so_b, ov_b = simulate(fb)
    so_i, ov_i = simulate(fi)

    roi_numerator = (so_b + ov_b) - (so_i + ov_i)
    denom = max(1e-9, so_b + ov_b)

    return pd.DataFrame(
        {
            "scenario": ["Baseline forecast", "Improved forecast"],
            "stockout_cost": [so_b, so_i],
            "overstock_cost": [ov_b, ov_i],
            "total_cost": [so_b + ov_b, so_i + ov_i],
            "savings_vs_baseline": [0.0, roi_numerator],
            "pct_savings": [0.0, 100.0 * roi_numerator / denom],
        }
    )


def project_roi_summary(
    annual_inventory_cost_baseline: float,
    annual_inventory_cost_improved: float,
    implementation_cost_one_time: float = 0.0,
    annual_maintenance_cost: float = 0.0,
) -> dict:
    """
    High-level ROI: (savings - capex) / capex if you want payback; here return dict of deltas.
    """
    savings = annual_inventory_cost_baseline - annual_inventory_cost_improved
    net_year1 = savings - implementation_cost_one_time - annual_maintenance_cost
    return {
        "annual_savings": float(savings),
        "year1_net_benefit": float(net_year1),
        "implementation_cost": float(implementation_cost_one_time),
        "annual_maintenance": float(annual_maintenance_cost),
    }
