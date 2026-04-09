"""
Inventory intelligence: Safety Stock, Reorder Point (ROP), Economic Order Quantity (EOQ).

Assumptions (state clearly in viva):
- Demand during lead time is approximately Normal with mean \\mu_L and std \\sigma_L.
- If daily demand has mean \\mu_d and std \\sigma_d (iid days), then over L days:
      \\mu_L = \\mu_d * L
      \\sigma_L = \\sigma_d * sqrt(L)   (square-root law for iid sums)

95% cycle service level (in-stock probability each cycle):
- z = \\Phi^{-1}(0.95) \\approx 1.645 for standard normal.

Safety stock:
      SS = z * \\sigma_L

Reorder point (continuous review, approx.):
      ROP = \\mu_L + SS

EOQ (classic Harris-Wilson, deterministic demand D per year, ordering cost S, holding cost H per unit per year):
      Q* = sqrt( (2 * D * S) / H )
Derived by minimizing ordering + holding cost per year under smooth approximations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class InventoryPolicy:
    safety_stock: float
    reorder_point: float
    eoq: float
    z_value: float
    sigma_lead_time: float
    mean_lead_time_demand: float


def service_level_to_z(service_level: float = 0.95) -> float:
    """One-sided critical value: P(Z <= z) = service_level."""
    return float(norm.ppf(service_level))


def safety_stock_normal(
    sigma_daily: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> tuple[float, float]:
    """
    SS = z * sigma_daily * sqrt(L).

    Returns (SS, sigma_lead_time).
    """
    z = service_level_to_z(service_level)
    sigma_l = float(sigma_daily * np.sqrt(max(lead_time_days, 0.0)))
    ss = z * sigma_l
    return ss, sigma_l


def reorder_point(
    mean_daily_demand: float,
    lead_time_days: float,
    sigma_daily: float,
    service_level: float = 0.95,
) -> InventoryPolicy:
    """
    ROP = mean demand during lead time + safety stock.
    """
    mu_l = float(mean_daily_demand * lead_time_days)
    ss, sigma_l = safety_stock_normal(sigma_daily, lead_time_days, service_level)
    z = service_level_to_z(service_level)
    rop = mu_l + ss
    return InventoryPolicy(
        safety_stock=ss,
        reorder_point=rop,
        eoq=float("nan"),  # filled by eoq function
        z_value=z,
        sigma_lead_time=sigma_l,
        mean_lead_time_demand=mu_l,
    )


def economic_order_quantity(
    annual_demand: float,
    ordering_cost_per_order: float,
    holding_cost_per_unit_per_year: float,
) -> float:
    """
    Q* = sqrt(2 D S / H). All units consistent: D annual, S per order, H per unit-year.
    """
    if annual_demand <= 0 or ordering_cost_per_order <= 0 or holding_cost_per_unit_per_year <= 0:
        return float("nan")
    return float(np.sqrt((2.0 * annual_demand * ordering_cost_per_order) / holding_cost_per_unit_per_year))


def full_policy(
    mean_daily_demand: float,
    sigma_daily: float,
    lead_time_days: float,
    ordering_cost: float,
    holding_cost_per_unit_per_year: float,
    service_level: float = 0.95,
) -> InventoryPolicy:
    """
    Combine ROP/SS with EOQ using annual demand = mean_daily * 365 (adjust for your business calendar).
    """
    base = reorder_point(mean_daily_demand, lead_time_days, sigma_daily, service_level)
    annual_d = float(mean_daily_demand * 365.0)
    q = economic_order_quantity(annual_d, ordering_cost, holding_cost_per_unit_per_year)
    return InventoryPolicy(
        safety_stock=base.safety_stock,
        reorder_point=base.reorder_point,
        eoq=q,
        z_value=base.z_value,
        sigma_lead_time=base.sigma_lead_time,
        mean_lead_time_demand=base.mean_lead_time_demand,
    )


def shopping_list(
    sku_forecast_mean_daily: dict[str, float],
    current_on_hand: dict[str, float],
    policy: dict[str, InventoryPolicy],
) -> pd.DataFrame:
    """
    Rows where on_hand < ROP => recommend order quantity (simple: EOQ or max(0, ROP + EOQ - on_hand)).

    Here: order_qty = max(0, ceil(ROP + EOQ - on_hand)) as a practical restock heuristic
    (treats EOQ as lot size when below ROP trigger). Adjust per your professor's preference.
    """
    rows = []
    for sku, mu_d in sku_forecast_mean_daily.items():
        oh = current_on_hand.get(sku, 0.0)
        pol = policy[sku]
        need = max(0.0, pol.reorder_point + pol.eoq - oh) if np.isfinite(pol.eoq) else max(0.0, pol.reorder_point - oh)
        below_rop = oh < pol.reorder_point
        rows.append(
            {
                "SKU": sku,
                "forecast_mean_daily": mu_d,
                "on_hand": oh,
                "ROP": pol.reorder_point,
                "EOQ": pol.eoq,
                "below_ROP": below_rop,
                "suggested_order_qty": float(np.ceil(need)),
            }
        )
    return pd.DataFrame(rows)
