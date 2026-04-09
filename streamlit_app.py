"""
Retail Demand Forecasting & Inventory Optimization — manager dashboard.

Run:  python -m streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_demo import make_demo_sales, make_multi_sku_demo
from data_io import aggregate_skus_to_total, latest_on_hand_by_sku, normalize_retail_dataframe
from feature_engineering import build_full_feature_table
from financial import FinancialParams, compare_scenarios, project_roi_summary
from forecasting import compare_models, time_series_cv_boosting
from inventory import full_policy, shopping_list
from pipeline import (
    comparison_to_dataframe,
    forecast_mean_daily_from_holdout,
    historical_mu_sigma,
    naive_rolling_forecast_series,
    select_best_model,
    xgboost_forecast_tail,
)


DATE_COL = "date"
TARGET = "demand"


def _ensure_session() -> None:
    if "work" not in st.session_state:
        st.session_state["work"] = {}


def _default_index(options: list[str], preferred: list[str]) -> int:
    """Pick first preferred option (case-insensitive), else 0."""
    low = [x.lower().strip() for x in options]
    for p in preferred:
        if p.lower() in low:
            return low.index(p.lower())
    return 0


def load_demo(use_multi: bool, seed: int) -> pd.DataFrame:
    if use_multi:
        return make_multi_sku_demo(n_days=400, seed=seed)
    return make_demo_sales(n_days=450, seed=seed)


def main() -> None:
    _ensure_session()
    st.set_page_config(page_title="Retail Demand & Inventory", layout="wide", initial_sidebar_state="expanded")
    st.title("Demand Forecasting for Retail Inventory Optimization")
    st.caption(
        "Hybrid models (SARIMA · XGBoost · LightGBM · LSTM) · Safety stock & EOQ · Financial impact · Shopping list"
    )

    # —— Sidebar: data ——
    st.sidebar.header("Data source")
    src = st.sidebar.radio("Choose source", ["Demo data", "Upload CSV"], horizontal=True)
    use_multi = False
    seed = int(st.sidebar.number_input("Random seed (demo)", 0, 9999, 42))

    raw: pd.DataFrame
    if src == "Demo data":
        use_multi = st.sidebar.checkbox("Multi-SKU demo (3 products)", value=True)
        raw = load_demo(use_multi, seed)
        st.sidebar.caption("Synthetic trend + weekly/annual seasonality + noise.")
    else:
        up = st.sidebar.file_uploader("CSV file", type=["csv"])
        if not up:
            st.info("Upload a CSV or switch to **Demo data** in the sidebar.")
            st.stop()
        def _read_uploaded_csv(buf: object) -> pd.DataFrame:
            try:
                df0 = pd.read_csv(buf, encoding="utf-8-sig")
            except UnicodeDecodeError:
                buf.seek(0)
                df0 = pd.read_csv(buf, encoding="latin-1")
            # European Excel often exports with ";" — reads as 1 column with comma separator
            if len(df0.columns) == 1:
                buf.seek(0)
                try:
                    df1 = pd.read_csv(buf, encoding="utf-8-sig", sep=";")
                except Exception:  # noqa: BLE001
                    buf.seek(0)
                    df1 = pd.read_csv(buf, encoding="latin-1", sep=";")
                if len(df1.columns) >= 2:
                    return df1
            return df0

        raw = _read_uploaded_csv(up)
        if raw.empty:
            st.error("The uploaded CSV has **no rows**. Export again or check the file.")
            st.stop()

    st.sidebar.subheader("Column mapping")
    cols = list(raw.columns)
    if len(cols) < 2:
        st.error(
            "Your file needs **at least two columns**: one for **dates** and one for **demand/quantity**. "
            "A single-column export cannot run this pipeline."
        )
        st.stop()
    d_idx = _default_index(cols, ["date", "ds", "day", "timestamp"])
    dcol = st.sidebar.selectbox("Date column", cols, index=d_idx)
    q_candidates = [c for c in cols if c != dcol]
    if not q_candidates:
        st.error("Pick a **different** column for demand than for date.")
        st.stop()
    q_idx = _default_index(
        q_candidates,
        ["demand", "qty", "quantity", "sales", "units", "y", "value"],
    )
    qcol = st.sidebar.selectbox("Demand / quantity column", q_candidates, index=q_idx)
    if dcol == qcol:
        st.error("**Date** and **demand** cannot be the same column — the app would wipe your dates.")
        st.stop()
    sku_candidates = [c for c in cols if c not in {dcol, qcol}]
    sku_col = st.sidebar.selectbox("SKU column (optional)", ["— none —"] + sku_candidates, index=0)
    sku_col = None if sku_col == "— none —" else sku_col
    oh_candidates = [c for c in cols if c not in {dcol, qcol, sku_col}]
    oh_col = st.sidebar.selectbox("On-hand column (optional)", ["— none —"] + oh_candidates, index=0)
    oh_col = None if oh_col == "— none —" else oh_col
    if not sku_candidates:
        st.sidebar.caption("No extra column available for SKU (using single-series mode).")
    if not oh_candidates:
        st.sidebar.caption("No extra column available for on-hand inventory.")

    if sku_col and sku_col in {dcol, qcol}:
        st.error("SKU column must be different from the selected date and demand columns.")
        st.stop()
    if oh_col and oh_col in {dcol, qcol}:
        st.error("On-hand column must be different from the selected date and demand columns.")
        st.stop()
    if sku_col and oh_col and sku_col == oh_col:
        st.error("SKU column and on-hand column must be different.")
        st.stop()

    upload_preview = raw.copy() if src == "Upload CSV" else None
    try:
        raw = normalize_retail_dataframe(raw, dcol, qcol, sku_col, oh_col)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    raw = raw.rename(columns={dcol: DATE_COL, qcol: TARGET})
    if sku_col:
        raw = raw.rename(columns={sku_col: "sku"})
    if oh_col:
        raw = raw.rename(columns={oh_col: "on_hand"})
    if raw.columns.duplicated().any():
        # Keep the first occurrence when header names collide after renaming.
        raw = raw.loc[:, ~raw.columns.duplicated(keep="first")].copy()
        st.sidebar.caption("Duplicate column names detected; kept the first instance per name.")

    if src == "Upload CSV" and len(raw) == 0:
        st.error(
            "**All rows were dropped** after parsing. Common causes:\n"
            "- Dates not recognized (use `YYYY-MM-DD`, `DD/MM/YYYY`, or Excel date columns).\n"
            "- Quantity column is text or empty → cannot convert to numbers.\n"
            "- Wrong **date** / **demand** columns selected in the sidebar.\n\n"
            "Preview of your **original** upload (check column names and cell formats):"
        )
        if upload_preview is not None:
            st.dataframe(upload_preview.head(20), use_container_width=True)
        st.stop()

    aggregate = False
    if "sku" in raw.columns and raw["sku"].nunique() > 1:
        aggregate = st.sidebar.checkbox("Aggregate all SKUs into one national series", value=False)

    def _resolve_series_columns(df: pd.DataFrame) -> tuple[str, str]:
        date_key = DATE_COL if DATE_COL in df.columns else (dcol if dcol in df.columns else "")
        target_key = TARGET if TARGET in df.columns else (qcol if qcol in df.columns else "")

        # Fallback: infer a date-like column if the chosen key disappeared after parsing.
        if not date_key:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    date_key = c
                    break
            if not date_key:
                for c in df.columns:
                    parsed = pd.to_datetime(df[c], errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        date_key = c
                        break

        # Fallback: infer a numeric demand-like column.
        if not target_key:
            numeric_candidates = [c for c in df.columns if c != date_key and pd.api.types.is_numeric_dtype(df[c])]
            if numeric_candidates:
                target_key = numeric_candidates[0]

        if not date_key or not target_key:
            st.error(
                "Could not identify usable date/demand columns after normalization. "
                "Please verify your sidebar mapping."
            )
            st.write("Available columns:", list(df.columns))
            st.stop()
        return date_key, target_key

    working = raw
    if aggregate and "sku" in working.columns:
        work_date_col, work_target_col = _resolve_series_columns(working)
        working = aggregate_skus_to_total(working, work_date_col, work_target_col, "sku")
        if work_date_col != DATE_COL:
            working = working.rename(columns={work_date_col: DATE_COL})
    elif "sku" in working.columns:
        work_date_col, work_target_col = _resolve_series_columns(working)
        skus = sorted(working["sku"].astype(str).unique())
        pick = st.sidebar.selectbox("SKU for forecasting", skus)
        working = working[working["sku"].astype(str) == pick][[work_date_col, work_target_col]].copy()
        working = working.rename(columns={work_date_col: DATE_COL, work_target_col: TARGET})

    # Ensure downstream feature engineering always receives canonical names.
    work_date_col, work_target_col = _resolve_series_columns(working)
    if work_date_col != DATE_COL or work_target_col != TARGET:
        working = working.rename(columns={work_date_col: DATE_COL, work_target_col: TARGET})
    # Keep only forecasting columns to avoid dropna() being dominated by optional sparse fields
    # like on_hand, which are not part of the modeling signal.
    working = working[[DATE_COL, TARGET]].copy()

    if len(working) == 0:
        st.error(
            "**No rows** in the working series (after SKU filter or aggregation). "
            "Choose another SKU or re-upload data."
        )
        st.stop()

    n_raw = len(working)
    feat_df = build_full_feature_table(working, DATE_COL, TARGET)
    feature_cols = [c for c in feat_df.columns if c not in (DATE_COL, TARGET)]
    feat_clean = feat_df.dropna().reset_index(drop=True)
    n_clean = len(feat_clean)

    MIN_CLEAN = 35
    if n_clean < MIN_CLEAN:
        if n_raw == 0:
            st.error(
                "**No usable daily rows** in the working series (`n_raw = 0`). "
                "That usually means the wrong **date** / **demand** columns, all dates or quantities invalid, "
                "or a CSV that opened as one column (try saving as **comma-separated** or **semicolon-separated** UTF-8)."
            )
        else:
            st.error(
                f"**Not enough rows after feature engineering.** "
                f"Raw days in series: **{n_raw}** → rows with full features (no NaNs): **{n_clean}**. "
                f"Need at least **{MIN_CLEAN}**.\n\n"
                "**Try:** more daily history (fewer gaps), one row per date, numeric demand, "
                "or pick a SKU with a longer series. For short files, use at least ~50–60 calendar days."
            )
        with st.expander("Debug: why rows are dropped"):
            st.write(
                "Lag/rolling features need past values; the first days have missing lags and are removed by `dropna()`. "
                "The app shortens lag windows on shorter series automatically."
            )
            st.dataframe(feat_df.head(40), use_container_width=True)
        st.stop()

    if n_clean < 80:
        st.warning(
            f"Short history: **{n_clean}** usable days (from **{n_raw}** raw). "
            "Use a **smaller hold-out** in Forecast & models (e.g. 14–21 days)."
        )

    # Data quality panel in sidebar for fast operational confidence.
    min_d = pd.to_datetime(working[DATE_COL]).min()
    max_d = pd.to_datetime(working[DATE_COL]).max()
    span_days = int((max_d - min_d).days + 1) if pd.notna(min_d) and pd.notna(max_d) else 0
    unique_days = int(pd.to_datetime(working[DATE_COL]).nunique())
    coverage = float(unique_days / span_days) if span_days > 0 else 0.0
    dup_days = int(pd.to_datetime(working[DATE_COL]).duplicated().sum())
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data quality check")
    st.sidebar.caption(
        f"Rows: **{n_raw}** | Usable rows: **{n_clean}**\n\n"
        f"Date span: **{span_days} days** | Coverage: **{coverage:.1%}** | Duplicate dates: **{dup_days}**"
    )

    st.session_state["work"] = {
        "raw": raw,
        "working": working,
        "feat_clean": feat_clean,
        "feature_cols": feature_cols,
        "has_sku": "sku" in raw.columns and not aggregate,
    }

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["Executive snapshot", "Forecast & models", "Inventory & shopping list", "Financial impact", "Data preview"]
    )

    with tab0:
        st.subheader("Executive snapshot")
        y_all = working[TARGET].astype(float)
        mean_d = float(y_all.mean()) if len(y_all) else 0.0
        std_d = float(y_all.std(ddof=0)) if len(y_all) else 0.0
        cv = float(std_d / mean_d) if mean_d > 1e-12 else 0.0
        recent_n = min(30, max(7, len(y_all) // 3))
        prev_n = min(recent_n, len(y_all) - recent_n)
        if prev_n > 0:
            prev_mean = float(y_all.iloc[-(recent_n + prev_n) : -recent_n].mean())
            recent_mean = float(y_all.iloc[-recent_n:].mean())
            trend_pct = (recent_mean - prev_mean) / prev_mean * 100.0 if abs(prev_mean) > 1e-12 else 0.0
        else:
            trend_pct = 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg daily demand", f"{mean_d:.1f}")
        k2.metric("Volatility (CV)", f"{cv:.2f}")
        k3.metric("Recent trend", f"{trend_pct:+.1f}%")
        k4.metric("Usable training days", f"{n_clean}")

        readiness = "High" if n_clean >= 120 else ("Medium" if n_clean >= 60 else "Low")
        st.info(
            f"Forecast readiness: **{readiness}**. "
            "For best performance, keep at least ~120 clean daily rows with consistent date coverage."
        )

        with st.expander("What this means for decisions"):
            st.write(
                "- Higher volatility (CV) implies wider safety stock buffers.\n"
                "- Positive trend suggests stock-up pressure; negative trend suggests tighter replenishment.\n"
                "- Coverage below ~90% may weaken seasonality and lag-based feature quality."
            )

    with tab4:
        st.subheader("Working series (forecast input)")
        st.dataframe(working.tail(20), use_container_width=True)
        if "sku" in raw.columns and not aggregate:
            st.caption("Forecasting uses the SKU selected in the sidebar.")
        st.download_button(
            "Download engineered features (sample)",
            feat_clean.head(500).to_csv(index=False).encode("utf-8"),
            "features_sample.csv",
            mime="text/csv",
        )

    with tab1:
        st.subheader("Hold-out model comparison")
        st.caption(f"Usable series length: **{n_clean}** days (after lags/rolling). Max hold-out ≈ **{max(7, n_clean - 36)}**.")
        c1, c2, c3 = st.columns(3)
        _max_h = max(7, min(120, n_clean - 36))
        _def_ts = min(30, _max_h)
        with c1:
            test_size = int(
                st.number_input(
                    "Hold-out days",
                    min_value=7,
                    max_value=max(7, _max_h),
                    value=_def_ts,
                    key="ts",
                    help="Training window needs ~35+ days beyond the hold-out.",
                )
            )
        with c2:
            lstm_ep = int(st.number_input("LSTM epochs (lower = faster)", 5, 80, 25, key="lstm_ep"))
        with c3:
            run_cmp = st.button("Run hybrid comparison", type="primary")

        if run_cmp:
            with st.spinner("Training SARIMA, XGBoost, LightGBM, LSTM — please wait…"):
                cmp = compare_models(
                    feat_clean[TARGET],
                    feat_clean,
                    feature_cols,
                    TARGET,
                    test_size=test_size,
                    lstm_epochs=lstm_ep,
                )
            st.session_state["cmp"] = cmp
            best_name, best_m = select_best_model(cmp)
            st.session_state["best_model"] = best_name
            st.session_state["best_metrics"] = best_m

        if "cmp" in st.session_state:
            cmp = st.session_state["cmp"]
            y_test = cmp["y_test"]
            idx = cmp["test_index"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx, y=y_test, name="Actual", mode="lines+markers", marker=dict(size=3)))
            for name, payload in cmp["models"].items():
                if payload.get("metrics"):
                    m = payload["metrics"]
                    fig.add_trace(
                        go.Scatter(
                            x=idx,
                            y=payload["pred"],
                            name=f"{name} (WMAPE={m['WMAPE']:.1f}%)",
                            mode="lines",
                        )
                    )
                elif payload.get("error"):
                    st.warning(f"{name}: {payload['error']}")
            fig.update_layout(
                title="Hold-out forecasts",
                xaxis_title="Time index",
                yaxis_title="Demand",
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            mdf = comparison_to_dataframe(cmp)
            st.dataframe(mdf, use_container_width=True)
            st.download_button(
                "Download metrics CSV",
                mdf.to_csv(index=False).encode("utf-8"),
                "model_comparison_metrics.csv",
                mime="text/csv",
            )

            bn, bm = select_best_model(cmp)
            if bn and bm:
                st.success(f"**Best model (lowest WMAPE):** {bn} — WMAPE={bm['WMAPE']:.2f}%")

        st.subheader("Time-series cross-validation (boosting)")
        cv_model = st.selectbox("Boosting model for CV", ["xgboost", "lightgbm"], index=0)
        n_splits = int(st.slider("CV folds", 3, 10, 5))
        if st.button("Run expanding-window CV"):
            with st.spinner("Cross-validating…"):
                cv = time_series_cv_boosting(
                    feat_clean, feature_cols, TARGET, n_splits=n_splits, model=cv_model
                )
            st.success(
                f"{cv.model_name} — Mean MAE={cv.mean_metrics['MAE']:.3f}, "
                f"RMSE={cv.mean_metrics['RMSE']:.3f}, WMAPE={cv.mean_metrics['WMAPE']:.2f}%"
            )

    with tab2:
        st.markdown(
            r"**Inventory policy (95% service level):** "
            r"$SS = z\sigma_L$, $ROP=\mu_L+SS$, $Q^*=\sqrt{2DS/H}$."
        )
        lead = float(st.number_input("Lead time (days)", 1, 90, 7, key="lead"))
        order_cost = float(st.number_input("Ordering cost per order ($)", 1.0, 500.0, 50.0, key="oc"))
        hold_h = float(st.number_input("Holding cost $ / unit / year", 0.1, 100.0, 2.0, key="hh"))
        svc = float(st.slider("Service level", 0.90, 0.99, 0.95, 0.01))

        mu_source = st.radio(
            "Expected daily demand μ for policy",
            ["From best model (hold-out mean prediction)", "Recent history (last 30 days)"],
            horizontal=True,
        )
        y = feat_clean[TARGET].values
        use_best = (
            mu_source.startswith("From best")
            and "cmp" in st.session_state
            and st.session_state.get("best_model")
        )
        if use_best:
            mu = forecast_mean_daily_from_holdout(st.session_state["cmp"], st.session_state["best_model"])
            if not np.isfinite(mu):
                st.warning("Could not read best-model forecast — using recent history for μ.")
                mu, _ = historical_mu_sigma(y, 30)
            st.caption("μ = mean of hold-out predictions from the winning model (Tab 1).")
        else:
            mu, _ = historical_mu_sigma(y, 30)
            st.caption("μ, σ from last 30 days of observed demand.")

        _, sigma = historical_mu_sigma(y, 30)

        pol = full_policy(mu, sigma, lead, order_cost, hold_h, service_level=float(svc))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Safety stock", f"{pol.safety_stock:.1f}")
        m2.metric("Reorder point", f"{pol.reorder_point:.1f}")
        m3.metric("EOQ", f"{pol.eoq:.1f}" if np.isfinite(pol.eoq) else "—")
        m4.metric("z", f"{pol.z_value:.3f}")

        st.subheader("Shopping list")

        inv_raw = st.session_state["work"]["raw"]
        has_sku = st.session_state["work"]["has_sku"]

        if has_sku and inv_raw["sku"].nunique() > 1:
            sku_stats: dict[str, tuple[float, float]] = {}
            for sku, g in inv_raw.groupby("sku"):
                arr = g.sort_values(DATE_COL)[TARGET].values.astype(float)
                if len(arr) < 14:
                    continue
                sku_stats[str(sku)] = historical_mu_sigma(arr, 30)

            policies: dict = {}
            forecast_means: dict[str, float] = {}
            for sku, (m_u, s_u) in sku_stats.items():
                policies[sku] = full_policy(
                    m_u, s_u, lead, order_cost, hold_h, service_level=float(svc)
                )
                forecast_means[sku] = m_u

            if "on_hand" in inv_raw.columns:
                oh_map = latest_on_hand_by_sku(inv_raw, "sku", "on_hand")
            else:
                st.caption("No **on_hand** column — using illustrative stock at 45% of each SKU’s ROP.")
                oh_map = {sku: policies[sku].reorder_point * 0.45 for sku in policies}

            for sku in oh_map:
                if sku not in policies:
                    policies[sku] = full_policy(mu, sigma, lead, order_cost, hold_h, service_level=float(svc))
                    forecast_means[sku] = mu

            sl = shopping_list(forecast_means, oh_map, policies)
            st.dataframe(sl, use_container_width=True)
            priority = sl[sl["below_ROP"]].copy()
            if len(priority):
                st.warning("**Below ROP today — prioritize:**")
                st.dataframe(priority, use_container_width=True)
            csv_bytes = sl.to_csv(index=False).encode("utf-8")
            st.download_button("Download shopping list CSV", csv_bytes, "shopping_list.csv", mime="text/csv")
        else:
            oh_min, oh_max = 0.0, 1e7
            oh_default = float(np.clip(pol.reorder_point * 0.5, oh_min, oh_max))
            # Clamp persisted widget state so a previous out-of-range value cannot crash reruns.
            if "oh" in st.session_state:
                st.session_state["oh"] = float(np.clip(st.session_state["oh"], oh_min, oh_max))
            oh_in = float(
                st.number_input("Current on-hand (units)", oh_min, oh_max, oh_default, key="oh")
            )
            single = {"PRIMARY": mu}
            pol_map = {"PRIMARY": pol}
            oh_map = {"PRIMARY": oh_in}
            sl = shopping_list(single, oh_map, pol_map)
            st.dataframe(sl, use_container_width=True)
            st.download_button(
                "Download shopping list CSV",
                sl.to_csv(index=False).encode("utf-8"),
                "shopping_list.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("Stockout vs overstock (simulation)")
        st.caption(
            "Toy ledger: each day, order up to forecast; compare naive rolling baseline vs XGBoost tail forecast."
        )
        tail = min(120, max(20, len(feat_clean) - 26))
        try:
            y_act, y_xgb = xgboost_forecast_tail(feat_clean, feature_cols, TARGET, tail=tail)
            base = naive_rolling_forecast_series(y_act, 7)
            fp = FinancialParams(
                unit_cost=float(st.number_input("Unit cost ($)", 0.1, 500.0, 5.0, key="uc")),
                selling_price=float(st.number_input("Selling price ($)", 0.5, 1000.0, 12.0, key="sp")),
                holding_cost_rate_per_year=float(
                    st.slider("Holding cost rate (% of unit cost / year)", 5, 50, 25, key="hr")
                )
                / 100.0,
                goodwill_stockout_penalty_per_unit=float(
                    st.number_input("Goodwill penalty $ / lost sale (optional)", 0.0, 50.0, 0.0, key="gw")
                ),
            )
            out = compare_scenarios(
                actual_demand=y_act,
                baseline_forecast=base,
                improved_forecast=y_xgb,
                starting_inventory=float(np.mean(y_act)),
                params=fp,
                fraction_of_year_per_step=1.0 / 365.0,
            )
            st.dataframe(out, use_container_width=True)

            tot_b = float(out.loc[out["scenario"] == "Baseline forecast", "total_cost"].iloc[0])
            tot_i = float(out.loc[out["scenario"] == "Improved forecast", "total_cost"].iloc[0])
            impl = float(st.number_input("One-time implementation cost ($)", 0.0, 1e6, 0.0))
            maint = float(st.number_input("Annual software maintenance ($)", 0.0, 1e6, 0.0))
            roi = project_roi_summary(tot_b * 3.0, tot_i * 3.0, impl, maint)
            st.subheader("Project ROI (scaled 3× for rough annualization of the simulated window)")
            st.json(roi)
        except Exception as ex:  # noqa: BLE001
            st.warning(f"Financial simulation needs enough history: {ex}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Run locally:** `python -m streamlit run streamlit_app.py`")


if __name__ == "__main__":
    main()
