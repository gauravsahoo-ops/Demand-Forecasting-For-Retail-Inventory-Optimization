# Demand Forecasting for Retail Inventory Optimization

Final-year style system: **feature engineering** (lags, rolling stats, harmonic seasonality), **hybrid forecasting** (SARIMA, XGBoost, LightGBM, LSTM), **inventory policy** (safety stock, ROP, EOQ), **financial comparison**, and a **Streamlit** dashboard with a downloadable **shopping list**.

## Requirements

- Python 3.10–3.12 recommended (TensorFlow for LSTM may not support Python 3.13 yet).

## Setup

```powershell
cd "c:\Project 1"
python -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
python -m streamlit run streamlit_app.py
```

If `streamlit` is not on your PATH, `python -m streamlit` still works.

## Deploy on Streamlit Community Cloud

1. Push the **entire** project to a GitHub repository — not only `streamlit_app.py`. You must include:
   - `streamlit_app.py`
   - `requirements.txt`
   - **All Python modules beside the app** (e.g. `data_demo.py`, `data_io.py`, `feature_engineering.py`, `forecasting.py`, `inventory.py`, `financial.py`, `metrics.py`, `pipeline.py`, `multi_sku.py`). They live in the **repo root** so they are obvious to commit and upload.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and click **New app**.
3. Select your repository, branch (usually `main`), and file path: `streamlit_app.py`.
4. Deploy.

If you see `ModuleNotFoundError`, open your repo on GitHub and confirm every `.py` file above is present (easy miss: only `streamlit_app.py` was uploaded).

### Recommended for reliable cloud builds

- Use Python 3.10-3.12 in your deployment settings.
- Keep `tensorflow` commented out in `requirements.txt` unless you explicitly need LSTM in cloud.
- If LSTM is unavailable, the app still runs with SARIMA, XGBoost, and LightGBM.

### First-time Git commands (PowerShell)

```powershell
cd "c:\Project 1"
git init
git add .
git commit -m "Initial Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Offline demo (no UI)

```powershell
python demo_run.py
```

## Sample data

- `data/sample_retail_daily.csv` — long format: `date`, `sku`, `demand`, `on_hand`
- Regenerate: `python scripts/export_sample_data.py`

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Project layout

| Path | Purpose |
|------|---------|
| `feature_engineering.py` | Calendar, harmonics, lags, past-only rolling |
| `metrics.py` | MAE, RMSE, WMAPE |
| `forecasting.py` | SARIMA, XGBoost, LightGBM, LSTM, CV |
| `inventory.py` | Safety stock, ROP, EOQ, shopping list |
| `financial.py` | Stockout vs holding cost scenarios |
| `data_io.py` | CSV normalization |
| `pipeline.py` | Best-model selection, tail XGB forecast for finance |
| `data_demo.py` | Synthetic single- and multi-SKU series |
| `streamlit_app.py` | Manager UI |

## Viva notes

State assumptions clearly: i.i.d. demand for \(\sigma_L = \sigma_d\sqrt{L}\), normal tail for \(z\); EOQ deterministic baseline; LSTM recursive multi-step error growth; financial tab is an illustrative ledger, not full enterprise simulation.
