"""Quick offline demo: prints metrics for hybrid comparison (no Streamlit)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_demo import make_demo_sales
from feature_engineering import build_full_feature_table
from forecasting import compare_models


def main() -> None:
    raw = make_demo_sales(n_days=450, seed=42)
    feat = build_full_feature_table(raw, "date", "demand").dropna().reset_index(drop=True)
    cols = [c for c in feat.columns if c not in ("date", "demand")]
    out = compare_models(feat["demand"], feat, cols, "demand", test_size=30, lstm_epochs=15)
    print("Hold-out metrics:")
    for name, payload in out["models"].items():
        if payload.get("metrics"):
            print(f"  {name}: {payload['metrics']}")
        else:
            print(f"  {name}: ERROR {payload.get('error')}")


if __name__ == "__main__":
    main()
