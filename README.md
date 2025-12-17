# Gold Regime Insight Engine

Builds a small set of gold/macro indicators into JSON files and PNG charts for use in dashboards, reports, or simple static sites.

The entrypoint script downloads market data, computes a few regime/ratio signals, and writes outputs into:

- [data/](data/) (machine-readable JSON)
- [charts/](charts/) (PNG charts)

## Data Sources

- Yahoo Finance via `yfinance` (GLD, SPY, IEF, DXY, ^VIX, ^VVIX, ^TNX)
- FRED (St. Louis Fed) for credit spread series (HY OAS). The script uses `pandas_datareader` when available and falls back to downloading the FRED CSV directly.

## Quickstart

Prerequisites: Python 3.10+.

1) Create a virtual environment and install dependencies

`python -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements.txt`

2) Generate / refresh outputs

`python src/build_gold_insights.py`

After it finishes, open [data/](data/) and [charts/](charts/) to see the refreshed artifacts.

## Notes

- Outputs are overwritten each run.
- Market data availability can vary by ticker and date; if a series is unavailable, the corresponding chart/JSON may be empty.