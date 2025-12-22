# Gold Regime Insight Engine

**Live dashboard:** https://gold-monitor.lovable.app

Generates a small set of gold + macro indicators as JSON files and PNG charts for use in dashboards, reports, or simple static sites.

The entrypoint script downloads market data, computes a handful of cross-asset views, and writes outputs into:

- [data/](data/) (machine-readable JSON)
- [charts/](charts/) (PNG charts)

The dashboard primarily uses:

- Gold vs USD (12M)
- Gold vs rates (12M)
- GLD/SPY ratio (12M)
- Gold/Oil ratio (12M)
- Credit risk premium (20Y)
- Gold value trend (since 2005)

For convenience, [data/latest.json](data/latest.json) acts as a single “manifest” pointing to the latest chart + JSON paths.

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