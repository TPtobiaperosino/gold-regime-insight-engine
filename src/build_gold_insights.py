from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Config:
    gold: str = "GLD"
    usd: str = "DX-Y.NYB"
    rates: str = "IEF"
    equity: str = "SPY"
    teny: str = "^TNX"

    momentum_weeks: int = 12
    

    history_period: str = "max"

    latest_path: str = "data/latest.json"

    gld_vs_usd_png_path: str = "charts/gld_vs_usd_12m.png"
    gld_vs_usd_json_path: str = "data/gld_vs_usd_12m.json"
    gld_vs_teny_png_path: str = "charts/gld_vs_10y_12m.png"
    gld_vs_teny_json_path: str = "data/gld_vs_10y_12m.json"

    regime_snapshot_png_path: str = "charts/regime_snapshot.png"
    regime_snapshot_json_path: str = "data/regime_snapshot.json"

    gld_spy_ratio_png_path: str = "charts/gld_spy_ratio_12m.png"
    gld_spy_ratio_json_path: str = "data/gld_spy_ratio_12m.json"

    gold_oil_ratio_png_path: str = "charts/gold_oil_ratio_12m.png"
    gold_oil_ratio_json_path: str = "data/gold_oil_ratio_12m.json"

    credit_risk_premium_20y_png_path: str = "charts/credit_risk_premium_20y.png"
    credit_risk_premium_20y_json_path: str = "data/credit_risk_premium_20y.json"

    gold_value_trend_2005_png_path: str = "charts/gold_value_trend_2005.png"
    gold_value_trend_2005_json_path: str = "data/gold_value_trend_2005.json"


CFG = Config()


def download_prices(tickers: List[str], period: str, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                group_by="column",
                auto_adjust=False,
                progress=False,
                threads=True,
            )

            if df.empty:
                raise RuntimeError("Download returned empty dataframe.")

            if "Adj Close" in df.columns:
                px = df["Adj Close"].copy()
            elif "Close" in df.columns:
                px = df["Close"].copy()
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    fields = df.columns.get_level_values(0).unique().tolist()
                    if "Adj Close" in fields:
                        px = df["Adj Close"].copy()
                    elif "Close" in fields:
                        px = df["Close"].copy()
                    else:
                        raise RuntimeError(f"Neither Adj Close nor Close found. Fields={fields}")
                else:
                    raise RuntimeError(f"Unexpected columns: {df.columns}")

            if isinstance(px, pd.Series):
                px = px.to_frame(name=tickers[0])

            px = px.sort_index()
            px = px.rename(columns=str)
            return px

        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(1.5 + random.random() * 1.5)

    raise RuntimeError("Unreachable.")


def align_and_clean(px: pd.DataFrame) -> pd.DataFrame:
    px = px.copy()
    px = px.sort_index()
    # Avoid global dropna across mixed calendars (indices/ETFs). Keep rows where at least
    # one ticker has data and bridge small gaps only.
    px = px.dropna(how="all")
    px = px.ffill(limit=5)
    return px


def to_weekly_prices(px_daily: pd.DataFrame) -> pd.DataFrame:
    weekly = px_daily.resample("W-FRI").last().dropna(how="all")
    if weekly.index[-1] > px_daily.index[-1]:
        weekly = weekly.iloc[:-1]
    return weekly


def pct_return(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna(how="all")


def momentum(series_weekly_price: pd.Series, weeks: int) -> pd.Series:
    return (series_weekly_price / series_weekly_price.shift(weeks) - 1).dropna()


def current_regime(px_weekly: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    w = cfg.momentum_weeks

    mom_usd = momentum(px_weekly[cfg.usd], w)
    mom_rates = momentum(px_weekly[cfg.rates], w)

    common_idx = mom_usd.index.intersection(mom_rates.index)
    mom_usd = mom_usd.loc[common_idx]
    mom_rates = mom_rates.loc[common_idx]

    last_dt = common_idx.max() if not common_idx.empty else None
    if last_dt is None:
        return {
            "asof": None,
            "usd": "unknown",
            "rates": "unknown",
            "mom_usd_12w": float("nan"),
            "mom_rates_12w": float("nan"),
        }

    usd_strong = bool(mom_usd.loc[last_dt] > 0)
    rates_up = bool(mom_rates.loc[last_dt] < 0)

    return {
        "asof": last_dt.date().isoformat(),
        "usd": "strong" if usd_strong else "weak",
        "rates": "up" if rates_up else "down",
        "mom_usd_12w": float(mom_usd.loc[last_dt]),
        "mom_rates_12w": float(mom_rates.loc[last_dt]),
    }


def build_regime_snapshot(px_weekly: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    """Build a compact regime snapshot aligned to the last common weekly date.

    Returns the structure requested by the UI contract (label + metrics).
    """
    w = cfg.momentum_weeks

    required = [cfg.usd, cfg.rates]
    if not all(r in px_weekly.columns for r in required):
        return {
            "label": "USD unknown · Bond prices unknown (IEF)",
            "metrics": {
                "usd_momentum_12w_pct": None,
                "rates_momentum_12w_pct": None,
            },
        }

    idx = px_weekly[cfg.usd].dropna().index
    idx = idx.intersection(px_weekly[cfg.rates].dropna().index)
    if idx.empty:
        return {
            "label": "USD unknown · Bond prices unknown (IEF)",
            "metrics": {
                "usd_momentum_12w_pct": None,
                "rates_momentum_12w_pct": None,
            },
        }

    last_dt = idx.max()

    usd_s = px_weekly[cfg.usd].dropna()
    if last_dt in usd_s.index and (usd_s.index.get_loc(last_dt) - w) >= 0:
        usd_mom = float(usd_s.loc[last_dt] / usd_s.shift(w).loc[last_dt] - 1)
    else:
        usd_mom = float("nan")
    usd_label = "strong" if usd_mom > 0 else "weak"

    rates_s = px_weekly[cfg.rates].dropna()
    if last_dt in rates_s.index and (rates_s.index.get_loc(last_dt) - w) >= 0:
        rates_mom = float(rates_s.loc[last_dt] / rates_s.shift(w).loc[last_dt] - 1)
    else:
        rates_mom = float("nan")
    rates_label = "up" if rates_mom < 0 else "down"

    # cfg.rates is IEF (bond price). We keep the historical bucketing key as-is,
    # but present it as bond prices (inverse-to-yields).
    bond_label = "down" if rates_label == "up" else "up"

    snapshot = {
        "label": f"USD {usd_label} · Bond prices {bond_label} (IEF)",
        "metrics": {
            "usd_momentum_12w_pct": round(usd_mom * 100.0, 2) if np.isfinite(usd_mom) else None,
            "rates_momentum_12w_pct": round(rates_mom * 100.0, 2) if np.isfinite(rates_mom) else None,
        },
    }

    return snapshot

def generate_insights(regime_meta: Dict[str, str]) -> List[Dict[str, object]]:
    usd = regime_meta["usd"]
    rates = regime_meta["rates"]

    bond_bucket = "unknown"
    if rates in ("up", "down"):
        bond_bucket = "down" if rates == "up" else "up"

    return [
        {
            "title": "Current macro regime",
            "evidence": f"USD={usd}, Bond prices (IEF)={bond_bucket} (asof {regime_meta['asof']})",
            "metric": {},
        }
    ]


def _sanitize_for_json(x: object) -> object:
    """Convert non-JSON-safe values (NaN/Inf, numpy scalars) into valid JSON values.

    Browsers' Response.json() rejects NaN/Infinity tokens, so we normalize them to null.
    """
    if x is None or isinstance(x, (str, bool, int)):
        return x

    # numpy scalars
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)

    # floats (catch NaN/Inf)
    if isinstance(x, float):
        return x if np.isfinite(x) else None

    if isinstance(x, dict):
        return {str(k): _sanitize_for_json(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_sanitize_for_json(v) for v in x]

    # pandas / numpy objects sometimes expose .item()
    try:
        item = getattr(x, "item", None)
        if callable(item):
            return _sanitize_for_json(item())
    except Exception:
        pass

    return x


def write_json(path: str, payload: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        safe_payload = _sanitize_for_json(payload)
        json.dump(safe_payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def _is_finite(x) -> bool:
    try:
        return x is not None and np.isfinite(x)
    except Exception:
        return False


def fmt_pct(x: float, decimals: int = 1) -> str | None:
    if not _is_finite(x):
        return None
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x: float, decimals: int = 2) -> str | None:
    if not _is_finite(x):
        return None
    return f"{x:.{decimals}f}"


def _tnx_to_yield_percent_with_meta(tnx_series: pd.Series) -> tuple[pd.Series, Dict[str, object]]:
    """Convert yfinance ^TNX quotes to 10Y yield in percent.

    yfinance commonly returns ^TNX as yield * 10 (e.g., 45.0 ~= 4.50%).
    This helper auto-detects scaling so we don't accidentally divide twice.

    Rules (based on last finite value):
      - 10..80   -> divide by 10  ("/10")
      - 0..2     -> treat as decimal yield (e.g., 0.045) and multiply by 100 ("*100")
      - else     -> assume already percent ("none")
    """
    s = pd.Series(tnx_series).dropna().astype(float)
    if s.empty:
        return s, {"scaling": "unknown", "tnx_raw_last": None, "yield_pct_last": None}

    raw_last = float(s.iloc[-1])
    if 10.0 <= raw_last <= 80.0:
        scaling = "/10"
        y = s / 10.0
    elif 0.0 < raw_last < 2.0:
        scaling = "*100"
        y = s * 100.0
    else:
        scaling = "none"
        y = s

    meta = {
        "scaling": scaling,
        "tnx_raw_last": raw_last,
        "yield_pct_last": float(y.iloc[-1]) if not y.empty else None,
    }
    return y.rename("teny_yield_pct"), meta


def tnx_to_yield_percent(tnx_series: pd.Series) -> pd.Series:
    """Convert ^TNX quotes to yield in percent."""
    y, _meta = _tnx_to_yield_percent_with_meta(tnx_series)
    return y


def _format_quarter_axis(ax) -> None:
    """Format x-axis to show quarters (Q1'24, Q2'24, etc.)."""
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter

    def quarter_formatter(x, pos):
        try:
            dt = mdates.num2date(x)
            q = (dt.month - 1) // 3 + 1
            return f"Q{q}'{dt.year % 100:02d}"
        except Exception:
            return ""

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(quarter_formatter))
    ax.tick_params(axis="x", rotation=0)


def save_gld_vs_usd_12m_chart(px_daily: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Save GLD vs DXY chart (12m) and JSON series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = px_daily.index.max()
    start = end - pd.DateOffset(years=1)

    gld = px_daily.loc[px_daily.index >= start, cfg.gold].dropna()
    dxy = px_daily.loc[px_daily.index >= start, cfg.usd].dropna()

    common_idx = gld.index.intersection(dxy.index)
    gld = gld.reindex(common_idx).dropna()
    dxy = dxy.reindex(common_idx).dropna()

    if gld.empty or dxy.empty:
        Path(cfg.gld_vs_usd_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.set_title("GLD vs USD (DXY) — last 12 months (dual-axis)")
        fig.tight_layout()
        fig.savefig(cfg.gld_vs_usd_png_path, dpi=180)
        plt.close(fig)
        write_json(cfg.gld_vs_usd_json_path, {"series": []})
        return {"png_path": cfg.gld_vs_usd_png_path, "json_path": cfg.gld_vs_usd_json_path}

    gld_color = "#1b6ca8"
    dxy_color = "#d35400"

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ln1 = ax.plot(gld.index, gld.values, label="GLD ($, left axis)", color=gld_color, linewidth=1.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("GLD ($)", color=gld_color)
    ax.tick_params(axis="y", colors=gld_color)
    for tl in ax.get_yticklabels():
        tl.set_color(gld_color)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

    ax2 = ax.twinx()
    ln2 = ax2.plot(dxy.index, dxy.values, label="DXY (index, right axis)", color=dxy_color, linewidth=1.6)
    ax2.set_ylabel("DXY (index)", color=dxy_color)
    ax2.tick_params(axis="y", colors=dxy_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(dxy_color)

    ax.set_title("GLD vs USD (DXY) — last 12 months (dual-axis)", fontsize=12, weight="bold")

    _format_quarter_axis(ax)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, fontsize=9)

    gld_current = float(gld.iloc[-1])
    dxy_current = float(dxy.iloc[-1])
    gld_12m = float((gld.iloc[-1] / gld.iloc[0] - 1.0) * 100.0) if len(gld) > 1 and gld.iloc[0] != 0 else float("nan")
    dxy_12m = float((dxy.iloc[-1] / dxy.iloc[0] - 1.0) * 100.0) if len(dxy) > 1 and dxy.iloc[0] != 0 else float("nan")
    as_of = pd.to_datetime(common_idx.max()).date().isoformat()

    footnote = (
        f"Current GLD: {gld_current:.2f}  |  Current DXY: {dxy_current:.2f}  |  "
        f"12m Δ GLD: {gld_12m:.2f}%  |  12m Δ DXY: {dxy_12m:.2f}%  |  "
        f"As of {as_of}"
    )
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=8, color="#555555")

    fig.tight_layout(rect=[0, 0.18, 1, 0.92])

    Path(cfg.gld_vs_usd_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.gld_vs_usd_png_path, dpi=180)
    if hasattr(cfg, "gold_12m_png_path"):
        try:
            Path(cfg.gold_12m_png_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(cfg.gold_12m_png_path, dpi=180)
        except Exception:
            pass

    plt.close(fig)

    df_out = pd.DataFrame(
        {
            "date": gld.index.date.astype(str),
            "gld": gld.values.astype(float),
            "dxy": dxy.values.astype(float),
        }
    )
    write_json(cfg.gld_vs_usd_json_path, {"series": df_out.to_dict(orient="records")})

    return {"png_path": cfg.gld_vs_usd_png_path, "json_path": cfg.gld_vs_usd_json_path}


def save_gld_vs_teny_12m_chart(px_daily: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Save GLD and 10Y yield (from ^TNX) as a dual-axis chart (last 12 months).

    Uses auto-detected scaling for ^TNX so yield is always in percent (e.g., 4.5).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = px_daily.index.max()
    start = end - pd.DateOffset(years=1)

    gld = px_daily.loc[px_daily.index >= start, cfg.gold].dropna()
    tnx = px_daily.loc[px_daily.index >= start, cfg.teny].dropna()

    common_idx = gld.index.intersection(tnx.index)
    gld = gld.reindex(common_idx).dropna()
    tnx = tnx.reindex(common_idx).dropna()

    if gld.empty or tnx.empty:
        Path(cfg.gld_vs_teny_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.set_title("GLD vs 10Y yield — last 12 months (dual-axis)")
        fig.tight_layout()
        fig.savefig(cfg.gld_vs_teny_png_path, dpi=180)
        plt.close(fig)
        write_json(cfg.gld_vs_teny_json_path, {"series": []})
        return {"png_path": cfg.gld_vs_teny_png_path, "json_path": cfg.gld_vs_teny_json_path}

    tny_pct, tnx_meta = _tnx_to_yield_percent_with_meta(tnx)

    gld_color = "#1b6ca8"
    tny_color = "#27ae60"

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(gld.index, gld.values, label="GLD ($, left axis)", color=gld_color, linewidth=1.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("GLD ($)", color=gld_color)
    ax.tick_params(axis="y", colors=gld_color)
    for tl in ax.get_yticklabels():
        tl.set_color(gld_color)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

    ax2 = ax.twinx()
    ax2.plot(tny_pct.index, tny_pct.values, label="10Y yield (%), right axis", color=tny_color, linewidth=1.6)
    ax2.set_ylabel("10Y yield (%)", color=tny_color)
    ax2.tick_params(axis="y", colors=tny_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(tny_color)

    ax.set_title("GLD vs 10Y yield — last 12 months (dual-axis)", fontsize=12, weight="bold")

    _format_quarter_axis(ax)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, fontsize=9)

    gld_current = float(gld.iloc[-1])
    y10_current = float(tny_pct.iloc[-1])
    gld_12m = float((gld.iloc[-1] / gld.iloc[0] - 1.0) * 100.0) if len(gld) > 1 and gld.iloc[0] != 0 else float("nan")
    y10_12m_bps = float((tny_pct.iloc[-1] - tny_pct.iloc[0]) * 100.0) if len(tny_pct) > 1 else float("nan")
    as_of = pd.to_datetime(common_idx.max()).date().isoformat()

    footnote = (
        f"Current GLD: {gld_current:.2f}  |  Current 10Y: {y10_current:.2f}%  |  "
        f"12m Δ GLD: {gld_12m:.2f}%  |  12m Δ 10Y: {y10_12m_bps:.0f} bps  |  "
        f"As of {as_of}"
    )
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=8, color="#555555")

    fig.tight_layout(rect=[0, 0.18, 1, 0.92])

    Path(cfg.gld_vs_teny_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.gld_vs_teny_png_path, dpi=180)
    plt.close(fig)

    df_out = pd.DataFrame(
        {
            "date": gld.index.date.astype(str),
            "gld": gld.values.astype(float),
            "teny_yield_pct": tny_pct.values.astype(float),
        }
    )
    write_json(
        cfg.gld_vs_teny_json_path,
        {
            "series": df_out.to_dict(orient="records"),
            "stats": {
                "tnx_raw_last": tnx_meta.get("tnx_raw_last"),
                "yield_pct_last": tnx_meta.get("yield_pct_last"),
                "scaling": tnx_meta.get("scaling"),
            },
        },
    )

    return {"png_path": cfg.gld_vs_teny_png_path, "json_path": cfg.gld_vs_teny_json_path}


def save_regime_snapshot(px_daily: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Create a compact table-style regime snapshot (PNG + JSON).

    Metrics computed using yfinance daily data (with weekly approximations where specified):
    - asof (last available date)
    - regime (USD strengthening/weakening; Yields rising/falling) based on 12-week changes
    - dxy_mom_12w (percent from weekly series)
    - y10_chg_12w_bps (12-week change in 10Y yield, in bps)
    - gld_ret_1w, gld_ret_1m, gld_ret_3m, gld_ret_12m (5d,21d,63d,252d returns)
    - gld_dd_12m (drawdown from 12m high, percent)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(cfg.regime_snapshot_png_path).parent.mkdir(parents=True, exist_ok=True)

    def _pct(x: float, decimals: int = 1) -> str:
        try:
            if x is None or not np.isfinite(x):
                return "NaN"
            return f"{x * 100:.{decimals}f}%"
        except Exception:
            return "NaN"

    def _num(x: float, decimals: int = 2) -> str:
        try:
            if x is None or not np.isfinite(x):
                return "NaN"
            return f"{x:.{decimals}f}"
        except Exception:
            return "NaN"

    asof = px_daily.index.max() if not px_daily.empty else pd.Timestamp("1970-01-01")

    gld = px_daily[cfg.gold].dropna() if cfg.gold in px_daily.columns else pd.Series(dtype=float)
    dxy = px_daily[cfg.usd].dropna() if cfg.usd in px_daily.columns else pd.Series(dtype=float)
    tnx = px_daily[cfg.teny].dropna() if cfg.teny in px_daily.columns else pd.Series(dtype=float)

    try:
        px_weekly = px_daily.resample("W-FRI").last().dropna(how="all")
    except Exception:
        px_weekly = pd.DataFrame()

    dxy_w = px_weekly[cfg.usd].dropna() if cfg.usd in px_weekly.columns else pd.Series(dtype=float)
    if cfg.teny in px_weekly.columns:
        tnx_w, _tnx_w_meta = _tnx_to_yield_percent_with_meta(px_weekly[cfg.teny])
        tnx_w = tnx_w.dropna()
    else:
        tnx_w = pd.Series(dtype=float)

    def _mom_w(series: pd.Series, weeks: int) -> float:
        try:
            if series is None or series.empty or len(series) <= weeks:
                return float("nan")
            return float(series.iloc[-1] / series.shift(weeks).iloc[-1] - 1)
        except Exception:
            return float("nan")

    def _yield_chg_bps(series: pd.Series, weeks: int) -> float:
        try:
            if series is None or series.empty or len(series) <= weeks:
                return float("nan")
            return float((series.iloc[-1] - series.shift(weeks).iloc[-1]) * 100)
        except Exception:
            return float("nan")

    dxy_mom_1w = _mom_w(dxy_w, 1)
    dxy_mom_4w = _mom_w(dxy_w, 4)
    dxy_mom_12w = _mom_w(dxy_w, 12)
    dxy_mom_52w = _mom_w(dxy_w, 52)

    y10_chg_1w_bps = _yield_chg_bps(tnx_w, 1)
    y10_chg_4w_bps = _yield_chg_bps(tnx_w, 4)
    y10_chg_12w_bps = _yield_chg_bps(tnx_w, 12)
    y10_chg_52w_bps = _yield_chg_bps(tnx_w, 52)

    usd_regime = "USD strengthening" if np.isfinite(dxy_mom_12w) and dxy_mom_12w > 0 else "USD weakening"
    y10_regime = "Yields rising" if np.isfinite(y10_chg_12w_bps) and y10_chg_12w_bps > 0 else "Yields falling"
    regime_str = f"{usd_regime} · {y10_regime}"

    def _ret(series: pd.Series, days: int) -> float:
        try:
            if series is None or series.empty or len(series) <= days:
                return float("nan")
            return float(series.iloc[-1] / series.shift(days).iloc[-1] - 1)
        except Exception:
            return float("nan")

    gld_ret_1w = _ret(gld, 5)
    gld_ret_4w = _ret(gld, 21)
    gld_ret_12w = _ret(gld, 63)
    gld_ret_12m = _ret(gld, 252)

    try:
        if gld is None or gld.empty:
            gld_dd_12m = float("nan")
        else:
            window = gld.iloc[-252:] if len(gld) >= 252 else gld
            rolling_max = window.max() if not window.empty else np.nan
            last_val = float(window.iloc[-1]) if not window.empty else np.nan
            gld_dd_12m = float(last_val / rolling_max - 1) if np.isfinite(rolling_max) and np.isfinite(last_val) else float("nan")
    except Exception:
        gld_dd_12m = float("nan")

    asof_str = asof.date().isoformat() if isinstance(asof, pd.Timestamp) else str(asof)

    raw_payload = {
        "asof": asof_str,
        "regime": regime_str,
        "dxy_mom_1w": dxy_mom_1w,
        "dxy_mom_4w": dxy_mom_4w,
        "dxy_mom_12w": dxy_mom_12w,
        "dxy_mom_12m": dxy_mom_52w,
        "y10_chg_1w_bps": y10_chg_1w_bps,
        "y10_chg_4w_bps": y10_chg_4w_bps,
        "y10_chg_12w_bps": y10_chg_12w_bps,
        "y10_chg_12m_bps": y10_chg_52w_bps,
        "gld_ret_1w": gld_ret_1w,
        "gld_ret_4w": gld_ret_4w,
        "gld_ret_12w": gld_ret_12w,
        "gld_ret_12m": gld_ret_12m,
        "gld_dd_12m": gld_dd_12m,
    }

    ui_rows = [
        {
            "period": "1W",
            "USD (DXY)": {"value": dxy_mom_1w, "display": _pct(dxy_mom_1w, 2)},
            "US 10Y yield Δ": {"value": y10_chg_1w_bps, "unit": "bps", "display": f"{_num(y10_chg_1w_bps, 1)} bps"},
            "Gold (GLD)": {"value": gld_ret_1w, "display": _pct(gld_ret_1w, 2)},
        },
        {
            "period": "4W",
            "USD (DXY)": {"value": dxy_mom_4w, "display": _pct(dxy_mom_4w, 2)},
            "US 10Y yield Δ": {"value": y10_chg_4w_bps, "unit": "bps", "display": f"{_num(y10_chg_4w_bps, 1)} bps"},
            "Gold (GLD)": {"value": gld_ret_4w, "display": _pct(gld_ret_4w, 2)},
        },
        {
            "period": "12W",
            "USD (DXY)": {"value": dxy_mom_12w, "display": _pct(dxy_mom_12w, 2)},
            "US 10Y yield Δ": {"value": y10_chg_12w_bps, "unit": "bps", "display": f"{_num(y10_chg_12w_bps, 1)} bps"},
            "Gold (GLD)": {"value": gld_ret_12w, "display": _pct(gld_ret_12w, 2)},
        },
        {
            "period": "12M",
            "USD (DXY)": {"value": dxy_mom_52w, "display": _pct(dxy_mom_52w, 2)},
            "US 10Y yield Δ": {"value": y10_chg_52w_bps, "unit": "bps", "display": f"{_num(y10_chg_52w_bps, 1)} bps"},
            "Gold (GLD)": {"value": gld_ret_12m, "display": _pct(gld_ret_12m, 2)},
        },
    ]

    ui_payload = {
        "asof": asof_str,
        "title": "Regime snapshot",
        "headline": regime_str,
        "table": {
            "columns": ["Period", "USD (DXY)", "US 10Y yield Δ", "Gold (GLD)"],
            "rows": ui_rows,
        },
        "highlights": [
            {
                "label": "Gold drawdown (12M)",
                "value": gld_dd_12m,
                "display": _pct(gld_dd_12m, 2),
            }
        ],
        "raw": raw_payload,
    }

    write_json(cfg.regime_snapshot_json_path, ui_payload)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    title = "Regime Snapshot"
    subtitle = f"As of {asof_str}"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.96)
    ax.set_title(subtitle, fontsize=11, loc="center", pad=8, color="#555555")

    col_x = [0.08, 0.30, 0.52, 0.78]
    y_start = 0.82
    y_step = 0.12

    headers = ["Period", "USD (DXY)", "10Y Δ (bps)", "GLD"]
    for j, hdr in enumerate(headers):
        ax.text(col_x[j], y_start + 0.08, hdr, transform=ax.transAxes, fontsize=11, fontweight="bold", color="#222222")

    ax.hlines(y_start + 0.04, col_x[0] - 0.02, 0.95, colors="#888888", linewidth=1.2, transform=ax.transAxes)

    data_rows = [
        ("1w", _pct(dxy_mom_1w, 2), f"{_num(y10_chg_1w_bps, 1)}", _pct(gld_ret_1w, 2)),
        ("4w", _pct(dxy_mom_4w, 2), f"{_num(y10_chg_4w_bps, 1)}", _pct(gld_ret_4w, 2)),
        ("12w", _pct(dxy_mom_12w, 2), f"{_num(y10_chg_12w_bps, 1)}", _pct(gld_ret_12w, 2)),
        ("12m", _pct(dxy_mom_52w, 2), f"{_num(y10_chg_52w_bps, 1)}", _pct(gld_ret_12m, 2)),
    ]

    for i, row in enumerate(data_rows):
        y = y_start - i * y_step
        for j, val in enumerate(row):
            ax.text(col_x[j], y, val, transform=ax.transAxes, fontsize=11, color="#000000")
        ax.hlines(y - 0.05, col_x[0] - 0.02, 0.95, colors="#e0e0e0", linewidth=0.8, transform=ax.transAxes)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    fig.savefig(cfg.regime_snapshot_png_path, dpi=180)
    plt.close(fig)

    return {"png_path": cfg.regime_snapshot_png_path, "json_path": cfg.regime_snapshot_json_path}


def save_gld_spy_ratio_12m_chart(px_daily: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Save GLD/SPY ratio chart (12m) and JSON series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = px_daily.index.max()
    start = end - pd.DateOffset(years=1)

    gld = px_daily.loc[px_daily.index >= start, cfg.gold].dropna() if cfg.gold in px_daily.columns else pd.Series(dtype=float)
    spy = px_daily.loc[px_daily.index >= start, cfg.equity].dropna() if cfg.equity in px_daily.columns else pd.Series(dtype=float)

    Path(cfg.gld_spy_ratio_png_path).parent.mkdir(parents=True, exist_ok=True)

    common_idx = gld.index.intersection(spy.index)
    gld = gld.reindex(common_idx).dropna()
    spy = spy.reindex(common_idx).dropna()

    if gld.empty or spy.empty or len(gld) < 50:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.set_title("GLD/SPY ratio — last 12 months", fontsize=12, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Ratio (GLD / SPY)", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        fig.tight_layout()
        fig.savefig(cfg.gld_spy_ratio_png_path, dpi=180)
        plt.close(fig)
        write_json(cfg.gld_spy_ratio_json_path, {"series": [], "stats": {}})
        return {"png_path": cfg.gld_spy_ratio_png_path, "json_path": cfg.gld_spy_ratio_json_path}

    ratio = gld / spy

    current_ratio = float(ratio.iloc[-1])
    percentile_12m = float((ratio <= current_ratio).mean() * 100)
    high_12m = float(ratio.max())
    drawdown_12m = float((current_ratio / high_12m - 1) * 100)
    as_of_date = ratio.index[-1].strftime("%Y-%m-%d")

    fig, ax = plt.subplots(figsize=(10, 4.6))

    avg_12m = float(ratio.mean())
    std_12m = float(ratio.std())
    if np.isfinite(avg_12m) and np.isfinite(std_12m) and std_12m > 0:
        ax.fill_between(
            ratio.index,
            avg_12m - std_12m,
            avg_12m + std_12m,
            color="#999999",
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )
    if np.isfinite(avg_12m):
        ax.axhline(avg_12m, color="#666666", linewidth=1.2, linestyle="--", label="12m avg", zorder=1)

    ax.plot(ratio.index, ratio.values, color="tab:blue", linewidth=1.8, label="GLD/SPY ratio", zorder=2)

    ax.set_title("GLD/SPY ratio — last 12 months", fontsize=12, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Ratio (GLD / SPY)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    for tl in ax.get_yticklabels():
        tl.set_color("tab:blue")

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)
    _format_quarter_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=9)

    footnote = (
        f"Current ratio: {current_ratio:.3f}  |  "
        f"Percentile (12m): {percentile_12m:.0f}%  |  "
        f"Drawdown from 12m high: {drawdown_12m:.2f}%  |  "
        f"As of {as_of_date}"
    )
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=8, color="#555555")

    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    fig.savefig(cfg.gld_spy_ratio_png_path, dpi=180)
    plt.close(fig)

    df_out = pd.DataFrame(
        {
            "date": ratio.index.date.astype(str),
            "ratio": ratio.values.astype(float),
        }
    )
    stats = {
        "current_ratio": current_ratio,
        "percentile_12m": percentile_12m,
        "high_12m": high_12m,
        "drawdown_12m_pct": drawdown_12m,
        "as_of": as_of_date,
    }
    write_json(cfg.gld_spy_ratio_json_path, {"series": df_out.to_dict(orient="records"), "stats": stats})

    return {"png_path": cfg.gld_spy_ratio_png_path, "json_path": cfg.gld_spy_ratio_json_path}


def plot_gold_oil_ratio_12m(cfg: Config = CFG) -> Dict[str, str] | None:
    """Plot Gold/Oil ratio (12m) and save chart + JSON."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        px = download_prices(tickers=["GC=F", "CL=F"], period="18mo")
        px = align_and_clean(px)

        if px.empty or "GC=F" not in px.columns or "CL=F" not in px.columns:
            return None

        end = px.index.max()
        start_12m = end - pd.DateOffset(years=1)

        gold = px["GC=F"].dropna()
        oil = px["CL=F"].dropna()
        common_idx = gold.index.intersection(oil.index)
        gold = gold.reindex(common_idx).dropna()
        oil = oil.reindex(common_idx).dropna()

        if gold.empty or oil.empty:
            return None

        ratio = (gold / oil).rename("gold_oil_ratio")
        ratio_12m = ratio.loc[ratio.index >= start_12m].dropna()

        if ratio_12m.empty:
            return None

        current_ratio = float(ratio_12m.iloc[-1])
        pctile_12m = float((ratio_12m <= current_ratio).mean() * 100.0)
        high_12m = float(ratio_12m.max())
        dd_from_high = (current_ratio / high_12m - 1.0) * 100.0 if high_12m != 0 else float("nan")

        lookback_days = 60
        if len(ratio_12m) > lookback_days:
            is_rising = current_ratio >= float(ratio_12m.iloc[-(lookback_days + 1)])
        else:
            is_rising = current_ratio >= float(ratio_12m.iloc[0])

        bias = "hedge / growth-stress / disinflation bias" if is_rising else "reflation / demand-driven inflation"
        as_of = pd.to_datetime(ratio_12m.index[-1]).date().isoformat()

        fig, ax = plt.subplots(figsize=(10, 4.6))

        avg_12m = float(ratio_12m.mean())
        std_12m = float(ratio_12m.std())
        if np.isfinite(avg_12m) and np.isfinite(std_12m) and std_12m > 0:
            ax.fill_between(
                ratio_12m.index,
                avg_12m - std_12m,
                avg_12m + std_12m,
                color="#999999",
                alpha=0.10,
                linewidth=0,
                zorder=0,
            )
        if np.isfinite(avg_12m):
            ax.axhline(avg_12m, color="#666666", linewidth=1.2, linestyle="--", label="12m avg", zorder=1)

        ax.plot(ratio_12m.index, ratio_12m.values, color="tab:blue", linewidth=1.8, label="Gold/Oil ratio", zorder=2)

        ax.set_title("Gold / Oil ratio — Inflation vs Hedge signal — last 12 months", fontsize=12, weight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Ratio (Gold / Oil)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

        _format_quarter_axis(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=9)

        footnote = (
            f"Current ratio: {current_ratio:.2f}  |  "
            f"Percentile (12m): {pctile_12m:.0f}%  |  "
            f"Drawdown from 12m high: {dd_from_high:.2f}%  |  "
            f"As of {as_of}"
        )
        fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=9, color="#666666")
        fig.tight_layout(rect=[0, 0.12, 1, 0.96])

        Path(cfg.gold_oil_ratio_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.gold_oil_ratio_png_path, dpi=180)
        plt.close(fig)

        df_out = pd.DataFrame(
            {
                "date": ratio_12m.index.date.astype(str),
                "ratio": ratio_12m.values.astype(float),
            }
        )
        stats = {
            "current_ratio": current_ratio,
            "percentile_12m": pctile_12m,
            "high_12m": high_12m,
            "drawdown_12m_pct": dd_from_high,
            "bias": bias,
            "as_of": as_of,
        }
        write_json(cfg.gold_oil_ratio_json_path, {"series": df_out.to_dict(orient="records"), "stats": stats})

        return {"png_path": cfg.gold_oil_ratio_png_path, "json_path": cfg.gold_oil_ratio_json_path}
    except Exception:
        return None


def plot_credit_risk_premium_20y(cfg: Config = CFG) -> Dict[str, str] | None:
    """Build and save a 20y HY credit spread chart using FRED (not yfinance).

    Textbook metric: High Yield Option-Adjusted Spread (HY OAS).

    Series:
      - HY OAS: BAMLH0A0HYM2 (% units)
      - Recession indicator (optional shading): USREC (0/1)

    Output:
      - charts/credit_risk_premium_20y.png
      - data/credit_risk_premium_20y.json
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from pandas_datareader.data import DataReader
    except Exception as e:
        print(f"[warn] pandas_datareader unavailable, falling back to FRED CSV: {e}")
        DataReader = None

    def _fred_series_via_csv(series_id: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.Series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        raw = pd.read_csv(url)
        date_col = None
        for candidate in ("DATE", "observation_date"):
            if candidate in raw.columns:
                date_col = candidate
                break
        if raw.empty or date_col is None or series_id not in raw.columns:
            raise ValueError(f"Unexpected FRED CSV format for {series_id}")
        s = raw.rename(columns={date_col: "date"}).copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s = s.dropna(subset=["date"]).set_index("date")
        out = pd.to_numeric(s[series_id], errors="coerce")
        out = out.loc[(out.index >= start_dt) & (out.index <= end_dt)]
        out.name = series_id
        return out

    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=20)

    try:
        if DataReader is not None:
            hy = DataReader("BAMLH0A0HYM2", "fred", start, end)
            try:
                usrec = DataReader("USREC", "fred", start, end)
            except Exception:
                usrec = None
        else:
            hy = _fred_series_via_csv("BAMLH0A0HYM2", start, end).to_frame()
            try:
                usrec = _fred_series_via_csv("USREC", start, end).to_frame()
            except Exception:
                usrec = None
    except Exception as e:
        print(f"[warn] FRED download failed: {e}")
        return None

    if hy is None or hy.empty:
        return None

    hy_oas = pd.to_numeric(hy.iloc[:, 0], errors="coerce").rename("hy_oas").sort_index()
    hy_oas = hy_oas.ffill(limit=5).dropna()
    if hy_oas.empty:
        return None

    median_20y = float(hy_oas.median())
    p25_20y = float(hy_oas.quantile(0.25))
    p75_20y = float(hy_oas.quantile(0.75))

    current_hy = float(hy_oas.iloc[-1])
    hy_pct = float((hy_oas <= current_hy).mean() * 100.0)
    risk_on_score = float(100.0 - hy_pct)
    as_of = pd.to_datetime(hy_oas.index[-1]).date().isoformat()

    fig, ax = plt.subplots(figsize=(10, 4.6))

    if np.isfinite(p25_20y) and np.isfinite(p75_20y):
        ax.axhspan(p25_20y, p75_20y, color="#999999", alpha=0.10, label="IQR (20y)")

    ax.plot(hy_oas.index, hy_oas.values, color="tab:blue", linewidth=1.8, label="HY OAS")
    if np.isfinite(median_20y):
        ax.axhline(median_20y, color="#666666", linestyle="--", linewidth=1.2, label="Median (20y)")

    ax.set_title("High yield credit spread (HY OAS) — last 20 years", fontsize=12, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (percentage points)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=True, fontsize=9)

    footnote = (
        f"Current HY OAS: {current_hy:.2f}%  |  Median (20y): {median_20y:.2f}%  |  Percentile (20y): {hy_pct:.0f}%\n"
        f"As of {as_of}"
    )
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=8, color="#555555")

    fig.tight_layout(rect=[0, 0.18, 1, 0.96])
    Path(cfg.credit_risk_premium_20y_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.credit_risk_premium_20y_png_path, dpi=180)
    plt.close(fig)

    df_out = pd.DataFrame({"date": hy_oas.index.date.astype(str), "hy_oas": hy_oas.values.astype(float)})
    stats = {
        "current_hy_oas_pct": current_hy,
        "hy_oas_percentile_20y": hy_pct,
        "risk_on_score_20y": risk_on_score,
        "median_20y": median_20y,
        "p25_20y": p25_20y,
        "p75_20y": p75_20y,
        "as_of": as_of,
        "start": pd.to_datetime(hy_oas.index.min()).date().isoformat(),
    }
    write_json(cfg.credit_risk_premium_20y_json_path, {"series": df_out.to_dict(orient="records"), "stats": stats})

    return {"png_path": cfg.credit_risk_premium_20y_png_path, "json_path": cfg.credit_risk_premium_20y_json_path}


def plot_gold_value_trend_2005(cfg: Config = CFG) -> Dict[str, str] | None:
    """Plot gold value trend since 2005 using GLD.

    Output:
      - charts/gold_value_trend_2005.png
      - data/gold_value_trend_2005.json
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    end = pd.Timestamp.today().normalize()
    start = pd.Timestamp("2005-01-01")

    gld_px = download_prices(tickers=[cfg.gold], period="max")
    gld_px = align_and_clean(gld_px)
    gld = gld_px[cfg.gold].dropna() if not gld_px.empty and cfg.gold in gld_px.columns else pd.Series(dtype=float)

    series = gld.loc[gld.index >= start].dropna()
    if series.empty:
        return None
    series_name = "gld"
    subtitle = "(GLD)"

    as_of = pd.to_datetime(series.index.max()).date().isoformat()
    start_used = pd.to_datetime(series.index.min()).date().isoformat()
    current_val = float(series.iloc[-1])
    start_val = float(series.iloc[0])
    years = max((series.index.max() - series.index.min()).days / 365.25, 0.0)
    cagr = float((current_val / start_val) ** (1 / years) - 1) if years > 0 and start_val > 0 else float("nan")
    peak = float(series.max())
    dd = float((current_val / peak - 1) * 100.0) if peak > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(series.index, series.values, color="#1b6ca8", linewidth=1.8, label="GLD")
    ax.set_title("Gold value trend — since 2005", fontsize=12, weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Level ($)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True, fontsize=9)

    footnote = (
        f"Current: {current_val:.2f}  |  CAGR: {cagr*100:.2f}%  |  Drawdown from peak: {dd:.2f}%\n"
        f"As of {as_of}  {subtitle}"
    )
    fig.text(0.5, 0.01, footnote, ha="center", va="bottom", fontsize=8, color="#555555")

    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    Path(cfg.gold_value_trend_2005_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.gold_value_trend_2005_png_path, dpi=180)
    plt.close(fig)

    df_out = pd.DataFrame({"date": series.index.date.astype(str), "value": series.values.astype(float)})
    stats = {
        "as_of": as_of,
        "start_used": start_used,
        "series": series_name,
        "current": current_val,
        "cagr": cagr,
        "drawdown_from_peak_pct": dd,
        "note": subtitle.strip(),
    }
    write_json(cfg.gold_value_trend_2005_json_path, {"series": df_out.to_dict(orient="records"), "stats": stats})

    return {"png_path": cfg.gold_value_trend_2005_png_path, "json_path": cfg.gold_value_trend_2005_json_path}


def main(cfg: Config = CFG) -> None:
    tickers = [cfg.gold, cfg.usd, cfg.rates, cfg.equity, cfg.teny]

    px_daily = download_prices(tickers=tickers, period=cfg.history_period)
    px_daily = align_and_clean(px_daily)

    if not px_daily.empty:
        start = pd.to_datetime(px_daily.index.min()).date().isoformat()
        end = pd.to_datetime(px_daily.index.max()).date().isoformat()
        counts = []
        for t in tickers:
            col = str(t)
            n = int(px_daily[col].notna().sum()) if col in px_daily.columns else 0
            counts.append(f"{col}:{n}")
        print(f"[diag] px_daily rows={len(px_daily)} range={start}->{end} nonnull=" + ",".join(counts))

    px_weekly = to_weekly_prices(px_daily)

    reg_now = current_regime(px_weekly=px_weekly, cfg=cfg)

    regime_snapshot_meta = build_regime_snapshot(px_weekly=px_weekly, cfg=cfg)

    insights = generate_insights(reg_now)

    chart_meta = save_gld_vs_usd_12m_chart(px_daily=px_daily, cfg=cfg)
    teny_meta = save_gld_vs_teny_12m_chart(px_daily=px_daily, cfg=cfg)
    regime_snap_meta = save_regime_snapshot(px_daily=px_daily, cfg=cfg)
    gld_spy_ratio_meta = save_gld_spy_ratio_12m_chart(px_daily=px_daily, cfg=cfg)
    gold_oil_ratio_meta = plot_gold_oil_ratio_12m(cfg=cfg)
    credit_risk_premium_meta = plot_credit_risk_premium_20y(cfg=cfg)
    gold_value_trend_meta = plot_gold_value_trend_2005(cfg=cfg)

    latest_payload = {
        "asof": reg_now["asof"],
        "regime": {"usd": reg_now["usd"], "rates": reg_now["rates"]},
        "regime_snapshot": regime_snap_meta,
        "regime_snapshot_meta": regime_snapshot_meta,
        "signals": {
            "mom_usd_12w": reg_now["mom_usd_12w"],
            "mom_rates_12w": reg_now["mom_rates_12w"],
        },

        "charts": {
            "gld_vs_usd_12m": chart_meta,
            "gld_vs_10y_12m": teny_meta,
            "gld_spy_ratio_12m": gld_spy_ratio_meta,
            "gold_oil_ratio_12m": gold_oil_ratio_meta,
            "credit_risk_premium_20y": credit_risk_premium_meta,
            "gold_value_trend_2005": gold_value_trend_meta,
        },
        "regime_snapshot": regime_snap_meta,
        "insights": insights,
    }

    write_json(cfg.latest_path, latest_payload)

    print("✅ Updated outputs:")
    print(f" - {cfg.latest_path}")
    print(f" - {cfg.gld_vs_usd_png_path}")
    print(f" - {cfg.gld_vs_usd_json_path}")
    print(f" - {cfg.gld_vs_teny_png_path}")
    print(f" - {cfg.gld_vs_teny_json_path}")
    print(f" - {cfg.regime_snapshot_png_path}")
    print(f" - {cfg.regime_snapshot_json_path}")
    print(f" - {cfg.gld_spy_ratio_png_path}")
    print(f" - {cfg.gld_spy_ratio_json_path}")
    if gold_oil_ratio_meta:
        print(f" - {gold_oil_ratio_meta['png_path']}")
        print(f" - {gold_oil_ratio_meta['json_path']}")
    if credit_risk_premium_meta:
        print(f" - {credit_risk_premium_meta['png_path']}")
        print(f" - {credit_risk_premium_meta['json_path']}")
    if gold_value_trend_meta:
        print(f" - {gold_value_trend_meta['png_path']}")
        print(f" - {gold_value_trend_meta['json_path']}")


if __name__ == "__main__":
    main()
