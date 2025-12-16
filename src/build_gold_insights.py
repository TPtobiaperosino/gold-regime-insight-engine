"""
Gold Regime Insight Engine — build_gold_insights.py

Output:
- data/latest.json              -> asof + regime corrente + expected impact + insight cards
- data/regimes.json             -> tabella 8 combinazioni regime con stats storiche
- data/rolling_corr.json        -> rolling corr (60 giorni) oro vs driver (USD, rates, SPY, VIX)
- data/gld_vs_usd_12m.png        -> GLD vs USD (UUP), last 12 months (dual-axis)
- data/gld_vs_usd_12m.json       -> series for the GLD vs USD chart
- data/gld_vs_10y_12m.png        -> GLD vs 10Y yield (from ^TNX, tnx/10 -> %), last 12 months (dual-axis)
- data/gld_vs_10y_12m.json       -> series for the GLD vs 10Y chart
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class Config:
    gold: str = "GLD"
    usd: str = "UUP"
    rates: str = "IEF"
    equity: str = "SPY"
    vix: str = "^VIX"
    teny: str = "^TNX"

    momentum_weeks: int = 12
    vix_lookback_weeks: int = 104
    vix_percentile: float = 0.75

    rolling_corr_days: int = 60
    history_period: str = "max"

    latest_path: str = "data/latest.json"
    regimes_path: str = "data/regimes.json"
    rolling_corr_path: str = "data/rolling_corr.json"

    gld_vs_usd_png_path: str = "data/gld_vs_usd_12m.png"
    gld_vs_usd_json_path: str = "data/gld_vs_usd_12m.json"
    gld_vs_teny_png_path: str = "data/gld_vs_10y_12m.png"
    gld_vs_teny_json_path: str = "data/gld_vs_10y_12m.json"
    regime_heatmap_path: str = "data/regime_heatmap.json"
    regime_heatmap_png_path: str = "data/regime_heatmap.png"


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
    px = px.dropna(how="all")
    px = px.dropna()
    return px


def to_weekly_prices(px_daily: pd.DataFrame) -> pd.DataFrame:
    weekly = px_daily.resample("W-FRI").last().dropna()
    if weekly.index[-1] > px_daily.index[-1]:
        weekly = weekly.iloc[:-1]
    return weekly


def pct_return(px: pd.DataFrame) -> pd.DataFrame:
    return px.pct_change().dropna()


def momentum(series_weekly_price: pd.Series, weeks: int) -> pd.Series:
    return (series_weekly_price / series_weekly_price.shift(weeks) - 1).dropna()


def rolling_corr(daily_returns: pd.DataFrame, target: str, drivers: List[str], window_days: int) -> pd.DataFrame:
    out = pd.DataFrame(index=daily_returns.index)
    for d in drivers:
        out[f"corr_{target}_{d}"] = daily_returns[target].rolling(window_days).corr(daily_returns[d])
    return out.dropna()


def current_regime(px_weekly: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    w = cfg.momentum_weeks

    mom_usd = momentum(px_weekly[cfg.usd], w)
    mom_rates = momentum(px_weekly[cfg.rates], w)

    common_idx = mom_usd.index.intersection(mom_rates.index)
    mom_usd = mom_usd.loc[common_idx]
    mom_rates = mom_rates.loc[common_idx]

    vix_lvl = px_weekly[cfg.vix].loc[common_idx].dropna()
    vix_thr = vix_lvl.rolling(cfg.vix_lookback_weeks).quantile(cfg.vix_percentile)
    vix_flag = (vix_lvl > vix_thr).dropna()

    last_dt = common_idx.max()

    if last_dt not in vix_flag.index:
        global_thr = vix_lvl.quantile(cfg.vix_percentile)
        risk_low = bool(vix_lvl.loc[last_dt] > global_thr) if last_dt in vix_lvl.index else False
        vix_level = float(vix_lvl.loc[last_dt]) if last_dt in vix_lvl.index else float("nan")
        vix_threshold = float(global_thr)
    else:
        risk_low = bool(vix_flag.loc[last_dt])
        vix_level = float(vix_lvl.loc[last_dt])
        vix_threshold = float(vix_thr.loc[last_dt])

    usd_strong = bool(mom_usd.loc[last_dt] > 0)
    rates_up = bool(mom_rates.loc[last_dt] < 0)

    return {
        "asof": last_dt.date().isoformat(),
        "usd": "strong" if usd_strong else "weak",
        "rates": "up" if rates_up else "down",
        "risk": "low" if risk_low else "high",
        "mom_usd_12w": float(mom_usd.loc[last_dt]),
        "mom_rates_12w": float(mom_rates.loc[last_dt]),
        "vix_level": vix_level,
        "vix_threshold": vix_threshold,
    }


def build_weekly_regimes(px_weekly: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    w = cfg.momentum_weeks

    mom_usd = momentum(px_weekly[cfg.usd], w)
    mom_rates = momentum(px_weekly[cfg.rates], w)

    vix_lvl = px_weekly[cfg.vix].dropna()
    vix_thr = vix_lvl.rolling(cfg.vix_lookback_weeks).quantile(cfg.vix_percentile)
    global_thr = vix_lvl.quantile(cfg.vix_percentile)

    idx = mom_usd.index.intersection(mom_rates.index).intersection(vix_lvl.index)
    df = pd.DataFrame(index=idx)

    df["usd"] = np.where(mom_usd.loc[idx] > 0, "strong", "weak")
    df["rates"] = np.where(mom_rates.loc[idx] < 0, "up", "down")
    thr = vix_thr.reindex(idx).fillna(global_thr)
    # risk appetite: low when VIX is above threshold, otherwise high
    df["risk"] = np.where(vix_lvl.loc[idx] > thr, "low", "high")

    df["regime"] = df["usd"] + "_" + df["rates"] + "_" + df["risk"]
    return df.dropna()


def build_regime_table(px_weekly: pd.DataFrame, weekly_ret: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    w = cfg.momentum_weeks

    mom_usd = momentum(px_weekly[cfg.usd], w)
    mom_rates = momentum(px_weekly[cfg.rates], w)

    vix_lvl = px_weekly[cfg.vix].dropna()
    vix_thr = vix_lvl.rolling(cfg.vix_lookback_weeks).quantile(cfg.vix_percentile)

    idx = weekly_ret.index.intersection(mom_usd.index).intersection(mom_rates.index).intersection(vix_lvl.index)
    df = pd.DataFrame(index=idx)

    df["gold_ret"] = weekly_ret.loc[idx, cfg.gold]
    df["usd"] = np.where(mom_usd.loc[idx] > 0, "strong", "weak")
    df["rates"] = np.where(mom_rates.loc[idx] < 0, "up", "down")

    global_thr = vix_lvl.quantile(cfg.vix_percentile)
    thr = vix_thr.reindex(idx)
    # risk appetite: low when VIX is above threshold, otherwise high
    df["risk"] = np.where(vix_lvl.loc[idx] > thr.fillna(global_thr), "low", "high")

    g = df.groupby(["usd", "rates", "risk"])
    table = g["gold_ret"].agg(
        avg_weekly_return="mean",
        vol_weekly="std",
        n_obs="count",
    ).reset_index()

    hit = g["gold_ret"].apply(lambda x: float((x > 0).mean())).reset_index(name="hit_rate")
    table = table.merge(hit, on=["usd", "rates", "risk"], how="left")

    table = table.sort_values(["risk", "usd", "rates"]).reset_index(drop=True)
    return table


def expected_impact_from_table(
    regime_table: pd.DataFrame, usd: str, rates: str, risk: str
) -> Tuple[str, Dict[str, float]]:
    row = regime_table[
        (regime_table["usd"] == usd)
        & (regime_table["rates"] == rates)
        & (regime_table["risk"] == risk)
    ]
    if row.empty:
        return "Neutral", {"avg_weekly_return": float("nan"), "hit_rate": float("nan")}

    avg = float(row["avg_weekly_return"].iloc[0])
    hit = float(row["hit_rate"].iloc[0])

    q1 = float(regime_table["avg_weekly_return"].quantile(0.33))
    q2 = float(regime_table["avg_weekly_return"].quantile(0.66))

    if avg <= q1:
        label = "Headwind"
    elif avg >= q2:
        label = "Tailwind"
    else:
        label = "Neutral"

    return label, {"avg_weekly_return": avg, "hit_rate": hit}


def _ordinal(n: int) -> str:
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def build_regime_snapshot(px_weekly: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    """Build a compact regime snapshot aligned to the last common weekly date.

    Returns the structure requested by the UI contract (label + metrics).
    """
    w = cfg.momentum_weeks

    required = [cfg.usd, cfg.rates, cfg.vix]
    if not all(r in px_weekly.columns for r in required):
        return {
            "label": "USD unknown · Rates unknown · Risk appetite unknown",
            "metrics": {
                "usd_momentum_12w_pct": None,
                "rates_momentum_12w_pct": None,
                "vix_percentile_104w": None,
                "vix_interpretation": "VIX data unavailable",
            },
        }

    idx = px_weekly[cfg.usd].dropna().index
    idx = idx.intersection(px_weekly[cfg.rates].dropna().index).intersection(px_weekly[cfg.vix].dropna().index)
    if idx.empty:
        return {
            "label": "USD unknown · Rates unknown · Risk appetite unknown",
            "metrics": {
                "usd_momentum_12w_pct": None,
                "rates_momentum_12w_pct": None,
                "vix_percentile_104w": None,
                "vix_interpretation": "VIX data unavailable",
            },
        }

    last_dt = idx.max()

    # USD momentum (UUP)
    usd_s = px_weekly[cfg.usd].dropna()
    if last_dt in usd_s.index and (usd_s.index.get_loc(last_dt) - w) >= 0:
        usd_mom = float(usd_s.loc[last_dt] / usd_s.shift(w).loc[last_dt] - 1)
    else:
        usd_mom = float("nan")
    usd_label = "strong" if usd_mom > 0 else "weak"

    # Rates momentum (IEF)
    rates_s = px_weekly[cfg.rates].dropna()
    if last_dt in rates_s.index and (rates_s.index.get_loc(last_dt) - w) >= 0:
        rates_mom = float(rates_s.loc[last_dt] / rates_s.shift(w).loc[last_dt] - 1)
    else:
        rates_mom = float("nan")
    rates_label = "up" if rates_mom < 0 else "down"

    # VIX percentile (104 weeks or available)
    vix_s = px_weekly[cfg.vix].dropna()
    vix_until = vix_s.loc[:last_dt].dropna()
    lookback_n = min(cfg.vix_lookback_weeks, len(vix_until))
    if lookback_n <= 0:
        pct = float("nan")
    else:
        hist = vix_until.iloc[-lookback_n:]
        vix_last = hist.iloc[-1]
        pct = float((hist <= vix_last).mean() * 100.0)

    if not np.isfinite(pct):
        risk_app = "unknown"
    elif pct >= 75.0:
        risk_app = "low"
    elif pct >= 40.0:
        risk_app = "medium"
    else:
        risk_app = "high"

    pct_round = round(pct, 1) if np.isfinite(pct) else None
    vix_interpretation = (
        f"VIX at {_ordinal(round(pct))} percentile → risk appetite {risk_app}"
        if np.isfinite(pct)
        else "VIX data unavailable"
    )

    snapshot = {
        "label": f"USD {usd_label} · Rates {rates_label} · Risk appetite {risk_app}",
        "metrics": {
            "usd_momentum_12w_pct": round(usd_mom * 100.0, 2) if np.isfinite(usd_mom) else None,
            "rates_momentum_12w_pct": round(rates_mom * 100.0, 2) if np.isfinite(rates_mom) else None,
            "vix_percentile_104w": pct_round,
            "vix_interpretation": vix_interpretation,
        },
    }

    return snapshot


def build_regime_heatmap(regime_table: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    """Build a Lovable-friendly regime heatmap JSON and optional static PNG.

    Returns a summary dict with current_cell, best_cell_by_avg_return, worst_cell_by_avg_return
    to be embedded into latest.json as `heatmap_summary`.
    """
    # orders required by UI
    usd_order = ["weak", "strong"]
    rates_order = ["down", "up"]
    risk_order = ["high", "low"]

    # copy and normalize risk names to risk_appetite
    df = regime_table.copy()
    def _map_risk(x: str) -> str:
        if x == "on":
            return "high"
        if x == "off":
            return "low"
        return x

    df["risk_appetite"] = df["risk"].apply(_map_risk)

    cells = []
    for usd in usd_order:
        for rates in rates_order:
            for risk_app in risk_order:
                sel = df[(df["usd"] == usd) & (df["rates"] == rates) & (df["risk_appetite"] == risk_app)]
                if not sel.empty:
                    row = sel.iloc[0]
                    avg = float(row["avg_weekly_return"]) if pd.notna(row["avg_weekly_return"]) else float("nan")
                    hit = float(row["hit_rate"]) if pd.notna(row["hit_rate"]) else float("nan")
                    n = int(row["n_obs"]) if pd.notna(row["n_obs"]) else 0
                else:
                    avg = float("nan")
                    hit = float("nan")
                    n = 0

                avg_disp = fmt_pct(avg, 2) if _is_finite(avg) else None
                hit_disp = fmt_pct(hit, 1) if _is_finite(hit) else None

                cells.append(
                    {
                        "usd": usd,
                        "rates": rates,
                        "risk_appetite": risk_app,
                        "avg_weekly_return": None if not _is_finite(avg) else avg,
                        "hit_rate": None if not _is_finite(hit) else hit,
                        "n_obs": n,
                        "avg_weekly_return_display": avg_disp,
                        "hit_rate_display": hit_disp,
                    }
                )

    payload = {
        "dimensions": {
            "usd_order": usd_order,
            "rates_order": rates_order,
            "risk_appetite_order": risk_order,
        },
        "cells": cells,
        "notes": {
            "avg_weekly_return": "Mean weekly GLD return in that regime.",
            "hit_rate": "Share of weeks with positive GLD return.",
            "n_obs": "Sample size; low n means less reliable.",
        },
    }

    # write JSON
    write_json(cfg.regime_heatmap_path, payload)

    # try to generate optional PNG (two panels, risk high and low)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        # prepare grids for avg_weekly_return for each risk_appetite
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        vmin = min([c["avg_weekly_return"] for c in cells if _is_finite(c["avg_weekly_return"])], default=0)
        vmax = max([c["avg_weekly_return"] for c in cells if _is_finite(c["avg_weekly_return"])], default=0)
        max_abs = max(abs(vmin), abs(vmax), 1e-6)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

        for ax, risk_app in zip(axes, risk_order):
            mat = np.zeros((len(usd_order), len(rates_order))) * np.nan
            ann = [["" for _ in rates_order] for __ in usd_order]
            for i, usd in enumerate(usd_order):
                for j, rates in enumerate(rates_order):
                    rec = next((c for c in cells if c["usd"] == usd and c["rates"] == rates and c["risk_appetite"] == risk_app), None)
                    if rec is None:
                        mat[i, j] = np.nan
                        ann[i][j] = "n=0"
                    else:
                        mat[i, j] = rec["avg_weekly_return"] if _is_finite(rec["avg_weekly_return"]) else 0.0
                        a_disp = rec["avg_weekly_return_display"] or ""
                        h_disp = rec["hit_rate_display"] or ""
                        ann[i][j] = f"{a_disp}\n{h_disp}\nn={rec['n_obs']}"

            im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
            # annotate
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=9)

            ax.set_xticks(range(len(rates_order)))
            ax.set_xticklabels(rates_order)
            ax.set_yticks(range(len(usd_order)))
            ax.set_yticklabels(usd_order)
            ax.set_title(f"Risk appetite = {risk_app}")

        fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", label="Avg weekly return")
        fig.tight_layout()
        Path(cfg.regime_heatmap_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.regime_heatmap_png_path, dpi=180)
        plt.close(fig)
        png_path = cfg.regime_heatmap_png_path
    except Exception:
        png_path = None

    # build summary
    # helper to find cell
    def _find_cell(u, r, ra):
        return next((c for c in cells if c["usd"] == u and c["rates"] == r and c["risk_appetite"] == ra), None)

    # current cell: determine current values using same logic as current_regime (but mapping risk if needed)
    # We'll pick latest by searching for max n_obs date is not available here; use reg_table aggregation
    # fallback: pick the cell matching the last regime computed elsewhere (caller will attach reg_now)

    # choose best/worst by avg_weekly_return with n_obs >= 30 else fallback to all
    def _pick_best_worst(cells_list):
        eligible = [c for c in cells_list if c["n_obs"] >= 30 and _is_finite(c["avg_weekly_return"])]
        if not eligible:
            eligible = [c for c in cells_list if _is_finite(c["avg_weekly_return"]) ]
        if not eligible:
            return None
        best = max(eligible, key=lambda x: x["avg_weekly_return"])
        worst = min(eligible, key=lambda x: x["avg_weekly_return"])
        return best, worst

    pick = _pick_best_worst(cells)
    best_cell = pick[0] if pick else None
    worst_cell = pick[1] if pick else None

    summary = {
        "current_cell": None,  # caller will populate if desired
        "best_cell_by_avg_return": best_cell,
        "worst_cell_by_avg_return": worst_cell,
        "json_path": cfg.regime_heatmap_path,
        "png_path": png_path,
    }

    return summary


def generate_insights(
    regime_meta: Dict[str, str],
    regime_table: pd.DataFrame,
    corr_latest: Dict[str, float],
) -> List[Dict[str, object]]:
    insights: List[Dict[str, object]] = []

    usd = regime_meta["usd"]
    rates = regime_meta["rates"]
    risk = regime_meta["risk"]

    impact, stats = expected_impact_from_table(regime_table, usd, rates, risk)

    insights.append(
        {
            "title": "Current macro regime",
            "evidence": f"USD={usd}, Rates={rates}, Risk={risk} (asof {regime_meta['asof']})",
            "metric": {
                "avg_weekly_return": stats["avg_weekly_return"],
                "hit_rate": stats["hit_rate"],
                "impact": impact,
            },
        }
    )

    c_usd = corr_latest.get("corr_GLD_UUP")
    if c_usd is not None and np.isfinite(c_usd) and c_usd < -0.40:
        insights.append(
            {
                "title": "USD pressure is dominant",
                "evidence": "Rolling 60d correlation Gold vs USD is strongly negative.",
                "metric": {"rolling_corr_60d": float(c_usd), "threshold": -0.40},
            }
        )

    c_rates = corr_latest.get("corr_GLD_IEF")
    if c_rates is not None and np.isfinite(c_rates) and c_rates > 0.30:
        insights.append(
            {
                "title": "Gold is behaving as a rates-sensitive asset",
                "evidence": "Positive rolling correlation with bond prices (IEF) implies inverse linkage to yields.",
                "metric": {"rolling_corr_60d": float(c_rates), "threshold": 0.30},
            }
        )

    avg_all = regime_table["avg_weekly_return"]
    if np.isfinite(stats["avg_weekly_return"]):
        bottom_quartile = float(avg_all.quantile(0.25))
        if stats["avg_weekly_return"] <= bottom_quartile:
            insights.append(
                {
                    "title": "Historically unfavorable regime for gold",
                    "evidence": "This regime sits in the bottom quartile of historical average gold weekly returns.",
                    "metric": {
                        "avg_weekly_return": stats["avg_weekly_return"],
                        "bottom_quartile": bottom_quartile,
                    },
                }
            )

    if np.isfinite(stats["hit_rate"]) and stats["hit_rate"] < 0.50:
        insights.append(
            {
                "title": "Low reliability in this regime",
                "evidence": "Gold was positive in less than 50% of weeks in similar macro conditions.",
                "metric": {"hit_rate": stats["hit_rate"], "threshold": 0.50},
            }
        )

    if risk == "low" and np.isfinite(stats["hit_rate"]) and stats["hit_rate"] < 0.55:
        insights.append(
            {
                "title": "Not a consistent crisis hedge (in this sample)",
                "evidence": "In low risk-appetite weeks (VIX elevated), gold did not deliver a high positive frequency historically.",
                "metric": {"hit_rate": stats["hit_rate"], "context": "risk=low"},
            }
        )

    return insights[:6]


def write_json(path: str, payload: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def save_gld_vs_usd_12m_chart(px_daily: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Save GLD and UUP raw price series for the last 12 months as a dual-axis chart and JSON series.

    Enhancements: color-coded axis labels/ticks, rolling 60d correlation annotation, and optional
    shading when rolling corr (60d) < -0.40.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = px_daily.index.max()
    start = end - pd.DateOffset(years=1)

    gld = px_daily.loc[px_daily.index >= start, cfg.gold].dropna()
    uup = px_daily.loc[px_daily.index >= start, cfg.usd].dropna()

    common_idx = gld.index.intersection(uup.index)
    gld = gld.reindex(common_idx).dropna()
    uup = uup.reindex(common_idx).dropna()

    if gld.empty or uup.empty:
        # create empty outputs
        Path(cfg.gld_vs_usd_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.set_title("GLD vs USD (UUP) — last 12 months (dual-axis)")
        fig.tight_layout()
        fig.savefig(cfg.gld_vs_usd_png_path, dpi=180)
        plt.close(fig)
        write_json(cfg.gld_vs_usd_json_path, {"series": []})
        return {"png_path": cfg.gld_vs_usd_png_path, "json_path": cfg.gld_vs_usd_json_path}

    # colors for series and corresponding axes
    gld_color = "#1b6ca8"
    uup_color = "#d35400"

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ln1 = ax.plot(gld.index, gld.values, label="GLD ($, left axis)", color=gld_color, linewidth=1.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("GLD ($)", color=gld_color)
    ax.tick_params(axis="y", colors=gld_color)
    for tl in ax.get_yticklabels():
        tl.set_color(gld_color)

    # primary axis grid only (subtle)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

    # twin axis for UUP
    ax2 = ax.twinx()
    ln2 = ax2.plot(uup.index, uup.values, label="UUP ($, right axis)", color=uup_color, linewidth=1.6)
    ax2.set_ylabel("UUP ($)", color=uup_color)
    ax2.tick_params(axis="y", colors=uup_color)
    for tl in ax2.get_yticklabels():
        tl.set_color(uup_color)

    ax.set_title("GLD vs USD (UUP) — last 12 months (dual-axis)", fontsize=12, weight="bold")

    # rolling correlation annotation and shading removed per request


    # combined legend (merge handles from both axes) placed below the x-axis label "Date"
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # place legend centered below the plot (below the x-label)
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, fontsize=9)

    # leave space at the bottom for the legend and at the top for the annotation
    fig.tight_layout(rect=[0, 0.14, 1, 0.92])

    # save to primary configured path
    Path(cfg.gld_vs_usd_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.gld_vs_usd_png_path, dpi=180)
    # also save to legacy gold_12m path if present so callers still find it
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
            "uup": uup.values.astype(float),
        }
    )
    write_json(cfg.gld_vs_usd_json_path, {"series": df_out.to_dict(orient="records")})

    return {"png_path": cfg.gld_vs_usd_png_path, "json_path": cfg.gld_vs_usd_json_path}


def save_gld_vs_teny_12m_chart(px_daily: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Save GLD and 10Y yield (from ^TNX) as a dual-axis chart (last 12 months).

    TNX is typically reported as yield*10; convert to percent with `tnx / 10`.
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

    # convert TNX to yield percent
    tny_pct = tnx / 10.0

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

    # combined legend below the plot
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, fontsize=9)

    fig.tight_layout(rect=[0, 0.14, 1, 0.92])

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
    write_json(cfg.gld_vs_teny_json_path, {"series": df_out.to_dict(orient="records")})

    return {"png_path": cfg.gld_vs_teny_png_path, "json_path": cfg.gld_vs_teny_json_path}







def main(cfg: Config = CFG) -> None:
    tickers = [cfg.gold, cfg.usd, cfg.rates, cfg.equity, cfg.vix, cfg.teny]

    px_daily = download_prices(tickers=tickers, period=cfg.history_period)
    px_daily = align_and_clean(px_daily)

    ret_daily = pct_return(px_daily)

    px_weekly = to_weekly_prices(px_daily)
    ret_weekly = pct_return(px_weekly)

    corr_df = rolling_corr(
        daily_returns=ret_daily,
        target=cfg.gold,
        drivers=[cfg.usd, cfg.rates, cfg.equity, cfg.vix],
        window_days=cfg.rolling_corr_days,
    )
    corr_df = corr_df.rename(columns={f"corr_{cfg.gold}_{cfg.vix}": f"corr_{cfg.gold}_VIX"})

    reg_now = current_regime(px_weekly=px_weekly, cfg=cfg)
    weekly_regimes = build_weekly_regimes(px_weekly=px_weekly, cfg=cfg)

    # compact regime snapshot for KPIs
    regime_snapshot = build_regime_snapshot(px_weekly=px_weekly, cfg=cfg)

    regime_table = build_regime_table(px_weekly=px_weekly, weekly_ret=ret_weekly, cfg=cfg)
    impact, stats_in_regime = expected_impact_from_table(
        regime_table, reg_now["usd"], reg_now["rates"], reg_now["risk"]
    )

    # heatmap JSON + optional PNG
    heatmap_summary = build_regime_heatmap(regime_table=regime_table, cfg=cfg)
    # map internal risk to risk_appetite for current cell lookup
    risk_map = {"on": "high", "off": "low"}
    current_risk_app = risk_map.get(reg_now["risk"], reg_now["risk"]) if isinstance(reg_now.get("risk"), str) else reg_now.get("risk")
    current_cell = None
    try:
        with open(cfg.regime_heatmap_path, "r", encoding="utf-8") as _f:
            hm = json.load(_f)
            current_cell = next((c for c in hm.get("cells", []) if c["usd"] == reg_now["usd"] and c["rates"] == reg_now["rates"] and c["risk_appetite"] == current_risk_app), None)
    except Exception:
        current_cell = None
    heatmap_summary["current_cell"] = current_cell

    corr_last_row = corr_df.iloc[-1].to_dict()
    corr_latest = {
        "corr_GLD_UUP": float(corr_last_row.get(f"corr_{cfg.gold}_{cfg.usd}", np.nan)),
        "corr_GLD_IEF": float(corr_last_row.get(f"corr_{cfg.gold}_{cfg.rates}", np.nan)),
        "corr_GLD_SPY": float(corr_last_row.get(f"corr_{cfg.gold}_{cfg.equity}", np.nan)),
        "corr_GLD_VIX": float(corr_last_row.get(f"corr_{cfg.gold}_VIX", np.nan)),
    }

    insights = generate_insights(reg_now, regime_table, corr_latest)

    stats_in_regime_display = {
        "avg_weekly_return": fmt_pct(stats_in_regime["avg_weekly_return"], 2),
        "hit_rate": fmt_pct(stats_in_regime["hit_rate"], 1),
    }

    rolling_corr_60d_latest_display = {
        "corr_GLD_UUP": fmt_num(corr_latest["corr_GLD_UUP"], 2),
        "corr_GLD_IEF": fmt_num(corr_latest["corr_GLD_IEF"], 2),
        "corr_GLD_SPY": fmt_num(corr_latest["corr_GLD_SPY"], 2),
        "corr_GLD_VIX": fmt_num(corr_latest["corr_GLD_VIX"], 2),
    }

    chart_meta = save_gld_vs_usd_12m_chart(px_daily=px_daily, cfg=cfg)
    teny_meta = save_gld_vs_teny_12m_chart(px_daily=px_daily, cfg=cfg)

    latest_payload = {
        "asof": reg_now["asof"],
        "regime": {"usd": reg_now["usd"], "rates": reg_now["rates"], "risk": reg_now["risk"]},
        "regime_snapshot": regime_snapshot,
        "signals": {
            "mom_usd_12w": reg_now["mom_usd_12w"],
            "mom_rates_12w": reg_now["mom_rates_12w"],
            "vix_level": reg_now["vix_level"],
            "vix_threshold": reg_now["vix_threshold"],
        },

        "expected_impact": impact,
        "stats_in_regime": stats_in_regime,
        "rolling_corr_60d_latest": corr_latest,
        "stats_in_regime_display": stats_in_regime_display,
        "rolling_corr_60d_latest_display": rolling_corr_60d_latest_display,
        "charts": {
            "gld_vs_usd_12m": chart_meta,
            "gld_vs_10y_12m": teny_meta,
        },
        "heatmap_summary": heatmap_summary,
        "insights": insights,
    }

    regimes_payload = regime_table.to_dict(orient="records")

    tmp = corr_df.reset_index()
    first_col = tmp.columns[0]
    tmp = tmp.rename(columns={first_col: "date"})
    if np.issubdtype(tmp["date"].dtype, np.datetime64):
        tmp["date"] = tmp["date"].dt.date.astype(str)
    else:
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.date.astype(str)

    rolling_corr_payload = {
        "window_days": cfg.rolling_corr_days,
        "series": tmp.to_dict(orient="records"),
    }

    write_json(cfg.latest_path, latest_payload)
    write_json(cfg.regimes_path, regimes_payload)
    write_json(cfg.rolling_corr_path, rolling_corr_payload)

    print("✅ Updated outputs:")
    print(f" - {cfg.latest_path}")
    print(f" - {cfg.regimes_path}")
    print(f" - {cfg.rolling_corr_path}")
    print(f" - {cfg.gld_vs_usd_png_path}")
    print(f" - {cfg.gld_vs_usd_json_path}")
    print(f" - {cfg.gld_vs_teny_png_path}")
    print(f" - {cfg.gld_vs_teny_json_path}")


if __name__ == "__main__":
    main()
