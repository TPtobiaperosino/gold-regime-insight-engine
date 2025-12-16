"""
Gold Regime Insight Engine — build_gold_insights.py

Output:
- data/latest.json              -> asof + regime corrente + expected impact + insight cards
- data/regimes.json             -> tabella 8 combinazioni regime con stats storiche
- data/rolling_corr.json        -> rolling corr (60 giorni) oro vs driver (USD, rates, SPY, VIX)
- data/gld_vs_usd_12m.png        -> GLD vs USD (UUP) normalized (base=100), last 12 months
- data/gld_vs_usd_12m.json       -> series for the GLD vs UUP normalized chart
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
    scatter_png_path: str = "data/scatter_gold_vs_yield.png"
    scatter_json_path: str = "data/scatter_gold_vs_yield.json"

    rolling_corr_12m_png_path: str = "data/rolling_corr_12m.png"
    rolling_corr_12m_json_path: str = "data/rolling_corr_12m.json"


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

    # Rolling 60d correlation of daily returns (proof)
    try:
        ret = px_daily.loc[common_idx, [cfg.gold, cfg.usd]].pct_change().dropna()
        rolling_corr = ret[cfg.gold].rolling(60).corr(ret[cfg.usd])
        rolling_corr_latest = rolling_corr.dropna().iloc[-1] if not rolling_corr.dropna().empty else float("nan")
        corr_text = f"Rolling corr (60d) latest: {rolling_corr_latest:.2f}" if np.isfinite(rolling_corr_latest) else "Rolling corr (60d) latest: N/A"
    except Exception:
        rolling_corr = pd.Series(dtype=float)
        rolling_corr_latest = float("nan")
        corr_text = "Rolling corr (60d) latest: N/A"

    # optional shading where rolling corr < -0.40 (very light)
    try:
        neg_mask = rolling_corr < -0.40
        if neg_mask.any():
            seg_start = None
            for dt, flag in neg_mask.iteritems():
                if flag and seg_start is None:
                    seg_start = dt
                elif not flag and seg_start is not None:
                    ax.axvspan(seg_start, dt, color="#fdecea", alpha=0.06, linewidth=0)
                    seg_start = None
            if seg_start is not None:
                ax.axvspan(seg_start, rolling_corr.index[-1], color="#fdecea", alpha=0.06, linewidth=0)
    except Exception:
        pass

    # place rolling-corr label above the axes to avoid overlapping the lines
    fig.text(
        0.99,
        0.985,
        corr_text,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#ffffff", alpha=0.8, edgecolor="#cccccc"),
    )

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


def build_scatter_gold_yield_risk(px_daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build weekly scatter dataset: weekly GLD return vs weekly Δ10Y (bps) and risk appetite bands."""
    px_weekly = to_weekly_prices(px_daily)

    if cfg.teny not in px_weekly.columns:
        raise RuntimeError(f"Missing tenor ticker {cfg.teny} in weekly prices")

    # weekly GLD returns (decimal)
    gld_ret = px_weekly[cfg.gold].pct_change()

    # 10y yield series: ^TNX typically reports yield*10, so divide by 10 to get percent
    y10 = px_weekly[cfg.teny] / 10.0
    dy10_bps = (y10 - y10.shift(1)) * 100.0

    vix = px_weekly[cfg.vix]
    q33 = vix.rolling(cfg.vix_lookback_weeks).quantile(1 / 3)
    q66 = vix.rolling(cfg.vix_lookback_weeks).quantile(2 / 3)
    global_q33 = vix.quantile(1 / 3)
    global_q66 = vix.quantile(2 / 3)

    idx = gld_ret.index.intersection(dy10_bps.index).intersection(vix.index)

    thr33 = q33.reindex(idx).fillna(global_q33)
    thr66 = q66.reindex(idx).fillna(global_q66)

    vix_loc = vix.reindex(idx)

    risk_app = np.where(
        vix_loc <= thr33, "High risk appetite", np.where(vix_loc <= thr66, "Medium risk appetite", "Low risk appetite")
    )

    df = pd.DataFrame(
        {
            "gld_ret": gld_ret.reindex(idx),
            "dy10_bps": dy10_bps.reindex(idx),
            "risk_appetite": risk_app,
        },
        index=idx,
    )

    df = df.dropna()

    return df


def save_scatter_gold_vs_yield(df: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Save simplified binned relationship: binned mean GLD weekly return vs dy10_bps plus linear fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(cfg.scatter_png_path).parent.mkdir(parents=True, exist_ok=True)

    n_bins = 20
    df2 = df.reset_index()
    if df2.empty:
        payload = {"type": "binned_line", "x": "dy10_bps", "y": "avg_gld_weekly_return", "fit": {}, "points": []}
        write_json(cfg.scatter_json_path, payload)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title("Gold weekly return vs Δ10Y yield (bps) — binned averages")
        fig.tight_layout()
        fig.savefig(cfg.scatter_png_path, dpi=180)
        plt.close(fig)
        return {"png_path": cfg.scatter_png_path, "json_path": cfg.scatter_json_path}

    # create fixed bins over full range
    try:
        bins = pd.cut(df2["dy10_bps"], bins=n_bins)
    except Exception:
        bins = pd.qcut(df2["dy10_bps"].rank(method="first"), q=n_bins)

    agg = df2.groupby(bins).agg(x_center=("dy10_bps", "mean"), y_mean=("gld_ret", "mean"), n=("gld_ret", "count")).reset_index()
    agg = agg.dropna()
    # drop small bins
    agg = agg[agg["n"] >= 10].sort_values("x_center")

    points = [{"x": float(r["x_center"]), "y": float(r["y_mean"]), "n": int(r["n"])} for _, r in agg.iterrows()]

    # Fit linear model on binned means (y in percent) using numpy.polyfit
    fit = {}
    if len(agg) >= 2:
        x_vals = agg["x_center"].values
        y_pct = (agg["y_mean"].values * 100.0)
        coef = np.polyfit(x_vals, y_pct, 1)
        slope = float(coef[0])
        intercept = float(coef[1])
        y_hat = np.polyval(coef, x_vals)
        ss_res = float(((y_pct - y_hat) ** 2).sum())
        ss_tot = float(((y_pct - y_pct.mean()) ** 2).sum())
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        fit = {"slope": slope, "intercept": intercept, "r2": r2}
    else:
        fit = {"slope": None, "intercept": None, "r2": None}

    payload = {"type": "binned_line", "x": "dy10_bps", "y": "avg_gld_weekly_return", "fit": fit, "points": points}
    write_json(cfg.scatter_json_path, payload)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    if not agg.empty:
        ax.plot(agg["x_center"], agg["y_mean"] * 100.0, marker="o", linestyle="-", color="#1b6ca8", linewidth=1.6, markersize=6, label="Binned avg")
        if fit.get("slope") is not None:
            x_vals = np.linspace(agg["x_center"].min(), agg["x_center"].max(), 100)
            y_fit = fit["slope"] * x_vals + fit["intercept"]
            ax.plot(x_vals, y_fit, linestyle="--", color="#d35400", linewidth=1.2, label="Linear fit")
            # annotation with slope and r2
            ann = f"slope={fit['slope']:.4f}%/bp\nr2={fit['r2']:.2f}"
            fig.text(0.99, 0.02, ann, ha="right", va="bottom", fontsize=9, bbox=dict(boxstyle="round", facecolor="#ffffff", alpha=0.8, edgecolor="#cccccc"))

    ax.set_xlabel("Δ 10Y yield (bps, weekly)")
    ax.set_ylabel("Avg GLD weekly return (%)")
    ax.set_title("Gold weekly return vs Δ10Y yield (bps) — binned averages")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

    # small legend
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(cfg.scatter_png_path, dpi=180)
    plt.close(fig)

    return {"png_path": cfg.scatter_png_path, "json_path": cfg.scatter_json_path}


def save_rolling_corr_12m_chart(corr_df: pd.DataFrame, cfg: Config) -> Dict[str, str]:
    """Save rolling 60d correlations (last 12 months) chart and a JSON snapshot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = corr_df.index.max()
    start = end - pd.DateOffset(years=1)

    sub = corr_df.loc[corr_df.index >= start].copy()

    # columns of interest if present
    cols = []
    mapping = {
        f"corr_{cfg.gold}_{cfg.usd}": "corr_GLD_UUP",
        f"corr_{cfg.gold}_{cfg.rates}": "corr_GLD_IEF",
        f"corr_{cfg.gold}_{cfg.equity}": "corr_GLD_SPY",
        f"corr_{cfg.gold}_VIX": "corr_GLD_VIX",
    }
    for k in [f"corr_{cfg.gold}_{cfg.usd}", f"corr_{cfg.gold}_{cfg.rates}", f"corr_{cfg.gold}_{cfg.equity}", f"corr_{cfg.gold}_VIX"]:
        if k in sub.columns:
            cols.append(k)

    if not cols:
        # no series to plot
        Path(cfg.rolling_corr_12m_png_path).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("Rolling correlation (60d) — last 12 months")
        fig.tight_layout()
        fig.savefig(cfg.rolling_corr_12m_png_path, dpi=180)
        plt.close(fig)
        write_json(cfg.rolling_corr_12m_json_path, {"window_days": cfg.rolling_corr_days, "series": [], "latest": {}})
        return {"png_path": cfg.rolling_corr_12m_png_path, "json_path": cfg.rolling_corr_12m_json_path}

    # prepare JSON series
    series = []
    for dt, row in sub.iterrows():
        rec = {"date": dt.date().isoformat()}
        for c in cols:
            rec[c] = None if not np.isfinite(row.get(c, np.nan)) else float(row[c])
        series.append(rec)

    # latest values (last non-all-NaN row)
    latest_row = sub.dropna(how="all")[cols].iloc[-1] if not sub.dropna(how="all").empty else pd.Series({c: None for c in cols})
    latest = {c: (None if not np.isfinite(latest_row.get(c, np.nan)) else float(latest_row[c])) for c in cols}

    payload = {"window_days": cfg.rolling_corr_days, "series": series, "latest": latest}
    write_json(cfg.rolling_corr_12m_json_path, payload)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {f"corr_{cfg.gold}_{cfg.usd}": "#1b6ca8", f"corr_{cfg.gold}_{cfg.rates}": "#d35400", f"corr_{cfg.gold}_{cfg.equity}": "#2ecc71", f"corr_{cfg.gold}_VIX": "#8e44ad"}

    for c in cols:
        ax.plot(sub.index, sub[c], label=mapping.get(c, c), color=colors.get(c, None), linewidth=1.4)

    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Rolling correlation (60d) — last 12 months", fontsize=12, weight="bold")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Date")

    # latest annotation box top-right
    ann_items = []
    label_map = {f"corr_{cfg.gold}_{cfg.usd}": "UUP", f"corr_{cfg.gold}_{cfg.rates}": "IEF", f"corr_{cfg.gold}_{cfg.equity}": "SPY"}
    for c in [f"corr_{cfg.gold}_{cfg.usd}", f"corr_{cfg.gold}_{cfg.rates}", f"corr_{cfg.gold}_{cfg.equity}"]:
        if c in latest:
            val = latest[c]
            if val is None:
                s = f"{label_map.get(c, c)}=N/A"
            else:
                s = f"{label_map.get(c, c)}={val:.2f}"
            ann_items.append(s)

    ann_text = " | ".join(ann_items)
    fig.text(0.99, 0.98, f"Latest: {ann_text}", ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="#ffffff", alpha=0.8, edgecolor="#cccccc"))

    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.18)

    fig.tight_layout()
    Path(cfg.rolling_corr_12m_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.rolling_corr_12m_png_path, dpi=180)
    plt.close(fig)

    return {"png_path": cfg.rolling_corr_12m_png_path, "json_path": cfg.rolling_corr_12m_json_path}


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

    # rolling corr 12m chart
    rolling_corr_meta = save_rolling_corr_12m_chart(corr_df=corr_df, cfg=cfg)

    reg_now = current_regime(px_weekly=px_weekly, cfg=cfg)
    weekly_regimes = build_weekly_regimes(px_weekly=px_weekly, cfg=cfg)

    # scatter dataset: prepare weekly GLD return vs Δ10Y (bps)
    df_scatter = build_scatter_gold_yield_risk(px_daily=px_daily, cfg=cfg)
    scatter_meta = save_scatter_gold_vs_yield(df_scatter, cfg)
    regime_table = build_regime_table(px_weekly=px_weekly, weekly_ret=ret_weekly, cfg=cfg)
    impact, stats_in_regime = expected_impact_from_table(
        regime_table, reg_now["usd"], reg_now["rates"], reg_now["risk"]
    )

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

    latest_payload = {
        "asof": reg_now["asof"],
        "regime": {"usd": reg_now["usd"], "rates": reg_now["rates"], "risk": reg_now["risk"]},
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
        "gld_vs_usd_chart_12m": chart_meta,
        "scatter_gold_vs_yield": scatter_meta,
        "rolling_corr_chart_12m": rolling_corr_meta,
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
    print(f" - {cfg.scatter_png_path}")
    print(f" - {cfg.scatter_json_path}")


if __name__ == "__main__":
    main()
