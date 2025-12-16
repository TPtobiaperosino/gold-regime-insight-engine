"""
Gold Regime Insight Engine — build_gold_insights.py

Output:
- data/latest.json              -> asof + regime corrente + expected impact + insight cards
- data/regimes.json             -> tabella 8 combinazioni regime con stats storiche
- data/rolling_corr.json        -> rolling corr (60 giorni) oro vs driver (USD, rates, SPY, VIX)
- data/gold_12m_regimes.png     -> GLD Adj Close (12 mesi) + regime bands (weekly)
- data/gold_12m_regimes.json    -> serie daily per chart (date, price, regime labels)
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

    momentum_weeks: int = 12
    vix_lookback_weeks: int = 104
    vix_percentile: float = 0.75

    rolling_corr_days: int = 60
    history_period: str = "max"

    latest_path: str = "data/latest.json"
    regimes_path: str = "data/regimes.json"
    rolling_corr_path: str = "data/rolling_corr.json"

    gold_12m_png_path: str = "data/gold_12m_regimes.png"
    gold_12m_json_path: str = "data/gold_12m_regimes.json"


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


def save_gold_12m_regime_chart(
    px_daily: pd.DataFrame,
    weekly_regimes: pd.DataFrame,
    reg_now: Dict[str, object],
    cfg: Config,
    expected_impact: str | None = None,
) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = px_daily.index.max()
    start = end - pd.DateOffset(years=1)

    gold = px_daily.loc[px_daily.index >= start, cfg.gold].dropna()

    reg_series = weekly_regimes["regime"].reindex(gold.index, method="ffill")
    reg_usd = weekly_regimes["usd"].reindex(gold.index, method="ffill")
    reg_rates = weekly_regimes["rates"].reindex(gold.index, method="ffill")
    reg_risk = weekly_regimes["risk"].reindex(gold.index, method="ffill")

    mask = reg_series.notna()
    gold = gold.loc[mask]
    reg_series = reg_series.loc[mask]
    reg_usd = reg_usd.loc[mask]
    reg_rates = reg_rates.loc[mask]
    reg_risk = reg_risk.loc[mask]

    # deterministic color mapping for the 8 regimes (risk part now uses 'low'/'high')
    regime_colors = {
        "strong_down_low": "#c0392b",
        "strong_down_high": "#e67e22",
        "strong_up_low": "#2980b9",
        "strong_up_high": "#5dade2",
        "weak_down_low": "#7f0000",
        "weak_down_high": "#d35400",
        "weak_up_low": "#1b4f72",
        "weak_up_high": "#85c1e9",
    }
    # fallback for any unexpected regimes
    unique_regimes = pd.Index(reg_series.unique()).sort_values()
    color_map = {r: regime_colors.get(r, "#bbbbbb") for r in unique_regimes}

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(gold.index, gold.values, color="#222222", linewidth=1.4)

    # Plot risk-only bands (risk appetite) as subtle background
    risk_color_map = {"high": "#d6eaf8", "low": "#f9d6d6"}
    dates = reg_risk.index
    labels = reg_risk.values

    seg_start = dates[0]
    prev = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != prev:
            seg_end = dates[i]
            ax.axvspan(seg_start, seg_end, alpha=0.12, color=risk_color_map.get(prev, "#eeeeee"))
            seg_start = dates[i]
            prev = labels[i]
    ax.axvspan(seg_start, dates[-1], alpha=0.12, color=risk_color_map.get(prev, "#eeeeee"))

    # add vertical line for asof date
    try:
        asof_dt = pd.to_datetime(reg_now.get("asof"))
        ax.axvline(asof_dt, color="#333333", linestyle="--", linewidth=1)
        ax.text(
            asof_dt,
            ax.get_ylim()[1],
            f" asof {asof_dt.date().isoformat()}",
            va="top",
            ha="left",
            fontsize=8,
            color="#333333",
        )
    except Exception:
        pass

    # title and concise subtitle
    ax.set_title("GLD (Adj Close) — last 12 months", fontsize=12, weight="bold")
    subtitle = f"Current regime: USD {reg_now['usd']} | Rates {reg_now['rates']} | Risk {reg_now['risk']} (asof {reg_now['asof']})"
    ax.text(0.01, 0.96, subtitle, transform=ax.transAxes, fontsize=9, va="top")

    # optional small textbox with expected impact
    if expected_impact:
        tb_text = f"Impact: {expected_impact}"
        ax.text(
            0.99,
            0.02,
            tb_text,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="#ffffff", alpha=0.7, edgecolor="#dddddd"),
        )

    # y-axis grid only
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#999999", alpha=0.4)
    ax.xaxis.grid(False)

    # format y-axis as dollars with 0 decimals
    try:
        from matplotlib.ticker import FuncFormatter

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
    except Exception:
        pass

    # legend: only show risk appetite legend for clarity
    import matplotlib.patches as mpatches

    risk_patches = [
        mpatches.Patch(color=risk_color_map["high"], label="Risk appetite: high"),
        mpatches.Patch(color=risk_color_map["low"], label="Risk appetite: low"),
    ]
    ax.legend(handles=risk_patches, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9)

    ax.set_xlabel("Date")
    ax.set_ylabel("GLD Adj Close")

    fig.tight_layout()
    Path(cfg.gold_12m_png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.gold_12m_png_path, dpi=180)
    plt.close(fig)

    df_out = pd.DataFrame(
        {
            "date": gold.index.date.astype(str),
            "gld_adj_close": gold.values.astype(float),
            "regime": reg_series.values.astype(str),
            "usd": reg_usd.values.astype(str),
            "rates": reg_rates.values.astype(str),
            "risk": reg_risk.values.astype(str),
        }
    )
    write_json(cfg.gold_12m_json_path, {"series": df_out.to_dict(orient="records")})

    return {"png_path": cfg.gold_12m_png_path, "json_path": cfg.gold_12m_json_path}


def main(cfg: Config = CFG) -> None:
    tickers = [cfg.gold, cfg.usd, cfg.rates, cfg.equity, cfg.vix]

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

    chart_meta = save_gold_12m_regime_chart(
        px_daily=px_daily,
        weekly_regimes=weekly_regimes,
        reg_now=reg_now,
        cfg=cfg,
        expected_impact=impact,
    )

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
        "price_chart_12m": chart_meta,
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
    print(f" - {cfg.gold_12m_png_path}")
    print(f" - {cfg.gold_12m_json_path}")


if __name__ == "__main__":
    main()
