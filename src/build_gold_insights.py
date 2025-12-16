"""
Gold Regime Insight Engine — build_gold_insights.py

Output:
- data/latest.json         -> "oggi" (asof) + regime corrente + expected impact + insight cards
- data/regimes.json        -> tabella delle 8 combinazioni regime con stats storiche
- data/rolling_corr.json   -> time series rolling corr (60 giorni) oro vs driver (USD, rates, SPY, VIX)

Note concetti:
- Adjusted Close: prezzo corretto per split/dividendi → returns più corretti
- Weekly returns: meno rumore, più “macro”
- Bond price ↑ => yield ↓ (inverso). Quindi "Rates up" ≈ bond ETF (IEF) in calo.
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# 0) CONFIG "da prodotto"
# -----------------------------

@dataclass(frozen=True)
class Config:
    # Ticker choices (Yahoo Finance)
    gold: str = "GLD"
    usd: str = "UUP"
    rates: str = "IEF"       # alternative: "TLT"
    equity: str = "SPY"
    vix: str = "^VIX"

    # Regime parameters
    momentum_weeks: int = 12       # ~ 60 trading days (≈ 3 mesi) ma in weekly
    vix_lookback_weeks: int = 104  # ~ 2 anni di weekly data
    vix_percentile: float = 0.75

    # Rolling correlation (daily)
    rolling_corr_days: int = 60

    # Data horizon
    history_period: str = "max"    # semplice, gratuito

    # Output paths
    data_dir: str = "data"
    latest_path: str = "data/latest.json"
    regimes_path: str = "data/regimes.json"
    rolling_corr_path: str = "data/rolling_corr.json"


CFG = Config()


# -----------------------------
# 1) Download data (robusto)
# -----------------------------

def download_prices(tickers: List[str], period: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Scarica serie prezzi da Yahoo Finance via yfinance.

    Perché 'Adj Close':
    - include aggiustamenti per dividendi/split
    - rende i return più "economicamente" corretti

    Edge case:
    - alcuni indici (es. ^VIX) a volte non hanno Adj Close pulito → fallback su Close.
    """
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

            # yfinance può tornare colonne multiindex (OHLCV)
            # Proviamo Adj Close, se manca usiamo Close.
            if "Adj Close" in df.columns:
                px = df["Adj Close"].copy()
            elif "Close" in df.columns:
                px = df["Close"].copy()
            else:
                # struttura alternativa: MultiIndex (Field, Ticker)
                # prova a estrarre 'Adj Close' o 'Close'
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

            # Se tickers singolo, diventa Series → normalizza a DF
            if isinstance(px, pd.Series):
                px = px.to_frame(name=tickers[0])

            # Standardizza: colonne = tickers, index = datetime
            px = px.sort_index()
            px = px.rename(columns=str)

            return px

        except Exception as e:
            if attempt == max_retries:
                raise
            # backoff leggero anti-rate-limit
            time.sleep(1.5 + random.random() * 1.5)

    raise RuntimeError("Unreachable.")


# -----------------------------
# 2) Cleaning + alignment
# -----------------------------

def align_and_clean(px: pd.DataFrame) -> pd.DataFrame:
    """
    Allinea per date comuni e rimuove missing.
    Motivazione:
    - le statistiche (corr, returns) richiedono date sincronizzate.
    """
    px = px.copy()
    px = px.dropna(how="all")
    px = px.dropna()  # drop righe dove manca qualunque ticker (semplice + robusto)
    return px


# -----------------------------
# 3) Feature engineering
# -----------------------------

def to_weekly_prices(px_daily: pd.DataFrame) -> pd.DataFrame:
    weekly = px_daily.resample("W-FRI").last().dropna()
    # Se l'ultima label weekly è > ultimo giorno realmente disponibile, la settimana è incompleta: droppala
    if weekly.index[-1] > px_daily.index[-1]:
        weekly = weekly.iloc[:-1]
    return weekly

def pct_return(px: pd.DataFrame) -> pd.DataFrame:
    """Return percentuale: (P_t / P_{t-1}) - 1"""
    return px.pct_change().dropna()


def momentum(series_weekly_price: pd.Series, weeks: int) -> pd.Series:
    """
    Momentum semplice su prezzi weekly:
    momentum_t = P_t / P_{t-weeks} - 1

    È spiegabile: "negli ultimi X settimane è salito/scese?"
    """
    return (series_weekly_price / series_weekly_price.shift(weeks) - 1).dropna()


def rolling_corr(daily_returns: pd.DataFrame, target: str, drivers: List[str], window_days: int) -> pd.DataFrame:
    """
    Rolling correlation su daily returns (60 giorni).
    Perché daily qui:
    - la rolling corr serve proprio per vedere come cambia *nel breve/medio* la relazione.
    """
    out = pd.DataFrame(index=daily_returns.index)
    for d in drivers:
        out[f"corr_{target}_{d}"] = daily_returns[target].rolling(window_days).corr(daily_returns[d])
    return out.dropna()


# -----------------------------
# 4) Regime rules (spiegabili)
# -----------------------------

def current_regime(px_weekly: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    """
    Determina regime corrente usando:
    - USD strong/weak via momentum UUP (12w)
    - Rates up/down via momentum IEF (12w) [bond ↓ ⇒ yields ↑]
    - Risk-off/on via VIX level vs percentile ultimi 104w
    """
    w = cfg.momentum_weeks

    mom_usd = momentum(px_weekly[cfg.usd], w)
    mom_rates = momentum(px_weekly[cfg.rates], w)

    # allineiamo agli stessi timestamp disponibili
    common_idx = mom_usd.index.intersection(mom_rates.index)
    mom_usd = mom_usd.loc[common_idx]
    mom_rates = mom_rates.loc[common_idx]

    # VIX: usiamo livelli weekly, percentile lookback
    vix_lvl = px_weekly[cfg.vix].loc[common_idx].dropna()
    # percentile rolling "lookback": threshold calcolato sulle ultime 104 settimane
    # (rolling quantile), poi confronti sul punto corrente.
    vix_thr = vix_lvl.rolling(cfg.vix_lookback_weeks).quantile(cfg.vix_percentile)
    vix_flag = (vix_lvl > vix_thr).dropna()

    # scegliamo l'ultima data dove abbiamo tutto
    last_dt = common_idx.max()
    # se VIX rolling non è ancora disponibile (dataset corto), fallback su quantile globale
    if last_dt not in vix_flag.index:
        global_thr = vix_lvl.quantile(cfg.vix_percentile)
        risk_off = bool(vix_lvl.loc[last_dt] > global_thr) if last_dt in vix_lvl.index else False
        vix_level = float(vix_lvl.loc[last_dt]) if last_dt in vix_lvl.index else float("nan")
        vix_threshold = float(global_thr)
    else:
        risk_off = bool(vix_flag.loc[last_dt])
        vix_level = float(vix_lvl.loc[last_dt])
        vix_threshold = float(vix_thr.loc[last_dt])

    usd_strong = bool(mom_usd.loc[last_dt] > 0)
    rates_up = bool(mom_rates.loc[last_dt] < 0)  # IEF down => yields up

    return {
        "asof": last_dt.date().isoformat(),
        "usd": "strong" if usd_strong else "weak",
        "rates": "up" if rates_up else "down",
        "risk": "off" if risk_off else "on",
        "mom_usd_12w": float(mom_usd.loc[last_dt]),
        "mom_rates_12w": float(mom_rates.loc[last_dt]),
        "vix_level": vix_level,
        "vix_threshold": vix_threshold,
    }


# -----------------------------
# 5) Stats per regime (8 combo)
# -----------------------------

def build_regime_table(px_weekly: pd.DataFrame, weekly_ret: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Crea dataset regime per ogni settimana e calcola stats dell'oro nel tempo.

    Regime per settimana t:
    - usd_strong_t = momentum(UUP, 12w) > 0
    - rates_up_t   = momentum(IEF, 12w) < 0
    - risk_off_t   = VIX_t > rolling 75% quantile (104w)

    Poi raggruppi per (usd, rates, risk) e calcoli:
    - avg weekly gold return
    - vol (std) weekly
    - hit rate (% gold_ret > 0)
    - n_obs
    """
    w = cfg.momentum_weeks

    mom_usd = momentum(px_weekly[cfg.usd], w)
    mom_rates = momentum(px_weekly[cfg.rates], w)

    # VIX threshold rolling
    vix_lvl = px_weekly[cfg.vix].dropna()
    vix_thr = vix_lvl.rolling(cfg.vix_lookback_weeks).quantile(cfg.vix_percentile)

    # Allineamento indice comune
    idx = weekly_ret.index.intersection(mom_usd.index).intersection(mom_rates.index).intersection(vix_lvl.index)
    df = pd.DataFrame(index=idx)
    df["gold_ret"] = weekly_ret.loc[idx, cfg.gold]

    df["usd"] = np.where(mom_usd.loc[idx] > 0, "strong", "weak")
    df["rates"] = np.where(mom_rates.loc[idx] < 0, "up", "down")  # IEF down => yields up

    # Se rolling threshold non disponibile in alcune date, fallback su quantile globale
    global_thr = vix_lvl.quantile(cfg.vix_percentile)
    thr = vix_thr.reindex(idx)
    df["risk"] = np.where(vix_lvl.loc[idx] > thr.fillna(global_thr), "off", "on")

    # Stats per regime
    g = df.groupby(["usd", "rates", "risk"])
    table = g["gold_ret"].agg(
        avg_weekly_return="mean",
        vol_weekly="std",
        n_obs="count",
    ).reset_index()

    # Hit rate: % settimane positive
    hit = g["gold_ret"].apply(lambda x: float((x > 0).mean())).reset_index(name="hit_rate")
    table = table.merge(hit, on=["usd", "rates", "risk"], how="left")

    # per comodità: ordina
    table = table.sort_values(["risk", "usd", "rates"]).reset_index(drop=True)
    return table


def expected_impact_from_table(regime_table: pd.DataFrame, usd: str, rates: str, risk: str) -> Tuple[str, Dict[str, float]]:
    """
    Traduce le stats storiche del regime in una label semplice:
    - Tailwind / Neutral / Headwind

    Regola: usa i tercili dell'avg_weekly_return.
    """
    row = regime_table[(regime_table["usd"] == usd) & (regime_table["rates"] == rates) & (regime_table["risk"] == risk)]
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


# -----------------------------
# 6) Insight rule-based
# -----------------------------

def generate_insights(
    regime_meta: Dict[str, str],
    regime_table: pd.DataFrame,
    corr_latest: Dict[str, float],
) -> List[Dict[str, object]]:
    """
    Genera 3–6 insight testuali, ciascuno con evidenza numerica.
    Niente LLM: solo regole + numeri.
    """
    insights = []

    usd = regime_meta["usd"]
    rates = regime_meta["rates"]
    risk = regime_meta["risk"]

    impact, stats = expected_impact_from_table(regime_table, usd, rates, risk)

    # Insight 1: regime summary
    insights.append({
        "title": "Current macro regime",
        "evidence": f"USD={usd}, Rates={rates}, Risk={risk} (asof {regime_meta['asof']})",
        "metric": {"avg_weekly_return": stats["avg_weekly_return"], "hit_rate": stats["hit_rate"], "impact": impact},
    })

    # Insight 2: USD pressure dominates (corr)
    c_usd = corr_latest.get("corr_GLD_UUP")
    if c_usd is not None and np.isfinite(c_usd) and c_usd < -0.40:
        insights.append({
            "title": "USD pressure is dominant",
            "evidence": "Rolling 60d correlation Gold vs USD is strongly negative.",
            "metric": {"rolling_corr_60d": float(c_usd), "threshold": -0.40},
        })

    # Insight 3: Rates sensitivity
    c_rates = corr_latest.get("corr_GLD_IEF")
    if c_rates is not None and np.isfinite(c_rates) and c_rates > 0.30:
        insights.append({
            "title": "Gold is behaving as a rates-sensitive asset",
            "evidence": "Positive rolling correlation with bond prices (IEF) implies inverse linkage to yields.",
            "metric": {"rolling_corr_60d": float(c_rates), "threshold": 0.30},
        })

    # Insight 4: regime historically unfavorable
    avg_all = regime_table["avg_weekly_return"]
    if np.isfinite(stats["avg_weekly_return"]):
        bottom_quartile = float(avg_all.quantile(0.25))
        if stats["avg_weekly_return"] <= bottom_quartile:
            insights.append({
                "title": "Historically unfavorable regime for gold",
                "evidence": "This regime sits in the bottom quartile of historical average gold weekly returns.",
                "metric": {"avg_weekly_return": stats["avg_weekly_return"], "bottom_quartile": bottom_quartile},
            })

    # Insight 5: low reliability
    if np.isfinite(stats["hit_rate"]) and stats["hit_rate"] < 0.50:
        insights.append({
            "title": "Low reliability in this regime",
            "evidence": "Gold was positive in less than 50% of weeks in similar macro conditions.",
            "metric": {"hit_rate": stats["hit_rate"], "threshold": 0.50},
        })

    # Insight 6: crisis hedge check (risk-off + low hit)
    if risk == "off" and np.isfinite(stats["hit_rate"]) and stats["hit_rate"] < 0.55:
        insights.append({
            "title": "Not a consistent crisis hedge (in this sample)",
            "evidence": "In risk-off weeks, gold did not deliver a high positive frequency historically.",
            "metric": {"hit_rate": stats["hit_rate"], "context": "risk=off"},
        })

    # Limita a 6 per UI pulita
    return insights[:6]


# -----------------------------
# IO: JSON writing
# -----------------------------

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
    """0.0042 -> '0.42%'"""
    if not _is_finite(x):
        return None
    return f"{x * 100:.{decimals}f}%"

def fmt_num(x: float, decimals: int = 2) -> str | None:
    """-0.1849 -> '-0.18'"""
    if not _is_finite(x):
        return None
    return f"{x:.{decimals}f}"

# -----------------------------
# MAIN
# -----------------------------

def main(cfg: Config = CFG) -> None:
    tickers = [cfg.gold, cfg.usd, cfg.rates, cfg.equity, cfg.vix]

    # (1) Download
    px_daily = download_prices(tickers=tickers, period=cfg.history_period)
    px_daily = align_and_clean(px_daily)

    # (3) Feature engineering
    ret_daily = pct_return(px_daily)

    px_weekly = to_weekly_prices(px_daily)
    ret_weekly = pct_return(px_weekly)

    # Rolling corr 60 giorni (daily)
    corr_df = rolling_corr(
        daily_returns=ret_daily,
        target=cfg.gold,
        drivers=[cfg.usd, cfg.rates, cfg.equity, cfg.vix],
        window_days=cfg.rolling_corr_days,
    )

    corr_df = corr_df.rename(columns={f"corr_{cfg.gold}_{cfg.vix}": f"corr_{cfg.gold}_VIX"})

    # (4) Regime current
    reg_now = current_regime(px_weekly=px_weekly, cfg=cfg)

    # (5) Regime table stats
    regime_table = build_regime_table(px_weekly=px_weekly, weekly_ret=ret_weekly, cfg=cfg)

    impact, stats_in_regime = expected_impact_from_table(
        regime_table, reg_now["usd"], reg_now["rates"], reg_now["risk"]
    )

    # Latest rolling correlations (ultimo punto disponibile)
    corr_last_row = corr_df.iloc[-1].to_dict()
    # rinomina chiavi in modo più pulito
    corr_latest = {
        "corr_GLD_UUP": float(corr_last_row.get(f"corr_{cfg.gold}_{cfg.usd}", np.nan)),
        "corr_GLD_IEF": float(corr_last_row.get(f"corr_{cfg.gold}_{cfg.rates}", np.nan)),
        "corr_GLD_SPY": float(corr_last_row.get(f"corr_{cfg.gold}_{cfg.equity}", np.nan)),
        "corr_GLD_VIX": float(corr_last_row.get(f"corr_{cfg.gold}_VIX", np.nan)),
    }

    # (6) Insights
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

    # Build JSON payloads
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
        "insights": insights,
    }

    regimes_payload = regime_table.to_dict(orient="records")

    rolling_corr_payload = {
        "window_days": cfg.rolling_corr_days,
        "series": (
            corr_df.reset_index()
            .rename(columns={"Date": "date", "index": "date"})
            .assign(date=lambda d: d["date"].dt.date.astype(str))
            .to_dict(orient="records")
        ),
    }

    # Write outputs
    write_json(cfg.latest_path, latest_payload)
    write_json(cfg.regimes_path, regimes_payload)
    write_json(cfg.rolling_corr_path, rolling_corr_payload)

    print("✅ Updated JSONs:")
    print(f" - {cfg.latest_path}")
    print(f" - {cfg.regimes_path}")
    print(f" - {cfg.rolling_corr_path}")


if __name__ == "__main__":
    main()
