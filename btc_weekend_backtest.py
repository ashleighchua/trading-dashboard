# btc_weekend_backtest.py
"""
BTC Weekend Pullback — Backtest
================================
Strategy: Buy BTC every Sunday when it has pulled back >=X% from its 7-day high.
Exit: Monday close. Trailing stop simulated: if Monday low <= entry*(1-trail_pct), exit at stop.

Parameter sweep:
  Pullback threshold: 3%, 5%, 7%, 10%
  Trailing stop:      2%, 3%, 5%
  Total combos:       12

Period: 2020-01-01 to 2026-03-28
Starting equity: $1,000
Position size:   20% of equity
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings("ignore")

START          = "2019-06-01"   # warmup for 7-day rolling high
BACKTEST_START = "2020-01-01"
END            = "2026-03-28"
INIT_EQ        = 1_000.0
SIZE_PCT       = 0.20

PULLBACK_THRESHOLDS = [3.0, 5.0, 7.0, 10.0]
TRAIL_PCTS          = [0.02, 0.03, 0.05]

print("Downloading BTC-USD daily bars 2019-06-01 to 2026-03-28...")
df_raw = yf.download("BTC-USD", start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)
df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
df_raw = df_raw.dropna()
print("Done. {} bars.\n".format(len(df_raw)))


def add_indicators(df):
    df = df.copy()
    df["high7"]        = df["High"].rolling(7).max()
    df["pullback_pct"] = (df["high7"] - df["Close"]) / df["high7"] * 100
    df["weekday"]      = df.index.weekday  # 6 = Sunday
    return df


def simulate_trade(df, entry_idx, entry_price, trail_pct):
    """Simulate Sunday close entry -> Monday close exit with trailing stop."""
    if entry_idx + 1 >= len(df):
        return 0.0, "no_next_bar"
    next_bar   = df.iloc[entry_idx + 1]
    stop_price = entry_price * (1 - trail_pct)
    if next_bar["Low"] <= stop_price:
        pnl_pct = (stop_price - entry_price) / entry_price * 100
        return pnl_pct, "stop"
    exit_price = next_bar["Close"]
    pnl_pct    = (exit_price - entry_price) / entry_price * 100
    return pnl_pct, "close"


def run(pullback_threshold, trail_pct):
    df     = add_indicators(df_raw)
    trades = []
    equity = INIT_EQ
    start_idx = df.index.searchsorted(pd.Timestamp(BACKTEST_START))

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        if row["weekday"] != 6:
            continue
        if pd.isna(row["pullback_pct"]) or row["pullback_pct"] < pullback_threshold:
            continue
        entry_price = row["Close"]
        if entry_price <= 0:
            continue
        trade_size      = equity * SIZE_PCT
        pnl_pct, reason = simulate_trade(df, i, entry_price, trail_pct)
        pnl_dollar      = pnl_pct / 100 * trade_size
        equity         += pnl_dollar
        trades.append({
            "date":    df.index[i],
            "entry":   round(entry_price, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_$":   round(pnl_dollar, 2),
            "reason":  reason,
            "win":     pnl_pct > 0,
            "equity":  round(equity, 2),
            "year":    df.index[i].year,
        })
    return pd.DataFrame(trades)


def stats(df):
    if df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
    n    = len(df)
    wins = df["win"].sum()
    wr   = wins / n * 100
    gp   = df.loc[df["win"],  "pnl_pct"].sum()
    gl   = abs(df.loc[~df["win"], "pnl_pct"].sum())
    pf   = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(df["pnl_$"].sum(), 2)}


# ── Sweep ──────────────────────────────────────────────────────────────────────
print("Sweeping parameters...\n")
results = []
for pb, trail in product(PULLBACK_THRESHOLDS, TRAIL_PCTS):
    trades = run(pb, trail)
    s      = stats(trades)
    s.update({"pullback": pb, "trail": trail})
    results.append(s)

results_df = pd.DataFrame(results).sort_values("pf", ascending=False)

print("=" * 62)
print("  ALL COMBOS BY PROFIT FACTOR")
print("=" * 62)
for _, row in results_df.iterrows():
    passed = row["wr"] >= 55 and row["pf"] >= 1.5 and row["n"] >= 30
    flag   = "✅" if passed else "⚠️ "
    print("  {} pb={:.0f}% stop={:.0f}% | {:3d} trades  {:.1f}% WR  PF {:.2f}  ${:+.0f}".format(
        flag, row["pullback"], row["trail"] * 100,
        int(row["n"]), row["wr"], row["pf"], row["pnl"]
    ))

# ── Year-by-year for best passing combo ────────────────────────────────────────
valid = results_df[
    (results_df["n"]  >= 30) &
    (results_df["wr"] >= 55) &
    (results_df["pf"] >= 1.5)
]
if not valid.empty:
    best = valid.iloc[0]
    print("\n{}\n  BEST: pb={:.0f}% stop={:.0f}%\n{}".format(
        "=" * 62, best["pullback"], best["trail"] * 100, "=" * 62))
    best_trades = run(best["pullback"], best["trail"])
    if not best_trades.empty:
        yearly = best_trades.groupby("year").agg(
            n   =("pnl_pct", "count"),
            wr  =("win",     lambda x: x.mean() * 100),
            pnl =("pnl_$",  "sum"),
        )
        for yr, row in yearly.iterrows():
            bar  = "█" * max(0, int(row["pnl"] / 5)) if row["pnl"] > 0 else "▒" * max(0, int(abs(row["pnl"]) / 5))
            flag = "✓" if row["wr"] >= 55 else "✗"
            print("    {}: {:2d} trades  {:4.1f}% WR  ${:+6.1f}  {}  {}".format(
                yr, int(row["n"]), row["wr"], row["pnl"], flag, bar))
else:
    print("\n⚠️  No combo passed acceptance criteria (>=30 trades, >=55% WR, PF >=1.5)")
    best = results_df.iloc[0]
    print("   Best available: pb={:.0f}% stop={:.0f}% | {} trades  {:.1f}% WR  PF {:.2f}".format(
        best["pullback"], best["trail"] * 100, int(best["n"]), best["wr"], best["pf"]))
