"""
Trend Pullback Long Backtest
=============================
Mirror of bear rally fade. Buy when EMA_fast > EMA_slow (uptrend)
and price has pulled back X% from 10-day high.

Parameter sweep:
  EMA combo:      (20, 50) vs (50, 200)
  Pullback depth: 2%, 3%, 4%, 5%
  Trailing stop:  1.5%, 2%, 3%
  Max hold:       5, 10, 15 days

Period: 2020-01-01 to 2026-03-28
Starting equity: $1,000
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings("ignore")

START          = "2019-06-01"
BACKTEST_START = "2020-01-01"
END            = "2026-03-28"
INIT_EQ        = 1_000.0

LONG_WATCHLIST = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "MSFT", "AMD"]

EMA_COMBOS    = [(20, 50), (50, 200)]
PULLBACK_PCTS = [2.0, 3.0, 4.0, 5.0]
TRAIL_PCTS    = [0.015, 0.020, 0.030]
MAX_HOLDS     = [5, 10, 15]

print(f"Downloading {len(LONG_WATCHLIST)} tickers {START}→{END}...")
raw = yf.download(LONG_WATCHLIST, start=START, end=END, auto_adjust=True, progress=False)
print("Done.\n")

def get_df(ticker):
    df = pd.DataFrame({
        "Open":   raw["Open"][ticker],
        "High":   raw["High"][ticker],
        "Low":    raw["Low"][ticker],
        "Close":  raw["Close"][ticker],
        "Volume": raw["Volume"][ticker],
    }).dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

data = {t: get_df(t) for t in LONG_WATCHLIST}


def add_indicators(df, ema_fast, ema_slow):
    df = df.copy()
    df["ema_fast"]    = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"]    = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["high10"]      = df["High"].rolling(10).max()
    df["pullback_pct"] = (df["high10"] - df["Close"]) / df["high10"] * 100
    return df


def is_pullback_long(df, i, pullback_threshold):
    if i < 200:
        return False
    row = df.iloc[i]
    if row["ema_fast"] <= row["ema_slow"]:
        return False
    window = df.iloc[max(0, i - 2):i + 1]
    return window["pullback_pct"].max() >= pullback_threshold


def simulate_long(df, entry_idx, entry_price, trail_pct=0.020, max_hold=10):
    stop = entry_price * (1 - trail_pct)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        high  = df.iloc[j]["High"]
        low_  = df.iloc[j]["Low"]
        close = df.iloc[j]["Close"]
        new_stop = close * (1 - trail_pct)
        if new_stop > stop:
            stop = new_stop
        if low_ <= stop:
            pnl_pct = (stop - entry_price) / entry_price * 100
            return pnl_pct, j, "stop"
    exit_price = df.iloc[min(entry_idx + max_hold, len(df) - 1)]["Close"]
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    return pnl_pct, min(entry_idx + max_hold, len(df) - 1), "maxhold"


def run_pullback_long(ema_fast, ema_slow, pullback_pct, trail_pct, max_hold):
    trades = []
    equity = INIT_EQ
    open_positions = {}

    all_dates = data["SPY"].index
    start_idx = all_dates.searchsorted(pd.Timestamp(BACKTEST_START))

    dfs = {t: add_indicators(data[t], ema_fast, ema_slow) for t in LONG_WATCHLIST}

    for i, date in enumerate(all_dates):
        if i < start_idx:
            continue
        open_positions = {t: ei for t, ei in open_positions.items() if ei > i}

        for ticker in LONG_WATCHLIST:
            if ticker in open_positions:
                continue
            df = dfs[ticker]
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx < 200:
                continue
            if not is_pullback_long(df, idx, pullback_pct):
                continue
            if idx + 1 >= len(df):
                continue
            entry_price = df.iloc[idx + 1]["Open"]
            if entry_price <= 0:
                continue

            trade_size = equity * 0.50
            pnl_pct, exit_idx, reason = simulate_long(df, idx + 1, entry_price, trail_pct, max_hold)
            pnl_dollar = pnl_pct / 100 * trade_size
            equity += pnl_dollar
            open_positions[ticker] = exit_idx + 1

            trades.append({
                "date":    date,
                "ticker":  ticker,
                "entry":   round(entry_price, 2),
                "pnl_pct": round(pnl_pct, 2),
                "pnl_$":   round(pnl_dollar, 2),
                "reason":  reason,
                "win":     pnl_pct > 0,
                "equity":  round(equity, 2),
                "year":    date.year,
            })
            break

    return pd.DataFrame(trades)


def stats(df, label):
    if df.empty:
        return {"label": label, "n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
    n    = len(df)
    wins = df["win"].sum()
    wr   = wins / n * 100
    gp   = df.loc[df["win"],  "pnl_pct"].sum()
    gl   = abs(df.loc[~df["win"], "pnl_pct"].sum())
    pf   = gp / gl if gl > 0 else float("inf")
    pnl  = df["pnl_$"].sum()
    return {"label": label, "n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(pnl, 2)}


# ── Parameter sweep ────────────────────────────────────────────────────────────
print("Running parameter sweep (this takes ~2-3 minutes)...\n")

results = []
for (ef, es), pb, trail, hold in product(EMA_COMBOS, PULLBACK_PCTS, TRAIL_PCTS, MAX_HOLDS):
    trades = run_pullback_long(ef, es, pb, trail, hold)
    s = stats(trades, f"EMA({ef},{es}) pb={pb}% stop={trail*100:.1f}% hold={hold}d")
    s.update({"ema_fast": ef, "ema_slow": es, "pullback": pb, "trail": trail, "hold": hold})
    results.append(s)

results_df = pd.DataFrame(results).sort_values("pf", ascending=False)

print("=" * 70)
print("  TOP 20 PARAMETER COMBOS BY PROFIT FACTOR  (min 30 trades)")
print("=" * 70)
top = results_df[results_df["n"] >= 30].head(20)
for _, row in top.iterrows():
    flag = "✅" if row["wr"] >= 55 and row["pf"] >= 1.5 else "⚠️ "
    print(f"  {flag} EMA({int(row['ema_fast'])},{int(row['ema_slow'])}) "
          f"pb={row['pullback']:.0f}% stop={row['trail']*100:.1f}% hold={int(row['hold'])}d "
          f"| {row['n']:3.0f} trades  {row['wr']:.1f}% WR  PF {row['pf']:.2f}  ${row['pnl']:+.0f}")

# ── Best combo full breakdown ──────────────────────────────────────────────────
valid = results_df[results_df["n"] >= 30]
if valid.empty:
    print("\nNo combos with >= 30 trades found.")
else:
    best = valid.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"  BEST COMBO: EMA({int(best['ema_fast'])},{int(best['ema_slow'])}) "
          f"pullback={best['pullback']:.0f}% stop={best['trail']*100:.1f}% hold={int(best['hold'])}d")
    print(f"{'=' * 70}")

    best_trades = run_pullback_long(
        int(best["ema_fast"]), int(best["ema_slow"]),
        best["pullback"], best["trail"], int(best["hold"])
    )

    if not best_trades.empty:
        by_ticker = best_trades.groupby("ticker").agg(
            n    =("pnl_pct", "count"),
            wr   =("win",     lambda x: x.mean() * 100),
            pf_n =("pnl_pct", lambda x: x[x > 0].sum()),
            pf_d =("pnl_pct", lambda x: abs(x[x < 0].sum())),
            pnl  =("pnl_$",   "sum"),
        )
        by_ticker["pf"] = by_ticker["pf_n"] / by_ticker["pf_d"].replace(0, np.nan)
        print("\n  Per-ticker breakdown:")
        for t, row in by_ticker.sort_values("pnl", ascending=False).iterrows():
            flag = "✓" if row["wr"] >= 55 and row["pf"] >= 1.5 else "✗"
            print(f"    {t:<6} {row['n']:2.0f} trades  {row['wr']:4.1f}% WR  "
                  f"PF {row['pf']:.2f}  ${row['pnl']:+.1f}  {flag}")

        yearly = best_trades.groupby("year").agg(
            n   =("pnl_pct", "count"),
            wr  =("win",     lambda x: x.mean() * 100),
            pnl =("pnl_$",  "sum"),
        )
        print("\n  Year-by-year (best combo, all tickers):")
        for yr, row in yearly.iterrows():
            bar  = "█" * max(0, int(row["pnl"] / 5)) if row["pnl"] > 0 else "▒" * max(0, int(abs(row["pnl"]) / 5))
            flag = "✓" if row["wr"] >= 55 else "✗"
            print(f"    {yr}: {row['n']:2.0f} trades  {row['wr']:4.1f}% WR  ${row['pnl']:+6.1f}  {flag}  {bar}")
