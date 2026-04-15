#!/usr/bin/env python3
"""
GLD (SPDR Gold ETF) Backtest
==============================
Tests two strategy families side-by-side:
  Part 1 — Day-of-week patterns (buy open, sell close)
  Part 2 — Trend pullback long (EMA uptrend + RSI oversold + pullback from high)

Data source: Alpaca → Tiingo → yfinance (via dashboard/data_provider.py)
Period: 2020-01-01 to present
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

SYMBOL         = "GLD"
START          = "2019-06-01"   # extra history for EMA warmup
END            = datetime.today().strftime("%Y-%m-%d")
BACKTEST_START = "2020-01-01"
SL_PCT         = 1.5            # 1.5% stop loss — GLD daily range ~0.5-1%
INIT_EQ        = 10_000.0

print(f"Downloading {SYMBOL} data ({START} → today)...")
# data_provider.download() takes period strings like "5y", "max"
# For backtests we need a specific start date, so use the Alpaca client directly
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

key    = os.environ.get("APCA_API_KEY_ID", "")
secret = os.environ.get("APCA_API_SECRET_KEY", "")
if not key or not secret:
    raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env")
client = StockHistoricalDataClient(key, secret)

req = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=datetime.strptime(START, "%Y-%m-%d"),
    end=datetime.strptime(END, "%Y-%m-%d"),
)
bars = client.get_stock_bars(req)
df_raw = bars.df

# Alpaca returns MultiIndex (symbol, timestamp) — flatten to single index
if isinstance(df_raw.index, pd.MultiIndex):
    df_raw = df_raw.xs(SYMBOL, level="symbol")
df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
df_raw = df_raw.rename(columns={
    "open": "Open", "high": "High", "low": "Low",
    "close": "Close", "volume": "Volume"
})
df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
print(f"Done. {len(df_raw)} bars ({df_raw.index[0].date()} → {df_raw.index[-1].date()})\n")

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — DAY-OF-WEEK PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

df_dow = df_raw.copy()
df_dow["Return"]     = (df_dow["Close"] - df_dow["Open"]) / df_dow["Open"] * 100
df_dow["Color"]      = df_dow["Return"].apply(lambda x: "Green" if x >= 0 else "Red")
df_dow["Prev_Color"] = df_dow["Color"].shift(1)
df_dow["Prev_Ret"]   = df_dow["Return"].shift(1)
df_dow["DayOfWeek"]  = df_dow.index.dayofweek
df_dow = df_dow[df_dow.index >= BACKTEST_START].dropna(subset=["Prev_Color"])


def backtest_dow(trades_df):
    """Run a day-of-week backtest with stop loss. Returns dict of stats."""
    capital = INIT_EQ
    log = []
    for _, row in trades_df.iterrows():
        stop_price = row["Open"] * (1 - SL_PCT / 100)
        exit_price = stop_price if row["Low"] <= stop_price else row["Close"]
        ret = (exit_price - row["Open"]) / row["Open"] * 100
        pnl = capital * (ret / 100)
        capital += pnl
        log.append({"ret": ret, "win": ret > 0, "pnl": pnl})
    if not log:
        return None
    tl = pd.DataFrame(log)
    n  = len(tl)
    wr = tl["win"].mean() * 100
    gp = tl.loc[tl["win"],  "ret"].sum()
    gl = abs(tl.loc[~tl["win"], "ret"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(tl["pnl"].sum(), 2)}


print("=" * 70)
print("  PART 1 — DAY-OF-WEEK PATTERNS")
print("=" * 70)

thresholds  = [0.0, 0.3, 0.5, 0.7, 1.0]
dow_signals = []

for dow in range(5):
    day_df = df_dow[df_dow["DayOfWeek"] == dow]
    for prev_color in ["Red", "Green"]:
        for thresh in thresholds:
            if prev_color == "Red":
                filtered = day_df[(day_df["Prev_Color"] == "Red") & (day_df["Prev_Ret"] <= -thresh)]
            else:
                filtered = day_df[(day_df["Prev_Color"] == "Green") & (day_df["Prev_Ret"] >= thresh)]
            if len(filtered) < 20:
                continue
            s = backtest_dow(filtered)
            if s is None:
                continue
            cond = f"Prev {prev_color} >={thresh:.1f}%" if prev_color == "Green" else f"Prev {prev_color} <=-{thresh:.1f}%"
            s.update({"day": day_names[dow], "condition": cond, "dow": dow})
            dow_signals.append(s)

# Print results — only WR >= 60% and PF >= 1.5
good_dow = [s for s in dow_signals if s["wr"] >= 60 and s["pf"] >= 1.5]
good_dow.sort(key=lambda x: x["pf"], reverse=True)

if good_dow:
    print(f"\n  {'Day':<12} {'Condition':<35} {'WR':>6} {'PF':>6} {'n':>5} {'P&L':>8}")
    print("  " + "-" * 75)
    for s in good_dow[:15]:
        flag = "✅" if s["n"] >= 20 else "⚠️ "
        print(f"  {flag} {s['day']:<10} {s['condition']:<35} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['n']:>5}  ${s['pnl']:>+.0f}")
else:
    print("\n  No strong day-of-week patterns found (WR<60% or PF<1.5 across all combos).")
    print("  This is expected for gold — it moves on macro events, not calendar effects.")
