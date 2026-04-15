# GLD Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive GLD backtest that tests day-of-week patterns and trend pullback long side-by-side, using Alpaca as the data source, and identifies the best strategy for scanner integration.

**Architecture:** Single script `gld_backtest.py` at project root. Downloads GLD data via `dashboard/data_provider.py` (Alpaca → Tiingo → yfinance fallback). Runs two independent strategy families and prints a clear winner. Saves chart to `gld_backtest.png`.

**Tech Stack:** Python, pandas, numpy, matplotlib, `dashboard/data_provider.py` (already exists), dotenv

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `gld_backtest.py` | Create | Main backtest script |
| `gld_backtest.png` | Create (output) | Chart saved by script |
| `dashboard/premarket_scanner.py` | Modify (post-backtest) | Add GLD if strategy validates |

---

## Task 1: Scaffold and data download

**Files:**
- Create: `gld_backtest.py`

- [ ] **Step 1: Create the script with imports, config, and data download**

```python
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

# Load .env so data_provider can find Alpaca/Tiingo keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Use the same data provider as the live scanner
sys.path.insert(0, str(Path(__file__).parent / "dashboard"))
from data_provider import download

SYMBOL         = "GLD"
START          = "2019-06-01"   # extra history for EMA warmup
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
client = StockHistoricalDataClient(key, secret)

req = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=datetime.strptime(START, "%Y-%m-%d"),
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
```

- [ ] **Step 2: Run the script to verify data downloads cleanly**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 gld_backtest.py
```

Expected output:
```
Downloading GLD data (2019-06-01 → today)...
Done. ~1600 bars (2019-06-03 → 2026-04-xx)
```

- [ ] **Step 3: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add gld_backtest.py
git commit -m "feat: scaffold GLD backtest with Alpaca data download"
```

---

## Task 2: Part 1 — Day-of-week pattern scan

**Files:**
- Modify: `gld_backtest.py`

- [ ] **Step 1: Add day-of-week helper columns and pattern scanner**

Append to `gld_backtest.py` after the data download block:

```python
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

# 2-day combo patterns
for d1 in range(4):
    for d2 in range(5):
        if d2 != d1 + 1:
            continue
        trade_dow = (d2 + 1) % 5  # day after the 2-day pattern
        if trade_dow == 0 and d2 != 4:  # skip if it would jump a weekend oddly
            continue
        d1_df = df_dow[df_dow["DayOfWeek"] == d1][["Color", "Ret" if "Ret" in df_dow.columns else "Return"]].rename(columns={"Return": "d1_ret", "Color": "d1_color"})
        d2_df = df_dow[df_dow["DayOfWeek"] == d2][["Color", "Return"]].rename(columns={"Return": "d2_ret", "Color": "d2_color"})
        combined = d2_df.join(d1_df.shift(-1), how="inner")  # align: d1 is yesterday of d2
        for c1, c2 in [("Red", "Red"), ("Green", "Green"), ("Red", "Green"), ("Green", "Red")]:
            filtered_combo = combined[(combined["d1_color"] == c1) & (combined["d2_color"] == c2)]
            trade_df = df_dow[df_dow["DayOfWeek"] == trade_dow]
            matched = trade_df[trade_df.index.isin(
                filtered_combo.index + pd.Timedelta(days=1 if trade_dow != 0 else 3)
            )]
            if len(matched) < 20:
                continue
            s = backtest_dow(matched)
            if s is None or s["wr"] < 60 or s["pf"] < 1.5:
                continue
            label = f"{day_names[d1]} {c1} + {day_names[d2]} {c2}"
            s.update({"day": day_names[trade_dow], "condition": label, "dow": trade_dow})
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
```

- [ ] **Step 2: Run and verify Part 1 output**

```bash
python3 gld_backtest.py
```

Expected: either no signals found (most likely) or a short table of signals. No errors.

- [ ] **Step 3: Commit**

```bash
git add gld_backtest.py
git commit -m "feat: add GLD day-of-week pattern scan (Part 1)"
```

---

## Task 3: Part 2 — Trend pullback long sweep

**Files:**
- Modify: `gld_backtest.py`

- [ ] **Step 1: Add indicator calculation and trend pullback sweep**

Append to `gld_backtest.py` after Part 1:

```python
# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — TREND PULLBACK LONG
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART 2 — TREND PULLBACK LONG")
print("=" * 70)

EMA_COMBOS    = [(20, 50), (50, 200)]
PULLBACK_PCTS = [2.0, 3.0, 4.0, 5.0]
RSI_FILTERS   = [35, 40, 45]
TRAIL_PCTS    = [0.015, 0.020]
MAX_HOLDS     = [5, 10, 15]


def add_indicators(df, ema_fast, ema_slow):
    df = df.copy()
    df["ema_fast"]     = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"]     = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["high10"]       = df["High"].rolling(10).max()
    df["pullback_pct"] = (df["high10"] - df["Close"]) / df["high10"] * 100
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df


def is_pullback_entry(df, i, pullback_threshold, rsi_max):
    if i < 200:
        return False
    row = df.iloc[i]
    if row["ema_fast"] <= row["ema_slow"]:
        return False
    window = df.iloc[max(0, i - 2):i + 1]
    if window["pullback_pct"].max() < pullback_threshold:
        return False
    if pd.isna(row["rsi14"]) or row["rsi14"] >= rsi_max:
        return False
    return True


def simulate_swing(df, entry_idx, entry_price, trail_pct, max_hold):
    stop = entry_price * (1 - trail_pct)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        high  = df.iloc[j]["High"]
        low_  = df.iloc[j]["Low"]
        close = df.iloc[j]["Close"]
        new_stop = close * (1 - trail_pct)
        if new_stop > stop:
            stop = new_stop
        if low_ <= stop:
            return (stop - entry_price) / entry_price * 100, j, "stop"
    exit_price = df.iloc[min(entry_idx + max_hold, len(df) - 1)]["Close"]
    return (exit_price - entry_price) / entry_price * 100, min(entry_idx + max_hold, len(df) - 1), "maxhold"


def run_pullback(ema_fast, ema_slow, pullback_pct, rsi_max, trail_pct, max_hold):
    df     = add_indicators(df_raw, ema_fast, ema_slow)
    trades = []
    equity = INIT_EQ
    in_trade_until = -1
    start_idx = df.index.searchsorted(pd.Timestamp(BACKTEST_START))

    for i in range(start_idx, len(df)):
        if i <= in_trade_until:
            continue
        if not is_pullback_entry(df, i, pullback_pct, rsi_max):
            continue
        if i + 1 >= len(df):
            continue
        entry_price = df.iloc[i + 1]["Open"]
        if entry_price <= 0:
            continue
        trade_size = equity * 0.5
        pnl_pct, exit_idx, reason = simulate_swing(df, i + 1, entry_price, trail_pct, max_hold)
        pnl_dollar = pnl_pct / 100 * trade_size
        equity += pnl_dollar
        in_trade_until = exit_idx
        trades.append({
            "date": df.index[i], "entry": round(entry_price, 2),
            "pnl_pct": round(pnl_pct, 2), "pnl_$": round(pnl_dollar, 2),
            "reason": reason, "win": pnl_pct > 0, "equity": round(equity, 2),
            "year": df.index[i].year,
        })
    return pd.DataFrame(trades)


def pullback_stats(df):
    if df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
    n  = len(df)
    wr = df["win"].mean() * 100
    gp = df.loc[df["win"],  "pnl_pct"].sum()
    gl = abs(df.loc[~df["win"], "pnl_pct"].sum())
    pf = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(df["pnl_$"].sum(), 2)}


print("\nSweeping parameters (this takes ~30 seconds)...")
pb_results = []
for (ef, es), pb, rsi, trail, hold in product(EMA_COMBOS, PULLBACK_PCTS, RSI_FILTERS, TRAIL_PCTS, MAX_HOLDS):
    trades = run_pullback(ef, es, pb, rsi, trail, hold)
    s = pullback_stats(trades)
    s.update({"ema_fast": ef, "ema_slow": es, "pullback": pb,
               "rsi_max": rsi, "trail": trail, "hold": hold})
    pb_results.append(s)

pb_df = pd.DataFrame(pb_results).sort_values("pf", ascending=False)

print(f"\n  {'Config':<55} {'WR':>6} {'PF':>6} {'n':>5} {'P&L':>8}")
print("  " + "-" * 85)
top_pb = pb_df[pb_df["n"] >= 15].head(15)
for _, row in top_pb.iterrows():
    flag = "✅" if row["wr"] >= 55 and row["pf"] >= 2.0 else "⚠️ "
    cfg  = (f"EMA({int(row['ema_fast'])},{int(row['ema_slow'])}) "
            f"pb={row['pullback']:.0f}% RSI<{int(row['rsi_max'])} "
            f"stop={row['trail']*100:.1f}% hold={int(row['hold'])}d")
    print(f"  {flag} {cfg:<55} {row['wr']:>5.1f}% {row['pf']:>5.2f} {int(row['n']):>5}  ${row['pnl']:>+.0f}")

# Best config year-by-year
valid_pb = pb_df[pb_df["n"] >= 15]
if not valid_pb.empty:
    best = valid_pb.iloc[0]
    best_trades = run_pullback(
        int(best["ema_fast"]), int(best["ema_slow"]),
        best["pullback"], best["rsi_max"], best["trail"], int(best["hold"])
    )
    print(f"\n  BEST CONFIG YEAR-BY-YEAR:")
    if not best_trades.empty:
        yearly = best_trades.groupby("year").agg(
            n  =("pnl_pct", "count"),
            wr =("win",     lambda x: x.mean() * 100),
            pnl=("pnl_$",  "sum"),
        )
        for yr, row in yearly.iterrows():
            bar  = "█" * max(0, int(row["pnl"] / 50))
            flag = "✓" if row["wr"] >= 55 else "✗"
            print(f"    {yr}: {int(row['n']):2d} trades  {row['wr']:4.1f}% WR  ${row['pnl']:+6.0f}  {flag}  {bar}")
```

- [ ] **Step 2: Run and verify Part 2 output**

```bash
python3 gld_backtest.py
```

Expected: table of parameter sweep results, best config year-by-year. No errors.

- [ ] **Step 3: Commit**

```bash
git add gld_backtest.py
git commit -m "feat: add GLD trend pullback long sweep (Part 2)"
```

---

## Task 4: Winner declaration and chart

**Files:**
- Modify: `gld_backtest.py`

- [ ] **Step 1: Add winner declaration and save chart**

Append to `gld_backtest.py`:

```python
# ══════════════════════════════════════════════════════════════════════════════
# WINNER + SCANNER READINESS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)

# Best day-of-week signal
best_dow = max(good_dow, key=lambda x: x["pf"]) if good_dow else None
# Best pullback config
best_pb  = valid_pb.iloc[0].to_dict() if not valid_pb.empty else None

def scanner_ready(s):
    if s is None:
        return False
    return s["pf"] >= 2.0 and s["wr"] >= 58 and s["n"] >= 15

dow_ready = scanner_ready(best_dow)
pb_ready  = scanner_ready(best_pb)

print()
if best_dow:
    flag = "✅ SCANNER READY" if dow_ready else "⚠️  not ready"
    print(f"  Day-of-week best:  WR={best_dow['wr']}%  PF={best_dow['pf']}  n={best_dow['n']}  [{flag}]")
else:
    print("  Day-of-week:  No signals found (expected for gold)")

if best_pb:
    flag = "✅ SCANNER READY" if pb_ready else "⚠️  not ready"
    cfg  = (f"EMA({int(best_pb['ema_fast'])},{int(best_pb['ema_slow'])}) "
            f"pb={best_pb['pullback']:.0f}% RSI<{int(best_pb['rsi_max'])} "
            f"stop={best_pb['trail']*100:.1f}% hold={int(best_pb['hold'])}d")
    print(f"  Trend pullback:    WR={best_pb['wr']}%  PF={best_pb['pf']}  n={int(best_pb['n'])}  [{flag}]")
    print(f"  Config: {cfg}")
else:
    print("  Trend pullback: No valid configs found")

# Pick overall winner
if pb_ready and (not dow_ready or best_pb["pf"] >= best_dow["pf"]):
    print("\n  🏆 WINNER: Trend Pullback Long")
    print("     Add GLD to LONG_WATCHLIST in premarket_scanner.py")
elif dow_ready:
    print("\n  🏆 WINNER: Day-of-Week Pattern")
    print(f"     Signal: {best_dow['day']} — {best_dow['condition']}")
else:
    print("\n  ❌ No strategy meets the bar. Do not add GLD to scanner.")
    print("     Consider revisiting in 6 months with more data.")

# ── Chart ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("GLD Backtest Results", fontsize=14, fontweight="bold")

# Equity curve for best pullback config
if best_pb and not valid_pb.empty:
    best_trades_chart = run_pullback(
        int(best_pb["ema_fast"]), int(best_pb["ema_slow"]),
        best_pb["pullback"], best_pb["rsi_max"], best_pb["trail"], int(best_pb["hold"])
    )
    if not best_trades_chart.empty:
        axes[0].plot(best_trades_chart["date"], best_trades_chart["equity"], color="#26a69a", linewidth=2)
        axes[0].axhline(y=INIT_EQ, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title(f"Trend Pullback Equity Curve (WR={best_pb['wr']}%, PF={best_pb['pf']})")
        axes[0].set_ylabel("Equity ($)")
        axes[0].grid(True, alpha=0.3)

# GLD price with buy signals
df_chart = add_indicators(df_raw, int(best_pb["ema_fast"]) if best_pb else 50,
                           int(best_pb["ema_slow"]) if best_pb else 200)
df_chart_trimmed = df_chart[df_chart.index >= BACKTEST_START]
axes[1].plot(df_chart_trimmed.index, df_chart_trimmed["Close"], color="#ef5350", linewidth=1, label="GLD Close")
axes[1].plot(df_chart_trimmed.index, df_chart_trimmed["ema_fast"], color="#42a5f5", linewidth=1, alpha=0.7,
             label=f"EMA{int(best_pb['ema_fast']) if best_pb else 50}")
axes[1].plot(df_chart_trimmed.index, df_chart_trimmed["ema_slow"], color="#ff9800", linewidth=1, alpha=0.7,
             label=f"EMA{int(best_pb['ema_slow']) if best_pb else 200}")
axes[1].set_title("GLD Price with EMAs")
axes[1].set_ylabel("Price ($)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
chart_path = Path(__file__).parent / "gld_backtest.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"\n  Chart saved to {chart_path}")
print("\nDone!")
```

- [ ] **Step 2: Run the full script end-to-end**

```bash
python3 gld_backtest.py
```

Expected: full output from Part 1 + Part 2 + Verdict + "Chart saved to gld_backtest.png". No errors.

- [ ] **Step 3: Commit**

```bash
git add gld_backtest.py gld_backtest.png
git commit -m "feat: complete GLD backtest with verdict and chart"
```

---

## Task 5: Scanner integration (conditional — only if strategy is scanner-ready)

**Skip this task if the backtest verdict says "Do not add GLD to scanner."**

**Files:**
- Modify: `dashboard/premarket_scanner.py`

- [ ] **Step 1: Read current LONG_WATCHLIST and score_ticker in premarket_scanner.py**

Check lines around:
```python
LONG_WATCHLIST = ["NVDA"]
```
and the `score_ticker()` function's trend pullback section (around line 210-234).

- [ ] **Step 2: Add GLD to LONG_WATCHLIST**

Change:
```python
LONG_WATCHLIST = ["NVDA"]
```
To:
```python
LONG_WATCHLIST = ["NVDA", "GLD"]
```

- [ ] **Step 3: Update score_ticker to use GLD-specific parameters**

In `score_ticker()`, the trend pullback block currently uses hardcoded parameters. After the backtest, replace with the validated GLD params. For example, if backtest shows EMA(50,200) pb=3% RSI<40 stop=1.5% hold=10d:

```python
# Trend pullback parameters — tuned per symbol
PULLBACK_PARAMS = {
    "NVDA": {"ema_fast": 50, "ema_slow": 200, "pullback_threshold": 3.0, "rsi_max": 35},
    "GLD":  {"ema_fast": 50, "ema_slow": 200, "pullback_threshold": 3.0, "rsi_max": 40},
}

# In score_ticker(), replace hardcoded trend_pullback() call:
params = PULLBACK_PARAMS.get(symbol, {"ema_fast": 50, "ema_slow": 200, "pullback_threshold": 3.0, "rsi_max": 35})
if (symbol in LONG_WATCHLIST and regime_allows_long(_regime, _regime_data)):
    tp = trend_pullback(bars, ema_fast=params["ema_fast"], ema_slow=params["ema_slow"],
                        pullback_threshold=params["pullback_threshold"], rsi_max=params["rsi_max"])
```

**Note:** Fill in the actual validated params from the backtest output before implementing this step.

- [ ] **Step 4: Run the scanner manually to verify GLD is now scanned**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "
import sys; sys.path.insert(0, '.')
from premarket_scanner import score_ticker
print(score_ticker('GLD'))
"
```

Expected: either `None` (no signal today) or a setup dict. No crash.

- [ ] **Step 5: Commit**

```bash
git add dashboard/premarket_scanner.py
git commit -m "feat: add GLD to premarket scanner (trend pullback long)"
```
