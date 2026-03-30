# Trend Pullback Long Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully automated long-side swing trade strategy that buys pullbacks in uptrending tickers, complementing the existing bear rally fade short strategy.

**Architecture:** Backtest first to find the best parameters (EMA combo, pullback depth, stop, hold period), then wire the validated signal into the live premarket scanner. Two new functions (`trend_pullback()` in ta_utils, `regime_allows_long()` in fred_client) mirror the existing short-side counterparts exactly.

**Tech Stack:** Python 3.9, yfinance (backtest), Alpaca SDK (live), existing ta_utils/fred_client/premarket_scanner infrastructure.

---

## File Map

| File | Change | Purpose |
|------|--------|---------|
| `trend_pullback_backtest.py` | CREATE | Sweep all parameter combos, print results, pick winners |
| `dashboard/ta_utils.py` | MODIFY | Add `trend_pullback()` — detects uptrend + pullback from 10-day high |
| `dashboard/fred_client.py` | MODIFY | Add `regime_allows_long()` — mirror of `regime_allows_short()` |
| `dashboard/premarket_scanner.py` | MODIFY | Add `LONG_WATCHLIST`, long setup block in `score_ticker()` |

---

## Task 1: Write the backtest script

**Files:**
- Create: `trend_pullback_backtest.py`

- [ ] **Step 1: Create the file with imports, constants, and data download**

```python
"""
Trend Pullback Long Backtest
=============================
Mirror of bear rally fade. Buy when EMA_fast > EMA_slow (uptrend)
and price has pulled back X% from 10-day high.

Parameter sweep:
  EMA combo:     (20, 50) vs (50, 200)
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

START          = "2019-06-01"    # buffer for EMA-200 warmup
BACKTEST_START = "2020-01-01"
END            = "2026-03-28"
INIT_EQ        = 1_000.0

LONG_WATCHLIST = ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "MSFT", "AMD"]

EMA_COMBOS     = [(20, 50), (50, 200)]
PULLBACK_PCTS  = [2.0, 3.0, 4.0, 5.0]
TRAIL_PCTS     = [0.015, 0.020, 0.030]
MAX_HOLDS      = [5, 10, 15]

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
```

- [ ] **Step 2: Add indicator builder**

```python
def add_indicators(df, ema_fast, ema_slow):
    df = df.copy()
    df["ema_fast"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["high10"]   = df["High"].rolling(10).max()
    # % pullback from 10-day high (positive = pulled back below high)
    df["pullback_pct"] = (df["high10"] - df["Close"]) / df["high10"] * 100
    return df
```

- [ ] **Step 3: Add pullback signal detection**

```python
def is_pullback_long(df, i, pullback_threshold):
    """
    True if:
      - ema_fast > ema_slow (uptrend)
      - price has pulled back >= pullback_threshold% from 10-day high
        within the last 3 bars (matches bear fade's 3-bar lookback)
    """
    if i < 200:
        return False
    row = df.iloc[i]
    if row["ema_fast"] <= row["ema_slow"]:
        return False
    window = df.iloc[max(0, i - 2):i + 1]
    return window["pullback_pct"].max() >= pullback_threshold
```

- [ ] **Step 4: Add long trade simulator**

```python
def simulate_long(df, entry_idx, entry_price, trail_pct=0.020, max_hold=10):
    """Simulate a LONG trade with trailing stop."""
    stop = entry_price * (1 - trail_pct)  # stop starts below entry

    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        high  = df.iloc[j]["High"]
        low_  = df.iloc[j]["Low"]
        close = df.iloc[j]["Close"]

        # Trail stop UP as price rises
        new_stop = close * (1 - trail_pct)
        if new_stop > stop:
            stop = new_stop

        # Stop hit if low touches or falls below stop
        if low_ <= stop:
            pnl_pct = (stop - entry_price) / entry_price * 100
            return pnl_pct, j, "stop"

    # Max hold exit at close
    exit_price = df.iloc[min(entry_idx + max_hold, len(df) - 1)]["Close"]
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    return pnl_pct, min(entry_idx + max_hold, len(df) - 1), "maxhold"
```

- [ ] **Step 5: Add strategy runner**

```python
def run_pullback_long(ema_fast, ema_slow, pullback_pct, trail_pct, max_hold):
    trades = []
    equity = INIT_EQ
    open_positions = {}   # ticker → exit_idx

    all_dates = data["SPY"].index
    start_idx = all_dates.searchsorted(pd.Timestamp(BACKTEST_START))

    # Pre-build indicator dataframes for this EMA combo
    dfs = {t: add_indicators(data[t], ema_fast, ema_slow) for t in LONG_WATCHLIST}

    for i, date in enumerate(all_dates):
        if i < start_idx:
            continue

        # Release expired positions
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

            # Entry: next open
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
            break  # one trade per day

    return pd.DataFrame(trades)
```

- [ ] **Step 6: Add stats helper and parameter sweep**

```python
def stats(df, label):
    if df.empty:
        return {"label": label, "n": 0, "wr": 0, "pf": 0, "pnl": 0}
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
    s.update({"ema_fast": ef, "ema_slow": es, "pullback": pb,
               "trail": trail, "hold": hold})
    results.append(s)

results_df = pd.DataFrame(results).sort_values("pf", ascending=False)

print("=" * 70)
print("  TOP 20 PARAMETER COMBOS BY PROFIT FACTOR")
print("=" * 70)
top = results_df[results_df["n"] >= 30].head(20)
for _, row in top.iterrows():
    flag = "✅" if row["wr"] >= 55 and row["pf"] >= 1.5 else "⚠️ "
    print(f"  {flag} EMA({int(row['ema_fast'])},{int(row['ema_slow'])}) "
          f"pb={row['pullback']:.0f}% stop={row['trail']*100:.1f}% hold={int(row['hold'])}d "
          f"| {row['n']:3.0f} trades  {row['wr']:.1f}% WR  PF {row['pf']:.2f}  ${row['pnl']:+.0f}")

# ── Best combo: per-ticker breakdown ──────────────────────────────────────────
best = results_df[results_df["n"] >= 30].iloc[0]
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

    # Year-by-year for best combo
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
```

- [ ] **Step 7: Run the backtest**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 trend_pullback_backtest.py
```

Expected: output showing top 20 parameter combos, per-ticker breakdown, year-by-year table. Takes 2–3 minutes.

- [ ] **Step 8: Record the best validated parameters**

From the output, note:
- Best EMA combo (20/50 or 50/200)
- Best pullback depth (2%, 3%, 4%, 5%)
- Best trailing stop %
- Best max hold days
- Which tickers pass (WR ≥ 55%, PF ≥ 1.5) — these become the live LONG_WATCHLIST

**Stop here and review results before continuing to Task 2.**

- [ ] **Step 9: Commit the backtest script**

```bash
git add trend_pullback_backtest.py
git commit -m "Add trend pullback long backtest — parameter sweep"
```

---

## Task 2: Add `trend_pullback()` to ta_utils.py

**Files:**
- Modify: `dashboard/ta_utils.py` (add after `trend_context()`, before `volume_analysis()`)

Uses the validated EMA combo and pullback depth from Task 1 results.
Replace `EMA_FAST`, `EMA_SLOW`, `PULLBACK_THRESHOLD` with actual numbers from backtest.

- [ ] **Step 1: Add the function**

Add this block after `trend_context()` (after line 159) in `dashboard/ta_utils.py`:

```python
def trend_pullback(bars, ema_fast=20, ema_slow=50, pullback_threshold=3.0):
    """
    Detect a pullback in an uptrending ticker — mirror of bear rally detection.

    Uptrend: EMA(ema_fast) > EMA(ema_slow)
    Signal:  price has pulled back >= pullback_threshold% from 10-day high
             within the last 3 bars.

    Parameters are set to validated backtest values (see trend_pullback_backtest.py).

    Returns dict:
      trend:               "uptrend" | "downtrend" | "choppy"
      pullback_long:       True if uptrend + pulled back enough to buy
      recent_pullback_pct: % below 10-day high right now
      high_10:             10-day high
    """
    if len(bars) < max(ema_slow + 1, 15):
        return {"trend": "insufficient data", "pullback_long": False,
                "recent_pullback_pct": 0.0, "high_10": 0.0}

    closes = [b["c"] for b in bars]

    # EMA using existing ema() helper
    ema_fast_vals = ema(closes, ema_fast)
    ema_slow_vals = ema(closes, ema_slow)

    latest_fast = ema_fast_vals[-1]
    latest_slow = ema_slow_vals[-1]

    if latest_fast > latest_slow * 1.005:
        trend = "uptrend"
    elif latest_fast < latest_slow * 0.995:
        trend = "downtrend"
    else:
        trend = "choppy"

    # 10-day high and current pullback
    high_10 = max(b["h"] for b in bars[-10:])
    current = closes[-1]
    recent_pullback_pct = (high_10 - current) / high_10 * 100

    # Check if pullback threshold was reached within last 3 bars
    pullback_3d = max(
        (high_10 - b["c"]) / high_10 * 100
        for b in bars[-3:]
    )
    pullback_long = (trend == "uptrend" and pullback_3d >= pullback_threshold)

    return {
        "trend":               trend,
        "pullback_long":       pullback_long,
        "recent_pullback_pct": round(recent_pullback_pct, 2),
        "high_10":             round(high_10, 2),
        "ema_fast":            round(latest_fast, 2),
        "ema_slow":            round(latest_slow, 2),
    }
```

- [ ] **Step 2: Verify the function works**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "
from ta_utils import trend_pullback
# Simulate uptrending bars: steadily rising closes
import random
random.seed(42)
price = 100.0
bars = []
for i in range(60):
    price *= (1 + random.uniform(-0.005, 0.012))
    bars.append({'o': price*0.99, 'h': price*1.01, 'l': price*0.98, 'c': price, 'v': 1000000, 't': '2025-01-01'})
# Force a pullback in last 3 bars
for i in range(-3, 0):
    bars[i]['c'] = bars[i]['c'] * 0.97
result = trend_pullback(bars)
print(result)
assert result['trend'] == 'uptrend', f'Expected uptrend, got {result[\"trend\"]}'
print('OK')
"
```

Expected output: dict with `trend: uptrend`, `pullback_long: True` (since we forced a 3% drop from highs).

- [ ] **Step 3: Commit**

```bash
git add dashboard/ta_utils.py
git commit -m "Add trend_pullback() to ta_utils — mirror of bear rally detection"
```

---

## Task 3: Add `regime_allows_long()` to fred_client.py

**Files:**
- Modify: `dashboard/fred_client.py` (add after `regime_allows_short()`)

- [ ] **Step 1: Add the function**

Add this block immediately after `regime_allows_short()` in `dashboard/fred_client.py`:

```python
def regime_allows_long(regime: str, regime_data: Optional[dict] = None) -> bool:
    """
    Trend pullback longs are allowed in NEUTRAL and RISK_ON regimes.
    In RISK_OFF (bear market), pullbacks extend rather than bounce — longs get trapped.
    During a steepening rally, longs are fine (curve normalising = bullish).
    """
    if regime == "RISK_OFF":
        return False
    return True
```

- [ ] **Step 2: Verify**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "
from fred_client import regime_allows_long
assert regime_allows_long('RISK_OFF') == False
assert regime_allows_long('NEUTRAL') == True
assert regime_allows_long('RISK_ON') == True
assert regime_allows_long('NEUTRAL', {'steepening_rally': True}) == True
print('All assertions passed')
"
```

Expected: `All assertions passed`

- [ ] **Step 3: Commit**

```bash
git add dashboard/fred_client.py
git commit -m "Add regime_allows_long() to fred_client — blocks longs in RISK_OFF"
```

---

## Task 4: Wire into premarket_scanner.py

**Files:**
- Modify: `dashboard/premarket_scanner.py`

Replace `EMA_FAST`, `EMA_SLOW`, `PULLBACK_PCT`, `TRAIL_PCT`, `MAX_HOLD_DAYS`
with the actual validated numbers from Task 1 before running this task.

- [ ] **Step 1: Add import and LONG_WATCHLIST constant**

At the top of `premarket_scanner.py`, add `trend_pullback` to the ta_utils import and `regime_allows_long` to the fred_client import:

```python
from ta_utils import full_analysis, trend_pullback
from fred_client import get_macro_regime, regime_allows_short, regime_allows_long
```

Add `LONG_WATCHLIST` constant after `FADE_WATCHLIST` (around line 62):

```python
# Tickers for trend pullback long strategy.
# Only tickers with positive PF over 6-year backtest (see trend_pullback_backtest.py).
# Update after each annual re-backtest.
LONG_WATCHLIST = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]  # replace with validated list from Task 1
```

- [ ] **Step 2: Add long setup block in score_ticker()**

Add this block in `score_ticker()` immediately after the Monday Reversal block (after line 218), before `return setup`:

```python
    # --- Trend Pullback Long ---
    # Uptrend + pulled back from recent high. Blocked in RISK_OFF.
    # Parameters validated via trend_pullback_backtest.py — update if re-backtested.
    PULLBACK_THRESHOLD = 3.0   # % pullback from 10-day high — replace with backtest winner
    EMA_FAST           = 20    # replace with backtest winner (20 or 50)
    EMA_SLOW           = 50    # replace with backtest winner (50 or 200)
    LONG_TRAIL_PCT     = 0.02  # trailing stop % — replace with backtest winner
    LONG_MAX_HOLD      = 10    # max hold days — replace with backtest winner

    pb = trend_pullback(bars, ema_fast=EMA_FAST, ema_slow=EMA_SLOW,
                        pullback_threshold=PULLBACK_THRESHOLD)

    if (symbol in LONG_WATCHLIST
            and pb.get("pullback_long")
            and regime_allows_long(_regime, _regime_data)
            and setup is None):   # don't override Monday Reversal or bear fade
        target_pct = 0.025
        pb_long_setup = {
            "symbol":    symbol,
            "side":      "long",
            "entry":     current_price,
            "target":    round(current_price * (1 + target_pct), 2),
            "stop_pct":  LONG_TRAIL_PCT,
            "thesis":    "Trend pullback: {} in uptrend, pulled back {:.1f}% from 10d high".format(
                             symbol, pb.get("recent_pullback_pct", 0)
                         ),
            "conviction": conviction + 2,
            "analysis":  analysis,
        }
        setup = pb_long_setup
```

- [ ] **Step 3: Update scan_list to include LONG_WATCHLIST tickers**

Find `scan_list` in `pick_best_setup()` (currently `["SPY"] + FADE_WATCHLIST`) and update:

```python
    scan_list = list(dict.fromkeys(["SPY"] + FADE_WATCHLIST + LONG_WATCHLIST))
```

- [ ] **Step 4: Verify scanner imports correctly**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "
import premarket_scanner
print('FADE_WATCHLIST:', premarket_scanner.FADE_WATCHLIST)
print('LONG_WATCHLIST:', premarket_scanner.LONG_WATCHLIST)
print('Import OK')
"
```

Expected: both watchlists printed, no import errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/premarket_scanner.py
git commit -m "Add trend pullback long strategy to premarket scanner"
```

---

## Task 5: Paper trade validation

- [ ] **Step 1: Monitor for 2 weeks**

Watch the premarket scanner Telegram alerts. For each long trade:
- Confirm it fires on a ticker in LONG_WATCHLIST
- Confirm regime is NEUTRAL or RISK_ON (not RISK_OFF)
- Confirm `pullback_long: True` makes sense visually (ticker is in uptrend, pulled back)

- [ ] **Step 2: Check postopen_stops handles longs correctly**

After any long fill, check Telegram for the "🛡️ TRAILING STOPS ATTACHED" message. Confirm:
- Stop side is SELL (not BUY)
- Trail % matches `LONG_TRAIL_PCT` from scanner

`postopen_stops.py` already handles longs correctly (uses `TRAIL_PERCENT_LONG = 3.0` for any BUY-side fill). If the backtest winner trail % differs from 3.0, update `TRAIL_PERCENT_LONG` in `postopen_stops.py` to match.

- [ ] **Step 3: Go live**

After 2 weeks of paper trades with no unexpected behaviour, fund the $5k Alpaca live account and switch `paper=True` to `paper=False` in `premarket_scanner.py` and `postopen_stops.py`.

```python
# premarket_scanner.py line 50
trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=False)

# postopen_stops.py line 37
trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=False)
```

---

## Self-Review Notes

- Task 1 must complete before Task 4 — parameters in Task 4 depend on backtest results.
- `setup is None` guard in Task 4 ensures Monday Reversal (conviction +4) and bear fade always win over pullback long if they fire on the same ticker same day.
- `TRAIL_PERCENT_LONG` in `postopen_stops.py` is currently hardcoded to 3.0 (Monday Reversal). If backtest picks a different trail %, update that constant too (noted in Task 5 Step 2).
- `trend_pullback()` uses the `ema()` helper already defined in ta_utils.py — no new dependency.
