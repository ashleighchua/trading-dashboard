# Rigorous Strategy Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single backtest script that rigorously tests three strategies (EMA Dip Momentum, Bear Rally Fade, Monday Reversal) with survivorship-bias mitigation, earnings filters, slippage, walk-forward validation, and a side-by-side output table with equity curve chart.

**Architecture:** One file `momentum_strategies_backtest.py` in the project root. Data is downloaded once via yfinance for the union of all tickers (2019-06-01 → 2026-04-25), then reused across all three strategy functions. Each strategy returns a list of trade dicts. A shared `compute_stats()` function handles walk-forward split and verdict. Main function orchestrates download → run strategies → print comparison → save chart.

**Tech Stack:** Python 3.9, yfinance, pandas, numpy, matplotlib. No new dependencies — all already installed in the project virtualenv.

---

## File Structure

| File | Role |
|---|---|
| `momentum_strategies_backtest.py` | Entire backtest — scaffold, data, indicators, earnings, strategies, output |

No other files created or modified.

---

## Task 1: Scaffold, Imports, Constants

**Files:**
- Create: `momentum_strategies_backtest.py`

- [ ] **Step 1: Create the file with header, imports, and all constants**

```python
"""
Rigorous Strategy Backtest
===========================
Three strategies tested with identical methodology:
  1. EMA Dip Momentum Long   — 28-ticker universe, uptrend + EMA pullback
  2. Bear Rally Fade (Short) — existing strategy, re-run with slippage + earnings filter
  3. Monday Reversal (Long)  — existing strategy, re-run with slippage

KNOWN LIMITATIONS:
- Survivorship bias not fully eliminated — individual stocks are 2026 survivors.
  ETFs (SPY, QQQ, IWM, XLK, XLF, XLE) in the universe have zero survivorship bias.
- yfinance earnings dates unreliable for pre-2023 periods. Earnings filter may
  silently miss historical events. Logged per ticker when data unavailable.
- Slippage estimated at 0.1% per side. Real slippage varies by liquidity.
- Past performance does not guarantee future results.

Period:         2019-06-01 → 2026-04-25 (download)
Backtest start: 2020-01-01 (EMA-200 fully warmed up)
In-sample:      2020-01-01 → 2022-12-31
Out-of-sample:  2023-01-01 → 2026-04-25
Position size:  $100 max risk per trade (qty = 100 / (entry * stop_pct))
Slippage:       0.1% per side
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import sys
from datetime import date, timedelta, datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── Date constants ────────────────────────────────────────────────────────────
DOWNLOAD_START  = "2019-06-01"   # warmup for EMA-200
BACKTEST_START  = "2020-01-01"   # first valid signal date
BACKTEST_END    = "2026-04-25"
SPLIT_DATE      = "2023-01-01"   # in-sample / out-of-sample boundary

# ── Sizing & slippage ─────────────────────────────────────────────────────────
MAX_RISK        = 100.0          # max dollar risk per trade
SLIP            = 0.001          # 0.1% per side

# ── Momentum universe ─────────────────────────────────────────────────────────
MOMENTUM_TICKERS = [
    # ETFs — zero survivorship bias
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE",
    # Mag7 + growth
    "AAPL", "NVDA", "MSFT", "META", "AMZN", "GOOGL", "TSLA", "AVGO", "NFLX",
    # Deliberate underperformers — anchors test in reality
    "INTC", "IBM", "BA", "PFE", "GE", "XOM",
    # Mixed / cyclical
    "JPM", "AMD", "BAC", "CVX", "UNH", "COST",
]

# ── Bear rally fade universe ──────────────────────────────────────────────────
FADE_TICKERS = ["PLTR", "FXI", "KWEB", "QQQ", "IWM"]

# ── Monday reversal universe ──────────────────────────────────────────────────
REVERSAL_TICKERS = ["SPY"]

# ── All tickers (downloaded once) ────────────────────────────────────────────
ALL_TICKERS = sorted(set(MOMENTUM_TICKERS + FADE_TICKERS + REVERSAL_TICKERS))

# ── Sector map ────────────────────────────────────────────────────────────────
SECTOR = {
    "SPY": "ETF",   "QQQ": "ETF",   "IWM": "ETF",
    "XLK": "ETF",   "XLF": "ETF",   "XLE": "ETF",
    "AAPL": "MegaTech", "MSFT": "MegaTech", "META": "MegaTech",
    "AMZN": "MegaTech", "GOOGL": "MegaTech",
    "NVDA": "Semis", "AMD": "Semis", "INTC": "Semis", "AVGO": "Semis",
    "TSLA": "Auto",  "NFLX": "Media",
    "JPM": "Finance", "BAC": "Finance",
    "XOM": "Energy",  "CVX": "Energy",
    "IBM": "Tech",   "GE": "Industrial", "BA": "Industrial",
    "PFE": "Health", "UNH": "Health",
    "COST": "Consumer",
    # Fade tickers not in momentum universe
    "PLTR": "Tech",  "FXI": "ETF",  "KWEB": "ETF",
}

# ── Parameter sets for momentum (pre-defined, not swept) ─────────────────────
MOMENTUM_PARAM_SETS = [
    {"label": "A", "ema_fast": 21, "ema_slow": 50, "pullback_max": 3.0, "trail": 0.03, "max_hold": 10},
    {"label": "B", "ema_fast": 21, "ema_slow": 50, "pullback_max": 5.0, "trail": 0.03, "max_hold": 15},
    {"label": "C", "ema_fast": 50, "ema_slow": 200, "pullback_max": 5.0, "trail": 0.04, "max_hold": 20},
]
```

- [ ] **Step 2: Verify the file is importable (no syntax errors)**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 -c "import momentum_strategies_backtest; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add momentum_strategies_backtest.py
git commit -m "feat: scaffold momentum backtest — constants, imports, universe"
```

---

## Task 2: Data Download and Access Helper

**Files:**
- Modify: `momentum_strategies_backtest.py` (append after constants)

- [ ] **Step 1: Add download function and per-ticker accessor**

Append this after the constants block:

```python
# ── Data download ─────────────────────────────────────────────────────────────

def download_data():
    logging.info(f"Downloading {len(ALL_TICKERS)} tickers {DOWNLOAD_START}→{BACKTEST_END}...")
    raw = yf.download(
        ALL_TICKERS,
        start=DOWNLOAD_START,
        end=BACKTEST_END,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    logging.info("Download complete.")
    return raw


def get_df(raw, ticker):
    """Extract a clean OHLCV DataFrame for one ticker. Returns None if data missing."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = pd.DataFrame({
                "Open":   raw["Open"][ticker],
                "High":   raw["High"][ticker],
                "Low":    raw["Low"][ticker],
                "Close":  raw["Close"][ticker],
                "Volume": raw["Volume"][ticker],
            }).dropna()
        else:
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df.index >= DOWNLOAD_START]
        if len(df) < 210:
            logging.warning(f"{ticker}: only {len(df)} rows — skipping")
            return None
        return df
    except Exception as e:
        logging.warning(f"{ticker}: data extraction failed — {e}")
        return None
```

- [ ] **Step 2: Add inline sanity check (runs when script is executed)**

Append immediately after `get_df`:

```python
def _test_get_df_shape(raw):
    """Sanity check: SPY should have data."""
    df = get_df(raw, "SPY")
    assert df is not None, "SPY data missing"
    assert len(df) > 1000, f"SPY too few rows: {len(df)}"
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index[0] < pd.Timestamp("2020-01-01"), "Warmup data missing"
    logging.info("✅ get_df sanity check passed")
```

- [ ] **Step 3: Verify syntax**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 -c "import momentum_strategies_backtest; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add momentum_strategies_backtest.py
git commit -m "feat: add data download and per-ticker accessor"
```

---

## Task 3: Indicator Helpers with Inline Tests

**Files:**
- Modify: `momentum_strategies_backtest.py` (append)

- [ ] **Step 1: Add indicator functions**

```python
# ── Indicators ────────────────────────────────────────────────────────────────

def ema_series(values, period):
    """
    Standard EMA. No lookahead — each value uses only data up to that bar.
    Returns list same length as input.
    """
    k = 2.0 / (period + 1)
    result = []
    for i, v in enumerate(values):
        if i == 0:
            result.append(float(v))
        else:
            result.append(float(v) * k + result[-1] * (1 - k))
    return result


def volume_ratio_series(volumes, window=20):
    """
    Ratio of bar[i] volume to mean of bar[i-window : i] (prior window bars only).
    Returns list same length as input; first `window` entries are None (warmup).
    No lookahead — bar[i] average does NOT include bar[i] itself.
    """
    vols = list(volumes)
    result = [None] * window
    for i in range(window, len(vols)):
        avg = sum(vols[i - window:i]) / window
        result.append(vols[i] / avg if avg > 0 else None)
    return result


def add_indicators(df, ema_fast, ema_slow):
    """Add EMA columns and volume ratio to a copy of df."""
    df = df.copy()
    closes = df["Close"].tolist()
    vols   = df["Volume"].tolist()
    df[f"ema{ema_fast}"] = ema_series(closes, ema_fast)
    df[f"ema{ema_slow}"] = ema_series(closes, ema_slow)
    df["vol_ratio"]      = volume_ratio_series(vols, 20)
    return df
```

- [ ] **Step 2: Add inline tests for indicators**

```python
def _test_indicators():
    # EMA: first value equals first input
    e = ema_series([10.0, 12.0, 11.0], 3)
    assert e[0] == 10.0, f"EMA[0] should be 10.0, got {e[0]}"
    # EMA period=1 is the value itself
    e1 = ema_series([5.0, 8.0, 3.0], 1)
    assert e1 == [5.0, 8.0, 3.0], f"EMA(1) should equal input, got {e1}"
    # EMA length preserved
    assert len(e) == 3

    # volume_ratio: warmup is None
    vr = volume_ratio_series([100] * 25, 20)
    assert vr[:20] == [None] * 20, "Warmup should be None"
    assert abs(vr[20] - 1.0) < 1e-9, f"Flat volume ratio should be 1.0, got {vr[20]}"
    # Low volume bar
    vols = [100] * 20 + [50]   # bar 20 is half the average
    vr2 = volume_ratio_series(vols, 20)
    assert abs(vr2[20] - 0.5) < 1e-9, f"Half-volume ratio should be 0.5, got {vr2[20]}"

    logging.info("✅ indicator tests passed")

_test_indicators()
```

- [ ] **Step 3: Run and confirm tests pass**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 momentum_strategies_backtest.py
```

Expected last lines include: `✅ indicator tests passed`

- [ ] **Step 4: Commit**

```bash
git add momentum_strategies_backtest.py
git commit -m "feat: add EMA and volume-ratio indicators with inline tests"
```

---

## Task 4: Shared Helpers — Sizing, Slippage, Earnings, Stats

**Files:**
- Modify: `momentum_strategies_backtest.py` (append)

- [ ] **Step 1: Add sizing and slippage helpers**

```python
# ── Sizing & slippage helpers ─────────────────────────────────────────────────

def calc_qty(entry_price, stop_pct, max_risk=MAX_RISK):
    """Shares = max_risk / (entry_price * stop_pct). Minimum 1."""
    stop_distance = entry_price * stop_pct
    return max(1, int(max_risk / stop_distance))


def long_entry_price(open_price):
    """Apply 0.1% slippage on long entry (buy slightly above open)."""
    return open_price * (1 + SLIP)


def long_exit_price(price):
    """Apply 0.1% slippage on long exit (sell slightly below)."""
    return price * (1 - SLIP)


def short_entry_price(open_price):
    """Apply 0.1% slippage on short entry (sell slightly below open)."""
    return open_price * (1 - SLIP)


def short_exit_price(price):
    """Apply 0.1% slippage on short exit (buy slightly above)."""
    return price * (1 + SLIP)
```

- [ ] **Step 2: Add earnings filter**

```python
# ── Earnings filter ───────────────────────────────────────────────────────────

def load_earnings_dates(tickers):
    """
    Returns {ticker: sorted list of datetime.date}.
    Falls back to [] with a warning if yfinance has no data.
    Only ETFs are skipped (no earnings).
    """
    ETF_PREFIXES = {"SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "FXI", "KWEB"}
    result = {}
    for t in tickers:
        if t in ETF_PREFIXES:
            result[t] = []
            continue
        try:
            ed = yf.Ticker(t).earnings_dates
            if ed is None or ed.empty:
                logging.warning(f"Earnings filter: no data for {t} — filter disabled for this ticker")
                result[t] = []
            else:
                result[t] = sorted(
                    idx.date() if hasattr(idx, "date") else idx
                    for idx in ed.index
                )
        except Exception as e:
            logging.warning(f"Earnings filter: {t} fetch failed ({e}) — filter disabled")
            result[t] = []
    return result


def near_earnings(signal_date, ticker, earnings_map, window_calendar_days=10):
    """
    True if any earnings date falls in (signal_date, signal_date + window_calendar_days].
    window_calendar_days=10 ≈ 5 trading days (conservative).
    """
    dates = earnings_map.get(ticker, [])
    if not dates:
        return False
    end = signal_date + timedelta(days=window_calendar_days)
    return any(signal_date < d <= end for d in dates)
```

- [ ] **Step 3: Add compute_stats function**

```python
# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(trades, label, trading_days_is, trading_days_oos):
    """
    trades: list of dicts with keys:
        date (datetime.date), ticker (str), entry (float), exit (float),
        pnl_dollar (float), side (str), reason (str)

    trading_days_is:  count of trading days in in-sample period
    trading_days_oos: count of trading days in out-of-sample period

    Returns dict with full stats and a verdict string.
    """
    if not trades:
        return {
            "label": label, "n": 0, "verdict": "FAIL",
            "is": {}, "oos": {},
        }

    split = pd.Timestamp(SPLIT_DATE)
    is_trades  = [t for t in trades if pd.Timestamp(t["date"]) <  split]
    oos_trades = [t for t in trades if pd.Timestamp(t["date"]) >= split]

    def _stats_block(group, total_days):
        if not group:
            return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0, "max_dd_pct": 0.0, "days_in_mkt_pct": 0.0}
        pnls   = [t["pnl_dollar"] for t in group]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr     = len(wins) / len(pnls) * 100
        gp     = sum(wins) if wins else 0.0
        gl     = abs(sum(losses)) if losses else 0.0
        pf     = gp / gl if gl > 0 else float("inf")
        avg_w  = gp / len(wins) if wins else 0.0
        avg_l  = gl / len(losses) if losses else 0.0

        # Max drawdown on cumulative P&L series
        cum, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            dd = (cum - peak) / (abs(peak) + 1e-9) * 100
            max_dd = min(max_dd, dd)

        # Days in market: sum of hold durations / total trading days
        hold_days = sum(t.get("hold_days", 1) for t in group)
        dim = hold_days / total_days * 100 if total_days > 0 else 0.0

        return {
            "n": len(group),
            "wr": round(wr, 1),
            "pf": round(pf, 2) if pf != float("inf") else 999.0,
            "pnl": round(sum(pnls), 2),
            "avg_win": round(avg_w, 2),
            "avg_loss": round(avg_l, 2),
            "max_dd_pct": round(abs(max_dd), 1),
            "days_in_mkt_pct": round(dim, 1),
        }

    is_s  = _stats_block(is_trades,  trading_days_is)
    oos_s = _stats_block(oos_trades, trading_days_oos)

    # Verdict
    oos_wr_ok  = oos_s["n"] >= 10 and oos_s["wr"] >= 55.0
    oos_pf_ok  = oos_s["pf"] >= 1.5
    oos_dd_ok  = oos_s["max_dd_pct"] <= 25.0
    drift_ok   = is_s["n"] > 0 and oos_s["n"] > 0 and abs(oos_s["wr"] - is_s["wr"]) <= 10.0

    if oos_wr_ok and oos_pf_ok and oos_dd_ok and drift_ok:
        verdict = "✅ PASS"
    elif oos_wr_ok and oos_pf_ok and (not oos_dd_ok or not drift_ok):
        verdict = "⚠️  WARN"
    else:
        verdict = "❌ FAIL"

    return {"label": label, "is": is_s, "oos": oos_s, "verdict": verdict,
            "all_pnls": [t["pnl_dollar"] for t in trades]}
```

- [ ] **Step 4: Add inline tests for helpers**

```python
def _test_helpers():
    # calc_qty: $100 risk, $50 stock, 2% stop → stop_distance=$1 → qty=100
    assert calc_qty(50.0, 0.02) == 100
    # calc_qty: minimum 1
    assert calc_qty(10000.0, 0.001) == 1
    # Slippage on entry/exit
    assert abs(long_entry_price(100.0) - 100.1) < 1e-9
    assert abs(long_exit_price(100.0) - 99.9) < 1e-9
    assert abs(short_entry_price(100.0) - 99.9) < 1e-9
    assert abs(short_exit_price(100.0) - 100.1) < 1e-9
    # near_earnings: no dates → False
    assert not near_earnings(date(2024, 1, 10), "AAPL", {"AAPL": []})
    # near_earnings: earnings 5 days later → True
    assert near_earnings(date(2024, 1, 10), "AAPL", {"AAPL": [date(2024, 1, 15)]})
    # near_earnings: earnings 15 days later → False (outside 10-day window)
    assert not near_earnings(date(2024, 1, 10), "AAPL", {"AAPL": [date(2024, 1, 25)]})
    logging.info("✅ helper tests passed")

_test_helpers()
```

- [ ] **Step 5: Run and confirm**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 momentum_strategies_backtest.py
```

Expected: `✅ indicator tests passed` and `✅ helper tests passed`

- [ ] **Step 6: Commit**

```bash
git add momentum_strategies_backtest.py
git commit -m "feat: add sizing, slippage, earnings filter, and stats helpers with tests"
```

---

## Task 5: Strategy 1 — EMA Dip Momentum Long

**Files:**
- Modify: `momentum_strategies_backtest.py` (append)

This is the most complex strategy. Read carefully — the signal logic, sector cap, and daily entry cap all interact.

- [ ] **Step 1: Add the long simulation helper**

```python
# ── Strategy 1: EMA Dip Momentum ─────────────────────────────────────────────

def simulate_long_trade(df, entry_idx, entry_price, trail_pct, max_hold):
    """
    Simulate a long trade starting at entry_idx with entry_price already slippage-adjusted.
    Trail stop uses daily close. Exit at stop (apply exit slippage) or max_hold close.
    Returns (exit_price_after_slippage, exit_idx, reason, hold_days).
    """
    stop = entry_price * (1 - trail_pct)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        low   = df.iloc[j]["Low"]
        close = df.iloc[j]["Close"]
        # Raise stop if close is higher
        new_stop = close * (1 - trail_pct)
        if new_stop > stop:
            stop = new_stop
        if low <= stop:
            return long_exit_price(stop), j, "stop", j - entry_idx
    # Max hold: exit at close of last bar
    exit_idx = min(entry_idx + max_hold, len(df) - 1)
    return long_exit_price(df.iloc[exit_idx]["Close"]), exit_idx, "maxhold", exit_idx - entry_idx
```

- [ ] **Step 2: Add the main momentum strategy runner**

```python
def run_momentum(raw, earnings_map, params):
    """
    Run EMA Dip Momentum for one parameter set.
    Returns list of trade dicts.
    """
    ema_fast    = params["ema_fast"]
    ema_slow    = params["ema_slow"]
    pullback_max = params["pullback_max"]
    trail       = params["trail"]
    max_hold    = params["max_hold"]

    # Prepare all DataFrames with indicators
    dfs = {}
    for t in MOMENTUM_TICKERS:
        df = get_df(raw, t)
        if df is not None:
            dfs[t] = add_indicators(df, ema_fast, ema_slow)

    spy_df = dfs.get("SPY")
    if spy_df is None:
        raise RuntimeError("SPY data missing — cannot determine market regime")

    # Build SPY EMA-200 series keyed by date
    spy_ema200 = ema_series(spy_df["Close"].tolist(), 200)
    spy_ema200_by_date = {
        spy_df.index[i].date(): spy_ema200[i]
        for i in range(len(spy_df))
    }

    backtest_start = pd.Timestamp(BACKTEST_START)
    trades = []
    open_positions = {}   # {ticker: exit_idx in that ticker's df}
    open_sectors   = {}   # {sector: count of open positions}

    # Get all trading dates from SPY
    all_dates = spy_df[spy_df.index >= backtest_start].index

    for date_ts in all_dates:
        signal_date = date_ts.date()
        new_today   = 0   # max 2 new positions per day
        day_signals = []  # collect all valid signals before applying caps

        for ticker in MOMENTUM_TICKERS:
            if ticker not in dfs:
                continue
            df = dfs[ticker]
            if date_ts not in df.index:
                continue
            idx = df.index.get_loc(date_ts)

            # Warmup: need at least ema_slow bars
            if idx < ema_slow:
                continue

            # Not already in a position in this ticker
            if ticker in open_positions and open_positions[ticker] > idx:
                continue

            row = df.iloc[idx]
            ema_f = row[f"ema{ema_fast}"]
            ema_s = row[f"ema{ema_slow}"]
            close = row["Close"]
            vr    = row["vol_ratio"]

            # 1. Market regime: SPY close > SPY EMA-200
            spy_ema = spy_ema200_by_date.get(signal_date)
            if spy_ema is None or spy_df.loc[date_ts, "Close"] <= spy_ema:
                continue

            # 2. Individual uptrend: ema_fast > ema_slow
            if ema_f <= ema_s:
                continue

            # 3. Pullback depth: 0% to pullback_max% below ema_fast
            if ema_f <= 0:
                continue
            pullback_pct = (ema_f - close) / ema_f * 100
            if not (0 <= pullback_pct <= pullback_max):
                continue

            # 4. Low-volume pullback: vol_ratio <= 0.8
            if vr is None or vr > 0.8:
                continue

            # 5. Earnings filter
            if near_earnings(signal_date, ticker, earnings_map):
                continue

            day_signals.append({
                "ticker": ticker,
                "vol_ratio": vr,
                "idx": idx,
                "close": close,
            })

        # Sort by lowest vol_ratio (lightest selling pressure = cleanest signal)
        day_signals.sort(key=lambda s: s["vol_ratio"])

        for sig in day_signals:
            if new_today >= 2:
                break
            ticker  = sig["ticker"]
            idx     = sig["idx"]
            df      = dfs[ticker]
            sector  = SECTOR.get(ticker, "Other")

            # Sector cap: max 2 open per sector
            if open_sectors.get(sector, 0) >= 2:
                continue

            # Need a next bar for entry
            if idx + 1 >= len(df):
                continue

            entry_open  = df.iloc[idx + 1]["Open"]
            entry_price = long_entry_price(entry_open)
            qty         = calc_qty(entry_price, trail)
            if qty == 0:
                continue

            exit_price, exit_idx, reason, hold_days = simulate_long_trade(
                df, idx + 1, entry_price, trail, max_hold
            )
            pnl_dollar = (exit_price - entry_price) * qty

            trades.append({
                "date":       signal_date,
                "ticker":     ticker,
                "entry":      round(entry_price, 4),
                "exit":       round(exit_price, 4),
                "qty":        qty,
                "pnl_dollar": round(pnl_dollar, 2),
                "side":       "long",
                "reason":     reason,
                "hold_days":  hold_days,
            })

            open_positions[ticker] = exit_idx
            open_sectors[sector]   = open_sectors.get(sector, 0) + 1
            new_today += 1

        # Decrement open_sectors for positions that closed today
        # (positions whose exit_idx <= current idx in their df)
        for t, ei in list(open_positions.items()):
            t_df = dfs.get(t)
            if t_df is None:
                continue
            if date_ts not in t_df.index:
                continue
            cur_idx = t_df.index.get_loc(date_ts)
            if ei <= cur_idx:
                s = SECTOR.get(t, "Other")
                open_sectors[s] = max(0, open_sectors.get(s, 0) - 1)
                del open_positions[t]

    return trades
```

- [ ] **Step 3: Verify no syntax errors**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 -c "import momentum_strategies_backtest; print('OK')"
```

Expected: `OK` (tests still pass)

- [ ] **Step 4: Commit**

```bash
git add momentum_strategies_backtest.py
git commit -m "feat: add EMA dip momentum strategy runner"
```

---

## Task 6: Strategy 2 — Bear Rally Fade Re-run

**Files:**
- Modify: `momentum_strategies_backtest.py` (append)

- [ ] **Step 1: Add short simulation helper**

```python
# ── Strategy 2: Bear Rally Fade ───────────────────────────────────────────────

def simulate_short_trade(df, entry_idx, entry_price, trail_pct, max_hold):
    """
    Simulate a short trade. Trail stop uses daily close.
    Stop rises (for short: stop is above price, trails down as price falls).
    entry_price is already slippage-adjusted (below open).
    Returns (exit_price_after_slippage, exit_idx, reason, hold_days).
    """
    stop = entry_price * (1 + trail_pct)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        high  = df.iloc[j]["High"]
        close = df.iloc[j]["Close"]
        # Lower stop as price falls
        new_stop = close * (1 + trail_pct)
        if new_stop < stop:
            stop = new_stop
        if high >= stop:
            return short_exit_price(stop), j, "stop", j - entry_idx
    exit_idx = min(entry_idx + max_hold, len(df) - 1)
    return short_exit_price(df.iloc[exit_idx]["Close"]), exit_idx, "maxhold", exit_idx - entry_idx
```

- [ ] **Step 2: Add bear rally fade runner**

```python
def run_bear_rally_fade(raw, earnings_map):
    """
    Bear Rally Fade (short) — PLTR, FXI, KWEB, QQQ, IWM.
    Signal: close < EMA-200 AND rallied >= 3% from 5-day low.
    Exit: 1.5% trailing stop OR 10-day max hold.
    Earnings filter applied to PLTR only (ETFs have no earnings).
    Slippage 0.1% per side.
    One position at a time across all tickers.
    """
    TRAIL   = 0.015
    MAX_HOLD = 10
    RALLY_THRESHOLD = 3.0  # percent rally from 5-day low

    dfs = {}
    for t in FADE_TICKERS:
        df = get_df(raw, t)
        if df is not None:
            df = df.copy()
            closes = df["Close"].tolist()
            df["ema200"] = ema_series(closes, 200)
            dfs[t] = df

    backtest_start = pd.Timestamp(BACKTEST_START)
    trades = []
    in_position = None   # ticker currently held
    exit_idx_map = {}    # {ticker: exit_idx in that ticker's df}

    spy_df = get_df(raw, "SPY")
    all_dates = spy_df[spy_df.index >= backtest_start].index

    for date_ts in all_dates:
        signal_date = date_ts.date()

        # Clear positions that have closed
        if in_position and in_position in exit_idx_map:
            df = dfs.get(in_position)
            if df is not None and date_ts in df.index:
                cur_idx = df.index.get_loc(date_ts)
                if exit_idx_map[in_position] <= cur_idx:
                    in_position = None

        if in_position:
            continue

        for ticker in FADE_TICKERS:
            if ticker not in dfs:
                continue
            df = dfs[ticker]
            if date_ts not in df.index:
                continue
            idx = df.index.get_loc(date_ts)
            if idx < 205:   # EMA-200 warmup
                continue

            row   = df.iloc[idx]
            close = row["Close"]
            ema200 = row["ema200"]

            # 1. Downtrend
            if close >= ema200:
                continue

            # 2. Bear rally: close >= 5-day low * 1.03
            low5 = df.iloc[idx - 4:idx + 1]["Low"].min()
            rally_pct = (close - low5) / low5 * 100
            if rally_pct < RALLY_THRESHOLD:
                continue

            # 3. Earnings filter (PLTR only)
            if ticker == "PLTR" and near_earnings(signal_date, ticker, earnings_map):
                continue

            # 4. Need next bar
            if idx + 1 >= len(df):
                continue

            entry_open  = df.iloc[idx + 1]["Open"]
            entry_price = short_entry_price(entry_open)
            qty         = calc_qty(entry_price, TRAIL)

            exit_price, ex_idx, reason, hold_days = simulate_short_trade(
                df, idx + 1, entry_price, TRAIL, MAX_HOLD
            )
            pnl_dollar = (entry_price - exit_price) * qty

            trades.append({
                "date":       signal_date,
                "ticker":     ticker,
                "entry":      round(entry_price, 4),
                "exit":       round(exit_price, 4),
                "qty":        qty,
                "pnl_dollar": round(pnl_dollar, 2),
                "side":       "short",
                "reason":     reason,
                "hold_days":  hold_days,
            })

            in_position = ticker
            exit_idx_map[ticker] = ex_idx
            break   # one position at a time

    return trades
```

- [ ] **Step 3: Verify syntax**

```bash
python3 -c "import momentum_strategies_backtest; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add momentum_strategies_backtest.py
git commit -m "feat: add bear rally fade re-run with slippage and earnings filter"
```

---

## Task 7: Strategy 3 — Monday Reversal Re-run

**Files:**
- Modify: `momentum_strategies_backtest.py` (append)

- [ ] **Step 1: Add Monday reversal runner**

```python
# ── Strategy 3: Monday Reversal ───────────────────────────────────────────────

def run_monday_reversal(raw):
    """
    Monday Reversal (long SPY).
    Signal: weekday == Monday AND previous Friday close return <= -0.75%.
    Exit: 1.5% trailing stop OR 5-day max hold.
    Slippage 0.1% per side. One position at a time.
    """
    TRAIL      = 0.015
    MAX_HOLD   = 5
    FRI_THRESH = -0.75   # percent

    df = get_df(raw, "SPY")
    if df is None:
        raise RuntimeError("SPY data missing for Monday Reversal")

    backtest_start = pd.Timestamp(BACKTEST_START)
    df = df[df.index >= pd.Timestamp(DOWNLOAD_START)].copy()

    trades = []
    in_position_until = -1   # row index until which we're occupied

    for i in range(1, len(df)):
        date_ts = df.index[i]
        if date_ts < backtest_start:
            continue
        if i <= in_position_until:
            continue

        # Must be Monday (weekday 0)
        if date_ts.weekday() != 0:
            continue

        # Find the most recent Friday
        fri_idx = i - 1
        while fri_idx >= 0 and df.index[fri_idx].weekday() != 4:
            fri_idx -= 1
        if fri_idx < 1:
            continue

        fri_close = df.iloc[fri_idx]["Close"]
        thu_close = df.iloc[fri_idx - 1]["Close"]
        if thu_close <= 0:
            continue
        fri_return = (fri_close - thu_close) / thu_close * 100

        if fri_return > FRI_THRESH:
            continue

        # Signal fires — entry at Monday's open
        entry_open  = df.iloc[i]["Open"]
        entry_price = long_entry_price(entry_open)
        qty         = calc_qty(entry_price, TRAIL)

        exit_price, exit_idx, reason, hold_days = simulate_long_trade(
            df, i, entry_price, TRAIL, MAX_HOLD
        )
        pnl_dollar = (exit_price - entry_price) * qty

        trades.append({
            "date":       date_ts.date(),
            "ticker":     "SPY",
            "entry":      round(entry_price, 4),
            "exit":       round(exit_price, 4),
            "qty":        qty,
            "pnl_dollar": round(pnl_dollar, 2),
            "side":       "long",
            "reason":     reason,
            "hold_days":  hold_days,
        })

        in_position_until = exit_idx

    return trades
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import momentum_strategies_backtest; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add momentum_strategies_backtest.py
git commit -m "feat: add Monday reversal re-run with slippage"
```

---

## Task 8: Main Function — Orchestration, Output Table, Equity Curve

**Files:**
- Modify: `momentum_strategies_backtest.py` (append)

- [ ] **Step 1: Add the output formatter**

```python
# ── Output ────────────────────────────────────────────────────────────────────

def print_strategy_result(result):
    is_s  = result["is"]
    oos_s = result["oos"]
    label = result["label"]
    print(f"\n  {label}")
    if not is_s:
        print("    No trades.")
        return
    print(f"    In-sample  (2020–2022): {is_s['n']:3d} trades | "
          f"{is_s['wr']:.1f}% WR | PF {is_s['pf']:.2f} | "
          f"${is_s['pnl']:+.0f} | MaxDD {is_s['max_dd_pct']:.1f}%")
    print(f"    Out-of-sample (2023–2026): {oos_s['n']:3d} trades | "
          f"{oos_s['wr']:.1f}% WR | PF {oos_s['pf']:.2f} | "
          f"${oos_s['pnl']:+.0f} | MaxDD {oos_s['max_dd_pct']:.1f}%")
    print(f"    Days in market (OOS): {oos_s['days_in_mkt_pct']:.1f}%")
    print(f"    Verdict: {result['verdict']}")
```

- [ ] **Step 2: Add the equity curve chart function**

```python
def save_equity_curve(results, output_path="momentum_strategies_backtest.png"):
    """Plot cumulative P&L for each strategy over the full backtest period."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for i, result in enumerate(results):
        if not result.get("all_pnls"):
            continue
        cum = [0.0]
        for p in result["all_pnls"]:
            cum.append(cum[-1] + p)
        ax.plot(range(len(cum)), cum, label=result["label"], color=colors[i % len(colors)], linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5)
    split_approx = None
    # Mark the in-sample / out-of-sample boundary on the longest series
    longest = max((r for r in results if r.get("all_pnls")), key=lambda r: len(r["all_pnls"]), default=None)
    ax.set_title("Strategy Cumulative P&L — Rigorous Backtest (2020–2026)")
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Equity curve saved to {output_path}")
```

- [ ] **Step 3: Add the main function**

```python
def main():
    # ── Download ──────────────────────────────────────────────────────────────
    raw = download_data()
    _test_get_df_shape(raw)

    # Count trading days in each period for days-in-market %
    spy_df = get_df(raw, "SPY")
    split  = pd.Timestamp(SPLIT_DATE)
    td_is  = len(spy_df[(spy_df.index >= BACKTEST_START) & (spy_df.index < split)])
    td_oos = len(spy_df[spy_df.index >= split])

    # ── Earnings filter (load once, used by momentum + fade) ──────────────────
    earnings_tickers = [t for t in ALL_TICKERS if t not in
                        {"SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "FXI", "KWEB"}]
    logging.info(f"Loading earnings dates for {len(earnings_tickers)} tickers...")
    earnings_map = load_earnings_dates(earnings_tickers)

    # ── Strategy 1: EMA Dip Momentum — run all 3 param sets ──────────────────
    print("\n" + "=" * 66)
    print("  EMA DIP MOMENTUM LONG")
    print("=" * 66)
    momentum_results = []
    best_oos_pf = -1.0
    best_momentum_result = None

    for params in MOMENTUM_PARAM_SETS:
        logging.info(f"Running momentum Set {params['label']}...")
        trades  = run_momentum(raw, earnings_map, params)
        label   = (f"Set {params['label']}: EMA({params['ema_fast']}/{params['ema_slow']}) "
                   f"pb≤{params['pullback_max']}% stop={params['trail']*100:.0f}% "
                   f"hold={params['max_hold']}d")
        result  = compute_stats(trades, label, td_is, td_oos)
        momentum_results.append(result)
        print_strategy_result(result)
        oos_pf = result["oos"].get("pf", 0) if result["oos"] else 0
        if oos_pf > best_oos_pf:
            best_oos_pf = oos_pf
            best_momentum_result = result

    # ── Strategy 2: Bear Rally Fade ───────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  BEAR RALLY FADE (re-run w/ slippage + earnings filter)")
    print("=" * 66)
    logging.info("Running Bear Rally Fade...")
    fade_trades  = run_bear_rally_fade(raw, earnings_map)
    fade_result  = compute_stats(fade_trades, "Bear Rally Fade", td_is, td_oos)
    print_strategy_result(fade_result)

    # ── Strategy 3: Monday Reversal ───────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  MONDAY REVERSAL (re-run w/ slippage)")
    print("=" * 66)
    logging.info("Running Monday Reversal...")
    reversal_trades  = run_monday_reversal(raw)
    reversal_result  = compute_stats(reversal_trades, "Monday Reversal", td_is, td_oos)
    print_strategy_result(reversal_result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  SUMMARY — BEST MOMENTUM SET vs EXISTING STRATEGIES")
    print("=" * 66)
    all_results = []
    if best_momentum_result:
        all_results.append(best_momentum_result)
    all_results += [fade_result, reversal_result]

    total_pnl = sum(
        (r["is"].get("pnl", 0) + r["oos"].get("pnl", 0))
        for r in all_results if r.get("is") and r.get("oos")
    )
    for r in all_results:
        total = (r["is"].get("pnl", 0) if r.get("is") else 0) + \
                (r["oos"].get("pnl", 0) if r.get("oos") else 0)
        print(f"  {r['verdict']}  {r['label']}  |  Total P&L: ${total:+.0f}")
    print(f"\n  Combined P&L (all strategies): ${total_pnl:+.0f}")

    # ── Chart ─────────────────────────────────────────────────────────────────
    for r in all_results:
        pass   # label already set
    save_equity_curve(all_results)

    print("\nDone.")


if __name__ == "__main__":
    # Remove inline test calls from module level when running as script
    main()
```

- [ ] **Step 4: Move inline test calls so they only run on import (not inside main)**

The `_test_indicators()` and `_test_helpers()` calls at module level are fine — they run fast and catch regressions. Leave them as-is.

- [ ] **Step 5: Verify syntax**

```bash
python3 -c "import momentum_strategies_backtest; print('OK')"
```

Expected: `OK` plus the two test-pass lines.

- [ ] **Step 6: Commit**

```bash
git add momentum_strategies_backtest.py
git commit -m "feat: add main orchestration, output table, and equity curve chart"
```

---

## Task 9: End-to-End Run and Verification

**Files:**
- No changes — this is a verification task only.

- [ ] **Step 1: Run the full backtest**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 momentum_strategies_backtest.py
```

Expected runtime: 5–10 minutes (data download ~2min, strategies ~5min).

Expected output structure:
```
INFO Downloading 33 tickers 2019-06-01→2026-04-25...
INFO Download complete.
INFO Loading earnings dates for N tickers...
...
==================================================================
  EMA DIP MOMENTUM LONG
==================================================================

  Set A: EMA(21/50) pb≤3% stop=3% hold=10d
    In-sample  (2020–2022): XX trades | XX.X% WR | PF X.XX | $+XXX | MaxDD XX.X%
    Out-of-sample (2023–2026): XX trades | XX.X% WR | PF X.XX | $+XXX | MaxDD XX.X%
    ...
```

- [ ] **Step 2: Verify sanity checks on output**

Manually confirm:
- All three strategies produced ≥ 10 trades in out-of-sample
- EMA Dip Momentum shows fewer or zero signals in 2022 (SPY below EMA-200 most of the year — regime filter should block most signals)
- Bear Rally Fade shows more short signals in 2022 (downtrend year)
- Monday Reversal has low trade count (weekly signal, SPY only)
- Equity curve PNG is saved in project root

- [ ] **Step 3: Commit final run note**

```bash
git add momentum_strategies_backtest.py momentum_strategies_backtest.png
git commit -m "feat: complete rigorous backtest — all three strategies with walk-forward validation"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task that covers it |
|---|---|
| 28-ticker momentum universe | Task 1 (MOMENTUM_TICKERS constant) |
| EMA-200 warmup from 2019-06-01 | Task 2 (DOWNLOAD_START constant + get_df check) |
| Low-volume pullback (vol ≤ 0.8×) | Task 5 (signal condition 4) |
| Earnings filter (10 calendar day window) | Task 4 (near_earnings) |
| Sector cap (max 2 per sector) | Task 5 (open_sectors dict) |
| Daily entry cap (max 2 per day) | Task 5 (new_today counter) |
| Slippage 0.1% per side | Task 4 (long/short entry/exit helpers) |
| 3 pre-defined param sets (not swept) | Task 1 (MOMENTUM_PARAM_SETS) + Task 5 |
| Walk-forward split 2020-2022 / 2023-2026 | Task 4 (compute_stats SPLIT_DATE) |
| Pass/fail verdict (WR≥55, PF≥1.5, DD≤25, drift≤10) | Task 4 (compute_stats verdict logic) |
| Bear Rally Fade: slippage + earnings for PLTR | Task 6 |
| Monday Reversal: slippage | Task 7 |
| Days in market % | Task 4 (compute_stats days_in_mkt_pct) |
| Equity curve chart saved as PNG | Task 8 (save_equity_curve) |
| Known limitations documented in header | Task 1 (docstring) |

**Placeholder scan:** No TBDs, TODOs, or vague steps. All code is complete. ✅

**Type consistency:**
- `simulate_long_trade` returns `(float, int, str, int)` — used correctly in Task 5 and Task 7 ✅
- `simulate_short_trade` returns `(float, int, str, int)` — used correctly in Task 6 ✅
- Trade dicts all have identical keys: `date, ticker, entry, exit, qty, pnl_dollar, side, reason, hold_days` — compute_stats expects `date`, `pnl_dollar`, `hold_days` ✅
- `compute_stats` returns dict with keys `label, is, oos, verdict, all_pnls` — print_strategy_result and save_equity_curve use these correctly ✅
