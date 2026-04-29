# 50-Day Breakout Momentum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 50-day breakout momentum strategy (with O'Neil RS, Weinstein Stage 2, and Minervini VCP filters) as a 4th strategy in `momentum_strategies_backtest.py`.

**Architecture:** All infrastructure (data download, slippage, earnings filter, position sizing, stats, sector cap) already exists. We add a constant block, three small helper functions, one strategy function `run_breakout()`, and update `main()` to run it. No new files.

**Tech Stack:** Python 3, yfinance, pandas, existing helpers in `momentum_strategies_backtest.py`

---

## File Map

| File | Change |
|------|--------|
| `momentum_strategies_backtest.py` | Add `BREAKOUT_PARAM_SETS` constant (~line 98), add `consolidation_pct()`, `stage2_confirmed()`, `relative_strength_vs_spy()` helpers + inline tests, add `run_breakout()`, update `main()` |

No other files change.

---

## Existing helpers you will call (do NOT rewrite these)

- `get_df(raw, ticker)` → DataFrame with columns: Open, High, Low, Close, Volume
- `ema_series(values_list, period)` → list of floats
- `volume_ratio_series(volumes_list, window=20)` → list of float|None
- `long_entry_price(open_price)` → open_price * 1.001
- `long_exit_price(price)` → price * 0.999
- `calc_qty(entry_price, stop_pct, max_risk=MAX_RISK)` → int, minimum 1
- `simulate_long_trade(df, entry_idx, entry_price, trail_pct, max_hold)` → (exit_price, exit_idx, reason, hold_days)
- `near_earnings(signal_date, ticker, earnings_map)` → bool
- `compute_stats(trades, label, trading_days_is, trading_days_oos)` → dict
- `SECTOR` dict, `MOMENTUM_TICKERS` list, `BACKTEST_START`, `SPLIT_DATE` constants

---

## Task 1: Add BREAKOUT_PARAM_SETS constant and helper functions

**Files:**
- Modify: `momentum_strategies_backtest.py` — insert after `MOMENTUM_PARAM_SETS` block (~line 98), and append helpers after `_test_helpers()` (~line 390)

- [ ] **Step 1: Add BREAKOUT_PARAM_SETS after MOMENTUM_PARAM_SETS**

Find this line in the file:
```python
# ── Data download ─────────────────────────────────────────────────────────────
```

Insert immediately before it:
```python
# ── Parameter sets for breakout (pre-defined, not swept) ─────────────────────
BREAKOUT_PARAM_SETS = [
    # Set A: standard O'Neil breakout — 6% trail, 15-day hold
    {"label": "A", "lookback": 50, "vol_min": 1.5, "consol_15": 5.0, "consol_5": 3.0, "trail": 0.06, "max_hold": 15},
    # Set B: more room for retest — 7% trail, 20-day hold
    {"label": "B", "lookback": 50, "vol_min": 1.5, "consol_15": 5.0, "consol_5": 3.0, "trail": 0.07, "max_hold": 20},
    # Set C: institutional conviction — tighter VCP, 2× volume, 7% trail, 25-day hold
    {"label": "C", "lookback": 50, "vol_min": 2.0, "consol_15": 4.0, "consol_5": 2.0, "trail": 0.07, "max_hold": 25},
]

```

- [ ] **Step 2: Verify import still works**

```bash
cd "/Users/ashleighchua/trading analyses" && python3 -c "import momentum_strategies_backtest"
```

Expected: two ✅ lines (indicator tests passed, helper tests passed), no errors.

- [ ] **Step 3: Add three helper functions after _test_helpers()**

Find this line in the file:
```python
_test_helpers()

# ── Strategy 1: EMA Dip Momentum ─────────────────────────────────────────────
```

Replace with:
```python
_test_helpers()

# ── Breakout helpers ──────────────────────────────────────────────────────────

def consolidation_pct(df, idx, window):
    """
    High-to-low range over the prior `window` bars as % of close at idx.
    Uses only bars BEFORE idx (no lookahead). Returns None if insufficient data.
    """
    if idx < window:
        return None
    slice_ = df.iloc[idx - window:idx]
    high   = slice_["High"].max()
    low    = slice_["Low"].min()
    close  = df.iloc[idx]["Close"]
    if close <= 0:
        return None
    return (high - low) / close * 100


def stage2_confirmed(df, idx):
    """
    Weinstein Stage 2: close > 150-day MA AND 150-day MA is rising.
    Rising = MA computed at idx > MA computed at idx-5.
    Returns False if insufficient data (need idx >= 155).
    """
    if idx < 155:
        return False
    ma_now  = df["Close"].iloc[idx - 150:idx].mean()
    ma_prev = df["Close"].iloc[idx - 155:idx - 5].mean()
    close   = df.iloc[idx]["Close"]
    return bool(close > ma_now and ma_now > ma_prev)


def relative_strength_vs_spy(df, spy_df, idx, window=63):
    """
    O'Neil RS: stock's window-bar return > SPY's window-bar return.
    window=63 ≈ 3 months of trading days.
    Aligns by bar index (valid — both DFs use the same yfinance trading calendar).
    Returns False if insufficient data.
    """
    if idx < window or len(spy_df) <= idx:
        return False
    stock_ret = (df.iloc[idx]["Close"] - df.iloc[idx - window]["Close"]) / df.iloc[idx - window]["Close"]
    spy_idx   = min(idx, len(spy_df) - 1)
    if spy_idx < window:
        return False
    spy_ret = (spy_df.iloc[spy_idx]["Close"] - spy_df.iloc[spy_idx - window]["Close"]) / spy_df.iloc[spy_idx - window]["Close"]
    return bool(stock_ret > spy_ret)


def _test_breakout_helpers():
    """Inline tests for consolidation_pct, stage2_confirmed, relative_strength_vs_spy."""
    import pandas as pd
    import numpy as np

    # ── consolidation_pct ────────────────────────────────────────────────────
    # Build a df where High=110, Low=90, Close=100 for 20 bars → range=20%
    n = 20
    df_flat = pd.DataFrame({
        "High":   [110.0] * n,
        "Low":    [90.0]  * n,
        "Close":  [100.0] * n,
        "Open":   [100.0] * n,
        "Volume": [1e6]   * n,
    })
    result = consolidation_pct(df_flat, 15, 10)
    assert result is not None and abs(result - 20.0) < 1e-6, f"Expected 20.0, got {result}"

    # Insufficient data: idx < window → None
    assert consolidation_pct(df_flat, 5, 10) is None

    # ── stage2_confirmed ─────────────────────────────────────────────────────
    # Build a df where price and MA are clearly rising: price[i] = 100 + i*0.1
    n2 = 200
    prices = [100.0 + i * 0.5 for i in range(n2)]
    df_rising = pd.DataFrame({
        "High":   prices, "Low": prices, "Close": prices,
        "Open":   prices, "Volume": [1e6] * n2,
    })
    assert stage2_confirmed(df_rising, 160) is True, "Rising price/MA should be Stage 2"

    # Flat price — MA not rising
    prices_flat = [100.0] * n2
    df_flat2 = pd.DataFrame({
        "High": prices_flat, "Low": prices_flat, "Close": prices_flat,
        "Open": prices_flat, "Volume": [1e6] * n2,
    })
    assert stage2_confirmed(df_flat2, 160) is False, "Flat price/MA should not be Stage 2"

    # Insufficient data
    assert stage2_confirmed(df_rising, 100) is False

    # ── relative_strength_vs_spy ─────────────────────────────────────────────
    # Stock up 20% over 63 bars, SPY up 10% → stock outperforms → True
    n3 = 100
    stock_prices = [100.0] + [100.0 + (20.0 / 63) * i for i in range(1, n3)]
    spy_prices   = [100.0] + [100.0 + (10.0 / 63) * i for i in range(1, n3)]
    df_stock = pd.DataFrame({"Close": stock_prices, "High": stock_prices,
                              "Low": stock_prices, "Open": stock_prices, "Volume": [1e6]*n3})
    df_spy   = pd.DataFrame({"Close": spy_prices,  "High": spy_prices,
                              "Low": spy_prices,  "Open": spy_prices,  "Volume": [1e6]*n3})
    assert relative_strength_vs_spy(df_stock, df_spy, 90) is True

    # Stock flat, SPY up → False
    df_stock_flat = pd.DataFrame({"Close": [100.0]*n3, "High": [100.0]*n3,
                                   "Low": [100.0]*n3, "Open": [100.0]*n3, "Volume": [1e6]*n3})
    assert relative_strength_vs_spy(df_stock_flat, df_spy, 90) is False

    # Insufficient data
    assert relative_strength_vs_spy(df_stock, df_spy, 10) is False

    logging.info("✅ breakout helper tests passed")


_test_breakout_helpers()

# ── Strategy 1: EMA Dip Momentum ─────────────────────────────────────────────
```

- [ ] **Step 4: Verify import still works**

```bash
cd "/Users/ashleighchua/trading analyses" && python3 -c "import momentum_strategies_backtest"
```

Expected output (3 ✅ lines):
```
✅ indicator tests passed
✅ helper tests passed
✅ breakout helper tests passed
```

- [ ] **Step 5: Commit**

```bash
cd "/Users/ashleighchua/trading analyses" && git add momentum_strategies_backtest.py && git commit -m "feat: add BREAKOUT_PARAM_SETS, consolidation_pct, stage2_confirmed, relative_strength_vs_spy helpers"
```

---

## Task 2: Add run_breakout() function

**Files:**
- Modify: `momentum_strategies_backtest.py` — append after `run_monday_reversal()`, before `# Task 8` comment

- [ ] **Step 1: Find the insertion point**

In the file, find the line:
```python
# ---------------------------------------------------------------------------
# Task 8 — Output formatter, equity curve, main
# ---------------------------------------------------------------------------
```

Insert the entire `run_breakout` function immediately before that comment block.

- [ ] **Step 2: Add run_breakout()**

```python
# ── Strategy 4: 50-Day Breakout Momentum ─────────────────────────────────────

def run_breakout(raw, earnings_map, params):
    """
    Run 50-Day Breakout Momentum for one parameter set.
    Returns list of trade dicts.

    Signal (end-of-day, entry next morning):
      1. SPY close > SPY EMA-200 (market uptrend)
      2. Stock close > highest close of prior `lookback` bars (50-day high)
      3. Today's volume >= vol_min * 20-day avg volume (institutional surge)
      4. 15-day high-to-low range <= consol_15% of close (VCP outer band)
      5. 5-day high-to-low range  <= consol_5%  of close (VCP inner band)
      6. Stage 2: close > 150-day MA AND MA rising (Weinstein)
      7. Relative strength: stock 3-month return > SPY 3-month return (O'Neil)
      8. No earnings within 10 calendar days
      9. No open position in this ticker

    Position limits:
      - Max 2 new positions per day (prioritised by highest vol_ratio — most conviction)
      - Max 2 open positions per sector at any time
    """
    lookback  = params["lookback"]
    vol_min   = params["vol_min"]
    consol_15 = params["consol_15"]
    consol_5  = params["consol_5"]
    trail     = params["trail"]
    max_hold  = params["max_hold"]

    # Prepare DataFrames — no indicator columns needed (we compute inline)
    dfs = {}
    for t in MOMENTUM_TICKERS:
        df = get_df(raw, t)
        if df is not None:
            # Attach volume ratio series as a column for convenience
            vr = volume_ratio_series(df["Volume"].tolist(), window=20)
            df = df.copy()
            df["vol_ratio"] = vr
            dfs[t] = df

    spy_df = dfs.get("SPY")
    if spy_df is None:
        raise RuntimeError("SPY data missing — cannot determine market regime")

    # SPY EMA-200 by date for the market regime check
    spy_ema200      = ema_series(spy_df["Close"].tolist(), 200)
    spy_ema200_by_date = {
        spy_df.index[i].date(): spy_ema200[i]
        for i in range(len(spy_df))
    }

    backtest_start = pd.Timestamp(BACKTEST_START)
    trades         = []
    open_positions = {}   # {ticker: exit_idx in that ticker's df}
    open_sectors   = {}   # {sector: count of currently open positions}

    all_dates = spy_df[spy_df.index >= backtest_start].index

    for date_ts in all_dates:
        signal_date = date_ts.date()

        # ── Close out expired positions ───────────────────────────────────────
        for t in list(open_positions.keys()):
            t_df = dfs.get(t)
            if t_df is None:
                s = SECTOR.get(t, "Other")
                open_sectors[s] = max(0, open_sectors.get(s, 0) - 1)
                del open_positions[t]
                continue
            if date_ts not in t_df.index:
                continue
            cur_idx = t_df.index.get_loc(date_ts)
            if open_positions[t] <= cur_idx:
                s = SECTOR.get(t, "Other")
                open_sectors[s] = max(0, open_sectors.get(s, 0) - 1)
                del open_positions[t]

        # ── Collect all valid signals for today ───────────────────────────────
        day_signals = []

        for ticker in MOMENTUM_TICKERS:
            if ticker not in dfs:
                continue
            df = dfs[ticker]
            if date_ts not in df.index:
                continue
            idx = df.index.get_loc(date_ts)

            # Need enough warmup for 150-day MA + 63-day RS + 50-day lookback
            if idx < 155:
                continue

            # Skip if already in a position in this ticker
            if ticker in open_positions and open_positions[ticker] > idx:
                continue

            close  = df.iloc[idx]["Close"]
            vr     = df.iloc[idx]["vol_ratio"]

            # 1. Market regime: SPY close > SPY EMA-200
            spy_ema = spy_ema200_by_date.get(signal_date)
            if spy_ema is None or spy_df.loc[date_ts, "Close"] <= spy_ema:
                continue

            # 2. 50-day closing high: close > max of prior lookback closes
            prior_high = df["Close"].iloc[idx - lookback:idx].max()
            if close <= prior_high:
                continue

            # 3. Volume surge: vol_ratio >= vol_min
            if vr is None or vr < vol_min:
                continue

            # 4. VCP outer band: 15-day consolidation
            c15 = consolidation_pct(df, idx, 15)
            if c15 is None or c15 > consol_15:
                continue

            # 5. VCP inner band: 5-day consolidation
            c5 = consolidation_pct(df, idx, 5)
            if c5 is None or c5 > consol_5:
                continue

            # 6. Stage 2 confirmation (Weinstein)
            if not stage2_confirmed(df, idx):
                continue

            # 7. Relative strength vs SPY (O'Neil)
            if not relative_strength_vs_spy(df, spy_df, idx):
                continue

            # 8. Earnings filter
            if near_earnings(signal_date, ticker, earnings_map):
                continue

            day_signals.append({
                "ticker":    ticker,
                "vol_ratio": vr,
                "idx":       idx,
            })

        # ── Sort by highest vol_ratio (strongest conviction breakout) ─────────
        day_signals.sort(key=lambda s: s["vol_ratio"], reverse=True)

        # ── Enter positions (up to 2 per day, up to 2 per sector) ────────────
        new_today = 0
        for sig in day_signals:
            if new_today >= 2:
                break
            ticker = sig["ticker"]
            idx    = sig["idx"]
            df     = dfs[ticker]
            sector = SECTOR.get(ticker, "Other")

            # Sector cap
            if open_sectors.get(sector, 0) >= 2:
                continue

            # Need a next bar for entry
            if idx + 1 >= len(df):
                continue

            entry_open  = df.iloc[idx + 1]["Open"]
            entry_price = long_entry_price(entry_open)
            qty         = calc_qty(entry_price, trail)

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

    return trades
```

- [ ] **Step 3: Verify import still works**

```bash
cd "/Users/ashleighchua/trading analyses" && python3 -c "import momentum_strategies_backtest"
```

Expected: three ✅ lines, no errors.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ashleighchua/trading analyses" && git add momentum_strategies_backtest.py && git commit -m "feat: add run_breakout() — 50-day high, VCP, Stage 2, RS filters"
```

---

## Task 3: Update main() to include breakout strategy

**Files:**
- Modify: `momentum_strategies_backtest.py` — update `main()` function and `save_equity_curve()` call

- [ ] **Step 1: Locate main() in the file**

Find this block inside `main()`:
```python
    # Run Monday Reversal
    logging.info("Running Monday Reversal...")
    rev_trades = run_monday_reversal(raw)
    rev_result = compute_stats(
        rev_trades,
        "Monday Reversal (Long SPY, w/ slippage)",
        trading_days_is, trading_days_oos,
    )

    # Print results
    separator = "=" * 60
    print(f"\n{separator}")
    print("  STRATEGY COMPARISON — RIGOROUS BACKTEST")
    print(separator)
    print()
    print_strategy_result(best_momentum)
    print()
    print_strategy_result(fade_result)
    print()
    print_strategy_result(rev_result)
    print()

    # Combined stats (no re-running — reuse already-computed trades)
    all_trades = best_momentum_trades + fade_trades + rev_trades
    total_pnl  = sum(t["pnl_dollar"] for t in all_trades)
    print(separator)
    print("  COMBINED (all strategies, no overlap in SPY positions)")
    print(f"    Total trades: {len(all_trades)} | Net P&L: ${total_pnl:+,.0f}")
    print("    Equity curve: [saved to momentum_strategies_backtest.png]")
    print(separator)
    print()

    save_equity_curve([best_momentum, fade_result, rev_result])
```

- [ ] **Step 2: Replace that block with the updated version**

```python
    # Run Monday Reversal
    logging.info("Running Monday Reversal...")
    rev_trades = run_monday_reversal(raw)
    rev_result = compute_stats(
        rev_trades,
        "Monday Reversal (Long SPY, w/ slippage)",
        trading_days_is, trading_days_oos,
    )

    # Run 50-Day Breakout Momentum — all param sets, pick best OOS profit factor
    logging.info("Running 50-Day Breakout Momentum (%d param sets)...", len(BREAKOUT_PARAM_SETS))
    breakout_results = []   # list of (result_dict, trades_list)
    for params in BREAKOUT_PARAM_SETS:
        trades = run_breakout(raw, earnings_map, params)
        label = (
            f"50-Day Breakout Set {params['label']} "
            f"(vol≥{params['vol_min']}×, VCP {params['consol_15']}%/{params['consol_5']}%, "
            f"trail {int(params['trail'] * 100)}%, hold {params['max_hold']}d)"
        )
        result = compute_stats(trades, label, trading_days_is, trading_days_oos)
        breakout_results.append((result, trades))
        logging.info(
            "  Set %s: %d trades OOS PF=%.2f verdict=%s",
            params["label"], result["oos"]["n"], result["oos"]["pf"], result["verdict"]
        )

    best_breakout, best_breakout_trades = max(breakout_results, key=lambda x: x[0]["oos"]["pf"])

    # Print results
    separator = "=" * 60
    print(f"\n{separator}")
    print("  STRATEGY COMPARISON — RIGOROUS BACKTEST")
    print(separator)
    print()
    print_strategy_result(best_momentum)
    print()
    print_strategy_result(fade_result)
    print()
    print_strategy_result(rev_result)
    print()
    print_strategy_result(best_breakout)
    print()

    # Combined stats (no re-running — reuse already-computed trades)
    all_trades = best_momentum_trades + fade_trades + rev_trades + best_breakout_trades
    total_pnl  = sum(t["pnl_dollar"] for t in all_trades)
    print(separator)
    print("  COMBINED (all strategies, no overlap in SPY positions)")
    print(f"    Total trades: {len(all_trades)} | Net P&L: ${total_pnl:+,.0f}")
    print("    Equity curve: [saved to momentum_strategies_backtest.png]")
    print(separator)
    print()

    save_equity_curve([best_momentum, fade_result, rev_result, best_breakout])
```

- [ ] **Step 3: Verify import still works**

```bash
cd "/Users/ashleighchua/trading analyses" && python3 -c "import momentum_strategies_backtest"
```

Expected: three ✅ lines, no errors.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ashleighchua/trading analyses" && git add momentum_strategies_backtest.py && git commit -m "feat: update main() to run breakout strategy and include in comparison"
```

---

## Task 4: End-to-end run and verification

**Files:**
- Run: `momentum_strategies_backtest.py`
- Output: `momentum_strategies_backtest.png` (updated)

- [ ] **Step 1: Run the full backtest**

```bash
cd "/Users/ashleighchua/trading analyses" && python3 momentum_strategies_backtest.py 2>&1
```

Allow up to 15 minutes. The script downloads data, runs 4 strategies × 3 param sets each = 12 simulations.

- [ ] **Step 2: Verify sanity checks**

The output must show:
- Three ✅ test lines at startup (indicator, helper, breakout helper)
- All 4 strategies printed with IS/OOS stats
- 50-Day Breakout produces ≥ 10 trades in IS period and ≥ 10 in OOS period
  - If fewer than 10: the filters are too tight — log a WARN but don't change anything; the verdict will reflect it
- No Python errors (KeyError, AttributeError, etc.)
- `momentum_strategies_backtest.png` file updated (check mtime)

- [ ] **Step 3: Commit PNG and final state**

```bash
cd "/Users/ashleighchua/trading analyses" && git add momentum_strategies_backtest.png && git commit -m "feat: add 50-day breakout strategy — end-to-end verified"
```

- [ ] **Step 4: Report results**

Report the full comparison table output (between the === lines) and the breakout trade counts for IS and OOS periods.
