# 50-Day Breakout Momentum Design

**Date:** 2026-04-29
**Status:** Approved for implementation

---

## Problem

The EMA Dip Momentum strategy (tested in `momentum_strategies_backtest.py`) achieved only 44% win rate — below the 55% threshold — because buying weakness (dips) produces too many false entries. The fix is to buy confirmed strength instead: stocks breaking out to new multi-month highs on high volume, after a quiet consolidation base. This is the core of William O'Neil's CANSLIM breakout methodology.

---

## Signal

Checked end-of-day. Entry at next morning's open.

All conditions must be true:

1. **Market uptrend:** SPY close > SPY EMA-200
2. **50-day closing high:** Today's close > highest close of the prior 50 trading days (not including today)
3. **Volume surge:** Today's volume ≥ 1.5× 20-day average volume — confirms institutional participation
4. **Consolidation base:** The stock's high-to-low range over the prior 10 trading days ≤ 6% of today's close — stock was coiling, not volatile
5. **Earnings filter:** No earnings within 10 calendar days
6. **No open position** in this ticker

**Expert rationale (O'Neil / Weinstein):**
- 50-day high = two months of resistance cleared — a meaningful level, not noise
- 1.5× volume = real buyers showing up, not a one-day fluke
- Consolidation base = stock was coiling quietly before the move; breakouts from tight bases are far more reliable than breakouts from volatile, choppy stocks

---

## Universe

Reuse the existing 28-ticker momentum universe already downloaded in `momentum_strategies_backtest.py`. No new data download required.

```
SPY, QQQ, IWM, XLK, XLF, XLE,
AAPL, NVDA, MSFT, META, AMZN, GOOGL, TSLA, AVGO, NFLX,
INTC, IBM, BA, PFE, GE, XOM,
JPM, AMD, BAC, CVX, UNH, COST
```

---

## Position Limits

- **Max 2 new positions opened per day** — prioritise the ticker with the highest volume ratio (strongest conviction)
- **Max 2 open positions per sector** at any time
- **Earnings filter:** skip if earnings within 10 calendar days (uses existing `near_earnings()` helper)

---

## Exit Rules

Trailing stop, updated daily on close. Entry price includes 0.1% slippage. Exit price includes 0.1% slippage.

O'Neil's rule: give breakouts room to retest — a 4% stop gets shaken out; 6–7% is the professional standard.

---

## Parameter Sets (3 pre-defined, not swept)

| Set | Lookback | Volume req | Consolidation (10-day range) | Trail stop | Max hold | Reasoning |
|-----|----------|-----------|------------------------------|-----------|----------|-----------|
| A | 50-day high | 1.5× | ≤ 6% | 6% | 15 days | Standard O'Neil breakout |
| B | 50-day high | 1.5× | ≤ 6% | 7% | 20 days | More room for retest |
| C | 50-day high | 2.0× | ≤ 5% | 7% | 25 days | Institutional-grade conviction only |

Winner selected by out-of-sample profit factor. If no set passes criteria, verdict is NO-GO.

---

## Pass Criteria (out-of-sample)

Same as all strategies in this backtest:
- Win rate ≥ 55%
- Profit factor ≥ 1.5
- Max drawdown ≤ 25% of $10k notional
- OOS win rate within 10 percentage points of IS win rate

---

## Walk-Forward Split

| Period | Label |
|--------|-------|
| 2020-01-01 → 2022-12-31 | In-sample |
| 2023-01-01 → 2026-04-25 | Out-of-sample |

---

## What Gets Built

### One function added: `run_breakout(raw, earnings_map, params)`

Added to `momentum_strategies_backtest.py`. Uses all existing helpers:
- `get_df()` — data access
- `add_indicators()` — EMA series (for SPY EMA-200)
- `volume_ratio_series()` — volume filter
- `near_earnings()` — earnings filter
- `calc_qty()`, `long_entry_price()`, `long_exit_price()` — sizing and slippage
- `simulate_long_trade()` — trailing stop simulation
- `compute_stats()` — IS/OOS stats and verdict

### `main()` updated

Runs `run_breakout()` for all 3 param sets, picks best by OOS profit factor, prints as a 4th strategy in the comparison table, includes in combined equity curve.

### No other files change

Live scanner, dashboard, and signal engine are untouched. This is backtest-only until the strategy passes the bar.

---

## Consolidation Filter Implementation

```python
def consolidation_range_pct(df, idx, window=10):
    """
    Returns the high-to-low range over the prior `window` bars
    as a percentage of the close at idx.
    Uses only bars before idx (no lookahead).
    """
    if idx < window:
        return None
    slice_ = df.iloc[idx - window:idx]
    high = slice_["High"].max()
    low  = slice_["Low"].min()
    close = df.iloc[idx]["Close"]
    if close <= 0:
        return None
    return (high - low) / close * 100
```

---

## Output (illustrative)

```
  50-Day Breakout Momentum Set B (best out-of-sample)
    In-sample  (2020–2022): XX trades | XX.X% WR | PF X.XX | $+X,XXX
    Out-of-sample (2023–2026): XX trades | XX.X% WR | PF X.XX | $+X,XXX
    Max drawdown: IS X.X% / OOS X.X% | Days in market: IS X% / OOS X%
    Verdict: ✅ PASS
```

---

## Known Limitations

- Survivorship bias not fully eliminated — individual stocks are 2026 survivors. ETFs have zero survivorship bias.
- yfinance earnings dates unreliable for pre-2023 periods.
- Slippage estimated at 0.1% per side. Real slippage on gap-up breakout opens may be higher.
- 50-day high breakouts in a strong bull market (2020–2021, 2023–2024) will generate many signals simultaneously — the 2-per-day cap and sector cap manage concentration risk but do not eliminate it.

---

## Files Changed

| File | Change |
|------|--------|
| `momentum_strategies_backtest.py` | Add `consolidation_range_pct()`, `run_breakout()`, update `main()` |

No changes to live system.

---

## Success Criteria

- Script runs end-to-end without errors
- Breakout strategy produces ≥ 20 trades in each period
- Output shows IS/OOS stats alongside existing three strategies
- Each parameter set gets an explicit PASS / WARN / FAIL verdict
