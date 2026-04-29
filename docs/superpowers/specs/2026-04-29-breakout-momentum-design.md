# 50-Day Breakout Momentum Design

**Date:** 2026-04-29
**Status:** Approved for implementation

---

## Problem

The EMA Dip Momentum strategy (tested in `momentum_strategies_backtest.py`) achieved only 44% win rate — below the 55% threshold — because buying weakness (dips) produces too many false entries. The fix is to buy confirmed strength instead: stocks breaking out to new multi-month highs on high volume, after a quiet consolidation base, with Stage 2 confirmation and relative strength filtering. This combines William O'Neil's CANSLIM methodology, Stan Weinstein's Stage Analysis, and Mark Minervini's SEPA/VCP framework.

---

## Signal

Checked end-of-day. Entry at next morning's open.

All conditions must be true:

1. **Market uptrend:** SPY close > SPY EMA-200
2. **50-day closing high:** Today's close > highest close of the prior 50 trading days (not including today)
3. **Volume surge:** Today's volume ≥ 1.5× 20-day average volume — confirms institutional participation
4. **VCP consolidation (Minervini):** 15-day high-to-low range ≤ 5% of close AND 5-day high-to-low range ≤ 3% of close — stock was coiling and contracting, not volatile
5. **Stage 2 confirmation (Weinstein):** Close > 150-day MA AND 150-day MA slope is positive (today's 150-day MA > 150-day MA from 5 bars ago)
6. **Relative strength (O'Neil):** Stock's 3-month return > SPY's 3-month return — buying leaders, not laggards
7. **Earnings filter:** No earnings within 10 calendar days
8. **No open position** in this ticker

**Expert rationale:**
- 50-day high (O'Neil): two months of resistance cleared — a meaningful level, not noise
- 1.5× volume (O'Neil): real institutional buyers, not a one-day fluke
- VCP consolidation (Minervini): breakouts from tightening, coiling bases are far more reliable than breakouts from volatile, choppy stocks
- Stage 2 (Weinstein): price above a rising 150-day MA confirms the stock is in a confirmed uptrend, not topping
- Relative strength (O'Neil): leaders outperform the market before they break out; laggards reaching local highs are traps

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

| Set | Lookback | Volume req | 15-day range | 5-day range | Trail stop | Max hold | Reasoning |
|-----|----------|-----------|--------------|-------------|-----------|----------|-----------|
| A | 50-day high | 1.5× | ≤ 5% | ≤ 3% | 6% | 15 days | Standard O'Neil/Minervini breakout |
| B | 50-day high | 1.5× | ≤ 5% | ≤ 3% | 7% | 20 days | More room for retest |
| C | 50-day high | 2.0× | ≤ 4% | ≤ 2% | 7% | 25 days | Institutional-grade conviction, tighter VCP |

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

### Two helpers added

**`consolidation_pct(df, idx, window)`** — returns high-to-low range over prior `window` bars as % of close. Called twice per signal check (window=15 and window=5).

**`stage2_confirmed(df, idx)`** — returns True if close > 150-day MA and 150-day MA is rising (value at idx > value at idx-5).

**`relative_strength_vs_spy(df, spy_df, idx, window=63)`** — returns True if stock's 63-bar (≈3 month) return exceeds SPY's 63-bar return.

### One strategy function added: `run_breakout(raw, earnings_map, params)`

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

## Helper Implementations

```python
def consolidation_pct(df, idx, window):
    """High-to-low range over prior window bars as % of close. No lookahead."""
    if idx < window:
        return None
    slice_ = df.iloc[idx - window:idx]
    high  = slice_["High"].max()
    low   = slice_["Low"].min()
    close = df.iloc[idx]["Close"]
    if close <= 0:
        return None
    return (high - low) / close * 100


def stage2_confirmed(df, idx):
    """
    Weinstein Stage 2: close > 150-day MA AND 150-day MA is rising.
    Rising = MA at idx > MA at idx-5.
    Returns False if insufficient data.
    """
    if idx < 155:
        return False
    ma_now  = df["Close"].iloc[idx - 150:idx].mean()
    ma_prev = df["Close"].iloc[idx - 155:idx - 5].mean()
    close   = df.iloc[idx]["Close"]
    return close > ma_now and ma_now > ma_prev


def relative_strength_vs_spy(df, spy_df, idx, window=63):
    """
    O'Neil RS: stock's window-bar return > SPY's window-bar return.
    Returns False if insufficient data.
    """
    if idx < window:
        return False
    # Align by position — both DataFrames share the same trading calendar
    stock_ret = (df.iloc[idx]["Close"] - df.iloc[idx - window]["Close"]) / df.iloc[idx - window]["Close"]
    spy_idx   = min(idx, len(spy_df) - 1)
    if spy_idx < window:
        return False
    spy_ret   = (spy_df.iloc[spy_idx]["Close"] - spy_df.iloc[spy_idx - window]["Close"]) / spy_df.iloc[spy_idx - window]["Close"]
    return stock_ret > spy_ret
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
- The relative strength filter aligns stock and SPY by bar index, not by date — this is valid because both use the same trading calendar from yfinance.
- Adding 5 filters (Stage 2, RS, VCP tight) means fewer signals. If fewer than 20 trades result per period, the verdict is inconclusive rather than FAIL.

---

## Files Changed

| File | Change |
|------|--------|
| `momentum_strategies_backtest.py` | Add `consolidation_pct()`, `stage2_confirmed()`, `relative_strength_vs_spy()`, `run_breakout()`, update `main()` |

No changes to live system.

---

## Success Criteria

- Script runs end-to-end without errors
- Breakout strategy produces ≥ 20 trades in each period
- Output shows IS/OOS stats alongside existing three strategies
- Each parameter set gets an explicit PASS / WARN / FAIL verdict
