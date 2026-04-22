# Scanner Redesign: Fade Watchlist + EMA-200 Filter

**Date:** 2026-04-23
**Status:** Approved for implementation

---

## Problem

The premarket scanner's bear-rally-fade strategy is producing too many low-quality short signals:

1. **Bad tickers in the watchlist.** KWEB (WR 29.8%, PF 0.61), TSLA (WR 40.8%, PF 1.14), and AMD (WR 39.0%, PF 1.33) are net drags. The scanner fired on these 332 times over 6 years — the majority were losing trades.

2. **No long-term trend filter.** The signal only checks the short-term trend (5-day MA vs 20-day MA). Without confirming the long-term downtrend, the scanner fires on temporary dips inside broader upswings. This inflates trade count but dilutes the edge.

3. **Result:** Combined across all 5 tickers, the strategy has a 41.7% win rate — below 50%. That means most trades lose.

---

## What the Backtest Found

Backtest period: 2020-01-01 → 2026-04-01 (6 years, ~1,500 trading days).
Signal: exact `ta_utils.trend_context()` logic. Entry: next-day open (mirrors live OPG order).

| Configuration | Trades | Win Rate | Profit Factor | Max Drawdown |
|---|---|---|---|---|
| All 5 tickers, no EMA-200 | 417 | 41.7% | 1.31 | -29.9% |
| PLTR + FXI only, no EMA-200 | 135 | 51.9% | 2.05 | -11.3% |
| All 5 tickers + EMA-200 filter | 216 | 44.4% | 1.39 | -19.6% |
| **PLTR + FXI + EMA-200 filter** | **74** | **54.1%** | **1.88** | **-8.4%** |

The combination of both fixes produces the cleanest signal: highest win rate, best profit factor, lowest drawdown. Minimum quality bar: 20+ trades, WR ≥ 52%, PF ≥ 1.5. Only PLTR+FXI+EMA-200 passes.

**Expected trade frequency:** ~12 trades/year across both tickers (6/year each). Fewer but higher quality.

### Honest caveat on the recent PLTR losses

The two PLTR losses on Apr 17 (-$2,890) and Apr 21 (-$2,728) occurred while PLTR was *below* its EMA-200. The EMA-200 filter would not have blocked them. They are normal losses within the strategy's 46% expected loss rate — not a signal that the filter is broken. PLTR crossed *above* EMA-200 on Apr 22, which is why the scanner pivoted to KWEB that day.

---

## Design

### Change 1: Shrink the fade watchlist

```python
# Before
FADE_WATCHLIST = ["KWEB", "FXI", "PLTR", "TSLA", "AMD"]

# After
FADE_WATCHLIST = ["PLTR", "FXI"]
```

Rationale per removed ticker:
- **KWEB:** 29.8% WR, PF 0.61 — actively losing money over 6 years
- **TSLA:** 40.8% WR, PF 1.14 — marginal, inconsistent volatility
- **AMD:** 39.0% WR, PF 1.33 — below quality bar, no meaningful edge

### Change 2: Add EMA-200 confirmation to the bear-rally-fade signal

The scanner's `score_ticker()` function currently checks:
- `trend.get("trend") == "downtrend"` — short-term (5-day vs 20-day MA)
- `trend.get("bear_rally") == True` — bounce ≥ 4% off 5-day low

**Add one more condition:** price must be below EMA-200.

This confirms the stock is in a genuine long-term downtrend, not just a short-term wobble inside a recovery.

Implementation: EMA-200 requires 260 bars of history (same amount already fetched for NVDA and GLD). The `bar_limit` for PLTR and FXI must increase from 25 to 260.

```python
# bar_limit logic (existing pattern, extend to FADE_WATCHLIST)
bar_limit = 260 if symbol in LONG_WATCHLIST or symbol in FADE_WATCHLIST else 25
```

The EMA-200 is calculated from the `bars` list already fetched inside `score_ticker()`, using a simple EMA loop (multiplier = 2/201). Add `and price_below_ema200` to the bear-rally-fade condition block. No new data fetches required — the 260 bars are already in scope.

### Change 3: Update bar_limit for FADE_WATCHLIST

Currently `LONG_WATCHLIST = ["NVDA", "GLD"]` triggers 260-bar fetch. PLTR and FXI need the same. The cleanest approach: combine LONG and FADE into one 260-bar group.

---

## What Does NOT Change

- Sizing logic (2% risk, max $100 loss per share)
- Trailing stop (1.5% for PLTR/FXI)
- OPG order placement (market order at open)
- Telegram alert format
- `record_entry()` DB write on placement
- `postopen_stops.py` stop guardian
- All other strategies (Monday Reversal, NVDA trend pullback, GLD pullback)

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| 74 trades over 6 years = ~12/year — very few | Acceptable. Quality > quantity. SPY Monday Reversal and NVDA/GLD pullback strategies continue running alongside. |
| PLTR/FXI short borrow occasionally unavailable (Apr 14, Apr 16 failures) | Log and alert when `"cannot be sold short"` — don't retry, wait for next signal |
| EMA-200 filter fires less often → scanner goes longer without trades | Expected and correct. The signal is rare by design. |
| KWEB open position (Apr 22 trade) — existing trade not covered by new design | KWEB is already in. Let it run to close naturally. New rule applies to future signals only. |

---

## Files to Change

| File | Change |
|---|---|
| `dashboard/premarket_scanner.py` | Shrink `FADE_WATCHLIST`, increase `bar_limit` for fade tickers, add EMA-200 check in `score_ticker()` |

No other files change.

---

## Success Criteria

- Scanner no longer fires on KWEB, TSLA, AMD (after current KWEB trade closes)
- Scanner only fades PLTR or FXI when price is below EMA-200
- All other strategies (Monday Reversal, NVDA, GLD) unaffected
- Backtest-validated: 54.1% WR, PF 1.88 over 74 trades (2020–2026)
