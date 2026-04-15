# GLD Backtest Design
**Date:** 2026-04-15
**Status:** Approved

## Goal

Find a profitable, automatable strategy for GLD (SPDR Gold ETF) that is uncorrelated to the existing SPY/QQQM/SPLG strategies. GLD moves on macro events (Fed, inflation, geopolitical risk) rather than market momentum, making it a natural diversifier. Once validated, the best strategy gets added to the premarket scanner alongside the existing setups.

## Data Source

Use `dashboard/data_provider.py` (Alpaca → Tiingo → yfinance fallback). This is the same source the live scanner uses, ensuring backtest results are trustworthy and consistent with live fills.

- Period: 2020–2026 (~6 years, same as other backtests)
- Symbol: `GLD`
- Stop loss baseline: 1.5% trailing (GLD daily range is ~0.5–1%, so 1.5% is firm without being too tight)

## Two Strategy Families

### Part 1 — Day-of-Week Patterns

Same methodology as `qqqm_splg_backtest.py`:
- Test all combinations: day of week × previous day color × threshold (0%, 0.3%, 0.5%, 0.7%, 1.0%)
- Also test 2-day combo patterns (e.g. Red Thursday + Red Friday → Buy Monday)
- Trade: buy at open, sell at close (same day)
- Only surface signals with WR ≥ 60% and n ≥ 20 trades
- Stop loss: 1.5%

**Honest expectation:** Gold is macro-driven, not day-of-week driven. This section may come up empty. That's a valid result — it tells us not to add noise.

### Part 2 — Trend Pullback Long

Same methodology as `nvda_pullback_backtest.py`:
- Entry condition: EMA(50) > EMA(200) (uptrend) + RSI below threshold + price pulled back ≥ X% from 10-day high
- RSI thresholds tested: 35, 40, 45
- Pullback depths tested: 2%, 3%, 4%, 5%
- Stop loss: 1.5% trailing
- Hold: up to 10 days (exit at stop or end of hold period)
- Only surface results with WR ≥ 55% and n ≥ 15 trades (fewer trades expected since signal is selective)

**Honest expectation:** Gold trends strongly. This is where we expect to find the edge.

## Output

- Print best signals from each part with: WR, profit factor, trade count, avg return
- Year-by-year breakdown of the best configuration to check consistency
- Declare a winner: day trade vs swing trade
- Flag "ready for scanner" if: PF > 2.0, WR ≥ 58%, n ≥ 15
- Save chart: `gld_backtest.png`

## Scanner Integration (post-backtest)

If the backtest finds a valid strategy:
- Add `GLD` to `LONG_WATCHLIST` in `premarket_scanner.py`
- Add the validated signal logic to `score_ticker()` (alongside NVDA trend pullback)
- GLD and SPY are uncorrelated so both can run simultaneously — no position conflict expected

## Files

| File | Purpose |
|------|---------|
| `gld_backtest.py` | New backtest script |
| `gld_backtest.png` | Output chart |
| `dashboard/premarket_scanner.py` | Add GLD after validation |
