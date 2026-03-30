# Trend Pullback Long — Design Spec
**Date:** 2026-03-31
**Status:** Awaiting backtest validation before live deployment

---

## Goal

Add a long-side swing trade strategy that fires 3–5x per week, is fully automated
(set-and-forget), and complements the existing bear rally fade. The two strategies
alternate naturally: fades fire in downtrends, longs fire in uptrends.

Target: more frequent trades on the $5k live account to compound returns before
September 2026.

---

## Strategy Logic

Mirror image of the bear rally fade:

| | Bear Rally Fade | Trend Pullback Long |
|---|---|---|
| Trend filter | EMA50 < EMA200 (downtrend) | EMA20 > EMA50 OR EMA50 > EMA200 (backtest decides) |
| Trigger | Bounced X%+ off 10-day low | Pulled back X% from 10-day high |
| Direction | SHORT | LONG |
| Regime gate | Blocked in RISK_ON | Blocked in RISK_OFF |

---

## Entry Conditions

1. Ticker is in the LONG_WATCHLIST
2. Uptrend confirmed (backtest sweeps EMA20>EMA50 vs EMA50>EMA200)
3. Price has pulled back X% from 10-day high (backtest sweeps 2%, 3%, 4%, 5%)
4. Macro regime is NEUTRAL or RISK_ON (RISK_OFF blocks longs — pullbacks become waterfalls)
5. No existing position or open order in that ticker

---

## Exit Conditions (backtest decides best combo)

- Trailing stop: sweep 1.5%, 2%, 3%
- Max hold: sweep 5 days, 10 days, 15 days
- Profit target: 2.5% (same as bear fade — backtest can override)

---

## Ticker Universe

**LONG_WATCHLIST:** SPY, QQQ, IWM, AAPL, NVDA, MSFT, AMD

Backtest will show which tickers have positive profit factor over 6 years.
Dead tickers (negative PF) will be excluded before live deployment — same
process as trimming FADE_WATCHLIST from 14 to 6.

---

## Position Sizing

Start at 50% of equity. If backtest shows significantly lower volatility than
bear fade, consider raising. Let data decide.

---

## Regime Gate

New `regime_allows_long(regime, regime_data)` function in `fred_client.py`:
- RISK_OFF → block (bear market, pullbacks extend not bounce)
- NEUTRAL → allow
- RISK_ON → allow (ideal — pullbacks get bought fast)
- Steepening rally → allow (curve normalising = bullish for longs)

Exact mirror of `regime_allows_short()`.

---

## Backtest Spec

**File:** `trend_pullback_backtest.py`
**Period:** 2020-01-01 to 2026-03-28 (same as existing backtests)
**Starting equity:** $1,000

**Parameter sweep:**
- EMA filter: (20,50) vs (50,200)
- Pullback depth: 2%, 3%, 4%, 5% from 10-day high
- Trailing stop: 1.5%, 2%, 3%
- Max hold: 5, 10, 15 days

**Acceptance criteria (same bar as existing strategies):**
- Win rate ≥ 55%
- Profit factor ≥ 1.5
- Minimum 30 trades over 6 years (enough sample size)
- Year-by-year breakdown must not show catastrophic single years

**Output:** best parameter combo per ticker, overall and per-ticker PF,
year-by-year table.

---

## Code Changes

### New code
- `trend_pullback_backtest.py` — standalone backtest script
- `regime_allows_long()` in `dashboard/fred_client.py`
- `trend_pullback()` function in `dashboard/ta_utils.py`

### Modified code
- `dashboard/premarket_scanner.py`
  - Add `LONG_WATCHLIST` constant
  - Add long pullback setup block in `score_ticker()`
  - Wire `regime_allows_long()` into long block

### Unchanged
- Scanner schedule (8:20 PM Bangkok / 13:20 UTC)
- `postopen_stops.py` (already handles longs by checking order side)
- `dashboard/tuesday_close.py` (SPY Monday Reversal unaffected)
- Telegram alert format

---

## Deployment Order

1. Write and run `trend_pullback_backtest.py`
2. Review results — trim LONG_WATCHLIST to positive-PF tickers only
3. Add validated parameters to live scanner
4. Paper trade 2 weeks minimum
5. Go live alongside bear fade + Monday Reversal

---

## Interaction With Existing Strategies

One trade per day maximum (existing rule stays). If a fade setup AND a pullback
long fire on the same day, scanner picks higher conviction score. In practice
conflict is rare — fades fire in downtrends, longs fire in uptrends. They
alternate with market regime.

Monday Reversal (SPY, Mondays only) is unaffected — it runs as a separate
check in `score_ticker()` and gets conviction +4 bonus, so it wins any
same-day conflict.
