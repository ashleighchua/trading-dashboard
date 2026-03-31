# BTC Weekend Pullback — Design Spec
**Date:** 2026-03-31
**Status:** Awaiting backtest validation before live deployment

---

## Goal

Add a crypto strategy that trades the documented BTC weekend dip/Monday recovery pattern.
Runs as a completely separate parallel system — does not compete with equity trades for
the one-trade-per-day slot. Fires once per week (Sunday entry, Monday exit).

Target: additional compounding income toward September 2026 South America trip goal.

---

## Strategy Logic

| | BTC Weekend Pullback |
|---|---|
| Asset | BTC/USD (Alpaca crypto) |
| Direction | LONG only |
| Entry timing | Sunday ~8 PM Bangkok = 13:00 UTC |
| Exit timing | Monday US open ~8:30 PM Bangkok = 13:30 UTC |
| Hold period | ~19 hours max |
| Signal | BTC pulled back ≥X% from 7-day high (backtest decides threshold) |
| Stop | Trailing stop placed immediately after fill (backtest decides %) |
| Regime gate | None — crypto moves independently of FRED yield curve |

**Pattern rationale:** BTC tends to drift down through weekends (low institutional liquidity,
retail panic) then recover into Monday as risk appetite returns. The pullback filter skips
weeks where BTC already ran up and avoids chasing pumps.

---

## Entry Conditions

1. Day is Sunday
2. BTC/USD has pulled back ≥X% from its 7-day high (backtest sweeps X)
3. No existing BTC position open

If conditions met: place market buy at ~13:00 UTC Sunday, attach trailing stop immediately.

---

## Exit Conditions

**Primary:** Monday exit script closes position at market at ~13:30 UTC (Monday US open).
**Secondary:** Trailing stop hit before Monday — position already closed, Monday script is a no-op.

The Monday exit script checks for an open BTC position. If found, close at market. No manual intervention needed.

---

## Position Sizing

20% of equity. Crypto is more volatile than equities — smaller allocation than the 50% used
for bear fade shorts. Do not raise above 20% until live data confirms backtest results.

---

## Backtest Spec

**File:** `btc_weekend_backtest.py`
**Period:** 2020-01-01 to 2026-03-28
**Starting equity:** $1,000
**Data:** Alpaca crypto bars (1-hour timeframe), BTC/USD

**Entry price:** 1-hour bar close at 13:00 UTC Sunday
**Exit price:** 1-hour bar close at 13:30 UTC Monday

**Parameter sweep:**
- Pullback threshold: 3%, 5%, 7%, 10% from 7-day high
- Trailing stop: 2%, 3%, 5%
- Total combos: 12

**Acceptance criteria (same bar as equity strategies):**
- Win rate ≥ 55%
- Profit factor ≥ 1.5
- Minimum 30 trades over 6 years
- Year-by-year breakdown must not show catastrophic single years

**Output:** best parameter combo, overall WR + PF, year-by-year table.
If no combo passes, do not deploy.

---

## Infrastructure

### New files
- `btc_weekend_backtest.py` — standalone backtest script (root level)
- `dashboard/crypto_scanner.py` — Sunday entry script (fetch bars, check signal, place order, Telegram alert)
- `dashboard/crypto_exit.py` — Monday exit script (close position, Telegram alert with P&L)
- `com.trading.crypto-entry.plist` — launchd: Sunday 13:00 UTC
- `com.trading.crypto-exit.plist` — launchd: Monday 13:30 UTC

### Unchanged
- `premarket_scanner.py` — equity scanner unaffected
- `postopen_stops.py` — not used; crypto_scanner.py attaches stop directly
- `signal_engine.py`, `fred_client.py` — unaffected

### Data source
`alpaca.data.historical.crypto` → `get_crypto_bars("BTC/USD", timeframe=TimeFrame.Hour)`
Already in the Alpaca SDK — no new dependencies.

---

## Order Flow

**Sunday (crypto_scanner.py):**
1. Fetch last 7 days of BTC/USD hourly bars
2. Calculate pullback from 7-day high
3. If pullback ≥ threshold AND no existing BTC position → place market buy
4. Immediately attach trailing stop (GTC)
5. Send Telegram alert

**Monday (crypto_exit.py):**
1. Check for open BTC/USD position
2. If found → close at market
3. Calculate P&L from fill prices
4. Send Telegram alert with result
5. If no position → send "no BTC position to close" log (stop already hit)

---

## Deployment Order

1. Write and run `btc_weekend_backtest.py`
2. Review results — confirm combo passes acceptance bar
3. Add validated parameters to `crypto_scanner.py` and `crypto_exit.py`
4. Paper trade 2 weeks minimum (2 Sunday/Monday cycles)
5. Go live alongside equity strategies

---

## Interaction With Existing Strategies

Fully independent — separate scripts, separate scheduler, separate position. The equity
scanner's one-trade-per-day rule does not apply here. BTC can be open simultaneously with
an equity position. No conflicts.
