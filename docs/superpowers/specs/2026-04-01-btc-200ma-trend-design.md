# BTC 200-Day MA Trend Following — Design Spec

**Date:** 2026-04-01
**Status:** Approved

---

## Goal

Automate a weekly BTC/USD trend-following check that buys when BTC is above its 200-day SMA and sells when it falls below. Runs alongside the existing equity system without interference.

---

## Strategy Logic

| Condition | Action |
|-----------|--------|
| BTC > 200-day SMA, no BTC position | Buy BTC — 20% of account equity |
| BTC < 200-day SMA, BTC position open | Sell full position (market order) |
| Already correctly positioned | Do nothing, log status |

**No stop loss.** The 200-day SMA crossing is the exit signal. The edge comes from holding through volatility — the backtest showed 11 trades over 6 years with PF 9.17. Cutting early destroys the thesis.

**Backtest results (btc_ma_backtest.py):**
- Period: 2018–2024
- Trades: 11
- Win rate: 45.5%
- Profit factor: 9.17
- $1k → $8,073 (100% sizing), $1k → $8,000+ (20% sizing)
- Current signal: LONG (BTC above 200-day MA as of 2026-04-01)

---

## Architecture

### New Files

**`dashboard/crypto_weekly.py`**
Standalone script. Same structural pattern as `tuesday_close.py`. Responsibilities:
1. Fetch last 201 daily bars of BTC/USD from Alpaca crypto API
2. Calculate 200-day SMA from closing prices
3. Compare current BTC price to SMA
4. Fetch open positions — check for existing BTC/USD
5. Fetch account equity for position sizing
6. Place buy or sell order as needed (or log no-op)
7. Send Telegram alert with action taken and key numbers

**`com.trading.crypto-weekly.plist`**
launchd agent. Runs every Monday at 20:20 Bangkok local time (= 13:20 UTC = 9:20 AM EDT). Same timeslot as equity premarket scanner — intentional, different market.

### Existing Files — No Changes

- `dashboard/signal_engine.py` — equity only, untouched
- `dashboard/premarket_scanner.py` — equity only, untouched
- `dashboard/tuesday_close.py` — SPY only, untouched

---

## Data Source

**Alpaca Crypto API** (`CryptoHistoricalDataClient`)
- Symbol: `BTC/USD`
- Timeframe: `TimeFrame.Day`
- Limit: 201 bars (200 for SMA + 1 current)
- No API key needed for crypto data — uses same Alpaca credentials from `.env`

---

## Position Sizing

```
equity = float(account.equity)
btc_price = current_bar.close
qty = (equity * 0.20) / btc_price   # rounded down to 6 decimal places
```

20% of portfolio. Matches backtest sizing. Alpaca supports fractional BTC.

---

## Orders

- **Buy:** `MarketOrderRequest(symbol="BTC/USD", qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)`
- **Sell:** `MarketOrderRequest(symbol="BTC/USD", qty=existing_qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)`
- **Client order ID:** `btc-ma-buy-YYYYMMDD` / `btc-ma-sell-YYYYMMDD` for traceability

---

## Telegram Alerts

| Event | Message |
|-------|---------|
| Buy | `🟢 BTC TREND — ENTERED LONG\nBTC: $X | 200-day MA: $Y\nBought X BTC @ ~$X\n20% equity deployed` |
| Sell | `🔴 BTC TREND — EXITED\nBTC: $X | 200-day MA: $Y\nSold X BTC\nP&L: $X` |
| Holding | `📊 BTC TREND CHECK\nBTC: $X | 200-day MA: $Y\nSignal: LONG — already positioned` |
| No position, signal flat | `📊 BTC TREND CHECK\nBTC: $X | 200-day MA: $Y\nSignal: FLAT — no position, staying out` |
| Error | `⚠️ BTC TREND ERROR\n<error message>` |

---

## Error Handling

- Alpaca API failure → log + Telegram error alert, exit cleanly
- Insufficient bars returned (< 201) → log warning, abort (don't trade on partial data)
- Position already correct direction → skip order, log status
- Account equity unavailable → abort (don't size blindly)
- Script is **idempotent**: safe to re-run; checks position before placing orders

---

## Scheduler (launchd)

```xml
<!-- com.trading.crypto-weekly.plist -->
<!-- Monday 20:20 Bangkok = 13:20 UTC = 9:20 AM EDT -->
<key>StartCalendarInterval</key>
<dict>
    <key>Weekday</key><integer>1</integer>  <!-- Monday Bangkok -->
    <key>Hour</key><integer>20</integer>
    <key>Minute</key><integer>20</integer>
</dict>
```

Logs: `crypto_weekly.log` / `crypto_weekly_error.log` in project root.

---

## Environment Variables Required

All already in `.env`:
- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

---

## Out of Scope

- Stop losses (strategy thesis requires holding through drawdowns)
- Multiple crypto assets (BTC only for now)
- Intraday checks (weekly only — this is a trend strategy, not a scalp)
- Dashboard integration (can add later; not needed for MVP)
