# BTC Weekend Pullback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully automated BTC/USD weekend pullback strategy — buy Sunday 13:00 UTC, exit Monday 13:30 UTC — running as a separate parallel system alongside the equity scanner.

**Architecture:** Two standalone scripts (`crypto_scanner.py` entry, `crypto_exit.py` exit) scheduled via launchd plists with `Weekday` keys. The backtest uses yfinance daily BTC-USD bars; the live scripts use Alpaca's crypto API. No dependencies on the equity scanner or FRED client.

**Tech Stack:** Python 3.9, Alpaca SDK (`CryptoHistoricalDataClient`, `TradingClient`), yfinance (backtest only), launchd plists, Telegram bot.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `btc_weekend_backtest.py` | Create | Sweep pullback + trail params, validate strategy |
| `dashboard/crypto_scanner.py` | Create | Sunday entry: fetch bars, check signal, place buy + stop |
| `dashboard/crypto_exit.py` | Create | Monday exit: close BTC position, send P&L alert |
| `com.trading.crypto-entry.plist` | Create | launchd: Sunday 13:00 UTC |
| `com.trading.crypto-exit.plist` | Create | launchd: Monday 13:30 UTC |

---

## Task 1: Write and run btc_weekend_backtest.py

**Files:**
- Create: `btc_weekend_backtest.py`

**Context:** Sweep pullback threshold (3%, 5%, 7%, 10%) and trailing stop (2%, 3%, 5%). Entry = Sunday close. Exit = Monday close or stop (if Monday low <= entry * (1 - trail_pct)). Position size = 20% of equity. Period 2020–2026. Same acceptance bar as all other strategies: ≥55% WR, PF ≥1.5, ≥30 trades.

- [ ] **Step 1: Create the backtest file**

```python
# btc_weekend_backtest.py
"""
BTC Weekend Pullback — Backtest
================================
Strategy: Buy BTC every Sunday when it has pulled back >=X% from its 7-day high.
Exit: Monday close. Trailing stop simulated: if Monday low <= entry*(1-trail_pct), exit at stop.

Parameter sweep:
  Pullback threshold: 3%, 5%, 7%, 10%
  Trailing stop:      2%, 3%, 5%
  Total combos:       12

Period: 2020-01-01 to 2026-03-28
Starting equity: $1,000
Position size:   20% of equity
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings("ignore")

START          = "2019-06-01"   # warmup for 7-day rolling high
BACKTEST_START = "2020-01-01"
END            = "2026-03-28"
INIT_EQ        = 1_000.0
SIZE_PCT       = 0.20

PULLBACK_THRESHOLDS = [3.0, 5.0, 7.0, 10.0]
TRAIL_PCTS          = [0.02, 0.03, 0.05]

print("Downloading BTC-USD daily bars 2019-06-01 to 2026-03-28...")
df_raw = yf.download("BTC-USD", start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)
df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
df_raw = df_raw.dropna()
print("Done. {} bars.\n".format(len(df_raw)))


def add_indicators(df):
    df = df.copy()
    df["high7"]       = df["High"].rolling(7).max()
    df["pullback_pct"] = (df["high7"] - df["Close"]) / df["high7"] * 100
    df["weekday"]     = df.index.weekday  # 6 = Sunday
    return df


def simulate_trade(df, entry_idx, entry_price, trail_pct):
    """Simulate Sunday close entry -> Monday close exit with trailing stop."""
    if entry_idx + 1 >= len(df):
        return 0.0, "no_next_bar"
    next_bar   = df.iloc[entry_idx + 1]
    stop_price = entry_price * (1 - trail_pct)
    if next_bar["Low"] <= stop_price:
        pnl_pct = (stop_price - entry_price) / entry_price * 100
        return pnl_pct, "stop"
    exit_price = next_bar["Close"]
    pnl_pct    = (exit_price - entry_price) / entry_price * 100
    return pnl_pct, "close"


def run(pullback_threshold, trail_pct):
    df     = add_indicators(df_raw)
    trades = []
    equity = INIT_EQ
    start_idx = df.index.searchsorted(pd.Timestamp(BACKTEST_START))

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        if row["weekday"] != 6:
            continue
        if pd.isna(row["pullback_pct"]) or row["pullback_pct"] < pullback_threshold:
            continue
        entry_price = row["Close"]
        if entry_price <= 0:
            continue
        trade_size     = equity * SIZE_PCT
        pnl_pct, reason = simulate_trade(df, i, entry_price, trail_pct)
        pnl_dollar     = pnl_pct / 100 * trade_size
        equity         += pnl_dollar
        trades.append({
            "date":    df.index[i],
            "entry":   round(entry_price, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_$":   round(pnl_dollar, 2),
            "reason":  reason,
            "win":     pnl_pct > 0,
            "equity":  round(equity, 2),
            "year":    df.index[i].year,
        })
    return pd.DataFrame(trades)


def stats(df):
    if df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
    n    = len(df)
    wins = df["win"].sum()
    wr   = wins / n * 100
    gp   = df.loc[df["win"],  "pnl_pct"].sum()
    gl   = abs(df.loc[~df["win"], "pnl_pct"].sum())
    pf   = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(df["pnl_$"].sum(), 2)}


# ── Sweep ──────────────────────────────────────────────────────────────────────
print("Sweeping parameters...\n")
results = []
for pb, trail in product(PULLBACK_THRESHOLDS, TRAIL_PCTS):
    trades = run(pb, trail)
    s      = stats(trades)
    s.update({"pullback": pb, "trail": trail})
    results.append(s)

results_df = pd.DataFrame(results).sort_values("pf", ascending=False)

print("=" * 62)
print("  ALL COMBOS BY PROFIT FACTOR")
print("=" * 62)
for _, row in results_df.iterrows():
    passed = row["wr"] >= 55 and row["pf"] >= 1.5 and row["n"] >= 30
    flag   = "✅" if passed else "⚠️ "
    print("  {} pb={:.0f}% stop={:.0f}% | {:3d} trades  {:.1f}% WR  PF {:.2f}  ${:+.0f}".format(
        flag, row["pullback"], row["trail"] * 100,
        int(row["n"]), row["wr"], row["pf"], row["pnl"]
    ))

# ── Year-by-year for best passing combo ────────────────────────────────────────
valid = results_df[
    (results_df["n"]  >= 30) &
    (results_df["wr"] >= 55) &
    (results_df["pf"] >= 1.5)
]
if not valid.empty:
    best = valid.iloc[0]
    print("\n{}\n  BEST: pb={:.0f}% stop={:.0f}%\n{}".format(
        "=" * 62, best["pullback"], best["trail"] * 100, "=" * 62))
    best_trades = run(best["pullback"], best["trail"])
    if not best_trades.empty:
        yearly = best_trades.groupby("year").agg(
            n   =("pnl_pct", "count"),
            wr  =("win",     lambda x: x.mean() * 100),
            pnl =("pnl_$",  "sum"),
        )
        for yr, row in yearly.iterrows():
            bar  = "█" * max(0, int(row["pnl"] / 5)) if row["pnl"] > 0 else "▒" * max(0, int(abs(row["pnl"]) / 5))
            flag = "✓" if row["wr"] >= 55 else "✗"
            print("    {}: {:2d} trades  {:4.1f}% WR  ${:+6.1f}  {}  {}".format(
                yr, int(row["n"]), row["wr"], row["pnl"], flag, bar))
else:
    print("\n⚠️  No combo passed acceptance criteria (>=30 trades, >=55% WR, PF >=1.5)")
    best = results_df.iloc[0]
    print("   Best available: pb={:.0f}% stop={:.0f}% | {} trades  {:.1f}% WR  PF {:.2f}".format(
        best["pullback"], best["trail"] * 100, int(best["n"]), best["wr"], best["pf"]))
```

- [ ] **Step 2: Run the backtest**

```bash
cd "/Users/ashleighchua/trading analyses"
python3 btc_weekend_backtest.py
```

Expected: prints combo table + year-by-year for best passing combo. If no combo passes, do not proceed to Task 2 — the strategy has no edge on this data. Record the best passing pullback_threshold and trail_pct values before moving on.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add btc_weekend_backtest.py
git commit -m "feat: add BTC weekend pullback backtest"
```

---

## Task 2: Write dashboard/crypto_scanner.py

**Files:**
- Create: `dashboard/crypto_scanner.py`

**Context:** Runs Sunday 13:00 UTC via launchd. Fetches last 14 days of BTC/USD daily bars from Alpaca crypto API, checks pullback from 7-day high, places market buy + trailing stop if signal fires. Uses validated parameters from Task 1 backtest. The stale guard checks both time drift AND day-of-week so it's safe to re-run manually.

Replace `PULLBACK_THRESHOLD` and `TRAIL_PERCENT` with the best values found in Task 1 before committing.

- [ ] **Step 1: Create crypto_scanner.py**

```python
#!/usr/bin/env python3
"""
Crypto Scanner — BTC Weekend Pullback Entry
============================================
Runs Sunday 13:00 UTC via launchd.
Fetches BTC/USD daily bars, checks pullback from 7-day high.
If signal fires: place market buy + trailing stop. Send Telegram alert.

Validated parameters (from btc_weekend_backtest.py):
  Pullback threshold: X% from 7-day high  ← replace with backtest result
  Trailing stop:      X%                  ← replace with backtest result
"""

import logging
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

BOT_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID     = os.environ["TELEGRAM_CHAT_ID"]
APCA_KEY    = os.environ["APCA_API_KEY_ID"]
APCA_SECRET = os.environ["APCA_API_SECRET_KEY"]

trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
crypto_client  = CryptoHistoricalDataClient(api_key=APCA_KEY, secret_key=APCA_SECRET)

SYMBOL             = "BTC/USD"
PULLBACK_THRESHOLD = 5.0    # % pullback from 7-day high — replace with backtest result
TRAIL_PERCENT      = 3.0    # % trailing stop — replace with backtest result
SIZE_PCT           = 0.20   # 20% of equity

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(text):
    url = "https://api.telegram.org/bot{}/sendMessage".format(BOT_TOKEN)
    for parse_mode in ("Markdown", None):
        try:
            params = {"chat_id": CHAT_ID, "text": text}
            if parse_mode:
                params["parse_mode"] = parse_mode
            data = urllib.parse.urlencode(params).encode()
            req  = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
            return
        except Exception as e:
            if parse_mode is None:
                logging.error("Telegram error: %s", e)

# ---------------------------------------------------------------------------
# Stale guard
# ---------------------------------------------------------------------------

def is_stale():
    """Return True if running outside Sunday 12:00–14:00 UTC window."""
    now = datetime.now(timezone.utc)
    if now.weekday() != 6:  # 6 = Sunday
        logging.info("Not Sunday (weekday=%d). Exiting.", now.weekday())
        return True
    scheduled_minutes = 13 * 60
    current_minutes   = now.hour * 60 + now.minute
    diff = abs(current_minutes - scheduled_minutes)
    if diff > 60:
        logging.info("Stale — running more than 60 min from 13:00 UTC. Exiting.")
        return True
    return False

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_btc_bars(days=14):
    """Fetch recent daily BTC/USD bars."""
    try:
        req  = CryptoBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=days),
        )
        resp = crypto_client.get_crypto_bars(req)
        raw  = resp.data if hasattr(resp, "data") else resp
        bars = raw.get(SYMBOL, [])
        return [
            {"h": float(b.high), "l": float(b.low), "c": float(b.close), "t": str(b.timestamp)}
            for b in bars
        ]
    except Exception as e:
        logging.error("fetch_btc_bars error: %s", e)
        return []


def check_pullback(bars):
    """Return (pullback_pct, high_7d, current_price) or None if no signal."""
    if len(bars) < 7:
        return None
    recent = bars[-7:]
    high_7d = max(b["h"] for b in recent)
    current = bars[-1]["c"]
    if high_7d <= 0:
        return None
    pullback_pct = (high_7d - current) / high_7d * 100
    return pullback_pct, high_7d, current


def has_btc_position():
    """Return True if BTC/USD position already open."""
    try:
        positions = trading_client.get_all_positions()
        return any(p.symbol == SYMBOL for p in positions)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def place_entry(current_price):
    """Place market buy for BTC/USD. Returns (order, qty)."""
    try:
        account    = trading_client.get_account()
        equity     = float(account.equity)
        trade_size = equity * SIZE_PCT
        qty        = round(trade_size / current_price, 6)  # fractional BTC

        req   = MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
            client_order_id="crypto-entry-btc-{}".format(
                datetime.now(timezone.utc).strftime("%Y%m%d")
            ),
        )
        order = trading_client.submit_order(req)
        return order, qty
    except Exception as e:
        logging.error("place_entry error: %s", e)
        return None, 0


def place_stop(qty, fill_price):
    """Attach trailing stop to BTC position."""
    try:
        req   = TrailingStopOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            trail_percent=TRAIL_PERCENT,
            client_order_id="crypto-stop-btc-{}".format(
                datetime.now(timezone.utc).strftime("%Y%m%d")
            ),
        )
        return trading_client.submit_order(req)
    except Exception as e:
        logging.error("place_stop error: %s", e)
        send_telegram("⚠️ *Crypto stop placement failed*\n`{}`".format(e))
        return None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.info("Crypto scanner starting...")

    if is_stale():
        return

    bars = fetch_btc_bars(days=14)
    if not bars:
        logging.info("No BTC bars fetched. Exiting.")
        return

    result = check_pullback(bars)
    if result is None:
        logging.info("Not enough bars for pullback check.")
        return

    pullback_pct, high_7d, current_price = result
    logging.info("BTC/USD: price=${:.0f}, 7d_high=${:.0f}, pullback={:.1f}%".format(
        current_price, high_7d, pullback_pct))

    if pullback_pct < PULLBACK_THRESHOLD:
        logging.info("Pullback {:.1f}% < threshold {:.1f}%. No signal.".format(
            pullback_pct, PULLBACK_THRESHOLD))
        send_telegram(
            "🌙 *BTC WEEKEND SCAN*\n\nNo signal. Pullback {:.1f}% < {:.0f}% threshold.".format(
                pullback_pct, PULLBACK_THRESHOLD)
        )
        return

    if has_btc_position():
        logging.info("BTC position already open. Skipping.")
        return

    logging.info("Signal! BTC pulled back {:.1f}% from ${:.0f}. Entering long.".format(
        pullback_pct, high_7d))

    order, qty = place_entry(current_price)
    if order is None:
        send_telegram("⚠️ *Crypto entry order FAILED* for BTC/USD")
        return

    # Small delay to allow fill before attaching stop
    import time
    time.sleep(3)
    stop = place_stop(qty, current_price)

    stop_price = round(current_price * (1 - TRAIL_PERCENT / 100), 0)
    msg = (
        "🌙 *BTC WEEKEND PULLBACK*\n\n"
        "📈 *LONG BTC/USD*\n"
        "Entry: ~${:.0f}\n"
        "7-day high: ${:.0f}\n"
        "Pullback: {:.1f}%\n"
        "Stop: {:.0f}% trailing (~${:.0f})\n"
        "Qty: {:.6f} BTC\n\n"
        "Exit: Monday 13:30 UTC (or stop)\n\n"
        "✅ Order #{}\n"
        "🛡️ Stop #{}"
    ).format(
        current_price, high_7d, pullback_pct,
        TRAIL_PERCENT, stop_price, qty,
        str(order.id)[:8] if order else "N/A",
        str(stop.id)[:8] if stop else "FAILED",
    )
    send_telegram(msg)
    logging.info("Entry placed. Order: %s", order.id if order else "FAILED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is syntactically correct**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "import ast; ast.parse(open('crypto_scanner.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add dashboard/crypto_scanner.py
git commit -m "feat: add BTC weekend pullback entry scanner"
```

---

## Task 3: Write dashboard/crypto_exit.py

**Files:**
- Create: `dashboard/crypto_exit.py`

**Context:** Runs Monday 13:30 UTC via launchd. Finds any open BTC/USD position and closes it at market. If no position exists (trailing stop already hit over the weekend), logs and exits cleanly. Calculates P&L from position's cost basis vs current price.

- [ ] **Step 1: Create crypto_exit.py**

```python
#!/usr/bin/env python3
"""
Crypto Exit — BTC Weekend Pullback Monday Close
================================================
Runs Monday 13:30 UTC via launchd.
Closes any open BTC/USD position at market.
If no position (stop already hit), logs and exits cleanly.
"""

import logging
import os
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

BOT_TOKEN   = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID     = os.environ["TELEGRAM_CHAT_ID"]
APCA_KEY    = os.environ["APCA_API_KEY_ID"]
APCA_SECRET = os.environ["APCA_API_SECRET_KEY"]

trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)

SYMBOL = "BTC/USD"

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(text):
    url = "https://api.telegram.org/bot{}/sendMessage".format(BOT_TOKEN)
    for parse_mode in ("Markdown", None):
        try:
            params = {"chat_id": CHAT_ID, "text": text}
            if parse_mode:
                params["parse_mode"] = parse_mode
            data = urllib.parse.urlencode(params).encode()
            req  = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
            return
        except Exception as e:
            if parse_mode is None:
                logging.error("Telegram error: %s", e)

# ---------------------------------------------------------------------------
# Stale guard
# ---------------------------------------------------------------------------

def is_stale():
    """Return True if running outside Monday 12:30–14:30 UTC window."""
    now = datetime.now(timezone.utc)
    if now.weekday() != 0:  # 0 = Monday
        logging.info("Not Monday (weekday=%d). Exiting.", now.weekday())
        return True
    scheduled_minutes = 13 * 60 + 30
    current_minutes   = now.hour * 60 + now.minute
    diff = abs(current_minutes - scheduled_minutes)
    if diff > 60:
        logging.info("Stale — running more than 60 min from 13:30 UTC. Exiting.")
        return True
    return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.info("Crypto exit script starting...")

    if is_stale():
        return

    # Find open BTC position
    try:
        positions = trading_client.get_all_positions()
        btc_pos   = next((p for p in positions if p.symbol == SYMBOL), None)
    except Exception as e:
        logging.error("Error fetching positions: %s", e)
        return

    if btc_pos is None:
        logging.info("No BTC/USD position found — stop already hit or no entry was placed.")
        send_telegram("🌅 *BTC EXIT*\n\nNo position to close (stop already triggered or no entry).")
        return

    qty         = float(btc_pos.qty)
    avg_entry   = float(btc_pos.avg_entry_price)
    current_val = float(btc_pos.market_value)
    cost_basis  = float(btc_pos.cost_basis)
    unrealized  = float(btc_pos.unrealized_pl)
    unrealized_pct = float(btc_pos.unrealized_plpc) * 100

    logging.info("Closing BTC/USD: qty=%.6f, entry=%.0f, current_val=%.2f, P&L=%.2f",
                 qty, avg_entry, current_val, unrealized)

    try:
        req   = MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            client_order_id="crypto-exit-btc-{}".format(
                datetime.now(timezone.utc).strftime("%Y%m%d")
            ),
        )
        order = trading_client.submit_order(req)
    except Exception as e:
        logging.error("Error placing exit order: %s", e)
        send_telegram("⚠️ *BTC exit order FAILED*\n`{}`".format(e))
        return

    emoji  = "✅" if unrealized >= 0 else "🔴"
    pnl_sign = "+" if unrealized >= 0 else ""
    msg = (
        "🌅 *BTC WEEKEND EXIT*\n\n"
        "{} *{}BTC/USD*\n"
        "Entry: ${:.0f}\n"
        "P&L: {}${:.2f} ({}{:.1f}%)\n"
        "Qty: {:.6f} BTC\n\n"
        "✅ Exit order #{}"
    ).format(
        emoji,
        "+" if unrealized >= 0 else "-",
        avg_entry,
        pnl_sign, abs(unrealized),
        pnl_sign, abs(unrealized_pct),
        qty,
        str(order.id)[:8],
    )
    send_telegram(msg)
    logging.info("Exit order placed: %s", order.id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "import ast; ast.parse(open('crypto_exit.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add dashboard/crypto_exit.py
git commit -m "feat: add BTC weekend pullback exit script"
```

---

## Task 4: Write launchd plists

**Files:**
- Create: `com.trading.crypto-entry.plist`
- Create: `com.trading.crypto-exit.plist`

**Context:** launchd supports `Weekday` key in `StartCalendarInterval` (0=Sunday, 1=Monday). The existing equity plists don't use this — they fire daily and rely on API calendar checks. The crypto plists use Weekday so they only fire on the correct day, no market calendar needed.

- [ ] **Step 1: Create com.trading.crypto-entry.plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading.crypto-entry</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/ashleighchua/trading analyses/dashboard/crypto_scanner.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/ashleighchua/trading analyses/dashboard</string>

    <!-- Sunday 13:00 UTC = 8:00 PM Bangkok (UTC+7). Weekday 0 = Sunday. -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>13</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/ashleighchua/trading analyses/crypto_entry.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ashleighchua/trading analyses/crypto_entry_error.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
        <key>PYTHONPATH</key>
        <string>/Users/ashleighchua/trading analyses/dashboard</string>
    </dict>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
```

- [ ] **Step 2: Create com.trading.crypto-exit.plist**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading.crypto-exit</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/ashleighchua/trading analyses/dashboard/crypto_exit.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/ashleighchua/trading analyses/dashboard</string>

    <!-- Monday 13:30 UTC = 8:30 PM Bangkok (UTC+7). Weekday 1 = Monday. -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>1</integer>
        <key>Hour</key>
        <integer>13</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/ashleighchua/trading analyses/crypto_exit.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/ashleighchua/trading analyses/crypto_exit_error.log</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
        <key>PYTHONPATH</key>
        <string>/Users/ashleighchua/trading analyses/dashboard</string>
    </dict>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
```

- [ ] **Step 3: Validate plist XML syntax**

```bash
cd "/Users/ashleighchua/trading analyses"
plutil -lint com.trading.crypto-entry.plist && echo "entry: OK"
plutil -lint com.trading.crypto-exit.plist && echo "exit: OK"
```

Expected:
```
com.trading.crypto-entry.plist: OK
entry: OK
com.trading.crypto-exit.plist: OK
exit: OK
```

- [ ] **Step 4: Load both plists into launchd**

```bash
launchctl load ~/Library/LaunchAgents/com.trading.crypto-entry.plist 2>/dev/null || \
  launchctl load "/Users/ashleighchua/trading analyses/com.trading.crypto-entry.plist"

launchctl load ~/Library/LaunchAgents/com.trading.crypto-exit.plist 2>/dev/null || \
  launchctl load "/Users/ashleighchua/trading analyses/com.trading.crypto-exit.plist"
```

Note: plists must be in `~/Library/LaunchAgents/` to be loaded by launchctl as a user agent. If you keep them in the project directory (like the other plists), copy them first:

```bash
cp "/Users/ashleighchua/trading analyses/com.trading.crypto-entry.plist" ~/Library/LaunchAgents/
cp "/Users/ashleighchua/trading analyses/com.trading.crypto-exit.plist" ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.trading.crypto-entry.plist
launchctl load ~/Library/LaunchAgents/com.trading.crypto-exit.plist
```

Verify loaded:
```bash
launchctl list | grep crypto
```

Expected: two lines containing `com.trading.crypto-entry` and `com.trading.crypto-exit`.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add com.trading.crypto-entry.plist com.trading.crypto-exit.plist
git commit -m "feat: add launchd plists for BTC weekend entry/exit"
```

---

## Deployment Checklist (after all tasks)

- [ ] Backtest passed acceptance bar (≥55% WR, PF ≥1.5, ≥30 trades)
- [ ] `PULLBACK_THRESHOLD` and `TRAIL_PERCENT` in `crypto_scanner.py` updated with backtest results
- [ ] Both plists loaded in launchd (`launchctl list | grep crypto`)
- [ ] Paper trade 2 Sunday/Monday cycles before going live
- [ ] Monitor `crypto_entry.log` and `crypto_exit.log` after first run
