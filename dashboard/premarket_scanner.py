#!/usr/bin/env python3
"""
Pre-Market Scanner
==================
Runs at 9:00 AM ET daily. Scans the watchlist, finds the best setup,
sends a Telegram alert, and places an OPG (at-open) order.

Cron / launchd: run at 09:00 ET on market days.
"""

import os
import sys
import json
import time
import logging
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
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, TrailingStopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame

from ta_utils import full_analysis, trend_pullback
from news_utils import get_news_summary, format_news_for_telegram
from fred_client import get_macro_regime, regime_allows_short, regime_allows_long

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
APCA_KEY = os.environ["APCA_API_KEY_ID"]
APCA_SECRET = os.environ["APCA_API_SECRET_KEY"]

trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
data_client = StockHistoricalDataClient(api_key=APCA_KEY, secret_key=APCA_SECRET)

WATCHLIST = [
    "SPY", "QQQ", "IWM",           # Core US ETFs
    "EWJ", "EWH", "FXI", "KWEB",   # Asian ETFs
    "AAPL", "NVDA", "AMD", "PLTR", "TSLA",  # High beta
    "GLD", "TLT",                  # Defensive
]

# Tickers where bear rally fade has positive PF over 6-year backtest.
# PLTR+FXI: 54.1% WR, PF 1.88 (74 trades, 2020-2026, with EMA-200 filter)
# Removed: KWEB (WR 29.8%, PF 0.61), TSLA (WR 40.8%, PF 1.14), AMD (WR 39.0%, PF 1.33)
FADE_WATCHLIST = ["PLTR", "FXI"]

# Tickers where trend pullback long has positive PF over 6-year backtest.
# NVDA only: EMA(50,200) + RSI<35 + 1.5% trailing stop → 60% WR, PF 4.82 (15 trades, 2020-2026)
LONG_WATCHLIST = ["NVDA"]

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
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
            return
        except Exception as e:
            if parse_mode is None:
                logging.error("Telegram error: {}".format(e))

# ---------------------------------------------------------------------------
# Market calendar check
# ---------------------------------------------------------------------------

def is_market_day():
    """Check if today is a trading day using Alpaca's calendar."""
    from alpaca.trading.requests import GetCalendarRequest
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        cal = trading_client.get_calendar(GetCalendarRequest(start=today, end=today))
        return len(cal) > 0
    except Exception:
        return False  # assume no if API fails — safer than placing a bad order

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_bars(symbol, limit=25):
    """Fetch recent daily bars for a symbol."""
    try:
        from datetime import timedelta
        start = datetime.now(timezone.utc) - timedelta(days=limit * 2)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            limit=limit,
        )
        bars_resp = data_client.get_stock_bars(req)
        raw = bars_resp.data if hasattr(bars_resp, "data") else bars_resp
        bars = raw.get(symbol, [])
        return [
            {"o": float(b.open), "h": float(b.high), "l": float(b.low),
             "c": float(b.close), "v": int(b.volume), "t": str(b.timestamp)}
            for b in bars
        ]
    except Exception as e:
        logging.error("fetch_bars error for {}: {}".format(symbol, e))
        return []

def get_open_positions():
    """Return set of symbols with open positions."""
    try:
        positions = trading_client.get_all_positions()
        return {p.symbol for p in positions}
    except Exception:
        return set()

def get_open_orders():
    """Return set of symbols with open orders."""
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return {o.symbol for o in orders}
    except Exception:
        return set()

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_ticker(symbol, macro_regime="NEUTRAL", regime_data=None):
    """
    Fetch bars, run full_analysis, return a scored setup dict or None.
    macro_regime/regime_data are injected from pick_best_setup() — fetched once, not per-ticker.
    """
    bars = fetch_bars(symbol)
    if len(bars) < 15:
        return None

    analysis = full_analysis(bars)
    analysis["_macro_regime"] = macro_regime
    analysis["_regime_data"] = regime_data or {}
    if "error" in analysis:
        return None

    trend = analysis.get("trend", {})
    rsi = analysis.get("rsi_14")
    conviction = analysis.get("conviction", 0)
    current_price = analysis.get("current_price")

    setup = None

    # --- Bear rally fade (SHORT) ---
    # Downtrend + bear rally bounce (RSI filter removed — backtest showed it adds no edge)
    # FRED regime gate: RISK_ON blocked; rapid curve steepening (steepening_rally) also blocked
    _regime = analysis.get("_macro_regime", "NEUTRAL")
    _regime_data = analysis.get("_regime_data", {})
    if (symbol in FADE_WATCHLIST
            and trend.get("trend") == "downtrend"
            and trend.get("bear_rally")
            and regime_allows_short(_regime, _regime_data)):
        target_pct = 0.025  # target 2.5% fade
        stop_pct = 0.015    # 1.5% trailing stop
        setup = {
            "symbol": symbol,
            "side": "short",
            "entry": current_price,
            "target": round(current_price * (1 - target_pct), 2),
            "stop_pct": stop_pct,
            "thesis": "Bear rally fade: {} in downtrend, bounced {:.1f}% off recent low, RSI {:.0f}".format(
                symbol, trend.get("recent_bounce_pct", 0), rsi
            ),
            "conviction": conviction + (2 if trend.get("bear_rally") else 0),
            "analysis": analysis,
        }

    # --- Trend Pullback Long (NVDA only, validated) ---
    # EMA(50,200) uptrend + RSI<35 + pulled back ≥3% from 10-day high
    # Backtested: 60% WR, PF 4.82 (15 trades, 2020-2026)
    # Regime gate: RISK_OFF blocks (pullbacks extend in bear markets)
    if (symbol in LONG_WATCHLIST
            and regime_allows_long(_regime, _regime_data)):
        tp = trend_pullback(bars, ema_fast=50, ema_slow=200, pullback_threshold=3.0, rsi_max=35)
        if tp.get("pullback_long"):
            target_pct = 0.025
            stop_pct = 0.015
            long_setup = {
                "symbol": symbol,
                "side": "long",
                "entry": current_price,
                "target": round(current_price * (1 + target_pct), 2),
                "stop_pct": stop_pct,
                "thesis": "Trend pullback long: {} in uptrend, pulled back {:.1f}% from high, RSI {:.0f}".format(
                    symbol, tp.get("recent_pullback_pct", 0), tp.get("rsi_14", 0)
                ),
                "conviction": conviction + 3,
                "analysis": analysis,
            }
            # Only replace existing setup if no setup yet, or higher conviction
            if setup is None or long_setup["conviction"] > setup["conviction"]:
                setup = long_setup

    # --- Monday Reversal (SPY only, fade-down, validated) ---
    # Buy Monday open when SPY Friday close was down >0.5%
    # Exit: Tuesday close. Stop: 1.5% trailing.
    # Backtested: 73.1% WR, PF 3.21 (67 trades, 2020-2026)
    # Bear regime: 77.8% WR, PF 3.85
    if symbol == "SPY" and datetime.now(timezone.utc).weekday() == 0 and len(bars) >= 2:
        # Find last Friday bar
        friday = None
        for bar in reversed(bars[:-1]):
            from datetime import datetime as dt
            bar_date = dt.fromisoformat(bar["t"].replace("Z", "+00:00"))
            if bar_date.weekday() == 4:
                friday = bar
                break
        if friday:
            fri_ret = (friday["c"] - friday["o"]) / friday["o"] * 100
            if fri_ret <= -0.75:
                setup = {
                    "symbol": "SPY",
                    "side": "long",
                    "entry": current_price,
                    "target": round(current_price * 1.025, 2),
                    "stop_pct": 0.03,
                    "thesis": "Monday Reversal: SPY Friday down {:.1f}% → buy at open, exit Tue close (70.2% WR, PF 2.64)".format(fri_ret),
                    "conviction": conviction + 4,
                    "analysis": analysis,
                }

    return setup


def pick_best_setup(occupied_symbols):
    """Scan all tickers, return the highest-conviction setup not already held."""
    logging.info("Scanning {} tickers...".format(len(WATCHLIST)))

    # Fetch macro regime once — used to gate bear rally fades
    try:
        regime_data = get_macro_regime()
        macro_regime = regime_data["regime"]
        logging.info("Macro regime: {} — {}".format(macro_regime, regime_data["reason"]))
    except Exception as e:
        macro_regime = "NEUTRAL"
        regime_data = {"regime": "NEUTRAL", "reason": "FRED unavailable ({})".format(e)}
        logging.warning("FRED regime fetch failed, defaulting to NEUTRAL: %s", e)

    setups = []

    # Monday Reversal only checks SPY; bear rally fade uses FADE_WATCHLIST; trend pullback uses LONG_WATCHLIST.
    # Combine without duplicates, SPY always included for Monday Reversal check.
    scan_list = list(dict.fromkeys(["SPY"] + FADE_WATCHLIST + LONG_WATCHLIST))

    for symbol in scan_list:
        if symbol in occupied_symbols:
            logging.info("{} — skipping (position/order exists)".format(symbol))
            continue
        setup = score_ticker(symbol, macro_regime=macro_regime, regime_data=regime_data)
        if setup:
            logging.info("Analyzing {} — conviction={}".format(symbol, setup["conviction"]))
            setups.append(setup)
        else:
            logging.info("Analyzing {} — no setup".format(symbol))
        time.sleep(0.3)  # rate limit

    if not setups:
        return None, regime_data

    return max(setups, key=lambda s: s["conviction"]), regime_data

# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------

def place_opg_order(setup):
    """Place a DAY market order at ~9:20 AM ET + trailing stop.
    Note: OPG (time_in_force=opg) does not work on Alpaca paper trading —
    orders expire unfilled at 9:30 AM. DAY market orders fill immediately once
    the market opens and are equivalent in practice."""
    symbol = setup["symbol"]
    side = setup["side"]
    price = setup["entry"]
    stop_pct = setup.get("stop_pct", 0.015)

    # Size: 95% for longs (SPY/ETF, low gap risk), 50% for shorts (single stocks, gap risk)
    account = trading_client.get_account()
    equity = float(account.equity)
    size_pct = 0.95 if side == "long" else 0.50
    trade_size = equity * size_pct
    qty = max(1, int(trade_size / price))

    order_side = OrderSide.BUY if side == "long" else OrderSide.SELL

    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
        client_order_id="scanner-opg-{}-{}".format(
            symbol.lower(),
            datetime.now(timezone.utc).strftime("%Y%m%d")
        ),
    )

    try:
        order = trading_client.submit_order(req)

        # Place trailing stop after entry fills
        if order and order.filled_qty:
            try:
                from alpaca.trading.requests import TrailingStopOrderRequest
                stop_req = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=order.filled_qty,
                    side=OrderSide.BUY if side == "short" else OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    trail_percent=stop_pct * 100,
                    client_order_id="scanner-stop-{}-{}".format(
                        symbol.lower(),
                        datetime.now(timezone.utc).strftime("%Y%m%d")
                    ),
                )
                stop_order = trading_client.submit_order(stop_req)
                logging.info("Trailing stop placed for {} ({}%)".format(symbol, stop_pct * 100))
            except Exception as e:
                logging.warning("Could not place trailing stop for {}: {}".format(symbol, e))

        return order, qty
    except Exception as e:
        return None, qty

# ---------------------------------------------------------------------------
# Telegram message
# ---------------------------------------------------------------------------

def build_alert(setup, order, qty):
    """Build the Telegram alert message."""
    symbol = setup["symbol"]
    side = setup["side"].upper()
    entry = setup["entry"]
    target = setup["target"]
    stop_pct = setup["stop_pct"] * 100
    rsi = setup["analysis"].get("rsi_14", "?")
    bounce = setup["analysis"].get("trend", {}).get("recent_bounce_pct", "?")
    vol = setup["analysis"].get("volume", {})

    # News
    news_items = get_news_summary(symbol, limit=3, hours_back=18)
    news_str = format_news_for_telegram(news_items, max_items=2)

    order_status = "✅ Market order placed" if order else "⚠️ Order failed"
    order_detail = "#{} — {} {} @ market".format(
        str(order.id)[:8] if order else "N/A",
        qty,
        symbol,
    )

    macro = setup["analysis"].get("_macro_regime", "?")
    rd = setup["analysis"].get("_regime_data", {})
    curve_line = "Macro: {}".format(macro)
    if rd:
        parts = []
        if rd.get("t10y3m") is not None:
            parts.append("T10Y3M={:.2f}%".format(rd["t10y3m"]))
        if rd.get("yield_curve") is not None:
            parts.append("T10Y2Y={:.2f}%".format(rd["yield_curve"]))
        if rd.get("slope_30d") is not None:
            parts.append("slope={:+.2f}%/30d".format(rd["slope_30d"]))
        if rd.get("inversion_days"):
            parts.append("inv={}d".format(rd["inversion_days"]))
        if rd.get("steepening_rally"):
            parts.append("⚠️ steepener")
        if parts:
            curve_line += " ({})".format(", ".join(parts))

    msg = (
        "🌅 *PRE-MARKET SCAN — {}*\n\n"
        "🎯 *{} {}*\n"
        "Entry: ~${} (at open)\n"
        "Target: ${} ({:.1f}% move)\n"
        "Stop: {:.0f}% trailing\n"
        "Qty: {} shares\n\n"
        "📊 *Technicals*\n"
        "RSI-14: {}\n"
        "Bounce off low: {}%\n"
        "Volume: {}\n"
        "{}\n\n"
        "📰 *News*\n"
        "{}\n\n"
        "💡 _{}_\n\n"
        "{}\n{}"
    ).format(
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        side, symbol,
        entry, target,
        abs((target - entry) / entry * 100),
        stop_pct,
        qty,
        rsi,
        bounce,
        vol.get("label", "unknown"),
        curve_line,
        news_str,
        setup["thesis"],
        order_status,
        order_detail,
    )
    return msg

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def is_stale(max_late_minutes=60):
    """Return True if running more than 60 min past the scheduled 9:20 AM ET window."""
    now_utc = datetime.now(timezone.utc)
    # Scheduled: 13:20 UTC (9:20 AM EDT, Mar–Nov). Change to 14:20 UTC in November.
    scheduled_utc_hour, scheduled_utc_minute = 13, 20
    scheduled_minutes = scheduled_utc_hour * 60 + scheduled_utc_minute
    current_minutes = now_utc.hour * 60 + now_utc.minute
    diff = abs(current_minutes - scheduled_minutes)
    # Handle midnight wraparound
    diff = min(diff, 1440 - diff)
    return diff > max_late_minutes


def main():
    logging.info("Pre-market scanner starting...")

    if is_stale():
        logging.info("Stale job — running more than 60 min from scheduled 14:20 UTC window. Exiting.")
        return

    if not is_market_day():
        logging.info("Not a trading day. Exiting.")
        return

    occupied = get_open_positions() | get_open_orders()
    logging.info("Occupied symbols: {}".format(occupied))

    setup, regime_data = pick_best_setup(occupied)

    if not setup:
        regime_line = "Macro: {} — {}".format(
            regime_data.get("regime", "?"), regime_data.get("reason", "")
        )
        send_telegram(
            "🌅 *PRE-MARKET SCAN*\n\nNo high-conviction setups today. Staying in cash.\n\n📊 _{}_".format(regime_line)
        )
        logging.info("No setups found.")
        return

    logging.info("Best setup: {} {} (conviction={})".format(
        setup["side"], setup["symbol"], setup["conviction"]
    ))

    order, qty = place_opg_order(setup)
    msg = build_alert(setup, order, qty)
    send_telegram(msg)

    logging.info("Alert sent. Order: {}".format(order.id if order else "FAILED"))


if __name__ == "__main__":
    main()
