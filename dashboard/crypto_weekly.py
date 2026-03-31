#!/usr/bin/env python3
"""
BTC 200-Day MA Trend Following
================================
Runs every Monday at 20:20 Bangkok (= 13:20 UTC = 9:20 AM EDT).
Buys BTC/USD when price > 200-day SMA, sells when below.

Backtest: 11 trades, 45.5% WR, PF 9.17, $1k -> $8,073 (2018-2024)
Position size: 20% of account equity.
"""

import os
import sys
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
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]
APCA_KEY  = os.environ["APCA_API_KEY_ID"]
APCA_SECRET = os.environ["APCA_API_SECRET_KEY"]

SYMBOL          = "BTC/USD"   # used for data API and order placement
POSITION_SYMBOL = "BTCUSD"    # Alpaca strips the slash in position responses
SMA_PERIOD  = 200
BARS_NEEDED = SMA_PERIOD + 1      # +1 so we have current price separate from SMA window
EQUITY_PCT  = 0.20

trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)
crypto_client  = CryptoHistoricalDataClient(APCA_KEY, APCA_SECRET)


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


def calculate_sma(closes, period=200):
    """Return SMA of last `period` values, or None if insufficient data."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def get_btc_position():
    """Return open BTC/USD position or None. Raises on API error."""
    positions = trading_client.get_all_positions()
    return next((p for p in positions if p.symbol == POSITION_SYMBOL), None)


def main():
    logging.info("BTC weekly MA check starting...")

    # --- Fetch price data ---
    try:
        bars_request = CryptoBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame.Day,
            limit=BARS_NEEDED,
        )
        bars_response = crypto_client.get_crypto_bars(bars_request)
        bars = bars_response[SYMBOL]
    except Exception as e:
        msg = "Could not fetch BTC bars: {}".format(e)
        logging.error(msg)
        send_telegram("⚠️ *BTC TREND ERROR*\n{}".format(msg))
        return

    if len(bars) < BARS_NEEDED:
        msg = "Only {} bars returned, need {}. Aborting.".format(len(bars), BARS_NEEDED)
        logging.warning(msg)
        send_telegram("⚠️ *BTC TREND ERROR*\n{}".format(msg))
        return

    closes = [bar.close for bar in bars]
    current_price = closes[-1]
    sma = calculate_sma(closes[:-1], period=SMA_PERIOD)   # SMA from prior 200 bars

    logging.info("BTC price: ${:.2f} | 200-day SMA: ${:.2f}".format(current_price, sma))

    signal_long = current_price > sma

    # --- Check existing position ---
    try:
        position = get_btc_position()
    except Exception as e:
        msg = "Could not fetch positions: {}".format(e)
        logging.error(msg)
        send_telegram("⚠️ *BTC TREND ERROR*\n{}".format(msg))
        return
    has_position = position is not None

    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    # --- Act ---
    if signal_long and not has_position:
        # BUY
        try:
            account = trading_client.get_account()
            equity = float(account.equity)
            qty = round((equity * EQUITY_PCT) / current_price, 6)

            # Check for pending BTC orders to avoid double-ordering
            open_orders = trading_client.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                symbols=[SYMBOL],
            ))
            if open_orders:
                logging.info("Open BTC order already exists ({}). Skipping buy.".format(open_orders[0].id))
                send_telegram(
                    "📊 *BTC TREND CHECK*\n\n"
                    "BTC: ${:.2f} | 200-day MA: ${:.2f}\n"
                    "Signal: LONG — pending order already open, skipping".format(current_price, sma)
                )
                return

            logging.info("Placing BUY order for {:.6f} BTC at market".format(qty))
            req = MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                client_order_id="btc-ma-buy-{}".format(today_str),
            )
            order = trading_client.submit_order(req)
            logging.info("BUY order placed: {} qty {}".format(order.id, qty))
            send_telegram(
                "🟢 *BTC TREND — ENTERED LONG*\n\n"
                "BTC: ${:.2f} | 200-day MA: ${:.2f}\n"
                "Bought {:.6f} BTC (~${:.0f})\n"
                "20% equity deployed".format(current_price, sma, qty, qty * current_price)
            )
        except Exception as e:
            logging.error("Error placing buy order: {}".format(e))
            send_telegram("⚠️ *BTC TREND ERROR*\nCould not place buy: `{}`".format(e))

    elif not signal_long and has_position:
        # SELL
        try:
            qty = float(position.qty)
            entry = float(position.avg_entry_price)
            pnl = float(position.unrealized_pl)

            logging.info("Placing SELL order for {:.6f} BTC at market".format(qty))
            req = MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id="btc-ma-sell-{}".format(today_str),
            )
            order = trading_client.submit_order(req)
            logging.info("SELL order placed: {} qty {}".format(order.id, qty))
            pnl_emoji = "✅" if pnl >= 0 else "🔴"
            send_telegram(
                "🔴 *BTC TREND — EXITED*\n\n"
                "BTC: ${:.2f} | 200-day MA: ${:.2f}\n"
                "Sold {:.6f} BTC\n"
                "Entry: ${:.2f}\n"
                "{} Unrealized P&L: ${:.2f}".format(
                    current_price, sma, qty, entry, pnl_emoji, pnl
                )
            )
        except Exception as e:
            logging.error("Error placing sell order: {}".format(e))
            send_telegram("⚠️ *BTC TREND ERROR*\nCould not place sell: `{}`".format(e))

    elif signal_long and has_position:
        # Already long, hold
        qty = float(position.qty)
        pnl = float(position.unrealized_pl)
        logging.info("Already long {:.6f} BTC. Holding.".format(qty))
        send_telegram(
            "📊 *BTC TREND CHECK*\n\n"
            "BTC: ${:.2f} | 200-day MA: ${:.2f}\n"
            "Signal: LONG — already positioned\n"
            "Holding {:.6f} BTC | P&L: ${:.2f}".format(current_price, sma, qty, pnl)
        )

    else:
        # Not long, no position, signal flat
        logging.info("Signal: FLAT. No position, staying out.")
        send_telegram(
            "📊 *BTC TREND CHECK*\n\n"
            "BTC: ${:.2f} | 200-day MA: ${:.2f}\n"
            "Signal: FLAT — no position, staying out".format(current_price, sma)
        )


if __name__ == "__main__":
    main()
