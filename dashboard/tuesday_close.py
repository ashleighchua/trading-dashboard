#!/usr/bin/env python3
"""
Tuesday Close — Monday Reversal Exit
======================================
Runs at 15:55 ET every Tuesday. Closes any open SPY long position
by placing a market-on-close (CLS) order for the closing auction.

This is the exit leg of the Monday Reversal strategy:
  Buy Monday open → Sell Tuesday close (73.1% WR, PF 3.21)
"""

import os
import sys
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
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
APCA_KEY = os.environ["APCA_API_KEY_ID"]
APCA_SECRET = os.environ["APCA_API_SECRET_KEY"]

trading_client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)


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


def main():
    logging.info("Tuesday close starting...")

    # Only run on Tuesdays
    today_dow = datetime.now(timezone.utc).weekday()
    if today_dow != 1:
        logging.info("Not Tuesday (weekday={}). Exiting.".format(today_dow))
        return

    # Check for open SPY long position
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        logging.error("Error fetching positions: {}".format(e))
        send_telegram("⚠️ *Tuesday Close Failed*\nCould not fetch positions: `{}`".format(e))
        return

    spy_position = next(
        (p for p in positions if p.symbol == "SPY" and float(p.qty) > 0),
        None
    )

    if not spy_position:
        logging.info("No open SPY long position found. Nothing to close.")
        return

    qty = float(spy_position.qty)
    entry_price = float(spy_position.avg_entry_price)
    current_price = float(spy_position.current_price)
    unrealized_pnl = float(spy_position.unrealized_pl)

    logging.info("Found SPY long: {} shares @ ${:.2f}, current ${:.2f}, P&L ${:.2f}".format(
        qty, entry_price, current_price, unrealized_pnl
    ))

    # Cancel any open trailing stops on SPY first
    try:
        open_orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
        spy_stops = [
            o for o in open_orders
            if o.symbol == "SPY" and str(o.order_type) in ("trailing_stop", "OrderType.trailing_stop")
        ]
        for stop in spy_stops:
            trading_client.cancel_order_by_id(stop.id)
            logging.info("Cancelled trailing stop {}".format(stop.id))
        if spy_stops:
            time.sleep(2)  # wait for Alpaca to release held qty before placing sell
    except Exception as e:
        logging.warning("Could not cancel trailing stops: {}".format(e))

    # Place market-on-close sell order
    try:
        req = MarketOrderRequest(
            symbol="SPY",
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.CLS,
            client_order_id="monday-reversal-close-{}".format(
                datetime.now(timezone.utc).strftime("%Y%m%d")
            ),
        )
        order = trading_client.submit_order(req)
        logging.info("CLS sell order placed: {}".format(order.id))
    except Exception as e:
        logging.error("Error placing close order: {}".format(e))
        send_telegram("⚠️ *Tuesday Close Failed*\nCould not place sell order: `{}`".format(e))
        return

    pnl_emoji = "✅" if unrealized_pnl >= 0 else "🔴"
    send_telegram(
        "🔔 *MONDAY REVERSAL — EXIT*\n\n"
        "Closing SPY at Tuesday close (CLS order)\n"
        "Entry: ${:.2f} | Now: ${:.2f}\n"
        "{} Unrealized P&L: ${:.2f}\n\n"
        "_Final P&L will update after close fills._".format(
            entry_price, current_price, pnl_emoji, unrealized_pnl
        )
    )


if __name__ == "__main__":
    main()
