#!/usr/bin/env python3
"""
Alpaca → Dashboard Trade Sync
==============================
Runs every 30 minutes. Keeps trades.db in sync with Alpaca:
  - Open positions: upserted with current entry price / qty
  - Closed positions: exit price + real P&L fetched from filled orders
  - Signal type preserved from scanner client_order_id where possible

launchd: com.trading.sync-trades (every 30 min)
"""

import os
import sys
import uuid
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

APCA_KEY = os.environ["APCA_API_KEY_ID"]
APCA_SECRET = os.environ["APCA_API_SECRET_KEY"]
DB_PATH = Path(__file__).parent / "trades.db"

client = TradingClient(APCA_KEY, APCA_SECRET, paper=True)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Signal type from client_order_id
# ---------------------------------------------------------------------------

def infer_signal_type(client_order_id, side):
    """Guess signal type from the order ID the scanner set."""
    cid = client_order_id or ""
    if "scanner" in cid or "opg" in cid:
        return "Bear Rally Fade" if side == "Short" else "Scanner Long"
    if "trail" in cid or "cover" in cid:
        return "Exit"
    return "Manual"


# ---------------------------------------------------------------------------
# Fetch closed orders to find real exit prices
# ---------------------------------------------------------------------------

def get_recent_filled_orders(days_back=30):
    """
    Returns a dict: symbol → list of filled orders (newest first).
    Includes both entry and exit fills.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    try:
        orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            after=since,
            limit=200,
        ))
    except Exception as e:
        logging.error("Failed to fetch orders: {}".format(e))
        return {}

    by_symbol = {}
    for o in orders:
        status = str(o.status).lower()
        if "filled" not in status:
            continue
        sym = o.symbol.replace("/", "")
        by_symbol.setdefault(sym, []).append(o)

    return by_symbol


def get_recent_fill_activities(days_back=30):
    """
    Returns a dict: symbol → list of fill activities (price, side, date).
    Used as a fallback when order intent matching fails (e.g. OCO orders).
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        activities = client.get_activities(activity_types="FILL", after=since)
    except Exception:
        try:
            # Fallback: use requests directly via trading client
            activities = client.get_activities(activity_type="FILL", after=since)
        except Exception as e:
            logging.warning("Could not fetch fill activities: {}".format(e))
            return {}

    by_symbol = {}
    for a in activities:
        sym = getattr(a, "symbol", "").replace("/", "")
        if not sym:
            continue
        by_symbol.setdefault(sym, []).append(a)
    return by_symbol


def find_exit_fill(symbol, entry_side, entry_date, filled_orders_by_symbol, fill_activities=None):
    """
    Given an entry (side + date), find the corresponding closing fill.
    Entry side "Short" closes with a "buy", "Long" closes with a "sell".
    First tries order objects; falls back to raw fill activities (handles OCO/bracket orders).
    Returns (exit_price, exit_date) or (None, None).
    """
    close_side = "buy" if entry_side == "Short" else "sell"

    # ── Primary: match via closed orders ──────────────────────────────────
    orders = filled_orders_by_symbol.get(symbol, [])
    for o in orders:
        o_side = str(o.side).lower()
        o_intent = str(getattr(o, "position_intent", "")).lower()

        is_closing = (
            close_side in o_side
            and ("close" in o_intent or "cover" in o_intent or "to_close" in o_intent)
        )
        if not is_closing:
            continue

        filled_at = o.filled_at
        if filled_at and str(filled_at.date()) >= entry_date:
            price = float(o.filled_avg_price) if o.filled_avg_price else None
            if price:
                return price, str(filled_at.date())

    # ── Fallback: match via raw fill activities (works for OCO/bracket) ───
    if fill_activities:
        activities = fill_activities.get(symbol, [])
        for a in activities:
            a_side = str(getattr(a, "side", "")).lower()
            if close_side not in a_side:
                continue
            txn_time = getattr(a, "transaction_time", None)
            if not txn_time:
                continue
            txn_date = str(txn_time)[:10]
            if txn_date >= entry_date:
                price = float(getattr(a, "price", 0) or 0)
                if price:
                    logging.info("Exit fill found via activity feed for {} @ ${}".format(symbol, price))
                    return price, txn_date

    return None, None


# ---------------------------------------------------------------------------
# Main sync
# ---------------------------------------------------------------------------

def sync():
    logging.info("Starting trade sync...")

    conn = get_db()
    filled_orders = get_recent_filled_orders(days_back=60)
    fill_activities = get_recent_fill_activities(days_back=60)

    # ── 1. Get current open positions from Alpaca ──────────────────────────
    try:
        positions = client.get_all_positions()
    except Exception as e:
        logging.error("Failed to fetch positions: {}".format(e))
        conn.close()
        return

    open_tickers = {}
    for p in positions:
        sym = p.symbol.replace("/", "")
        open_tickers[sym] = p

    # ── 2. Get existing DB trades ─────────────────────────────────────────
    existing_open = {}   # ticker → row (open trades in DB)
    for row in conn.execute(
        "SELECT * FROM trades WHERE exit_price IS NULL AND signal_type != 'Exit'"
    ).fetchall():
        existing_open[row["ticker"]] = dict(row)

    # ── 3. Upsert open positions ──────────────────────────────────────────
    for sym, p in open_tickers.items():
        qty = abs(float(p.qty))
        entry = float(p.avg_entry_price)
        side = "Long" if float(p.qty) > 0 else "Short"
        unreal_pl = float(p.unrealized_pl)

        if sym in existing_open:
            conn.execute(
                "UPDATE trades SET entry_price=?, shares=?, direction=?, pnl=? WHERE id=?",
                (entry, qty, side, unreal_pl, existing_open[sym]["id"])
            )
            logging.info("Updated open: {} {} @ ${:.2f}".format(side, sym, entry))
        else:
            # New position — find matching entry order for signal type
            signal_type = "Alpaca Sync"
            entry_orders = [
                o for o in filled_orders.get(sym, [])
                if ("buy" in str(o.side).lower() and side == "Long")
                or ("sell" in str(o.side).lower() and side == "Short")
            ]
            if entry_orders:
                signal_type = infer_signal_type(
                    str(entry_orders[0].client_order_id or ""), side
                )

            trade_id = uuid.uuid4().hex[:12]
            date_str = datetime.now().strftime("%Y-%m-%d")
            conn.execute("""
                INSERT INTO trades
                    (id, date, ticker, direction, signal_type,
                     entry_price, shares, pnl, notes, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, date_str, sym, side, signal_type,
                entry, qty, unreal_pl,
                "Auto-synced from Alpaca", 3,
            ))
            logging.info("Inserted open: {} {} @ ${:.2f}".format(side, sym, entry))

    # ── 4. Close trades no longer in Alpaca positions ─────────────────────
    for sym, row in existing_open.items():
        if sym in open_tickers:
            continue  # still open

        # Position closed — find real exit price from orders
        exit_price, exit_date = find_exit_fill(
            sym, row["direction"], row["date"], filled_orders, fill_activities
        )

        if exit_price is None:
            # Fallback: use last known entry as exit (better than 0)
            logging.warning("No exit fill found for {} — using entry as fallback".format(sym))
            exit_price = row["entry_price"]
            exit_date = datetime.now().strftime("%Y-%m-%d")

        qty = row["shares"] or 1
        entry_price = row["entry_price"] or exit_price

        if row["direction"] == "Short":
            pnl = (entry_price - exit_price) * qty
        else:
            pnl = (exit_price - entry_price) * qty

        return_pct = (pnl / (entry_price * qty)) * 100 if entry_price else 0

        conn.execute("""
            UPDATE trades
            SET exit_price=?, pnl=?, return_pct=?, exit_type=?, notes=?
            WHERE id=?
        """, (
            exit_price,
            round(pnl, 2),
            round(return_pct, 2),
            "Auto-closed",
            "Closed position synced from Alpaca (exit @ ${:.2f})".format(exit_price),
            row["id"],
        ))
        logging.info("Closed: {} {} P&L=${:.2f} ({:.1f}%)".format(
            row["direction"], sym, pnl, return_pct
        ))

    conn.commit()
    conn.close()
    logging.info("Sync complete.")


if __name__ == "__main__":
    sync()
