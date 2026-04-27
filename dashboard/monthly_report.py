#!/usr/bin/env python3
"""
Monthly Strategy Performance Report
=====================================
Runs on the 1st of every month at 9am Bangkok.
Queries trades.db, sends Telegram summary, writes monthly_stats.json.

cron: 0 9 1 * *
"""

import os
import json
import sqlite3
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
DB_PATH = Path(__file__).parent / "trades.db"
STATS_PATH = Path(__file__).parent / "monthly_stats.json"

STRATEGIES = ["Bear Rally Fade", "GLD Pullback", "Monday Reversal"]

HEALTH_THRESHOLDS = {
    "Bear Rally Fade": 45.0,
    "GLD Pullback":    55.0,
    "Monday Reversal": 60.0,
}


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
                print("Telegram error: {}".format(e))


def compute_stats(trades):
    """Compute WR, PF, total P&L from a list of trade dicts."""
    if not trades:
        return {"trades": 0, "wr": None, "pf": None, "pnl": 0.0}
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    wr  = len(wins) / len(trades) * 100
    pf  = (gross_win / gross_loss) if gross_loss > 0 else None  # None = infinite PF
    pnl = sum(t["pnl"] for t in trades)
    return {
        "trades": len(trades),
        "wr":     round(wr, 1),
        "pf":     round(pf, 2) if pf is not None else None,
        "pnl":    round(pnl, 2),
    }


def health_check(strategy, last20_trades):
    """Return 'ok', 'warn', or 'skip' (insufficient data)."""
    real = [t for t in last20_trades if t["pnl"] != 0]
    if len(real) < 10:
        return "skip"
    wr = sum(1 for t in real if t["pnl"] > 0) / len(real) * 100
    threshold = HEALTH_THRESHOLDS.get(strategy, 50.0)
    return "ok" if wr >= threshold else "warn"


def main():
    now = datetime.now(timezone.utc)
    first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_end   = first_of_this_month - timedelta(seconds=1)
    last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_start_str  = last_month_start.strftime("%Y-%m-%d")
    month_end_str    = last_month_end.strftime("%Y-%m-%d")
    month_label      = last_month_start.strftime("%B %Y")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    strategy_results = {}
    health_flags = {}

    for strategy in STRATEGIES:
        monthly_rows = conn.execute("""
            SELECT pnl FROM trades
            WHERE exit_price IS NOT NULL
              AND pnl IS NOT NULL
              AND signal_type = ?
              AND date >= ?
              AND date <= ?
        """, (strategy, month_start_str, month_end_str)).fetchall()
        monthly_trades = [dict(r) for r in monthly_rows]

        last20_rows = conn.execute("""
            SELECT pnl FROM trades
            WHERE exit_price IS NOT NULL
              AND pnl IS NOT NULL
              AND signal_type = ?
            ORDER BY date DESC
            LIMIT 20
        """, (strategy,)).fetchall()
        last20_trades = [dict(r) for r in last20_rows]

        strategy_results[strategy] = compute_stats(monthly_trades)
        health_flags[strategy]     = health_check(strategy, last20_trades)

    conn.close()

    lines = ["📊 *{} — Monthly Strategy Report*".format(month_label), ""]

    total_pnl    = 0.0
    total_trades = 0

    for strategy in STRATEGIES:
        s  = strategy_results[strategy]
        total_pnl    += s["pnl"]
        total_trades += s["trades"]

        if s["trades"] == 0:
            lines.append("*{}*".format(strategy))
            lines.append("  No trades this month")
        else:
            pf_str   = "{:.2f}".format(s["pf"]) if s["pf"] is not None else "inf"
            pnl_icon = "🟢" if s["pnl"] >= 0 else "🔴"
            lines.append("*{}*".format(strategy))
            lines.append("  Trades: {} | WR: {:.0f}% | PF: {} | P&L: {} ${:+,.0f}".format(
                s["trades"], s["wr"], pf_str, pnl_icon, s["pnl"]
            ))
        lines.append("")

    pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
    lines.append("*Total: {} trades | Net P&L: {} ${:+,.0f}*".format(
        total_trades, pnl_icon, total_pnl
    ))
    lines.append("")

    for strategy in STRATEGIES:
        hf        = health_flags[strategy]
        threshold = HEALTH_THRESHOLDS[strategy]
        if hf == "warn":
            lines.append("⚠️ *{}*: WR below {:.0f}% threshold over last 20 trades — consider pausing".format(
                strategy, threshold
            ))
        elif hf == "ok":
            lines.append("✅ *{}*: WR on track".format(strategy))
        else:
            lines.append("— *{}*: insufficient data for health check".format(strategy))

    send_telegram("\n".join(lines))

    stats_payload = {
        "month":      month_label,
        "generated":  now.strftime("%Y-%m-%d %H:%M UTC"),
        "strategies": {},
    }
    for strategy in STRATEGIES:
        s  = strategy_results[strategy]
        hf = health_flags[strategy]
        stats_payload["strategies"][strategy] = {
            "trades":  s["trades"],
            "wr":      s["wr"],
            "pf":      s["pf"],
            "pnl":     s["pnl"],
            "health":  hf,
        }

    with open(STATS_PATH, "w") as f:
        json.dump(stats_payload, f, indent=2)

    print("Monthly report sent for {}. Total trades: {}, Net P&L: ${:+,.2f}".format(
        month_label, total_trades, total_pnl
    ))


if __name__ == "__main__":
    main()
