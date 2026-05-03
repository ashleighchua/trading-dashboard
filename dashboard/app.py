"""
Trading Journal Dashboard
==========================
Flask server with SQLite storage for logging and reviewing trades.
Run: python3 app.py → opens http://localhost:5050
"""

import json
import os
import sys
import csv
import sqlite3
import uuid
import logging
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# Load .env file if present (for Alpaca keys)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from data_provider import download, download_multi

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "trades.db"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
SIGNAL_LOG = BASE_DIR.parent / "signal_log.csv"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload



# ── Database ─────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL DEFAULT 'UPRO',
            direction TEXT NOT NULL DEFAULT 'Long',
            signal_type TEXT,
            entry_price REAL,
            stop_loss REAL,
            exit_price REAL,
            exit_type TEXT,
            shares REAL,
            pnl REAL,
            return_pct REAL,
            emotion TEXT,
            confidence INTEGER,
            followed_plan INTEGER,
            notes TEXT,
            screenshot_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/trades", methods=["GET"])
def get_trades():
    conn = get_db()
    trades = conn.execute("SELECT * FROM trades ORDER BY date DESC, created_at DESC").fetchall()
    conn.close()
    return jsonify([dict(t) for t in trades])


@app.route("/api/trades", methods=["POST"])
def add_trade():
    data = request.form.to_dict()

    # Handle screenshot upload
    screenshot_path = None
    if 'screenshot' in request.files:
        file = request.files['screenshot']
        if file.filename:
            ext = Path(file.filename).suffix or '.png'
            filename = f"{uuid.uuid4().hex[:12]}{ext}"
            file.save(str(UPLOAD_DIR / filename))
            screenshot_path = f"uploads/{filename}"

    # Calculate P&L if prices provided
    entry = float(data.get('entry_price') or 0)
    exit_p = float(data.get('exit_price') or 0)
    shares = float(data.get('shares') or 0)
    direction = data.get('direction', 'Long')

    if entry and exit_p and shares:
        if direction == 'Long':
            pnl = (exit_p - entry) * shares
            return_pct = (exit_p - entry) / entry * 100
        else:
            pnl = (entry - exit_p) * shares
            return_pct = (entry - exit_p) / entry * 100
    else:
        pnl = float(data.get('pnl') or 0)
        return_pct = float(data.get('return_pct') or 0)

    trade_id = uuid.uuid4().hex[:12]
    conn = get_db()
    conn.execute("""
        INSERT INTO trades (id, date, ticker, direction, signal_type,
            entry_price, stop_loss, exit_price, exit_type,
            shares, pnl, return_pct,
            emotion, confidence, followed_plan,
            notes, screenshot_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_id,
        data.get('date', datetime.now().strftime('%Y-%m-%d')),
        data.get('ticker', 'UPRO'),
        direction,
        data.get('signal_type', ''),
        entry or None,
        float(data.get('stop_loss') or 0) or None,
        exit_p or None,
        data.get('exit_type', ''),
        shares or None,
        pnl,
        return_pct,
        data.get('emotion', ''),
        int(data.get('confidence') or 3),
        1 if data.get('followed_plan') == 'yes' else 0,
        data.get('notes', ''),
        screenshot_path,
    ))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok", "id": trade_id})


@app.route("/api/trades/<trade_id>", methods=["DELETE"])
def delete_trade(trade_id):
    conn = get_db()
    conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "ok"})


@app.route("/api/monthly-stats")
def get_monthly_stats():
    stats_path = BASE_DIR / "monthly_stats.json"
    if not stats_path.exists():
        return jsonify({
            "month": None,
            "generated": None,
            "strategies": {
                "Bear Rally Fade": {"trades": 0, "wr": None, "pf": None, "pnl": 0.0, "health": "skip"},
                "GLD Pullback":    {"trades": 0, "wr": None, "pf": None, "pnl": 0.0, "health": "skip"},
                "Monday Reversal": {"trades": 0, "wr": None, "pf": None, "pnl": 0.0, "health": "skip"},
            }
        })
    with open(stats_path) as f:
        return jsonify(json.load(f))


@app.route("/api/stats")
def get_stats():
    conn = get_db()
    trades = conn.execute("SELECT * FROM trades ORDER BY date ASC").fetchall()
    conn.close()

    if not trades:
        return jsonify({
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "profit_factor": 0, "avg_win": 0, "avg_loss": 0,
            "best_trade": 0, "worst_trade": 0, "current_streak": 0,
            "by_signal": {}, "by_emotion": {}, "by_confidence": {},
            "equity_curve": [], "calendar": {},
        })

    trades = [dict(t) for t in trades]
    # Separate closed (have P&L) from open (no exit yet)
    open_trades = [t for t in trades if t.get('pnl') is None or t.get('exit_price') is None]
    trades = [t for t in trades if t.get('pnl') is not None and t.get('exit_price') is not None]

    if not trades:
        return jsonify({
            "total_trades": 0, "win_rate": 0, "total_pnl": 0,
            "profit_factor": 0, "avg_win": 0, "avg_loss": 0,
            "best_trade": 0, "worst_trade": 0, "current_streak": 0,
            "by_signal": {}, "by_emotion": {}, "by_confidence": {},
            "equity_curve": [], "calendar": {},
            "open_positions": len(open_trades),
            "open_trades": [{"ticker": t["ticker"], "direction": t["direction"],
                           "entry_price": t["entry_price"], "shares": t["shares"],
                           "date": t["date"]} for t in open_trades],
        })

    pnls = [t['pnl'] for t in trades]
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_profit = sum(t['pnl'] for t in wins)
    gross_loss = abs(sum(t['pnl'] for t in losses))

    # Current streak
    streak = 0
    for t in reversed(trades):
        if t['pnl'] > 0:
            streak += 1
        else:
            if streak == 0:
                streak = -1
                for t2 in reversed(trades):
                    if t2['pnl'] <= 0:
                        streak -= 1
                    else:
                        break
                streak += 1
            break

    # By signal type
    by_signal = {}
    for t in trades:
        sig = t.get('signal_type') or 'Unknown'
        if sig not in by_signal:
            by_signal[sig] = {'trades': 0, 'wins': 0, 'pnl': 0}
        by_signal[sig]['trades'] += 1
        by_signal[sig]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_signal[sig]['wins'] += 1

    # By emotion
    by_emotion = {}
    for t in trades:
        emo = t.get('emotion') or 'Unknown'
        if emo not in by_emotion:
            by_emotion[emo] = {'trades': 0, 'wins': 0, 'pnl': 0}
        by_emotion[emo]['trades'] += 1
        by_emotion[emo]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_emotion[emo]['wins'] += 1

    # By confidence
    by_confidence = {}
    for t in trades:
        conf = str(t.get('confidence') or '3')
        if conf not in by_confidence:
            by_confidence[conf] = {'trades': 0, 'wins': 0, 'pnl': 0}
        by_confidence[conf]['trades'] += 1
        by_confidence[conf]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_confidence[conf]['wins'] += 1

    # Equity curve
    running = 0
    equity = []
    for t in trades:
        running += t['pnl']
        equity.append({'date': t['date'], 'equity': round(running, 2)})

    # Calendar (daily P&L)
    calendar = {}
    for t in trades:
        d = t['date']
        if d not in calendar:
            calendar[d] = 0
        calendar[d] += t['pnl']

    return jsonify({
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(sum(pnls), 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "avg_win": round(sum(t['pnl'] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t['pnl'] for t in losses) / len(losses), 2) if losses else 0,
        "best_trade": round(max(pnls), 2),
        "worst_trade": round(min(pnls), 2),
        "current_streak": streak,
        "by_signal": by_signal,
        "by_emotion": by_emotion,
        "by_confidence": by_confidence,
        "equity_curve": equity,
        "calendar": calendar,
        "open_positions": len(open_trades),
        "open_trades": [{"ticker": t["ticker"], "direction": t["direction"],
                       "entry_price": t["entry_price"], "shares": t["shares"],
                       "date": t["date"]} for t in open_trades],
    })


@app.route("/api/signal")
def get_signal():
    """Read the latest signal from signal_log.csv."""
    if not SIGNAL_LOG.exists():
        return jsonify({"signal": None})
    try:
        with open(SIGNAL_LOG) as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) > 1:
                last = rows[-1]
                return jsonify({"signal": ",".join(last)})
    except Exception:
        pass
    return jsonify({"signal": None})


def _market_status():
    """Return market open/close status for US, SGX, HKEX, TSE."""
    from datetime import datetime
    import pytz

    now_utc = datetime.now(pytz.utc)
    status = {}

    # US market: 9:30 AM - 4:00 PM ET
    et = pytz.timezone('US/Eastern')
    now_et = now_utc.astimezone(et)
    us_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    us_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    us_is_open = now_et.weekday() < 5 and us_open <= now_et <= us_close
    status['US'] = {
        'open': us_is_open,
        'label': 'US Market OPEN' if us_is_open else 'US Market CLOSED',
        'close_time': '4:00 PM ET',
        'local_time': now_et.strftime('%I:%M %p ET'),
    }

    # SGX: 9:00 AM - 5:00 PM SGT
    sgt = pytz.timezone('Asia/Singapore')
    now_sgt = now_utc.astimezone(sgt)
    sgx_open = now_sgt.replace(hour=9, minute=0, second=0, microsecond=0)
    sgx_close = now_sgt.replace(hour=17, minute=0, second=0, microsecond=0)
    sgx_is_open = now_sgt.weekday() < 5 and sgx_open <= now_sgt <= sgx_close
    status['SGX'] = {
        'open': sgx_is_open,
        'label': 'SGX OPEN' if sgx_is_open else 'SGX CLOSED',
        'close_time': '5:00 PM SGT',
        'local_time': now_sgt.strftime('%I:%M %p SGT'),
    }

    # HKEX: 9:30 AM - 4:00 PM HKT
    hkt = pytz.timezone('Asia/Hong_Kong')
    now_hkt = now_utc.astimezone(hkt)
    hk_open = now_hkt.replace(hour=9, minute=30, second=0, microsecond=0)
    hk_close = now_hkt.replace(hour=16, minute=0, second=0, microsecond=0)
    hk_is_open = now_hkt.weekday() < 5 and hk_open <= now_hkt <= hk_close
    status['HKEX'] = {
        'open': hk_is_open,
        'label': 'HKEX OPEN' if hk_is_open else 'HKEX CLOSED',
        'close_time': '4:00 PM HKT',
        'local_time': now_hkt.strftime('%I:%M %p HKT'),
    }

    # TSE: 9:00 AM - 3:00 PM JST
    jst = pytz.timezone('Asia/Tokyo')
    now_jst = now_utc.astimezone(jst)
    tse_open = now_jst.replace(hour=9, minute=0, second=0, microsecond=0)
    tse_close = now_jst.replace(hour=15, minute=0, second=0, microsecond=0)
    tse_is_open = now_jst.weekday() < 5 and tse_open <= now_jst <= tse_close
    status['TSE'] = {
        'open': tse_is_open,
        'label': 'TSE OPEN' if tse_is_open else 'TSE CLOSED',
        'close_time': '3:00 PM JST',
        'local_time': now_jst.strftime('%I:%M %p JST'),
    }

    any_open = any(s['open'] for s in status.values())
    status['any_open'] = any_open

    return status


@app.route("/api/playbook")
def get_playbook_signals():
    """Scan SPY, QQQ, IWM for today's active signals."""
    try:
        import pandas as pd

        signals = []
        DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        ticker_list = ['SPY', 'QQQ', 'IWM', 'QQQM', 'SPLG', 'SPYM']
        raw = download_multi(ticker_list, period="10d")
        data = {}
        for name, df in raw.items():
            if df.empty:
                continue
            df['Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['Color'] = df['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')
            df['DayOfWeek'] = df.index.dayofweek
            df['PrevClose'] = df['Close'].shift(1)
            df['GapPct'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100
            data[name] = df

        spy = data['SPY']
        qqq = data['QQQ']
        iwm = data['IWM']
        qqqm = data.get('QQQM')
        splg = data.get('SPLG')
        spym = data.get('SPYM')

        if len(spy) == 0:
            return jsonify({"signals": [], "week": [], "error": "No data"})

        today = spy.iloc[-1]
        today_dow = int(today['DayOfWeek'])
        today_name = DAY_NAMES[today_dow] if today_dow < 5 else 'Weekend'
        today_date = spy.index[-1].strftime('%Y-%m-%d')

        def get_day(df, offset):
            idx = len(df) - 1 + offset
            if 0 <= idx < len(df):
                return df.iloc[idx]
            return None

        # Build recent days info
        week = []
        for i in range(max(0, len(spy)-5), len(spy)):
            row = spy.iloc[i]
            d = spy.index[i]
            ret = float(row['Return'])
            week.append({
                'date': d.strftime('%m/%d'),
                'day': DAY_NAMES[d.dayofweek] if d.dayofweek < 5 else 'Weekend',
                'color': 'green' if ret >= 0 else 'red',
                'return': round(ret, 2),
            })

        # Helper to get returns
        def ret(df, offset=0):
            r = get_day(df, offset)
            return float(r['Return']) if r is not None else 0

        def color(df, offset=0):
            r = get_day(df, offset)
            return r['Color'] if r is not None else ''

        tomorrow_dow = (today_dow + 1) % 5 if today_dow < 4 else 0
        tomorrow_name = DAY_NAMES[tomorrow_dow]

        # ── SPY signals for tomorrow ──
        # Monday: Fri Red >= 0.5%
        if tomorrow_dow == 0 and color(spy) == 'Red' and abs(ret(spy)) >= 0.5:
            signals.append({'ticker': 'SPY', 'day': 'Monday', 'signal': f'Fri Red ({ret(spy):+.2f}%) → Buy Mon', 'wr': 72.5, 'sl': 1.9, 'tier': 'STRONG'})
        # Wednesday: Mon Red + Tue Green >= 0.5%
        if tomorrow_dow == 2 and color(spy) == 'Green' and ret(spy) >= 0.5 and color(spy, -1) == 'Red':
            signals.append({'ticker': 'SPY', 'day': 'Wednesday', 'signal': f'Mon Red + Tue Grn ({ret(spy):+.2f}%) → Buy Wed', 'wr': 68.0, 'sl': 1.9, 'tier': 'STRONG'})
        # Thursday: Tue+Wed Green >= 0.5%
        if tomorrow_dow == 3 and color(spy) == 'Green' and ret(spy) >= 0.5 and color(spy, -1) == 'Green' and ret(spy, -1) >= 0.5:
            signals.append({'ticker': 'SPY', 'day': 'Thursday', 'signal': f'Tue+Wed Green → Buy Thu', 'wr': 65.5, 'sl': 1.1, 'tier': 'STRONG'})
        # Friday: Wed Green >= 1.0%
        if tomorrow_dow == 4 and color(spy) == 'Green' and ret(spy) >= 1.0:
            signals.append({'ticker': 'SPY', 'day': 'Friday', 'signal': f'Wed Grn ({ret(spy):+.2f}%) → Buy Fri', 'wr': 65.6, 'sl': 1.9, 'tier': 'STRONG'})
        # Multi-day: Red today signals
        if color(spy) == 'Red':
            multi = {0: ['Wednesday'], 1: ['Friday'], 2: ['Friday', 'Monday'], 3: ['Friday'], 4: ['Monday']}
            if today_dow in multi:
                for trade_day in multi[today_dow]:
                    sig_text = f'Red {today_name} ({ret(spy):+.2f}%) → Buy {trade_day}'
                    if not any(s['ticker'] == 'SPY' and s['day'] == trade_day for s in signals):
                        signals.append({'ticker': 'SPY', 'day': trade_day, 'signal': sig_text, 'wr': 58.4, 'sl': 1.9, 'tier': 'MULTI-DAY'})

        # ── QQQ signals for tomorrow ──
        # Monday: Fri Red >= 1.0%
        if tomorrow_dow == 0 and color(qqq) == 'Red' and abs(ret(qqq)) >= 1.0:
            signals.append({'ticker': 'QQQ', 'day': 'Monday', 'signal': f'Fri Red ({ret(qqq):+.2f}%) → Buy Mon', 'wr': 62.9, 'sl': 2.0, 'tier': 'STRONG'})
        # Tuesday: Mon Red >= 1.0%
        if tomorrow_dow == 1 and color(qqq) == 'Red' and abs(ret(qqq)) >= 1.0:
            signals.append({'ticker': 'QQQ', 'day': 'Tuesday', 'signal': f'Mon Red ({ret(qqq):+.2f}%) → Buy Tue', 'wr': 63.5, 'sl': 2.0, 'tier': 'STRONG'})
        # Wednesday: Mon+Tue Red >= 0.5%
        if tomorrow_dow == 2 and color(qqq) == 'Red' and abs(ret(qqq)) >= 0.5 and color(qqq, -1) == 'Red' and abs(ret(qqq, -1)) >= 0.5:
            signals.append({'ticker': 'QQQ', 'day': 'Wednesday', 'signal': f'Mon+Tue Red → Buy Wed', 'wr': 72.7, 'sl': 2.0, 'tier': 'STRONG'})
        # Thursday: Tue+Wed Red >= 0.3%
        if tomorrow_dow == 3 and color(qqq) == 'Red' and abs(ret(qqq)) >= 0.3 and color(qqq, -1) == 'Red' and abs(ret(qqq, -1)) >= 0.3:
            signals.append({'ticker': 'QQQ', 'day': 'Thursday', 'signal': f'Tue+Wed Red → Buy Thu', 'wr': 60.0, 'sl': 2.0, 'tier': 'STRONG'})
        # Friday: Wed Green + Thu Red >= 0.5%
        if tomorrow_dow == 4 and color(qqq) == 'Red' and abs(ret(qqq)) >= 0.5 and color(qqq, -1) == 'Green' and ret(qqq, -1) >= 0.5:
            signals.append({'ticker': 'QQQ', 'day': 'Friday', 'signal': f'Wed Grn + Thu Red → Buy Fri', 'wr': 64.7, 'sl': 2.0, 'tier': 'STRONG'})

        # ── QQQM signals for tomorrow ──
        if qqqm is not None and len(qqqm) >= 2:
            # Monday: Fri Red >= 1.0%
            if tomorrow_dow == 0 and color(qqqm) == 'Red' and abs(ret(qqqm)) >= 1.0:
                signals.append({'ticker': 'QQQM', 'day': 'Monday', 'signal': f'Fri Red ({ret(qqqm):+.2f}%) → Buy Mon', 'wr': 74.3, 'sl': 2.0, 'tier': 'STRONG'})
            # Monday combo: Thu Grn + Fri Red >= 0.3%
            if tomorrow_dow == 0 and color(qqqm) == 'Red' and abs(ret(qqqm)) >= 0.3 and color(qqqm, -1) == 'Green' and ret(qqqm, -1) >= 0.3:
                if not any(s['ticker'] == 'QQQM' and s['day'] == 'Monday' for s in signals):
                    signals.append({'ticker': 'QQQM', 'day': 'Monday', 'signal': f'Thu Grn + Fri Red → Buy Mon', 'wr': 70.0, 'sl': 2.0, 'tier': 'STRONG'})
            # Wednesday: Tue Green >= 1.0%
            if tomorrow_dow == 2 and color(qqqm) == 'Green' and ret(qqqm) >= 1.0:
                signals.append({'ticker': 'QQQM', 'day': 'Wednesday', 'signal': f'Tue Grn ({ret(qqqm):+.2f}%) → Buy Wed', 'wr': 62.5, 'sl': 2.0, 'tier': 'STRONG'})
            # Wednesday combo: Mon Red + Tue Red >= 0.3%
            if tomorrow_dow == 2 and color(qqqm) == 'Red' and abs(ret(qqqm)) >= 0.3 and color(qqqm, -1) == 'Red' and abs(ret(qqqm, -1)) >= 0.5:
                if not any(s['ticker'] == 'QQQM' and s['day'] == 'Wednesday' for s in signals):
                    signals.append({'ticker': 'QQQM', 'day': 'Wednesday', 'signal': f'Mon+Tue Red → Buy Wed', 'wr': 76.5, 'sl': 2.0, 'tier': 'STRONG'})
            # Friday: Wed Grn + Thu Red >= 0.3%
            if tomorrow_dow == 4 and color(qqqm) == 'Red' and abs(ret(qqqm)) >= 0.3 and color(qqqm, -1) == 'Green' and ret(qqqm, -1) >= 0.3:
                signals.append({'ticker': 'QQQM', 'day': 'Friday', 'signal': f'Wed Grn + Thu Red → Buy Fri', 'wr': 71.4, 'sl': 2.0, 'tier': 'STRONG'})
            # Friday: Thu Red >= 1.0%
            if tomorrow_dow == 4 and color(qqqm) == 'Red' and abs(ret(qqqm)) >= 1.0:
                if not any(s['ticker'] == 'QQQM' and s['day'] == 'Friday' for s in signals):
                    signals.append({'ticker': 'QQQM', 'day': 'Friday', 'signal': f'Thu Red ({ret(qqqm):+.2f}%) → Buy Fri', 'wr': 61.1, 'sl': 2.0, 'tier': 'STRONG'})

        # ── SPLG signals for tomorrow ──
        if splg is not None and len(splg) >= 2:
            # Monday: Fri Red >= 0.5%
            if tomorrow_dow == 0 and color(splg) == 'Red' and abs(ret(splg)) >= 0.5:
                signals.append({'ticker': 'SPLG', 'day': 'Monday', 'signal': f'Fri Red ({ret(splg):+.2f}%) → Buy Mon', 'wr': 72.0, 'sl': 1.9, 'tier': 'STRONG'})
            # Monday combo: Thu Red + Fri Red >= 0.5%
            if tomorrow_dow == 0 and color(splg) == 'Red' and abs(ret(splg)) >= 0.5 and color(splg, -1) == 'Red' and abs(ret(splg, -1)) >= 0.5:
                if not any(s['ticker'] == 'SPLG' and s['day'] == 'Monday' for s in signals):
                    signals.append({'ticker': 'SPLG', 'day': 'Monday', 'signal': f'Thu+Fri Red → Buy Mon', 'wr': 81.2, 'sl': 1.9, 'tier': 'STRONG'})
            # Wednesday: Mon Red + Tue Grn >= 0.5%
            if tomorrow_dow == 2 and color(splg) == 'Green' and ret(splg) >= 0.5 and color(splg, -1) == 'Red':
                signals.append({'ticker': 'SPLG', 'day': 'Wednesday', 'signal': f'Mon Red + Tue Grn → Buy Wed', 'wr': 80.0, 'sl': 1.9, 'tier': 'STRONG'})
            # Wednesday: Tue Grn >= 0.7%
            if tomorrow_dow == 2 and color(splg) == 'Green' and ret(splg) >= 0.7:
                if not any(s['ticker'] == 'SPLG' and s['day'] == 'Wednesday' for s in signals):
                    signals.append({'ticker': 'SPLG', 'day': 'Wednesday', 'signal': f'Tue Grn ({ret(splg):+.2f}%) → Buy Wed', 'wr': 70.6, 'sl': 1.9, 'tier': 'STRONG'})
            # Friday: Wed Red + Thu Grn >= 0.3%
            if tomorrow_dow == 4 and color(splg) == 'Green' and ret(splg) >= 0.3 and color(splg, -1) == 'Red' and abs(ret(splg, -1)) >= 0.5:
                signals.append({'ticker': 'SPLG', 'day': 'Friday', 'signal': f'Wed Red + Thu Grn → Buy Fri', 'wr': 76.5, 'sl': 1.9, 'tier': 'STRONG'})
            # Friday: Thu Red >= 0.5%
            if tomorrow_dow == 4 and color(splg) == 'Red' and abs(ret(splg)) >= 0.5:
                if not any(s['ticker'] == 'SPLG' and s['day'] == 'Friday' for s in signals):
                    signals.append({'ticker': 'SPLG', 'day': 'Friday', 'signal': f'Thu Red ({ret(splg):+.2f}%) → Buy Fri', 'wr': 62.7, 'sl': 1.9, 'tier': 'STRONG'})

        # ── SPYM signals for tomorrow ──
        if spym is not None and len(spym) >= 2:
            # Monday: Fri Red >= 0.5%
            if tomorrow_dow == 0 and color(spym) == 'Red' and abs(ret(spym)) >= 0.5:
                signals.append({'ticker': 'SPYM', 'day': 'Monday', 'signal': f'Fri Red ({ret(spym):+.2f}%) → Buy Mon', 'wr': 72.0, 'sl': 1.9, 'tier': 'STRONG'})
            # Monday combo: Thu Red + Fri Red >= 0.5%
            if tomorrow_dow == 0 and color(spym) == 'Red' and abs(ret(spym)) >= 0.5 and color(spym, -1) == 'Red' and abs(ret(spym, -1)) >= 0.5:
                if not any(s['ticker'] == 'SPYM' and s['day'] == 'Monday' for s in signals):
                    signals.append({'ticker': 'SPYM', 'day': 'Monday', 'signal': f'Thu+Fri Red → Buy Mon', 'wr': 81.2, 'sl': 1.9, 'tier': 'STRONG'})
            # Wednesday: Tue Green >= 0.8%
            if tomorrow_dow == 2 and color(spym) == 'Green' and ret(spym) >= 0.8:
                signals.append({'ticker': 'SPYM', 'day': 'Wednesday', 'signal': f'Tue Grn ({ret(spym):+.2f}%) → Buy Wed', 'wr': 75.0, 'sl': 1.9, 'tier': 'STRONG'})
            # Wednesday combo: Mon Red + Tue Grn >= 0.5%
            if tomorrow_dow == 2 and color(spym) == 'Green' and ret(spym) >= 0.5 and color(spym, -1) == 'Red':
                if not any(s['ticker'] == 'SPYM' and s['day'] == 'Wednesday' for s in signals):
                    signals.append({'ticker': 'SPYM', 'day': 'Wednesday', 'signal': f'Mon Red + Tue Grn → Buy Wed', 'wr': 70.0, 'sl': 1.9, 'tier': 'STRONG'})
            # Friday: Wed Red + Thu Grn >= 0.5%
            if tomorrow_dow == 4 and color(spym) == 'Green' and ret(spym) >= 0.5 and color(spym, -1) == 'Red' and abs(ret(spym, -1)) >= 0.5:
                signals.append({'ticker': 'SPYM', 'day': 'Friday', 'signal': f'Wed Red + Thu Grn → Buy Fri', 'wr': 77.8, 'sl': 1.9, 'tier': 'STRONG'})
            # Friday: Thu Red >= 0.5%
            if tomorrow_dow == 4 and color(spym) == 'Red' and abs(ret(spym)) >= 0.5:
                if not any(s['ticker'] == 'SPYM' and s['day'] == 'Friday' for s in signals):
                    signals.append({'ticker': 'SPYM', 'day': 'Friday', 'signal': f'Thu Red ({ret(spym):+.2f}%) → Buy Fri', 'wr': 62.7, 'sl': 1.9, 'tier': 'STRONG'})

        # ── IWM signals for tomorrow ──
        # Monday: Thu Green + Fri Red >= 0.3%
        if tomorrow_dow == 0 and color(iwm) == 'Red' and abs(ret(iwm)) >= 0.3 and color(iwm, -1) == 'Green' and ret(iwm, -1) >= 0.3:
            signals.append({'ticker': 'IWM', 'day': 'Monday', 'signal': f'Thu Grn + Fri Red → Buy Mon', 'wr': 60.0, 'sl': 2.5, 'tier': 'STRONG'})
        # Tuesday: Fri Green + Mon Red >= 0.7%
        if tomorrow_dow == 1 and color(iwm) == 'Red' and abs(ret(iwm)) >= 0.7 and color(iwm, -1) == 'Green' and ret(iwm, -1) >= 0.7:
            signals.append({'ticker': 'IWM', 'day': 'Tuesday', 'signal': f'Fri Grn + Mon Red → Buy Tue', 'wr': 63.6, 'sl': 2.5, 'tier': 'STRONG'})
        # Wednesday: Mon Red + Tue Green >= 0.3%
        if tomorrow_dow == 2 and color(iwm) == 'Green' and ret(iwm) >= 0.3 and color(iwm, -1) == 'Red':
            signals.append({'ticker': 'IWM', 'day': 'Wednesday', 'signal': f'Mon Red + Tue Grn → Buy Wed', 'wr': 60.8, 'sl': 2.5, 'tier': 'STRONG'})
        # Thursday: Tue+Wed Green >= 0.7%
        if tomorrow_dow == 3 and color(iwm) == 'Green' and ret(iwm) >= 0.7 and color(iwm, -1) == 'Green' and ret(iwm, -1) >= 0.7:
            signals.append({'ticker': 'IWM', 'day': 'Thursday', 'signal': f'Tue+Wed Green → Buy Thu', 'wr': 64.5, 'sl': 2.5, 'tier': 'STRONG'})
        # Friday: Wed+Thu Red >= 0.5%
        if tomorrow_dow == 4 and color(iwm) == 'Red' and abs(ret(iwm)) >= 0.5 and color(iwm, -1) == 'Red' and abs(ret(iwm, -1)) >= 0.5:
            signals.append({'ticker': 'IWM', 'day': 'Friday', 'signal': f'Wed+Thu Red → Buy Fri', 'wr': 62.0, 'sl': 2.5, 'tier': 'STRONG'})

        # ── Backup signals ──
        # Gap fill (check tomorrow morning)
        gap = float(today['GapPct']) if 'GapPct' in today.index else 0
        if gap <= -0.3:
            signals.append({'ticker': 'SPY', 'day': today_name, 'signal': f'Gap down {gap:.2f}% → Gap Fill', 'wr': 55.7, 'sl': 0.5, 'tier': 'BACKUP'})

        # 2-day red bounce
        if color(spy) == 'Red' and color(spy, -1) == 'Red':
            signals.append({'ticker': 'SPY', 'day': tomorrow_name, 'signal': '2 consecutive red days → Bounce', 'wr': 56.3, 'sl': 2.5, 'tier': 'BACKUP'})

        # 3-day red bounce
        if color(spy) == 'Red' and color(spy, -1) == 'Red' and color(spy, -2) == 'Red':
            # Remove 2-day, upgrade to 3-day
            signals = [s for s in signals if s.get('signal') != '2 consecutive red days → Bounce']
            signals.append({'ticker': 'SPY', 'day': tomorrow_name, 'signal': '3 consecutive red days → Bounce', 'wr': 55.3, 'sl': 1.0, 'tier': 'BACKUP'})

        # Overnight hold (Red Tuesday)
        if today_dow == 1 and color(spy) == 'Red':
            signals.append({'ticker': 'SPY', 'day': 'Tue→Wed', 'signal': f'Red Tue ({ret(spy):+.2f}%) → Hold overnight', 'wr': 64.1, 'sl': 0, 'tier': 'OVERNIGHT'})

        # ── Rare But Powerful ──
        # VIX Spike (VIX > 35)
        try:
            vix = download('^VIX', period='5d')  # VIX uses yfinance fallback (non-US symbol)
            if len(vix) > 0:
                vix_val = float(vix.iloc[-1]['Close'])
                if vix_val >= 35:
                    signals.append({'ticker': 'SPY', 'day': tomorrow_name, 'signal': f'VIX Spike ({vix_val:.1f}) → Buy SPY', 'wr': 90.9, 'sl': 3.0, 'tier': 'RARE'})
                elif vix_val >= 28:
                    signals.append({'ticker': 'SPY', 'day': tomorrow_name, 'signal': f'VIX Elevated ({vix_val:.1f}) → Watch for entry', 'wr': 75.0, 'sl': 2.5, 'tier': 'RARE'})
        except Exception:
            pass

        # RSI Oversold (RSI-14 < 30 on SPY)
        try:
            spy_3mo = download('SPY', period='3mo')
            if len(spy_3mo) >= 14:
                delta = spy_3mo['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = float(rsi.iloc[-1])
                if rsi_val < 30:
                    signals.append({'ticker': 'SPY', 'day': tomorrow_name, 'signal': f'RSI Oversold ({rsi_val:.1f}) → Buy SPY', 'wr': 83.3, 'sl': 2.0, 'tier': 'RARE'})
        except Exception:
            pass

        # Earnings Season Drift (Jan, Apr, Jul, Oct)
        import calendar as cal_mod
        current_month = pd.Timestamp.now().month
        if current_month in [1, 4, 7, 10]:
            signals.append({'ticker': 'SPY', 'day': 'This month', 'signal': f'Earnings season ({cal_mod.month_name[current_month]}) → Bullish drift', 'wr': 78.0, 'sl': 2.0, 'tier': 'RARE'})

        # MA Crossover (Golden Cross: SMA-50 crosses above SMA-200)
        try:
            if len(spy_3mo) >= 50:
                sma50 = spy_3mo['Close'].rolling(50).mean()
                sma200 = spy_3mo['Close'].rolling(200).mean() if len(spy_3mo) >= 200 else None
                if sma200 is not None and not sma200.isna().all():
                    # Check if SMA-50 just crossed above SMA-200 in last 3 days
                    recent = pd.DataFrame({'sma50': sma50, 'sma200': sma200}).dropna().tail(5)
                    if len(recent) >= 2:
                        if recent['sma50'].iloc[-1] > recent['sma200'].iloc[-1] and recent['sma50'].iloc[-3] <= recent['sma200'].iloc[-3]:
                            signals.append({'ticker': 'SPY', 'day': tomorrow_name, 'signal': 'Golden Cross (SMA-50 > SMA-200) → Bullish', 'wr': 75.0, 'sl': 2.0, 'tier': 'RARE'})
        except Exception:
            pass

        # Sort: STRONG first, then by win rate
        tier_order = {'RARE': 0, 'STRONG': 1, 'MULTI-DAY': 2, 'OVERNIGHT': 3, 'BACKUP': 4}
        signals.sort(key=lambda s: (tier_order.get(s['tier'], 9), -s['wr']))

        market_status = _market_status()

        return jsonify({
            "signals": signals,
            "week": week,
            "today": today_name,
            "today_date": today_date,
            "today_color": 'green' if ret(spy) >= 0 else 'red',
            "today_return": round(float(today['Return']), 2),
            "market_status": market_status,
        })

    except Exception as e:
        return jsonify({"signals": [], "week": [], "error": str(e)})


@app.route("/api/asian-signals")
def get_asian_signals():
    """Scan Asian ETFs for today's active signals."""
    try:
        import pandas as pd

        DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        ASIAN_TICKERS = {
            'ES3.SI': {'name': 'STI ETF', 'market': 'SGX', 'sl': 1.5},
            'G3B.SI': {'name': 'Nikko STI', 'market': 'SGX', 'sl': 1.5},
            'CLR.SI': {'name': 'HSTECH SGX', 'market': 'SGX', 'sl': 2.5},
            '2800.HK': {'name': 'Tracker HK', 'market': 'HKEX', 'sl': 2.0},
            '3067.HK': {'name': 'HS Tech', 'market': 'HKEX', 'sl': 2.5},
            '2828.HK': {'name': 'HSI ETF', 'market': 'HKEX', 'sl': 2.0},
            '1306.T': {'name': 'Nikkei 225', 'market': 'TSE', 'sl': 2.0},
            'EWJ': {'name': 'Japan', 'market': 'US-Asia', 'sl': 2.0},
            'EWS': {'name': 'Singapore', 'market': 'US-Asia', 'sl': 2.0},
            'EWH': {'name': 'HK', 'market': 'US-Asia', 'sl': 2.0},
            'FXI': {'name': 'China LC', 'market': 'US-Asia', 'sl': 2.5},
            'KWEB': {'name': 'CN Internet', 'market': 'US-Asia', 'sl': 3.0},
            'EWT': {'name': 'Taiwan', 'market': 'US-Asia', 'sl': 2.0},
            'EWY': {'name': 'S Korea', 'market': 'US-Asia', 'sl': 2.5},
        }

        # Signal configs (same as spy_signal_checker.py)
        ASIAN_DAY_SIGNALS = {
            'G3B.SI': {0: ('GREEN', 0.5, 'Tuesday', 67.6)},
            '2800.HK': {2: ('RED', 1.0, 'Thursday', 67.6)},
            '3067.HK': {2: ('RED', 1.5, 'Thursday', 59.0)},
            '2828.HK': {3: ('RED', 0.8, 'Friday', 66.7)},
            '1306.T': {3: ('RED', 0.5, 'Friday', 68.8)},
            'EWJ': {0: ('GREEN', 0.5, 'Tuesday', 68.6), 4: ('RED', 0.0, 'Monday', 64.5)},
            'EWS': {2: ('RED', 0.8, 'Thursday', 66.7)},
            'EWH': {1: ('GREEN', 0.5, 'Wednesday', 65.3)},
            'FXI': {1: ('RED', 1.0, 'Wednesday', 61.3), 2: ('RED', 1.0, 'Thursday', 61.3)},
            'KWEB': {1: ('RED', 1.5, 'Wednesday', 60.0)},
            'EWT': {0: ('RED', 0.5, 'Tuesday', 71.4), 2: ('RED', 1.0, 'Thursday', 71.4)},
            'EWY': {0: ('RED', 0.8, 'Tuesday', 72.7), 1: ('GREEN', 0.8, 'Wednesday', 73.3), 2: ('RED', 1.0, 'Thursday', 65.4)},
        }

        ASIAN_COMBO_SIGNALS = {
            'ES3.SI': [(3,'Green','Green',80.0),(4,'Green','Green',80.0),(1,'Green','Green',75.0)],
            'G3B.SI': [(4,'Green','Green',76.2),(1,'Green','Green',75.0),(3,'Green','Red',68.2)],
            '2800.HK': [(2,'Red','Green',65.2),(3,'Green','Red',63.9),(0,'Red','Red',60.9)],
            '3067.HK': [(4,'Green','Green',61.9)],
            '2828.HK': [(3,'Green','Red',65.2),(4,'Red','Green',65.0)],
            '1306.T': [(3,'Red','Red',75.0),(4,'Red','Red',74.2),(0,'Red','Green',69.2)],
            'EWJ': [(0,'Red','Red',77.3),(4,'Green','Red',75.0),(1,'Green','Green',75.0),(3,'Green','Red',70.8),(2,'Red','Red',66.7)],
            'EWS': [(0,'Red','Green',83.9),(4,'Green','Green',71.4),(2,'Red','Green',65.0),(3,'Green','Green',64.3)],
            'EWH': [(2,'Red','Green',76.2),(1,'Green','Green',66.7),(3,'Red','Red',66.7),(0,'Green','Green',65.8)],
            'FXI': [(3,'Red','Red',75.0),(0,'Red','Green',70.4),(2,'Green','Green',65.0)],
            'KWEB': [(0,'Red','Green',68.0),(3,'Red','Red',67.4)],
            'EWT': [(3,'Red','Red',80.0),(4,'Red','Green',72.0),(0,'Red','Green',70.0),(2,'Red','Red',69.6),(1,'Red','Red',66.7)],
            'EWY': [(2,'Green','Green',72.0),(0,'Red','Red',68.2),(4,'Green','Green',66.7),(1,'Green','Red',66.7)],
        }

        ASIAN_MULTIDAY_SIGNALS = {
            '2828.HK': [(2,0.5,3,0,76.9)],
            'EWJ': [(2,0.5,3,0,71.8)],
            'EWH': [(2,0.5,3,0,72.2),(0,0.5,2,2,65.5)],
            'EWT': [(2,0.5,3,0,71.4)],
            'EWY': [(2,1.0,3,0,69.6),(2,1.0,4,1,69.6)],
            'EWS': [(2,0.5,3,0,65.8)],
        }

        ASIAN_OVERNIGHT_SIGNALS = {
            'ES3.SI': [(0,'Red',68.6),(2,'Red',67.5)],
            'G3B.SI': [(0,'Red',73.1),(2,'Red',68.1)],
            'CLR.SI': [(4,'Green',72.2),(1,'Red',69.0),(2,'Green',67.9),(0,'Red',66.4)],
            '1306.T': [(0,'Red',66.1),(4,'Green',65.2)],
            '2828.HK': [(0,None,66.7)],
        }

        signals = []
        ticker_data = {}

        raw = download_multi(list(ASIAN_TICKERS.keys()), period="10d")
        for ticker, df in raw.items():
            if df.empty:
                continue
            df['Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['Color'] = df['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')
            ticker_data[ticker] = df

        for ticker, info in ASIAN_TICKERS.items():
            if ticker not in ticker_data or len(ticker_data[ticker]) == 0:
                continue
            df = ticker_data[ticker]
            today = df.iloc[-1]
            today_dow = int(df.index[-1].dayofweek)
            if today_dow > 4:
                continue
            today_ret = float(today['Return'])
            today_color = 'Green' if today_ret >= 0 else 'Red'
            today_name = DAY_NAMES[today_dow]
            sl = info['sl']
            tomorrow_dow = (today_dow + 1) % 5 if today_dow < 4 else 0

            # Single-day signals
            day_sigs = ASIAN_DAY_SIGNALS.get(ticker, {})
            ds = day_sigs.get(today_dow)
            if ds:
                cond, thresh, trade_day, wr = ds
                if today_color.upper() == cond and abs(today_ret) >= thresh:
                    signals.append({
                        'ticker': ticker, 'name': info['name'], 'market': info['market'],
                        'type': 'DAY', 'day': trade_day,
                        'signal': f"{today_color} {today_name} ({today_ret:+.2f}%) → Buy {trade_day}",
                        'wr': wr, 'sl': sl,
                    })

            # Combo signals
            combos = ASIAN_COMBO_SIGNALS.get(ticker, [])
            if len(df) >= 2 and combos:
                prev = df.iloc[-2]
                prev_ret = float(prev['Return'])
                prev_color = 'Green' if prev_ret >= 0 else 'Red'
                for combo in combos:
                    trade_dow, p2c, p1c, wr = combo
                    if tomorrow_dow == trade_dow and prev_color == p2c and today_color == p1c:
                        trade_day = DAY_NAMES[trade_dow]
                        if not any(s['ticker'] == ticker and s['day'] == trade_day for s in signals):
                            _combo_desc = {
                            ('Green', 'Green'): "2 up-days in a row → momentum carries into",
                            ('Red',   'Red'):   "2 down-days in a row → oversold bounce on",
                            ('Green', 'Red'):   "Rally then pullback → dip entry on",
                            ('Red',   'Green'): "Selloff then recovery → follow-through on",
                        }.get((p2c, p1c), f"{p2c} + {p1c} →")
                        signals.append({
                                'ticker': ticker, 'name': info['name'], 'market': info['market'],
                                'type': 'COMBO', 'day': trade_day,
                                'signal': f"{_combo_desc} {trade_day}",
                                'wr': wr, 'sl': sl,
                            })

            # Multi-day signals
            multidays = ASIAN_MULTIDAY_SIGNALS.get(ticker, [])
            for md in multidays:
                sig_dow, thresh, offset, trd_dow, wr = md
                if today_dow == sig_dow and today_color == 'Red' and abs(today_ret) >= thresh:
                    trade_day = DAY_NAMES[trd_dow]
                    if not any(s['ticker'] == ticker and s['day'] == trade_day for s in signals):
                        signals.append({
                            'ticker': ticker, 'name': info['name'], 'market': info['market'],
                            'type': 'MULTI-DAY', 'day': trade_day,
                            'signal': f"Red {today_name} ({today_ret:+.2f}%) → Buy {trade_day} ({offset}d)",
                            'wr': wr, 'sl': sl,
                        })

            # Overnight signals
            overnights = ASIAN_OVERNIGHT_SIGNALS.get(ticker, [])
            for ov in overnights:
                sig_dow, color_filter, wr = ov
                if today_dow == sig_dow:
                    if color_filter is None or today_color == color_filter:
                        next_dow = (today_dow + 1) % 5 if today_dow < 4 else 0
                        signals.append({
                            'ticker': ticker, 'name': info['name'], 'market': info['market'],
                            'type': 'OVERNIGHT', 'day': f"{today_name}→{DAY_NAMES[next_dow]}",
                            'signal': f"Buy {today_color} {today_name} close → Sell {DAY_NAMES[next_dow]} open",
                            'wr': wr, 'sl': sl,
                        })

        signals.sort(key=lambda s: -s['wr'])

        market_status = _market_status()

        return jsonify({
            "signals": signals,
            "scan_time": datetime.now().isoformat(),
            "tickers_scanned": len(ticker_data),
            "market_status": market_status,
        })

    except Exception as e:
        return jsonify({"signals": [], "error": str(e)})


@app.route("/api/dip-scanner")
def get_dip_signals():
    """Scan all tickers for dip-buy signals and return scored results."""
    try:
        sys.path.insert(0, str(BASE_DIR.parent))
        from dip_scanner import scan_dip_signals
        results = scan_dip_signals()
        return jsonify({"results": results, "scan_time": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


# ── FinRL Endpoints ─────────────────────────────────────────────────────────

@app.route("/api/finrl-signals")
def get_finrl_signals():
    """Get FinRL RL-based trading signals for US tickers."""
    try:
        from finrl_engine import get_finrl_signals as _get_signals
        tickers = request.args.get("tickers", "SPY,QQQ,IWM,QQQM,SPLG")
        ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        period = request.args.get("period", "1y")
        retrain = request.args.get("retrain", "false").lower() == "true"

        signals = _get_signals(tickers=ticker_list, training_period=period, retrain=retrain)
        return jsonify({
            "signals": signals,
            "scan_time": datetime.now().isoformat(),
            "model": "PPO (FinRL)",
        })
    except Exception as e:
        logging.exception("FinRL signals error")
        return jsonify({"signals": [], "error": str(e)})


@app.route("/api/finrl-backtest")
def get_finrl_backtest():
    """Run a FinRL backtest for a single ticker."""
    try:
        from finrl_engine import get_finrl_backtest as _backtest
        ticker = request.args.get("ticker", "SPY").upper()
        period = request.args.get("period", "1y")
        result = _backtest(ticker=ticker, period=period)
        return jsonify(result)
    except Exception as e:
        logging.exception("FinRL backtest error")
        return jsonify({"error": str(e)})


@app.route("/api/unified-backtest")
def get_unified_backtest():
    """Run a unified strategy backtest."""
    try:
        from unified_backtest import run_unified_backtest

        tickers_str = request.args.get("tickers", "SPY,QQQ,IWM")
        tickers = [t.strip() for t in tickers_str.split(",")]
        period = request.args.get("period", "6mo")
        threshold = int(request.args.get("threshold", "60"))

        result = run_unified_backtest(
            tickers=tickers,
            period=period,
            score_threshold=threshold,
        )
        return jsonify(result)
    except Exception as e:
        logging.exception("Unified backtest error")
        return jsonify({"error": str(e)})


@app.route("/api/unified-signals")
def get_unified_signals():
    """Unified signal engine — composite scores from all sources."""
    try:
        from signal_engine import get_unified_signals as _unified
        result = _unified()
        return jsonify(result)
    except Exception as e:
        logging.exception("Unified signals error")
        return jsonify({"signals": [], "error": str(e)})


@app.route("/api/last-telegram-report")
def get_last_telegram_report():
    """Return the last report that was sent to Telegram."""
    import os
    cache_path = os.path.join(os.path.dirname(__file__), "..", "last_telegram_report.json")
    cache_path = os.path.abspath(cache_path)
    if not os.path.exists(cache_path):
        return jsonify({"error": "No report sent yet"}), 404
    with open(cache_path) as f:
        return jsonify(json.load(f))


# ── Alpaca Sync ──────────────────────────────────────────────────────────────

@app.route("/api/sync-alpaca")
def sync_alpaca():
    """Sync current Alpaca positions to the dashboard as open trades."""
    try:
        from alpaca.trading.client import TradingClient

        key = os.environ.get("APCA_API_KEY_ID", "")
        secret = os.environ.get("APCA_API_SECRET_KEY", "")
        if not key or not secret:
            return jsonify({"error": "Alpaca keys not set"}), 400

        client = TradingClient(key, secret, paper=True)
        positions = client.get_all_positions()

        conn = get_db()

        # Get existing open trades (no exit_price) already in DB
        existing_open = {}
        for row in conn.execute(
            "SELECT id, ticker FROM trades WHERE exit_price IS NULL AND signal_type = 'Alpaca Sync'"
        ).fetchall():
            existing_open[row["ticker"]] = row["id"]

        # Current Alpaca tickers
        alpaca_tickers = set()

        synced = 0
        for p in positions:
            ticker = p.symbol.replace("/", "")
            alpaca_tickers.add(ticker)
            qty = abs(float(p.qty))
            entry = float(p.avg_entry_price)
            side = "Long" if float(p.qty) > 0 else "Short"

            if ticker in existing_open:
                # Update existing open trade
                conn.execute(
                    "UPDATE trades SET entry_price=?, shares=?, direction=? WHERE id=?",
                    (entry, qty, side, existing_open[ticker])
                )
            else:
                # Insert new open position
                trade_id = uuid.uuid4().hex[:12]
                conn.execute("""
                    INSERT INTO trades (id, date, ticker, direction, signal_type,
                        entry_price, shares, notes, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_id,
                    datetime.now().strftime("%Y-%m-%d"),
                    ticker,
                    side,
                    "Alpaca Sync",
                    entry,
                    qty,
                    "Open position synced from Alpaca",
                    3,
                ))
                synced += 1

        # Close trades that are no longer in Alpaca positions
        for ticker, trade_id in existing_open.items():
            if ticker not in alpaca_tickers:
                # Position was closed on Alpaca — mark as closed
                conn.execute(
                    "UPDATE trades SET exit_price = entry_price, pnl = 0, "
                    "return_pct = 0, exit_type = 'Closed on Alpaca', "
                    "notes = 'Position closed — synced from Alpaca' "
                    "WHERE id = ?",
                    (trade_id,)
                )

        conn.commit()
        conn.close()

        pos_list = []
        for p in positions:
            pos_list.append({
                "ticker": p.symbol.replace("/", ""),
                "qty": float(p.qty),
                "side": str(p.side),
                "avg_entry": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
            })

        return jsonify({
            "synced": synced,
            "positions": pos_list,
        })

    except Exception as e:
        logging.exception("Alpaca sync error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  Trading Journal Dashboard")
    print(f"  Open: http://localhost:{port}")
    print(f"  Database: {DB_PATH}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
