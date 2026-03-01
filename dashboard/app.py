"""
Trading Journal Dashboard
==========================
Flask server with SQLite storage for logging and reviewing trades.
Run: python3 app.py → opens http://localhost:5050
"""

import os
import csv
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

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


@app.route("/api/playbook")
def get_playbook_signals():
    """Scan SPY, QQQ, IWM for today's active signals."""
    try:
        import yfinance as yf
        import pandas as pd

        signals = []
        DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        tickers = {'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM'}
        data = {}
        for name, sym in tickers.items():
            df = yf.download(sym, period="10d", auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel('Ticker')
            df['Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
            df['Color'] = df['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')
            df['DayOfWeek'] = df.index.dayofweek
            df['PrevClose'] = df['Close'].shift(1)
            df['GapPct'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100
            data[name] = df

        spy = data['SPY']
        qqq = data['QQQ']
        iwm = data['IWM']

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

        # Sort: STRONG first, then by win rate
        tier_order = {'STRONG': 0, 'MULTI-DAY': 1, 'OVERNIGHT': 2, 'BACKUP': 3}
        signals.sort(key=lambda s: (tier_order.get(s['tier'], 9), -s['wr']))

        return jsonify({
            "signals": signals,
            "week": week,
            "today": today_name,
            "today_date": today_date,
            "today_color": 'green' if ret(spy) >= 0 else 'red',
            "today_return": round(float(today['Return']), 2),
        })

    except Exception as e:
        return jsonify({"signals": [], "week": [], "error": str(e)})


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == "__main__":
    print(f"\n  Trading Journal Dashboard")
    print(f"  Open: http://localhost:5050")
    print(f"  Database: {DB_PATH}\n")
    app.run(host="127.0.0.1", port=5050, debug=True)
