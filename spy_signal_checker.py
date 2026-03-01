"""
SPY Day-of-Week Signal Checker
================================
Runs after market close, checks today's SPY candle, and sends
a macOS notification if there's a trade signal for tomorrow.

Two tiers of signals:
  STRONG (from multi-day pair analysis, 58%+ WR):
    Red Monday    → Buy Wednesday at open
    Red Tuesday   → Buy Friday at open
    Red Wednesday → Buy Friday at open + next Monday at open
    Red Thursday  → Buy Friday at open
    Red Friday    → Buy Monday at open

  DAILY (from next-day filter analysis):
    Monday:    BUY if prev Friday was RED    (WR: 66.0%)
    Tuesday:   SKIP — no reliable edge
    Wednesday: BUY if prev Tuesday was GREEN (WR: 54.3%, optional)
    Thursday:  SKIP — no reliable edge
    Friday:    BUY if prev Thursday was RED  (WR: 58.4%)

Trade rules:
  - Buy at market open
  - Stop loss: 1.9% below entry
  - No take profit
  - Sell at market close
"""

import subprocess
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
STOP_LOSS_PCT = 1.9
LOG_FILE = Path(__file__).parent / "signal_log.csv"

# Multi-day signals: if today (signal day) is RED → trade on these future days
# 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
MULTI_DAY_SIGNALS = {
    0: ["Wednesday"],         # Red Monday → Buy Wednesday
    1: ["Friday"],            # Red Tuesday → Buy Friday
    2: ["Friday", "Monday"],  # Red Wednesday → Buy Friday + Monday
    3: ["Friday"],            # Red Thursday → Buy Friday
    4: ["Monday"],            # Red Friday → Buy Monday
}

# Next-day signals: should we buy TOMORROW based on today's color?
# Format: {today_dow: {'condition': 'RED'|'GREEN'|None, 'tomorrow': day_name, 'wr': win_rate, 'tier': 'STRONG'|'OPTIONAL'}}
NEXT_DAY_SIGNALS = {
    0: {'condition': None, 'tomorrow': 'Tuesday', 'tier': None},          # Monday → Tuesday: SKIP
    1: {'condition': 'GREEN', 'tomorrow': 'Wednesday', 'wr': 54.3, 'tier': 'OPTIONAL'},  # Green Tue → Wed
    2: {'condition': None, 'tomorrow': 'Thursday', 'tier': None},         # Wednesday → Thursday: SKIP
    3: {'condition': 'RED', 'tomorrow': 'Friday', 'wr': 58.4, 'tier': 'STRONG'},   # Red Thu → Friday
    4: {'condition': 'RED', 'tomorrow': 'Monday', 'wr': 66.0, 'tier': 'STRONG'},   # Red Fri → Monday
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def send_notification(title, message, subtitle=""):
    """Send a macOS notification using osascript."""
    # Escape double quotes for AppleScript
    title = title.replace('"', '\\"')
    message = message.replace('"', '\\"')
    subtitle = subtitle.replace('"', '\\"')

    script = f'display notification "{message}" with title "{title}"'
    if subtitle:
        script = f'display notification "{message}" with title "{title}" subtitle "{subtitle}"'

    subprocess.run(["osascript", "-e", script], capture_output=True)


def send_sound_notification(title, message, subtitle=""):
    """Send notification with sound."""
    title = title.replace('"', '\\"')
    message = message.replace('"', '\\"')
    subtitle = subtitle.replace('"', '\\"')

    script = f'display notification "{message}" with title "{title}" subtitle "{subtitle}" sound name "Glass"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def log_signal(date, day_name, color, signal_text):
    """Append signal to CSV log."""
    header = not LOG_FILE.exists()
    with open(LOG_FILE, "a") as f:
        if header:
            f.write("date,day,color,open,close,return_pct,signal\n")
        f.write(f"{date},{day_name},{color},{signal_text}\n")


def check_signals():
    """Main function: check today's SPY candle and alert if signal found."""
    print(f"SPY Signal Checker — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)

    # Download today's data
    spy = yf.download("SPY", period="5d", auto_adjust=True)
    if isinstance(spy.columns, __import__('pandas').MultiIndex):
        spy = spy.droplevel('Ticker', axis=1)

    if len(spy) == 0:
        print("No data available. Market might be closed.")
        send_notification("SPY Signal Checker", "No data available. Market might be closed.")
        return

    today = spy.iloc[-1]
    today_date = spy.index[-1]
    today_dow = today_date.dayofweek  # 0=Mon ... 4=Fri
    today_name = DAY_NAMES[today_dow]

    open_price = today['Open']
    close_price = today['Close']
    day_return = (close_price - open_price) / open_price * 100
    color = "GREEN" if day_return >= 0 else "RED"

    print(f"\n  Today: {today_name} {today_date.strftime('%Y-%m-%d')}")
    print(f"  Open:  ${open_price:.2f}")
    print(f"  Close: ${close_price:.2f}")
    print(f"  Return: {day_return:+.2f}%")
    print(f"  Color: {color}")

    # Check if today is a weekday
    if today_dow > 4:
        print("\n  Weekend — no signals.")
        return

    signals_found = []

    # ── CHECK 1: Multi-day pair signals (Red today → trade later this week) ──
    if color == "RED" and today_dow in MULTI_DAY_SIGNALS:
        trade_days = MULTI_DAY_SIGNALS[today_dow]
        for trade_day in trade_days:
            signals_found.append({
                'type': 'MULTI-DAY',
                'trade_day': trade_day,
                'reason': f"Red {today_name} ({day_return:+.2f}%)",
                'tier': 'STRONG',
            })

    # ── CHECK 2: Next-day signal (should we trade tomorrow?) ────────────────
    next_day_info = NEXT_DAY_SIGNALS.get(today_dow)
    if next_day_info and next_day_info['condition'] is not None:
        if color == next_day_info['condition']:
            # Check if this signal already exists from multi-day (avoid duplicate)
            already = any(s['trade_day'] == next_day_info['tomorrow'] for s in signals_found)
            if not already:
                signals_found.append({
                    'type': 'NEXT-DAY',
                    'trade_day': next_day_info['tomorrow'],
                    'reason': f"{color.capitalize()} {today_name} ({day_return:+.2f}%) → WR: {next_day_info['wr']:.1f}%",
                    'tier': next_day_info['tier'],
                })

    # ── OUTPUT ───────────────────────────────────────────────────────────────
    if signals_found:
        print(f"\n  🔔 {len(signals_found)} SIGNAL(S) DETECTED!")

        for sig in signals_found:
            est_entry = close_price
            stop_price = est_entry * (1 - STOP_LOSS_PCT / 100)
            tier_icon = "🟢" if sig['tier'] == 'STRONG' else "🟡"

            print(f"\n  {tier_icon} [{sig['tier']}] {sig['type']}: Buy SPY at {sig['trade_day']} open")
            print(f"     Reason: {sig['reason']}")
            print(f"     Stop loss: {STOP_LOSS_PCT}% below entry (~${stop_price:.2f})")
            print(f"     Exit: Sell at {sig['trade_day']} close")

            # Send macOS notification
            send_sound_notification(
                title=f"{'🟢' if sig['tier'] == 'STRONG' else '🟡'} SPY: Buy {sig['trade_day']}! [{sig['tier']}]",
                message=f"Buy SPY at {sig['trade_day']} open. SL: {STOP_LOSS_PCT}% (~${stop_price:.2f}). Sell at close.",
                subtitle=sig['reason'],
            )

        # Log all signals
        all_trades = ", ".join(f"{s['trade_day']}({s['tier']})" for s in signals_found)
        log_signal(today_date.strftime('%Y-%m-%d'), today_name, color, all_trades)

    else:
        print(f"\n  ✅ No trade signals for tomorrow.")
        # Check why: wrong color, or just a skip day?
        next_day_info = NEXT_DAY_SIGNALS.get(today_dow)
        if next_day_info and next_day_info['condition'] is None:
            skip_tomorrow = DAY_NAMES[(today_dow + 1) % 7]
            reason = f"{skip_tomorrow} has no reliable edge — skip day"
        elif next_day_info and color != next_day_info['condition']:
            reason = f"Needed {next_day_info['condition']} {today_name}, got {color}"
        else:
            reason = "No matching signal"

        print(f"  Reason: {reason}")
        send_notification(
            "SPY Signal Checker",
            f"{today_name} {color} ({day_return:+.2f}%) — {reason}"
        )

    # Show upcoming context
    print(f"\n  --- Recent SPY Days ---")
    for i in range(max(0, len(spy)-5), len(spy)):
        row = spy.iloc[i]
        d = spy.index[i]
        ret = (row['Close'] - row['Open']) / row['Open'] * 100
        c = "🟢" if ret >= 0 else "🔴"
        print(f"  {c} {DAY_NAMES[d.dayofweek]:>10} {d.strftime('%m/%d')}: ${row['Open']:.2f} → ${row['Close']:.2f} ({ret:+.2f}%)")

    print(f"\n  Signal log: {LOG_FILE}")
    print("  Done!")


if __name__ == "__main__":
    check_signals()
