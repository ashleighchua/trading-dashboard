"""
Daily Multi-ETF Signal Checker
================================
Run this every night at 9:00 PM Thailand time (30 min before US open).
Scans SPY, QQQ, IWM for all active signals and tells you exactly what to trade.

Sends macOS notifications with sound for each signal found.

Usage:
  python3 daily_signal_checker.py          # Normal run
  python3 daily_signal_checker.py --test   # Test with sample notifications

Signals scanned (all backtested on 10 years):

  TIER 1 — SPY Day-of-Week (60%+ WR):
    Mon: Fri Red >= 0.5%           → 72.5% WR, SL 1.9%
    Wed: Mon Red + Tue Grn >= 0.5% → 68.0% WR, SL 1.9%
    Thu: Tue+Wed Grn >= 0.5%       → 65.5% WR, SL 1.1%
    Fri: Wed Grn >= 1.0%           → 65.6% WR, SL 1.9%

  TIER 1 — QQQ Day-of-Week (60%+ WR):
    Mon: Fri Red >= 1.0%           → 62.9% WR, SL 2.0%
    Tue: Mon Red >= 1.0%           → 63.5% WR, SL 2.0%
    Wed: Mon+Tue Red >= 0.5%       → 72.7% WR, SL 2.0%
    Thu: Tue+Wed Red >= 0.3%       → 60.0% WR, SL 2.0%
    Fri: Wed Grn + Thu Red >= 0.5% → 64.7% WR, SL 2.0%

  TIER 1 — IWM Day-of-Week (60%+ WR):
    Mon: Thu Grn + Fri Red >= 0.3% → 60.0% WR, SL 2.5%
    Tue: Fri Grn + Mon Red >= 0.7% → 63.6% WR, SL 2.5%
    Wed: Mon Red + Tue Grn >= 0.3% → 60.8% WR, SL 2.5%
    Thu: Tue+Wed Grn >= 0.7%       → 64.5% WR, SL 2.5%
    Fri: Wed+Thu Red >= 0.5%       → 62.0% WR, SL 2.5%

  TIER 2 — SPY Multi-Day Pairs:
    Red Mon → Buy Wed              → 58.4% WR, SL 1.9%
    Red Tue → Buy Fri              → 58.4% WR, SL 1.9%
    Red Wed → Buy Fri + Mon        → 58.4% WR, SL 1.9%
    Red Thu → Buy Fri              → 58.4% WR, SL 1.9%
    Red Fri → Buy Mon              → 58.4% WR, SL 1.9%

  TIER 3 — Backup (any day):
    Overnight Hold: Red Tue close → sell Wed open  → 64.1% WR
    2-Day Red Bounce: 2 red days in a row → buy    → 56.3% WR, SL 2.5%
    Gap Fill: SPY gaps down >= 0.3%                → 55.7% WR, SL 0.5%

  RARE BUT POWERFUL (checked every night):
    VIX Spike: VIX > 35 → Buy SPY (hold til VIX < 25) → 90.9% WR (~1-2/year)
    RSI Oversold: RSI(14) < 30 → Buy SPY             → 83.3% WR (~2/year)
    Earnings Drift: Jan/Apr/Jul/Oct → Buy SPY         → 78.0% WR (~4/year)
    Golden Cross: 50-SMA crosses above 200-SMA         → 75.0% WR (~2/year)
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "numpy"])
    import yfinance as yf
    import pandas as pd
    import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).parent / "signal_log.csv"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ── Notification ──────────────────────────────────────────────────────────────
def notify(title, message, subtitle="", sound=True):
    """Send a macOS notification."""
    title = title.replace('"', '\\"')
    message = message.replace('"', '\\"')
    subtitle = subtitle.replace('"', '\\"')
    snd = ' sound name "Glass"' if sound else ''
    sub = f' subtitle "{subtitle}"' if subtitle else ''
    script = f'display notification "{message}" with title "{title}"{sub}{snd}'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def log_signal(date, signals_text):
    """Append to signal log CSV."""
    header = not LOG_FILE.exists()
    with open(LOG_FILE, "a") as f:
        if header:
            f.write("date,time,signals\n")
        f.write(f'{date},{datetime.now().strftime("%H:%M")},"{signals_text}"\n')


# ── Data Helpers ──────────────────────────────────────────────────────────────
def get_return(df, offset=0):
    """Get intraday return % for a given day offset from latest."""
    idx = len(df) - 1 + offset
    if 0 <= idx < len(df):
        row = df.iloc[idx]
        return float(row['Return'])
    return 0.0

def get_color(df, offset=0):
    """Get color (Green/Red) for a given day offset from latest."""
    idx = len(df) - 1 + offset
    if 0 <= idx < len(df):
        return df.iloc[idx]['Color']
    return ''

def get_close(df, offset=0):
    """Get close price for a given day offset from latest."""
    idx = len(df) - 1 + offset
    if 0 <= idx < len(df):
        return float(df.iloc[idx]['Close'])
    return 0.0


# ── Signal Detection ─────────────────────────────────────────────────────────
def scan_signals(spy, qqq, iwm):
    """Scan all 3 ETFs for active signals. Returns list of signal dicts."""
    signals = []

    today_dow = int(spy.iloc[-1]['DayOfWeek'])
    today_name = DAY_NAMES[today_dow] if today_dow < 5 else 'Weekend'
    tomorrow_dow = (today_dow + 1) % 5 if today_dow < 4 else 0
    tomorrow_name = DAY_NAMES[tomorrow_dow]

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1: SPY DAY-OF-WEEK PLAYBOOK
    # ═══════════════════════════════════════════════════════════════════════

    # Monday: Fri Red >= 0.5%
    if tomorrow_dow == 0:
        if get_color(spy) == 'Red' and abs(get_return(spy)) >= 0.5:
            signals.append({
                'tier': 1, 'ticker': 'SPY', 'trade_day': 'Monday',
                'signal': f'Fri Red ({get_return(spy):+.2f}%) → Buy Mon open',
                'wr': 72.5, 'sl': 1.9, 'type': 'DAY-OF-WEEK',
            })

    # Wednesday: Mon Red + Tue Green >= 0.5%
    if tomorrow_dow == 2:
        if get_color(spy) == 'Green' and get_return(spy) >= 0.5 and get_color(spy, -1) == 'Red':
            signals.append({
                'tier': 1, 'ticker': 'SPY', 'trade_day': 'Wednesday',
                'signal': f'Mon Red + Tue Grn ({get_return(spy):+.2f}%) → Buy Wed',
                'wr': 68.0, 'sl': 1.9, 'type': 'DAY-OF-WEEK',
            })

    # Thursday: Tue+Wed Green >= 0.5%
    if tomorrow_dow == 3:
        if (get_color(spy) == 'Green' and get_return(spy) >= 0.5
                and get_color(spy, -1) == 'Green' and get_return(spy, -1) >= 0.5):
            signals.append({
                'tier': 1, 'ticker': 'SPY', 'trade_day': 'Thursday',
                'signal': f'Tue+Wed Green (both >=0.5%) → Buy Thu',
                'wr': 65.5, 'sl': 1.1, 'type': 'DAY-OF-WEEK',
            })

    # Friday: Wed Green >= 1.0%
    if tomorrow_dow == 4:
        if get_color(spy) == 'Green' and get_return(spy) >= 1.0:
            signals.append({
                'tier': 1, 'ticker': 'SPY', 'trade_day': 'Friday',
                'signal': f'Wed Grn ({get_return(spy):+.2f}%) → Buy Fri',
                'wr': 65.6, 'sl': 1.9, 'type': 'DAY-OF-WEEK',
            })

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1: QQQ DAY-OF-WEEK PLAYBOOK
    # ═══════════════════════════════════════════════════════════════════════

    # Monday: Fri Red >= 1.0%
    if tomorrow_dow == 0:
        if get_color(qqq) == 'Red' and abs(get_return(qqq)) >= 1.0:
            signals.append({
                'tier': 1, 'ticker': 'QQQ', 'trade_day': 'Monday',
                'signal': f'Fri Red ({get_return(qqq):+.2f}%) → Buy Mon',
                'wr': 62.9, 'sl': 2.0, 'type': 'DAY-OF-WEEK',
            })

    # Tuesday: Mon Red >= 1.0%
    if tomorrow_dow == 1:
        if get_color(qqq) == 'Red' and abs(get_return(qqq)) >= 1.0:
            signals.append({
                'tier': 1, 'ticker': 'QQQ', 'trade_day': 'Tuesday',
                'signal': f'Mon Red ({get_return(qqq):+.2f}%) → Buy Tue',
                'wr': 63.5, 'sl': 2.0, 'type': 'DAY-OF-WEEK',
            })

    # Wednesday: Mon+Tue Red >= 0.5%
    if tomorrow_dow == 2:
        if (get_color(qqq) == 'Red' and abs(get_return(qqq)) >= 0.5
                and get_color(qqq, -1) == 'Red' and abs(get_return(qqq, -1)) >= 0.5):
            signals.append({
                'tier': 1, 'ticker': 'QQQ', 'trade_day': 'Wednesday',
                'signal': f'Mon+Tue Red (both >=0.5%) → Buy Wed',
                'wr': 72.7, 'sl': 2.0, 'type': 'DAY-OF-WEEK',
            })

    # Thursday: Tue+Wed Red >= 0.3%
    if tomorrow_dow == 3:
        if (get_color(qqq) == 'Red' and abs(get_return(qqq)) >= 0.3
                and get_color(qqq, -1) == 'Red' and abs(get_return(qqq, -1)) >= 0.3):
            signals.append({
                'tier': 1, 'ticker': 'QQQ', 'trade_day': 'Thursday',
                'signal': f'Tue+Wed Red (both >=0.3%) → Buy Thu',
                'wr': 60.0, 'sl': 2.0, 'type': 'DAY-OF-WEEK',
            })

    # Friday: Wed Green + Thu Red >= 0.5%
    if tomorrow_dow == 4:
        if (get_color(qqq) == 'Red' and abs(get_return(qqq)) >= 0.5
                and get_color(qqq, -1) == 'Green' and get_return(qqq, -1) >= 0.5):
            signals.append({
                'tier': 1, 'ticker': 'QQQ', 'trade_day': 'Friday',
                'signal': f'Wed Grn + Thu Red → Buy Fri',
                'wr': 64.7, 'sl': 2.0, 'type': 'DAY-OF-WEEK',
            })

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 1: IWM DAY-OF-WEEK PLAYBOOK
    # ═══════════════════════════════════════════════════════════════════════

    # Monday: Thu Green + Fri Red >= 0.3%
    if tomorrow_dow == 0:
        if (get_color(iwm) == 'Red' and abs(get_return(iwm)) >= 0.3
                and get_color(iwm, -1) == 'Green' and get_return(iwm, -1) >= 0.3):
            signals.append({
                'tier': 1, 'ticker': 'IWM', 'trade_day': 'Monday',
                'signal': f'Thu Grn + Fri Red → Buy Mon',
                'wr': 60.0, 'sl': 2.5, 'type': 'DAY-OF-WEEK',
            })

    # Tuesday: Fri Green + Mon Red >= 0.7%
    if tomorrow_dow == 1:
        if (get_color(iwm) == 'Red' and abs(get_return(iwm)) >= 0.7
                and get_color(iwm, -1) == 'Green' and get_return(iwm, -1) >= 0.7):
            signals.append({
                'tier': 1, 'ticker': 'IWM', 'trade_day': 'Tuesday',
                'signal': f'Fri Grn + Mon Red → Buy Tue',
                'wr': 63.6, 'sl': 2.5, 'type': 'DAY-OF-WEEK',
            })

    # Wednesday: Mon Red + Tue Green >= 0.3%
    if tomorrow_dow == 2:
        if get_color(iwm) == 'Green' and get_return(iwm) >= 0.3 and get_color(iwm, -1) == 'Red':
            signals.append({
                'tier': 1, 'ticker': 'IWM', 'trade_day': 'Wednesday',
                'signal': f'Mon Red + Tue Grn → Buy Wed',
                'wr': 60.8, 'sl': 2.5, 'type': 'DAY-OF-WEEK',
            })

    # Thursday: Tue+Wed Green >= 0.7%
    if tomorrow_dow == 3:
        if (get_color(iwm) == 'Green' and get_return(iwm) >= 0.7
                and get_color(iwm, -1) == 'Green' and get_return(iwm, -1) >= 0.7):
            signals.append({
                'tier': 1, 'ticker': 'IWM', 'trade_day': 'Thursday',
                'signal': f'Tue+Wed Green (both >=0.7%) → Buy Thu',
                'wr': 64.5, 'sl': 2.5, 'type': 'DAY-OF-WEEK',
            })

    # Friday: Wed+Thu Red >= 0.5%
    if tomorrow_dow == 4:
        if (get_color(iwm) == 'Red' and abs(get_return(iwm)) >= 0.5
                and get_color(iwm, -1) == 'Red' and abs(get_return(iwm, -1)) >= 0.5):
            signals.append({
                'tier': 1, 'ticker': 'IWM', 'trade_day': 'Friday',
                'signal': f'Wed+Thu Red → Buy Fri',
                'wr': 62.0, 'sl': 2.5, 'type': 'DAY-OF-WEEK',
            })

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 2: SPY MULTI-DAY PAIR SIGNALS
    # ═══════════════════════════════════════════════════════════════════════
    multi_day_map = {
        0: ['Wednesday'],         # Red Monday → Buy Wednesday
        1: ['Friday'],            # Red Tuesday → Buy Friday
        2: ['Friday', 'Monday'],  # Red Wednesday → Buy Friday + Monday
        3: ['Friday'],            # Red Thursday → Buy Friday
        4: ['Monday'],            # Red Friday → Buy Monday
    }

    if get_color(spy) == 'Red' and today_dow in multi_day_map:
        for trade_day in multi_day_map[today_dow]:
            # Don't duplicate if already covered by Tier 1
            already = any(s['ticker'] == 'SPY' and s['trade_day'] == trade_day for s in signals)
            if not already:
                signals.append({
                    'tier': 2, 'ticker': 'SPY', 'trade_day': trade_day,
                    'signal': f'Red {today_name} ({get_return(spy):+.2f}%) → Buy {trade_day}',
                    'wr': 58.4, 'sl': 1.9, 'type': 'MULTI-DAY',
                })

    # ═══════════════════════════════════════════════════════════════════════
    # TIER 3: BACKUP SIGNALS
    # ═══════════════════════════════════════════════════════════════════════

    # Overnight Hold: Buy at Red Tuesday close, sell Wednesday open
    if today_dow == 1 and get_color(spy) == 'Red':
        signals.append({
            'tier': 3, 'ticker': 'SPY', 'trade_day': 'Tue→Wed overnight',
            'signal': f'Red Tue ({get_return(spy):+.2f}%) → Buy close, sell Wed open',
            'wr': 64.1, 'sl': 0, 'type': 'OVERNIGHT',
        })

    # 2-Day Red Bounce
    if get_color(spy) == 'Red' and get_color(spy, -1) == 'Red':
        # Check if 3-day
        if get_color(spy, -2) == 'Red':
            signals.append({
                'tier': 3, 'ticker': 'SPY', 'trade_day': tomorrow_name,
                'signal': f'3 consecutive red days → Buy {tomorrow_name}',
                'wr': 55.3, 'sl': 1.0, 'type': 'RED-BOUNCE',
            })
        else:
            signals.append({
                'tier': 3, 'ticker': 'SPY', 'trade_day': tomorrow_name,
                'signal': f'2 consecutive red days → Buy {tomorrow_name}',
                'wr': 56.3, 'sl': 2.5, 'type': 'RED-BOUNCE',
            })

    # Gap Fill — can only check at market open, flag if today had gap down
    today_gap = 0
    if 'GapPct' in spy.columns:
        today_gap = float(spy.iloc[-1]['GapPct'])
    if today_gap <= -0.3:
        signals.append({
            'tier': 3, 'ticker': 'SPY', 'trade_day': f'{today_name} (was gap)',
            'signal': f'Today gapped down {today_gap:.2f}% — gap fill was active',
            'wr': 55.7, 'sl': 0.5, 'type': 'GAP-FILL',
        })

    # Sort: Tier 1 first, then by win rate
    signals.sort(key=lambda s: (s['tier'], -s['wr']))
    return signals


# ── Rare But Powerful Strategy Detection ─────────────────────────────────────
def calc_rsi(series, period=14):
    """Calculate RSI for a price series."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def scan_rare_signals():
    """Scan for rare but powerful signals: VIX Spike, RSI Oversold, Earnings Drift, MA Crossover."""
    rare = []

    # Need longer history for SMA(200) and RSI
    print("  Scanning rare strategies (VIX, RSI, MA, Earnings)...")
    spy_long = yf.download("SPY", period="1y", auto_adjust=True)
    if isinstance(spy_long.columns, pd.MultiIndex):
        spy_long.columns = spy_long.columns.droplevel('Ticker')

    vix = yf.download("^VIX", period="5d", auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel('Ticker')

    # ── VIX SPIKE: VIX > 35 → Buy SPY (90.9% WR) ──
    if len(vix) > 0:
        vix_close = float(vix.iloc[-1]['Close'])
        if vix_close > 35:
            rare.append({
                'name': 'VIX SPIKE',
                'signal': f'VIX at {vix_close:.1f} (> 35) → BUY SPY immediately!',
                'wr': 90.9, 'sl': 3.0,
                'action': 'Buy SPY at open, hold 1-5 days until VIX drops below 25',
                'icon': '🚨',
                'urgency': 'VERY HIGH',
            })
        elif vix_close > 28:
            rare.append({
                'name': 'VIX ELEVATED',
                'signal': f'VIX at {vix_close:.1f} — approaching spike zone (35+)',
                'wr': 0, 'sl': 0,
                'action': 'Watch closely. If VIX crosses 35, buy SPY immediately.',
                'icon': '👀',
                'urgency': 'WATCH',
            })

    # ── RSI OVERSOLD: RSI(14) < 30 → Buy SPY (83.3% WR) ──
    if len(spy_long) >= 14:
        spy_long['RSI'] = calc_rsi(spy_long['Close'])
        current_rsi = float(spy_long['RSI'].iloc[-1])
        if current_rsi < 30:
            rare.append({
                'name': 'RSI OVERSOLD',
                'signal': f'SPY RSI(14) = {current_rsi:.1f} (< 30) → BUY SPY!',
                'wr': 83.3, 'sl': 2.0,
                'action': 'Buy SPY at open, hold 2-5 days until RSI crosses back above 40',
                'icon': '🚨',
                'urgency': 'VERY HIGH',
            })
        elif current_rsi < 35:
            rare.append({
                'name': 'RSI LOW',
                'signal': f'SPY RSI(14) = {current_rsi:.1f} — approaching oversold (30)',
                'wr': 0, 'sl': 0,
                'action': 'Watch for further drop. Buy if RSI dips below 30.',
                'icon': '👀',
                'urgency': 'WATCH',
            })

    # ── MA CROSSOVER / GOLDEN CROSS: 50 SMA crosses above 200 SMA (75% WR) ──
    if len(spy_long) >= 200:
        spy_long['SMA50'] = spy_long['Close'].rolling(50).mean()
        spy_long['SMA200'] = spy_long['Close'].rolling(200).mean()

        sma50_now = float(spy_long['SMA50'].iloc[-1])
        sma200_now = float(spy_long['SMA200'].iloc[-1])
        sma50_prev = float(spy_long['SMA50'].iloc[-2])
        sma200_prev = float(spy_long['SMA200'].iloc[-2])

        # Golden Cross: 50 SMA just crossed above 200 SMA
        if sma50_prev <= sma200_prev and sma50_now > sma200_now:
            rare.append({
                'name': 'GOLDEN CROSS',
                'signal': f'SPY 50-SMA (${sma50_now:.0f}) crossed ABOVE 200-SMA (${sma200_now:.0f})!',
                'wr': 75.0, 'sl': 3.0,
                'action': 'Buy SPY at open, hold 5-20 days. Strong bullish signal.',
                'icon': '🚨',
                'urgency': 'HIGH',
            })
        # Death Cross: 50 SMA just crossed below 200 SMA
        elif sma50_prev >= sma200_prev and sma50_now < sma200_now:
            rare.append({
                'name': 'DEATH CROSS',
                'signal': f'SPY 50-SMA (${sma50_now:.0f}) crossed BELOW 200-SMA (${sma200_now:.0f})!',
                'wr': 0, 'sl': 0,
                'action': 'AVOID buying SPY. Bearish signal — consider reducing positions.',
                'icon': '⚠️',
                'urgency': 'WARNING',
            })
        else:
            # Show current status
            gap_pct = (sma50_now - sma200_now) / sma200_now * 100
            if abs(gap_pct) < 1.0:
                status = "CONVERGING" if abs(gap_pct) < abs((sma50_prev - sma200_prev) / sma200_prev * 100) else "near each other"
                rare.append({
                    'name': 'MA STATUS',
                    'signal': f'50-SMA (${sma50_now:.0f}) vs 200-SMA (${sma200_now:.0f}) — gap: {gap_pct:+.2f}%, {status}',
                    'wr': 0, 'sl': 0,
                    'action': 'Watch for potential crossover in coming days.',
                    'icon': '👀',
                    'urgency': 'WATCH',
                })

    # ── EARNINGS SEASON DRIFT: Buy SPY in Jan/Apr/Jul/Oct (78% WR) ──
    current_month = datetime.now().month
    earnings_months = {1: 'Q4', 4: 'Q1', 7: 'Q2', 10: 'Q3'}
    if current_month in earnings_months:
        quarter = earnings_months[current_month]
        rare.append({
            'name': 'EARNINGS SEASON',
            'signal': f'{quarter} earnings season ({datetime.now().strftime("%B")}) → Buy SPY!',
            'wr': 78.0, 'sl': 2.0,
            'action': 'Buy SPY at start of month, hold through earnings season (~3-4 weeks).',
            'icon': '📊',
            'urgency': 'ACTIVE',
        })
    # Heads-up for next month
    next_month = (current_month % 12) + 1
    if next_month in earnings_months:
        quarter = earnings_months[next_month]
        month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][next_month - 1]
        rare.append({
            'name': 'EARNINGS UPCOMING',
            'signal': f'{quarter} earnings season starts next month ({month_name})',
            'wr': 0, 'sl': 0,
            'action': f'Prepare to buy SPY at start of {month_name}.',
            'icon': '📅',
            'urgency': 'UPCOMING',
        })

    return rare


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now = datetime.now()
    print()
    print("=" * 66)
    print("  DAILY MULTI-ETF SIGNAL CHECKER")
    print(f"  {now.strftime('%Y-%m-%d %H:%M')} Thailand time")
    print("=" * 66)

    # Download data for all 3 ETFs
    print("\n  Downloading SPY, QQQ, IWM data...")
    data = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        df = yf.download(ticker, period="10d", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel('Ticker')
        df['DayOfWeek'] = df.index.dayofweek
        df['Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
        df['Color'] = df['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')
        df['PrevClose'] = df['Close'].shift(1)
        df['GapPct'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100
        data[ticker] = df

    spy, qqq, iwm = data['SPY'], data['QQQ'], data['IWM']

    if len(spy) == 0:
        print("\n  No data available. Market might be closed.")
        notify("Signal Checker", "No market data available")
        return

    # Show this week's candles
    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │                    THIS WEEK'S CANDLES                      │")
    print("  ├────────────┬─────────────┬─────────────┬────────────────────┤")
    print("  │    Day     │     SPY     │     QQQ     │       IWM          │")
    print("  ├────────────┼─────────────┼─────────────┼────────────────────┤")

    for i in range(max(0, len(spy) - 5), len(spy)):
        d = spy.index[i]
        dow = d.dayofweek
        if dow > 4:
            continue
        day = DAY_NAMES[dow]

        spy_ret = get_return(spy, i - len(spy) + 1)
        spy_icon = "🟢" if spy_ret >= 0 else "🔴"

        qqq_ret = get_return(qqq, i - len(qqq) + 1)
        qqq_icon = "🟢" if qqq_ret >= 0 else "🔴"

        iwm_ret = get_return(iwm, i - len(iwm) + 1)
        iwm_icon = "🟢" if iwm_ret >= 0 else "🔴"

        print(f"  │ {day:>10} │ {spy_icon} {spy_ret:+6.2f}%  │ {qqq_icon} {qqq_ret:+6.2f}%  │ {iwm_icon} {iwm_ret:+6.2f}%           │")

    print("  └────────────┴─────────────┴─────────────┴────────────────────┘")

    # Latest prices
    spy_price = get_close(spy)
    qqq_price = get_close(qqq)
    iwm_price = get_close(iwm)
    print(f"\n  Latest close: SPY ${spy_price:.2f}  |  QQQ ${qqq_price:.2f}  |  IWM ${iwm_price:.2f}")

    today_dow = int(spy.iloc[-1]['DayOfWeek'])
    today_name = DAY_NAMES[today_dow] if today_dow < 5 else 'Weekend'
    tomorrow_dow = (today_dow + 1) % 5 if today_dow < 4 else 0
    tomorrow_name = DAY_NAMES[tomorrow_dow]

    print(f"  Today (US): {today_name}  |  Next trading day: {tomorrow_name}")

    # Scan signals
    signals = scan_signals(spy, qqq, iwm)

    # ── OUTPUT ────────────────────────────────────────────────────────────
    if not signals:
        print("\n  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  ✅  NO DAY-OF-WEEK SIGNALS — checking rare strategies...   ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        notify("Signal Checker", f"No day-of-week signals for {tomorrow_name}.", f"{today_name} done")

        # Still check rare signals even if no regular signals
        rare_signals = scan_rare_signals()
        if rare_signals:
            actionable = [r for r in rare_signals if r['urgency'] in ('VERY HIGH', 'HIGH', 'ACTIVE')]
            watchlist = [r for r in rare_signals if r['urgency'] in ('WATCH', 'UPCOMING', 'WARNING')]

            if actionable:
                print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
                print(f"  ║  🚨  RARE SIGNALS ACTIVE — HIGH WIN RATE!                   ║")
                print(f"  ╚══════════════════════════════════════════════════════════════╝")
                for r in actionable:
                    print(f"\n  {r['icon']} [{r['name']}] — {r['wr']}% Win Rate")
                    print(f"     {r['signal']}")
                    print(f"     Action: {r['action']}")
                    if r['sl'] > 0:
                        print(f"     Stop Loss: {r['sl']}%")
                    notify(
                        title=f"{r['icon']} RARE: {r['name']}! ({r['wr']}% WR)",
                        message=r['signal'],
                        subtitle=r['action'],
                    )

            if watchlist:
                print(f"\n  ── WATCHLIST (not active yet) ───────────────────────────────")
                for r in watchlist:
                    print(f"  {r['icon']} [{r['name']}] {r['signal']}")
                    if r['action']:
                        print(f"     → {r['action']}")
        else:
            print(f"  No rare signals either. Full rest day.")

        # Log
        rare_active = [f"RARE:{r['name']}({r['wr']}%)" for r in rare_signals if r['urgency'] in ('VERY HIGH', 'HIGH', 'ACTIVE')]
        sig_summary = " | ".join(rare_active) if rare_active else "No signals"
        log_signal(spy.index[-1].strftime('%Y-%m-%d'), sig_summary)

        print(f"\n  Signal log: {LOG_FILE}")
        print(f"  Done! ✅")
        print()
        return

    tier1 = [s for s in signals if s['tier'] == 1]
    tier2 = [s for s in signals if s['tier'] == 2]
    tier3 = [s for s in signals if s['tier'] == 3]

    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  🔔  {len(signals)} SIGNAL(S) DETECTED!                               ║")
    print(f"  ╚══════════════════════════════════════════════════════════════╝")

    # ── TIER 1 ──
    if tier1:
        print(f"\n  ── TIER 1: PRIMARY SIGNALS (Trade these!) ──────────────────")
        for i, s in enumerate(tier1, 1):
            price = {'SPY': spy_price, 'QQQ': qqq_price, 'IWM': iwm_price}[s['ticker']]
            sl_price = price * (1 - s['sl'] / 100) if s['sl'] > 0 else 0

            print(f"\n  [{i}] {s['ticker']} — {s['signal']}")
            print(f"      Win Rate: {s['wr']}%  |  Stop Loss: {s['sl']}% (~${sl_price:.2f})")
            print(f"      Action:   BUY {s['ticker']} at {s['trade_day']} market open")
            print(f"      Exit:     Sell at {s['trade_day']} market close")
            print(f"      Est. Entry: ~${price:.2f}  |  SL Price: ~${sl_price:.2f}")
            print(f"      Risk per share: ~${price - sl_price:.2f}")

            # Send notification
            notify(
                title=f"🟢 BUY {s['ticker']} {s['trade_day']}! ({s['wr']}% WR)",
                message=f"{s['signal']}. SL: {s['sl']}% (~${sl_price:.2f}). Sell at close.",
                subtitle=f"Tier 1 | {s['type']}",
            )

    # ── TIER 2 ──
    if tier2:
        print(f"\n  ── TIER 2: SECONDARY SIGNALS ────────────────────────────────")
        for s in tier2:
            price = spy_price
            sl_price = price * (1 - s['sl'] / 100)
            print(f"\n  [T2] {s['ticker']} — {s['signal']}")
            print(f"       WR: {s['wr']}%  |  SL: {s['sl']}% (~${sl_price:.2f})")

            notify(
                title=f"🟡 {s['ticker']}: {s['trade_day']} ({s['wr']}% WR)",
                message=f"{s['signal']}. SL: {s['sl']}%.",
                subtitle="Tier 2 | Multi-day",
            )

    # ── TIER 3 ──
    if tier3:
        print(f"\n  ── TIER 3: BACKUP SIGNALS ───────────────────────────────────")
        for s in tier3:
            print(f"\n  [T3] {s['ticker']} — {s['signal']}")
            print(f"       WR: {s['wr']}%  |  SL: {s['sl'] if s['sl'] else 'None'}%")

    # ── TRADE PLAN SUMMARY ──
    print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║                    TONIGHT'S TRADE PLAN                     ║")
    print(f"  ╠══════════════════════════════════════════════════════════════╣")

    if tier1:
        best = tier1[0]
        price = {'SPY': spy_price, 'QQQ': qqq_price, 'IWM': iwm_price}[best['ticker']]
        sl_price = price * (1 - best['sl'] / 100) if best['sl'] > 0 else 0

        print(f"  ║  BEST SIGNAL: {best['ticker']} — {best['wr']}% win rate              ")
        print(f"  ║                                                            ║")
        print(f"  ║  1. Open TradingView at 9:30 PM TH time                    ║")
        print(f"  ║  2. Buy 1 {best['ticker']} at MARKET ORDER                        ║")
        print(f"  ║  3. Set stop loss at {best['sl']}% below entry (~${sl_price:.2f})       ")
        print(f"  ║  4. Set alarm for 3:45 AM TH time                          ║")
        print(f"  ║  5. Sell at market close (4:00 AM TH)                      ║")
        print(f"  ╚══════════════════════════════════════════════════════════════╝")

        if len(tier1) > 1:
            print(f"\n  💡 You have {len(tier1)} Tier 1 signals. Pick the one with highest WR")
            print(f"     or spread across multiple ETFs if you have enough capital.")
    else:
        best = tier2[0] if tier2 else tier3[0]
        print(f"  ║  No Tier 1 signals. Best backup: {best['ticker']} ({best['wr']}% WR) ")
        print(f"  ║  Consider skipping or using backup signal cautiously.      ║")
        print(f"  ╚══════════════════════════════════════════════════════════════╝")

    # ═══════════════════════════════════════════════════════════════════════
    # RARE BUT POWERFUL STRATEGIES
    # ═══════════════════════════════════════════════════════════════════════
    rare_signals = scan_rare_signals()

    if rare_signals:
        actionable = [r for r in rare_signals if r['urgency'] in ('VERY HIGH', 'HIGH', 'ACTIVE')]
        watchlist = [r for r in rare_signals if r['urgency'] in ('WATCH', 'UPCOMING', 'WARNING')]

        if actionable:
            print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
            print(f"  ║  🚨  RARE SIGNALS ACTIVE — HIGH WIN RATE!                   ║")
            print(f"  ╚══════════════════════════════════════════════════════════════╝")

            for r in actionable:
                print(f"\n  {r['icon']} [{r['name']}] — {r['wr']}% Win Rate")
                print(f"     {r['signal']}")
                print(f"     Action: {r['action']}")
                if r['sl'] > 0:
                    print(f"     Stop Loss: {r['sl']}%")

                # Send notification for actionable rare signals
                notify(
                    title=f"{r['icon']} RARE: {r['name']}! ({r['wr']}% WR)",
                    message=r['signal'],
                    subtitle=r['action'],
                )

        if watchlist:
            print(f"\n  ── WATCHLIST (not active yet) ───────────────────────────────")
            for r in watchlist:
                print(f"  {r['icon']} [{r['name']}] {r['signal']}")
                if r['action']:
                    print(f"     → {r['action']}")
    else:
        print(f"\n  ── Rare Strategies: No special conditions detected ──────────")

    # Log
    all_sigs = [f"{s['ticker']}:{s['trade_day']}({s['wr']}%)" for s in signals]
    rare_active = [f"RARE:{r['name']}({r['wr']}%)" for r in rare_signals if r['urgency'] in ('VERY HIGH', 'HIGH', 'ACTIVE')]
    all_sigs.extend(rare_active)
    sig_summary = " | ".join(all_sigs) if all_sigs else "No signals"
    log_signal(spy.index[-1].strftime('%Y-%m-%d'), sig_summary)

    print(f"\n  Signal log: {LOG_FILE}")
    print(f"  Done! ✅")
    print()


if __name__ == "__main__":
    if "--test" in sys.argv:
        print("Sending test notification...")
        notify("Signal Checker Test", "This is a test notification!", "Test mode")
        print("Done!")
    else:
        main()
