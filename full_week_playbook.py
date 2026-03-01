"""
Full Week Playbook — Find Best Signal For Every Day
=====================================================
For each day Mon–Fri, exhaustively tests all combinations of:
  - Previous 1, 2, or 3 days' colors
  - Magnitude thresholds (0%, 0.3%, 0.5%, 0.7%, 1.0%)
  - Long only (going with the signal)

Goal: Find signals with >60% win rate and meaningful sample size (n>=20)
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# ── Download SPY data ─────────────────────────────────────────────────────────
print("Downloading SPY data (10 years)...")
spy = yf.download("SPY", period="10y", auto_adjust=True)
spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy

spy['DayOfWeek'] = spy.index.dayofweek
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

print(f"Data: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)")

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# ── Build full week rows: each row = Mon through Fri of same week ─────────────
def build_week_data(spy):
    """Build a dataframe where each row is one week with all 5 days."""
    rows = []
    mondays = spy[spy['DayOfWeek'] == 0]

    for mon_date, mon_row in mondays.iterrows():
        week = {'Mon_Date': mon_date}
        current = mon_date
        valid = True

        for dow, day in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']):
            day_data = spy[(spy.index >= current) & (spy['DayOfWeek'] == dow)]
            day_data = day_data[day_data.index >= mon_date]
            if len(day_data) == 0:
                valid = False
                break
            row = day_data.iloc[0]
            if (row.name - mon_date).days > 6:
                valid = False
                break
            week[f'{day}_Date']   = row.name
            week[f'{day}_Return'] = row['Return']
            week[f'{day}_Color']  = row['Color']
            week[f'{day}_Open']   = row['Open']
            week[f'{day}_High']   = row['High']
            week[f'{day}_Low']    = row['Low']
            week[f'{day}_Close']  = row['Close']
            current = row.name

        if valid and all(f'{d}_Return' in week for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']):
            rows.append(week)

    return pd.DataFrame(rows)

print("Building weekly data...")
weeks = build_week_data(spy)
print(f"Complete weeks: {len(weeks)}")

# ── Also build cross-week pairs (Fri → next Mon) ─────────────────────────────
def build_cross_week(spy):
    fridays = spy[spy['DayOfWeek'] == 4]
    mondays = spy[spy['DayOfWeek'] == 0]
    rows = []
    for fri_date, fri_row in fridays.iterrows():
        next_mons = mondays[mondays.index > fri_date]
        if len(next_mons) == 0:
            continue
        mon = next_mons.iloc[0]
        if (mon.name - fri_date).days > 5:
            continue
        rows.append({
            'Fri_Date':    fri_date,
            'Fri_Return':  fri_row['Return'],
            'Fri_Color':   fri_row['Color'],
            'Mon_Date':    mon.name,
            'Mon_Return':  mon['Return'],
            'Mon_Color':   mon['Color'],
            'Mon_Open':    mon['Open'],
            'Mon_High':    mon['High'],
            'Mon_Low':     mon['Low'],
            'Mon_Close':   mon['Close'],
        })
    return pd.DataFrame(rows)

cross = build_cross_week(spy)

# ── EXHAUSTIVE SIGNAL SEARCH ──────────────────────────────────────────────────
print("\n" + "="*70)
print("EXHAUSTIVE SIGNAL SEARCH — All Days, All Combinations")
print("="*70)

MIN_SAMPLE = 20
MIN_WIN_RATE = 60.0
thresholds = [0.0, 0.3, 0.5, 0.7, 1.0]

best_signals = {}  # day → list of valid signals

# ─── MONDAY (signal from previous Friday, cross-week) ────────────────────────
print(f"\n{'─'*70}")
print(f"MONDAY — Testing signals from previous Friday")
print(f"{'─'*70}")
print(f"Baseline Monday green rate: {(cross['Mon_Color']=='Green').sum()/len(cross)*100:.1f}%\n")

mon_signals = []
for color in ['Red', 'Green']:
    for thresh in thresholds:
        if color == 'Red':
            mask = cross['Fri_Return'] <= -thresh
        else:
            mask = cross['Fri_Return'] >= thresh

        sub = cross[mask]
        if len(sub) < MIN_SAMPLE:
            continue
        wr = (sub['Mon_Color'] == 'Green').sum() / len(sub) * 100
        avg = sub['Mon_Return'].mean()
        if wr >= MIN_WIN_RATE:
            label = f"Fri {color} >= {thresh}%" if thresh > 0 else f"Fri {color}"
            mon_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg,
                                 'color': color, 'thresh': thresh, 'day': 'Monday'})
            print(f"  ✅ {label:<30} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

if not mon_signals:
    print("  ❌ No signals found with >60% win rate for Monday")
best_signals['Monday'] = sorted(mon_signals, key=lambda x: x['win_rate'], reverse=True)

# ─── TUESDAY ────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"TUESDAY — Testing signals from Monday")
print(f"{'─'*70}")
print(f"Baseline Tuesday green rate: {(weeks['Tue_Color']=='Green').sum()/len(weeks)*100:.1f}%\n")

tue_signals = []
for color in ['Red', 'Green']:
    for thresh in thresholds:
        if color == 'Red':
            mask = weeks['Mon_Return'] <= -thresh
        else:
            mask = weeks['Mon_Return'] >= thresh

        sub = weeks[mask]
        if len(sub) < MIN_SAMPLE:
            continue
        wr = (sub['Tue_Color'] == 'Green').sum() / len(sub) * 100
        avg = sub['Tue_Return'].mean()
        if wr >= MIN_WIN_RATE:
            label = f"Mon {color} >= {thresh}%" if thresh > 0 else f"Mon {color}"
            tue_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg,
                                  'color': color, 'thresh': thresh, 'day': 'Tuesday'})
            print(f"  ✅ {label:<30} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

if not tue_signals:
    print("  ❌ No single-day signals found. Testing 2-day combinations...")
    # Test Fri+Mon combinations
    for fri_color in ['Red', 'Green']:
        for mon_color in ['Red', 'Green']:
            for thresh in thresholds:
                merged = pd.merge(cross[['Mon_Date', 'Fri_Return', 'Fri_Color']],
                                  weeks[['Mon_Date', 'Tue_Color', 'Tue_Return', 'Mon_Return']],
                                  on='Mon_Date')
                if fri_color == 'Red':
                    m1 = merged['Fri_Return'] <= -thresh
                else:
                    m1 = merged['Fri_Return'] >= thresh
                if mon_color == 'Red':
                    m2 = merged['Mon_Return'] <= -thresh
                else:
                    m2 = merged['Mon_Return'] >= thresh

                sub = merged[m1 & m2]
                if len(sub) < MIN_SAMPLE:
                    continue
                wr = (sub['Tue_Color'] == 'Green').sum() / len(sub) * 100
                avg = sub['Tue_Return'].mean()
                if wr >= MIN_WIN_RATE:
                    label = f"Fri {fri_color} + Mon {mon_color} >={thresh}%"
                    tue_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg, 'day': 'Tuesday'})
                    print(f"  ✅ {label:<35} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

if not tue_signals:
    print("  ❌ No signals found with >60% win rate for Tuesday")
best_signals['Tuesday'] = sorted(tue_signals, key=lambda x: x['win_rate'], reverse=True)

# ─── WEDNESDAY ────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"WEDNESDAY — Testing signals from Mon, Tue, Mon+Tue")
print(f"{'─'*70}")
print(f"Baseline Wednesday green rate: {(weeks['Wed_Color']=='Green').sum()/len(weeks)*100:.1f}%\n")

wed_signals = []

# Single day signals
for prev_day, prev_col in [('Mon', 'Mon_Return'), ('Tue', 'Tue_Return')]:
    for color in ['Red', 'Green']:
        for thresh in thresholds:
            if color == 'Red':
                mask = weeks[prev_col] <= -thresh
            else:
                mask = weeks[prev_col] >= thresh
            sub = weeks[mask]
            if len(sub) < MIN_SAMPLE:
                continue
            wr = (sub['Wed_Color'] == 'Green').sum() / len(sub) * 100
            avg = sub['Wed_Return'].mean()
            if wr >= MIN_WIN_RATE:
                label = f"{prev_day} {color} >= {thresh}%" if thresh > 0 else f"{prev_day} {color}"
                wed_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg, 'day': 'Wednesday'})
                print(f"  ✅ {label:<30} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

# Two day combinations Mon+Tue
for mon_color in ['Red', 'Green']:
    for tue_color in ['Red', 'Green']:
        for thresh in thresholds:
            if mon_color == 'Red':
                m1 = weeks['Mon_Return'] <= -thresh
            else:
                m1 = weeks['Mon_Return'] >= thresh
            if tue_color == 'Red':
                m2 = weeks['Tue_Return'] <= -thresh
            else:
                m2 = weeks['Tue_Return'] >= thresh
            sub = weeks[m1 & m2]
            if len(sub) < MIN_SAMPLE:
                continue
            wr = (sub['Wed_Color'] == 'Green').sum() / len(sub) * 100
            avg = sub['Wed_Return'].mean()
            if wr >= MIN_WIN_RATE:
                label = f"Mon {mon_color} + Tue {tue_color} >={thresh}%"
                # avoid duplicates
                if not any(s['signal'] == label for s in wed_signals):
                    wed_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg, 'day': 'Wednesday'})
                    print(f"  ✅ {label:<35} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

if not wed_signals:
    print("  ❌ No signals found with >60% win rate for Wednesday")
best_signals['Wednesday'] = sorted(wed_signals, key=lambda x: x['win_rate'], reverse=True)

# ─── THURSDAY ────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"THURSDAY — Testing signals from Tue, Wed, Tue+Wed")
print(f"{'─'*70}")
print(f"Baseline Thursday green rate: {(weeks['Thu_Color']=='Green').sum()/len(weeks)*100:.1f}%\n")

thu_signals = []

# Single day signals
for prev_day, prev_col in [('Tue', 'Tue_Return'), ('Wed', 'Wed_Return')]:
    for color in ['Red', 'Green']:
        for thresh in thresholds:
            if color == 'Red':
                mask = weeks[prev_col] <= -thresh
            else:
                mask = weeks[prev_col] >= thresh
            sub = weeks[mask]
            if len(sub) < MIN_SAMPLE:
                continue
            wr = (sub['Thu_Color'] == 'Green').sum() / len(sub) * 100
            avg = sub['Thu_Return'].mean()
            if wr >= MIN_WIN_RATE:
                label = f"{prev_day} {color} >= {thresh}%" if thresh > 0 else f"{prev_day} {color}"
                thu_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg, 'day': 'Thursday'})
                print(f"  ✅ {label:<30} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

# Two day combinations
for tue_color in ['Red', 'Green']:
    for wed_color in ['Red', 'Green']:
        for thresh in thresholds:
            if tue_color == 'Red':
                m1 = weeks['Tue_Return'] <= -thresh
            else:
                m1 = weeks['Tue_Return'] >= thresh
            if wed_color == 'Red':
                m2 = weeks['Wed_Return'] <= -thresh
            else:
                m2 = weeks['Wed_Return'] >= thresh
            sub = weeks[m1 & m2]
            if len(sub) < MIN_SAMPLE:
                continue
            wr = (sub['Thu_Color'] == 'Green').sum() / len(sub) * 100
            avg = sub['Thu_Return'].mean()
            if wr >= MIN_WIN_RATE:
                label = f"Tue {tue_color} + Wed {wed_color} >={thresh}%"
                if not any(s['signal'] == label for s in thu_signals):
                    thu_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg, 'day': 'Thursday'})
                    print(f"  ✅ {label:<35} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

if not thu_signals:
    print("  ❌ No signals found with >60% win rate for Thursday")
best_signals['Thursday'] = sorted(thu_signals, key=lambda x: x['win_rate'], reverse=True)

# ─── FRIDAY ────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"FRIDAY — Testing signals from Wed, Thu, Wed+Thu")
print(f"{'─'*70}")
print(f"Baseline Friday green rate: {(weeks['Fri_Color']=='Green').sum()/len(weeks)*100:.1f}%\n")

fri_signals = []

# Single day signals
for prev_day, prev_col in [('Wed', 'Wed_Return'), ('Thu', 'Thu_Return')]:
    for color in ['Red', 'Green']:
        for thresh in thresholds:
            if color == 'Red':
                mask = weeks[prev_col] <= -thresh
            else:
                mask = weeks[prev_col] >= thresh
            sub = weeks[mask]
            if len(sub) < MIN_SAMPLE:
                continue
            wr = (sub['Fri_Color'] == 'Green').sum() / len(sub) * 100
            avg = sub['Fri_Return'].mean()
            if wr >= MIN_WIN_RATE:
                label = f"{prev_day} {color} >= {thresh}%" if thresh > 0 else f"{prev_day} {color}"
                fri_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg, 'day': 'Friday'})
                print(f"  ✅ {label:<30} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

# Two day combinations
for wed_color in ['Red', 'Green']:
    for thu_color in ['Red', 'Green']:
        for thresh in thresholds:
            if wed_color == 'Red':
                m1 = weeks['Wed_Return'] <= -thresh
            else:
                m1 = weeks['Wed_Return'] >= thresh
            if thu_color == 'Red':
                m2 = weeks['Thu_Return'] <= -thresh
            else:
                m2 = weeks['Thu_Return'] >= thresh
            sub = weeks[m1 & m2]
            if len(sub) < MIN_SAMPLE:
                continue
            wr = (sub['Fri_Color'] == 'Green').sum() / len(sub) * 100
            avg = sub['Fri_Return'].mean()
            if wr >= MIN_WIN_RATE:
                label = f"Wed {wed_color} + Thu {thu_color} >={thresh}%"
                if not any(s['signal'] == label for s in fri_signals):
                    fri_signals.append({'signal': label, 'n': len(sub), 'win_rate': wr, 'avg_ret': avg, 'day': 'Friday'})
                    print(f"  ✅ {label:<35} n={len(sub):>4}  WR={wr:.1f}%  AvgRet={avg:+.3f}%")

if not fri_signals:
    print("  ❌ No signals found with >60% win rate for Friday")
best_signals['Friday'] = sorted(fri_signals, key=lambda x: x['win_rate'], reverse=True)

# ── FULL PLAYBOOK SUMMARY ─────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("FULL WEEK PLAYBOOK — Best Signal Per Day (>60% Win Rate)")
print(f"{'='*70}")

playbook = []
for day in DAY_NAMES:
    sigs = best_signals.get(day, [])
    # Pick best signal with largest sample size among top win rates
    # Filter to n>=20 and pick highest win rate, tie-break by sample size
    valid = [s for s in sigs if s['n'] >= MIN_SAMPLE]
    if valid:
        best = valid[0]  # already sorted by win_rate desc
        playbook.append(best)
        print(f"\n  {day.upper()}")
        print(f"    Signal:   {best['signal']}")
        print(f"    Win Rate: {best['win_rate']:.1f}%")
        print(f"    Avg Return: {best['avg_ret']:+.3f}%")
        print(f"    Sample: {best['n']} trades")
    else:
        playbook.append(None)
        print(f"\n  {day.upper()}")
        print(f"    ❌ No signal found with >60% win rate and n>=20")

# ── BACKTEST BEST SIGNAL PER DAY ──────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("BACKTEST — Best Signal Per Day, 1.5% Stop Loss")
print(f"{'='*70}")

STOP_LOSS = 1.5
day_col_map = {
    'Monday':    ('Mon_Open', 'Mon_High', 'Mon_Low', 'Mon_Close', 'Mon_Return'),
    'Tuesday':   ('Tue_Open', 'Tue_High', 'Tue_Low', 'Tue_Close', 'Tue_Return'),
    'Wednesday': ('Wed_Open', 'Wed_High', 'Wed_Low', 'Wed_Close', 'Wed_Return'),
    'Thursday':  ('Thu_Open', 'Thu_High', 'Thu_Low', 'Thu_Close', 'Thu_Return'),
    'Friday':    ('Fri_Open', 'Fri_High', 'Fri_Low', 'Fri_Close', 'Fri_Return'),
}

all_equity = []
all_bt = []
capital = 10000
initial = capital

for day, sig in zip(DAY_NAMES, playbook):
    if sig is None:
        continue

    open_col, high_col, low_col, close_col, ret_col = day_col_map[day]

    # Rebuild the signal mask
    signal_label = sig['signal']

    # Monday uses cross-week data
    if day == 'Monday':
        color = sig.get('color', 'Red')
        thresh = sig.get('thresh', 0.5)
        if color == 'Red':
            mask = cross['Fri_Return'] <= -thresh
        else:
            mask = cross['Fri_Return'] >= thresh
        trade_data = cross[mask][['Mon_Date', 'Mon_Open', 'Mon_High', 'Mon_Low', 'Mon_Close']].copy()
        trade_data = trade_data.rename(columns={'Mon_Date': 'Date', 'Mon_Open': 'Open',
                                                 'Mon_High': 'High', 'Mon_Low': 'Low', 'Mon_Close': 'Close'})
    else:
        # Parse signal to get mask
        parts = signal_label.split('+')
        mask = pd.Series([True] * len(weeks), index=weeks.index)
        for part in parts:
            part = part.strip()
            tokens = part.split()
            prev_d = tokens[0]  # Mon, Tue, Wed, Thu, Fri
            color = tokens[1]   # Red or Green
            thresh_str = tokens[-1].replace('%', '').replace('>=', '')
            try:
                thresh = float(thresh_str)
            except:
                thresh = 0.0
            col = f'{prev_d}_Return'
            if col not in weeks.columns:
                mask = mask & pd.Series([False] * len(weeks), index=weeks.index)
                continue
            if color == 'Red':
                mask = mask & (weeks[col] <= -thresh)
            else:
                mask = mask & (weeks[col] >= thresh)

        trade_data = weeks[mask][[f'{day[:3]}_Date', open_col, high_col, low_col, close_col]].copy()
        trade_data.columns = ['Date', 'Open', 'High', 'Low', 'Close']

    trades = []
    for _, row in trade_data.iterrows():
        entry = row['Open']
        stop_price = entry * (1 - STOP_LOSS / 100)
        if row['Low'] <= stop_price:
            exit_price = stop_price
            exit_type = 'STOP'
        else:
            exit_price = row['Close']
            exit_type = 'CLOSE'
        ret = (exit_price - entry) / entry
        pnl = capital * ret
        capital += pnl
        trades.append({
            'Date': row['Date'], 'Day': day, 'Signal': signal_label,
            'Entry': entry, 'Exit': exit_price, 'Exit_Type': exit_type,
            'Return_Pct': ret * 100, 'PnL': pnl, 'Capital': capital, 'Win': ret > 0,
        })
        all_equity.append({'Date': row['Date'], 'Capital': capital})

    if trades:
        t = pd.DataFrame(trades)
        all_bt.extend(trades)
        wr = t['Win'].mean() * 100
        cum = t['Return_Pct'].sum()
        print(f"\n  {day} ({signal_label})")
        print(f"    Trades: {len(t)} | Win Rate: {wr:.1f}% | Cum Return: {cum:+.2f}%")

# Combined stats
if all_bt:
    bt = pd.DataFrame(all_bt).sort_values('Date').reset_index(drop=True)
    eq = pd.DataFrame(all_equity).sort_values('Date').reset_index(drop=True)
    eq['Peak'] = eq['Capital'].cummax()
    eq['DD'] = (eq['Capital'] - eq['Peak']) / eq['Peak'] * 100

    total_ret = (capital - initial) / initial * 100
    years = (bt['Date'].iloc[-1] - bt['Date'].iloc[0]).days / 365.25
    annual_ret = ((capital / initial) ** (1 / years) - 1) * 100
    overall_wr = bt['Win'].mean() * 100
    max_dd = eq['DD'].min()
    sharpe = (bt['Return_Pct'].mean() / bt['Return_Pct'].std()) * np.sqrt(len(bt) / years) if bt['Return_Pct'].std() > 0 else 0

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  COMBINED PLAYBOOK PERFORMANCE                       │
  ├──────────────────────────────────────────────────────┤
  │  Starting Capital:     ${initial:>10,.2f}             │
  │  Ending Capital:       ${capital:>10,.2f}             │
  │  Total Return:          {total_ret:>+9.2f}%             │
  │  Annualized Return:     {annual_ret:>+9.2f}%             │
  │  Max Drawdown:          {max_dd:>+9.2f}%             │
  │  Sharpe Ratio:           {sharpe:>8.2f}              │
  │  Total Trades:           {len(bt):>8}              │
  │  Trades/Year:            {len(bt)/years:>8.1f}              │
  │  Overall Win Rate:       {overall_wr:>7.1f}%              │
  └──────────────────────────────────────────────────────┘
""")

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Full Week Playbook — Best Signal Per Day (SPY Long Only)',
             fontsize=15, fontweight='bold')

# 1. Win rates by day
ax1 = fig.add_subplot(2, 3, 1)
days_with_signals = [(d, p) for d, p in zip(DAY_NAMES, playbook) if p is not None]
day_labels = [d[:3] for d, _ in days_with_signals]
day_wrs = [p['win_rate'] for _, p in days_with_signals]
day_ns = [p['n'] for _, p in days_with_signals]
colors_days = ['#26a69a' if w >= 60 else '#ef5350' for w in day_wrs]
bars = ax1.bar(day_labels, day_wrs, color=colors_days, width=0.5)
ax1.axhline(y=60, color='#ff9800', linestyle='--', alpha=0.8, label='60% target')
ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Baseline')
ax1.set_ylabel('Win Rate %')
ax1.set_title('Best Signal Win Rate Per Day')
ax1.legend(fontsize=8)
ax1.set_ylim(0, 100)
for bar, w, n in zip(bars, day_wrs, day_ns):
    ax1.text(bar.get_x() + bar.get_width()/2, w + 1,
             f'{w:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=8)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Equity curve combined
if all_bt:
    ax2 = fig.add_subplot(2, 3, (2, 3))
    ax2.plot(eq['Date'], eq['Capital'], color='#26a69a', linewidth=2)
    ax2.fill_between(eq['Date'], eq['Capital'], initial,
                     where=eq['Capital'] >= initial, alpha=0.15, color='#26a69a')
    ax2.fill_between(eq['Date'], eq['Capital'], initial,
                     where=eq['Capital'] < initial, alpha=0.15, color='#ef5350')
    ax2.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title(f'Combined Equity Curve ($10K → ${capital:,.0f})', fontsize=12)
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.grid(True, alpha=0.3)

    # 3. Win rate per day (backtest actual)
    ax3 = fig.add_subplot(2, 3, 4)
    bt_by_day = bt.groupby('Day').agg(
        WR=('Win', lambda x: x.mean() * 100),
        Count=('Win', 'count'),
        TotalRet=('Return_Pct', 'sum')
    ).reindex(DAY_NAMES).dropna()
    colors_bt = ['#26a69a' if w >= 60 else '#ef5350' for w in bt_by_day['WR']]
    bars3 = ax3.bar(bt_by_day.index.str[:3], bt_by_day['WR'], color=colors_bt, width=0.5)
    ax3.axhline(y=60, color='#ff9800', linestyle='--', alpha=0.8, label='60% target')
    ax3.set_ylabel('Win Rate %')
    ax3.set_title('Actual Backtest Win Rate Per Day\n(with 1.5% stop loss)')
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 100)
    for bar, (_, row) in zip(bars3, bt_by_day.iterrows()):
        ax3.text(bar.get_x() + bar.get_width()/2, row['WR'] + 1,
                 f'{row["WR"]:.1f}%', ha='center', va='bottom', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Individual trades
    ax4 = fig.add_subplot(2, 3, 5)
    colors_pnl = ['#26a69a' if r > 0 else '#ef5350' for r in bt['Return_Pct']]
    ax4.bar(range(len(bt)), bt['Return_Pct'], color=colors_pnl, width=1.0)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_title('All Trades — Individual Returns', fontsize=12)
    ax4.set_xlabel('Trade #')
    ax4.set_ylabel('Return %')
    ax4.grid(True, alpha=0.3)

    # 5. Drawdown
    ax5 = fig.add_subplot(2, 3, 6)
    ax5.fill_between(eq['Date'], eq['DD'], 0, color='#ef5350', alpha=0.4)
    ax5.plot(eq['Date'], eq['DD'], color='#ef5350', linewidth=1)
    ax5.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontsize=12)
    ax5.set_ylabel('Drawdown %')
    ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/ashleighchua/trading analyses/full_week_playbook.png', dpi=150, bbox_inches='tight')
print("\nChart saved to: full_week_playbook.png")
print("✅ Done!")
