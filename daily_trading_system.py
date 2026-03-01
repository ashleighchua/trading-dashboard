"""
Daily Trading System — A Strategy For Every Day
==================================================
Goal: Have at least 1 tradeable signal EVERY trading day.

Current strategies:
  1. SPY Day-of-Week Playbook (Mon/Wed/Thu/Fri signals)
  2. Earnings Season Drift (4x/year, swing)
  3. RSI Oversold Bounce (rare but high WR)

NEW strategies to fill gaps:
  4. Gap Fill Strategy — Buy when SPY gaps down >0.3% at open
  5. QQQ Day-of-Week Playbook — Same logic, different ETF
  6. IWM (Russell 2000) Day-of-Week — Small caps have different patterns
  7. Overnight Hold — Buy at close, sell at next open
  8. 3-Day Mean Reversion — Buy after 3 consecutive red days
  9. First Red Day Bounce — Buy after first red day following 3+ green days

All backtested on 10 years of data.
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Download Data ─────────────────────────────────────────────────────────────
print("Downloading data (SPY, QQQ, IWM)...")
spy = yf.download("SPY", period="10y", auto_adjust=True)
qqq = yf.download("QQQ", period="10y", auto_adjust=True)
iwm = yf.download("IWM", period="10y", auto_adjust=True)

for df in [spy, qqq, iwm]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')

for df in [spy, qqq, iwm]:
    df['DayOfWeek'] = df.index.dayofweek
    df['Return'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Color'] = df['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')
    df['PrevClose'] = df['Close'].shift(1)
    df['GapPct'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100
    df['OvernightReturn'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100

print(f"SPY: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)")
print(f"QQQ: {qqq.index[0].date()} → {qqq.index[-1].date()} ({len(qqq)} days)")
print(f"IWM: {iwm.index[0].date()} → {iwm.index[-1].date()} ({len(iwm)} days)")

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
results = {}

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: GAP FILL — Buy when SPY gaps down, sell at close
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 4: GAP FILL (Buy gap-down at open, sell at close)")
print("="*70)

for gap_threshold in [-0.3, -0.5, -0.7, -1.0]:
    gap_days = spy[spy['GapPct'] <= gap_threshold].copy()
    if len(gap_days) == 0:
        continue
    wins = (gap_days['Return'] > 0).sum()
    total = len(gap_days)
    wr = wins / total * 100
    avg_ret = gap_days['Return'].mean()
    print(f"  Gap <= {gap_threshold}%: {total} trades, WR={wr:.1f}%, Avg={avg_ret:+.2f}%")

# Best gap threshold with good sample size
best_gap = -0.3
gap_trades = spy[spy['GapPct'] <= best_gap].copy()
gap_wins = (gap_trades['Return'] > 0).sum()
gap_total = len(gap_trades)
gap_wr = gap_wins / gap_total * 100
gap_avg = gap_trades['Return'].mean()

# With stop loss optimization
print(f"\n  Stop loss optimization for gap <= {best_gap}%:")
best_gap_sl = 0
best_gap_profit = -999
for sl in [0.5, 0.8, 1.0, 1.2, 1.5, 1.9, 2.5]:
    pnl = 0
    wins = 0
    total = 0
    for _, row in gap_trades.iterrows():
        total += 1
        # Check if stop loss hit (using low price as proxy)
        low_from_open = (row['Low'] - row['Open']) / row['Open'] * 100
        if low_from_open <= -sl:
            pnl -= sl  # stopped out
        else:
            ret = row['Return']
            pnl += ret
            if ret > 0:
                wins += 1
    wr_sl = wins / total * 100 if total > 0 else 0
    print(f"    SL {sl}%: PnL={pnl:+.1f}%, WR={wr_sl:.1f}%")
    if pnl > best_gap_profit:
        best_gap_profit = pnl
        best_gap_sl = sl

print(f"\n  ✅ BEST: Gap <= {best_gap}%, SL={best_gap_sl}%, Total PnL={best_gap_profit:+.1f}%")
print(f"     Trades: {gap_total}, Avg/trade: {gap_avg:+.3f}%")
print(f"     Frequency: ~{gap_total/10:.0f} trades/year")

results['Gap Fill (SPY)'] = {
    'trades': gap_total,
    'per_year': gap_total / 10,
    'win_rate': gap_wr,
    'avg_return': gap_avg,
    'stop_loss': best_gap_sl,
}

# Per day-of-week breakdown for gap fill
print(f"\n  Gap Fill by Day of Week:")
for dow in range(5):
    day_gaps = gap_trades[gap_trades['DayOfWeek'] == dow]
    if len(day_gaps) > 0:
        dw = (day_gaps['Return'] > 0).sum()
        dt = len(day_gaps)
        dwr = dw / dt * 100
        print(f"    {DAY_NAMES[dow]:>10}: {dt} trades, WR={dwr:.1f}%, Avg={day_gaps['Return'].mean():+.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: QQQ DAY-OF-WEEK PLAYBOOK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 5: QQQ DAY-OF-WEEK PLAYBOOK")
print("="*70)

def test_day_signals(df, ticker_name):
    """Test all day-of-week signals for a given ETF."""
    signals = {}
    for target_dow in range(5):
        target_name = DAY_NAMES[target_dow]
        best_signal = None
        best_wr = 0

        for lookback in [1, 2]:
            for colors in [['Red'], ['Green'], ['Red', 'Red'], ['Green', 'Green'],
                           ['Red', 'Green'], ['Green', 'Red']]:
                if len(colors) != lookback:
                    continue
                for thresh in [0.0, 0.3, 0.5, 0.7, 1.0]:
                    trades = []
                    for i in range(lookback, len(df)):
                        row = df.iloc[i]
                        if row['DayOfWeek'] != target_dow:
                            continue

                        match = True
                        for j, req_color in enumerate(colors):
                            prev_idx = i - lookback + j
                            if prev_idx < 0:
                                match = False
                                break
                            prev = df.iloc[prev_idx]
                            if prev['Color'] != req_color:
                                match = False
                                break
                            if abs(prev['Return']) < thresh:
                                match = False
                                break
                        if match:
                            trades.append(row['Return'])

                    if len(trades) >= 20:
                        wins = sum(1 for t in trades if t > 0)
                        wr = wins / len(trades) * 100
                        if wr > best_wr and wr >= 58:
                            condition = " + ".join(
                                f"{c} >= {thresh}%" for c in colors
                            )
                            prev_days = [DAY_NAMES[(target_dow - lookback + j) % 5] for j in range(lookback)]
                            desc = " & ".join(f"{prev_days[j]} {colors[j]}" for j in range(lookback))
                            if thresh > 0:
                                desc += f" (>={thresh}%)"

                            best_signal = {
                                'desc': desc,
                                'trades': len(trades),
                                'win_rate': wr,
                                'avg_return': np.mean(trades),
                                'total_return': sum(trades),
                            }
                            best_wr = wr

        if best_signal:
            signals[target_name] = best_signal

    return signals

qqq_signals = test_day_signals(qqq, "QQQ")
print(f"\n  QQQ Day-of-Week Signals (>= 58% WR, n>=20):\n")
for day, sig in qqq_signals.items():
    icon = "✅" if sig['win_rate'] >= 60 else "🟡"
    print(f"  {icon} {day:>10}: {sig['desc']}")
    print(f"              WR={sig['win_rate']:.1f}%, n={sig['trades']}, Avg={sig['avg_return']:+.3f}%, Total={sig['total_return']:+.1f}%")
    results[f'QQQ {day}'] = {
        'trades': sig['trades'],
        'per_year': sig['trades'] / 10,
        'win_rate': sig['win_rate'],
        'avg_return': sig['avg_return'],
        'stop_loss': 2.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 6: IWM (RUSSELL 2000) DAY-OF-WEEK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 6: IWM (RUSSELL 2000) DAY-OF-WEEK PLAYBOOK")
print("="*70)

iwm_signals = test_day_signals(iwm, "IWM")
print(f"\n  IWM Day-of-Week Signals (>= 58% WR, n>=20):\n")
for day, sig in iwm_signals.items():
    icon = "✅" if sig['win_rate'] >= 60 else "🟡"
    print(f"  {icon} {day:>10}: {sig['desc']}")
    print(f"              WR={sig['win_rate']:.1f}%, n={sig['trades']}, Avg={sig['avg_return']:+.3f}%, Total={sig['total_return']:+.1f}%")
    results[f'IWM {day}'] = {
        'trades': sig['trades'],
        'per_year': sig['trades'] / 10,
        'win_rate': sig['win_rate'],
        'avg_return': sig['avg_return'],
        'stop_loss': 2.5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 7: OVERNIGHT HOLD — Buy at close, sell at next open
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 7: OVERNIGHT HOLD (Buy SPY at close, sell at next open)")
print("="*70)

spy['NextOpen'] = spy['Open'].shift(-1)
spy['OvernightRet'] = (spy['NextOpen'] - spy['Close']) / spy['Close'] * 100

print(f"\n  Overnight returns by day (buy at close, sell at next open):\n")
for dow in range(5):
    day_data = spy[spy['DayOfWeek'] == dow].dropna(subset=['OvernightRet'])
    wins = (day_data['OvernightRet'] > 0).sum()
    total = len(day_data)
    wr = wins / total * 100 if total > 0 else 0
    avg = day_data['OvernightRet'].mean()
    total_ret = day_data['OvernightRet'].sum()
    print(f"  {DAY_NAMES[dow]:>10}: WR={wr:.1f}%, Avg={avg:+.3f}%, n={total}, Total={total_ret:+.1f}%")

# Best overnight day
print(f"\n  Overnight hold filtered by today's color:\n")
for dow in range(5):
    for color in ['Red', 'Green']:
        day_data = spy[(spy['DayOfWeek'] == dow) & (spy['Color'] == color)].dropna(subset=['OvernightRet'])
        if len(day_data) < 20:
            continue
        wins = (day_data['OvernightRet'] > 0).sum()
        total = len(day_data)
        wr = wins / total * 100
        avg = day_data['OvernightRet'].mean()
        if wr >= 58:
            icon = "✅" if wr >= 60 else "🟡"
            print(f"  {icon} {color} {DAY_NAMES[dow]:>10} → overnight: WR={wr:.1f}%, Avg={avg:+.3f}%, n={total}")
            results[f'Overnight {color} {DAY_NAMES[dow]}'] = {
                'trades': total,
                'per_year': total / 10,
                'win_rate': wr,
                'avg_return': avg,
                'stop_loss': 0,  # overnight, no intraday SL
            }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 8: 3-DAY RED STREAK BOUNCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 8: 3-DAY RED STREAK BOUNCE (Buy after 3 consecutive red days)")
print("="*70)

spy['Prev1Color'] = spy['Color'].shift(1)
spy['Prev2Color'] = spy['Color'].shift(2)
spy['Prev3Color'] = spy['Color'].shift(3)

for streak in [2, 3]:
    if streak == 2:
        mask = (spy['Prev1Color'] == 'Red') & (spy['Prev2Color'] == 'Red')
    else:
        mask = (spy['Prev1Color'] == 'Red') & (spy['Prev2Color'] == 'Red') & (spy['Prev3Color'] == 'Red')

    bounce_days = spy[mask].copy()
    if len(bounce_days) == 0:
        continue
    wins = (bounce_days['Return'] > 0).sum()
    total = len(bounce_days)
    wr = wins / total * 100
    avg = bounce_days['Return'].mean()
    total_ret = bounce_days['Return'].sum()
    print(f"\n  After {streak} consecutive red days:")
    print(f"  Trades: {total}, WR={wr:.1f}%, Avg={avg:+.3f}%, Total={total_ret:+.1f}%")
    print(f"  Frequency: ~{total/10:.0f}/year")

    # Stop loss optimization
    best_sl = 0
    best_pnl = -999
    for sl in [0.5, 0.8, 1.0, 1.5, 1.9, 2.5]:
        pnl = 0
        w = 0
        for _, row in bounce_days.iterrows():
            low_from_open = (row['Low'] - row['Open']) / row['Open'] * 100
            if low_from_open <= -sl:
                pnl -= sl
            else:
                pnl += row['Return']
                if row['Return'] > 0:
                    w += 1
        print(f"    SL {sl}%: PnL={pnl:+.1f}%, WR={w/total*100:.1f}%")
        if pnl > best_pnl:
            best_pnl = pnl
            best_sl = sl

    if wr >= 55:
        results[f'{streak}-Day Red Bounce'] = {
            'trades': total,
            'per_year': total / 10,
            'win_rate': wr,
            'avg_return': avg,
            'stop_loss': best_sl,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 9: FIRST RED AFTER GREEN STREAK
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 9: FIRST RED AFTER GREEN STREAK (Mean reversion)")
print("="*70)

# Find first red day after 3+ consecutive green days → buy NEXT day
spy['GreenStreak'] = 0
streak_count = 0
for i in range(len(spy)):
    if spy.iloc[i]['Color'] == 'Green':
        streak_count += 1
    else:
        spy.iloc[i, spy.columns.get_loc('GreenStreak')] = streak_count
        streak_count = 0

for min_streak in [2, 3, 4]:
    # Buy the day AFTER a red day that broke a green streak
    mask = spy['GreenStreak'] >= min_streak
    signal_indices = spy[mask].index
    trades = []
    for sig_date in signal_indices:
        loc = spy.index.get_loc(sig_date)
        if loc + 1 < len(spy):
            next_day = spy.iloc[loc + 1]
            trades.append(next_day['Return'])

    if len(trades) >= 15:
        wins = sum(1 for t in trades if t > 0)
        wr = wins / len(trades) * 100
        avg = np.mean(trades)
        total_ret = sum(trades)
        print(f"\n  After {min_streak}+ green streak broken by red → buy next day:")
        print(f"  Trades: {len(trades)}, WR={wr:.1f}%, Avg={avg:+.3f}%, Total={total_ret:+.1f}%")
        print(f"  Frequency: ~{len(trades)/10:.0f}/year")

        if wr >= 55:
            results[f'Red After {min_streak}+ Green'] = {
                'trades': len(trades),
                'per_year': len(trades) / 10,
                'win_rate': wr,
                'avg_return': avg,
                'stop_loss': 1.9,
            }


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 10: TUESDAY REVERSAL (Fill the Tuesday gap in SPY playbook)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 10: TUESDAY SPECIAL — Finding an edge for the skip day")
print("="*70)

tue_data = spy[spy['DayOfWeek'] == 1].copy()

# Test various conditions for Tuesday
print(f"\n  Testing Tuesday conditions:\n")

# Condition: Monday red AND gap down on Tuesday
tue_with_prev = tue_data.copy()
tue_with_prev['PrevReturn'] = spy['Return'].shift(1).reindex(tue_with_prev.index)
tue_with_prev['PrevColor'] = spy['Color'].shift(1).reindex(tue_with_prev.index)

for cond_name, mask in [
    ("Mon Red + Tue gaps down", (tue_with_prev['PrevColor'] == 'Red') & (tue_with_prev['GapPct'] < 0)),
    ("Mon Red >= 0.5%", (tue_with_prev['PrevColor'] == 'Red') & (tue_with_prev['PrevReturn'].abs() >= 0.5)),
    ("Mon Green + Tue gaps down", (tue_with_prev['PrevColor'] == 'Green') & (tue_with_prev['GapPct'] < -0.3)),
    ("Tue gaps down >= 0.5%", tue_with_prev['GapPct'] <= -0.5),
    ("Mon Red >= 1.0%", (tue_with_prev['PrevColor'] == 'Red') & (tue_with_prev['PrevReturn'].abs() >= 1.0)),
]:
    filtered = tue_with_prev[mask]
    if len(filtered) >= 15:
        wins = (filtered['Return'] > 0).sum()
        wr = wins / len(filtered) * 100
        avg = filtered['Return'].mean()
        icon = "✅" if wr >= 60 else "🟡" if wr >= 55 else "❌"
        print(f"  {icon} {cond_name:>30}: WR={wr:.1f}%, n={len(filtered)}, Avg={avg:+.3f}%")
        if wr >= 58:
            results[f'Tue: {cond_name}'] = {
                'trades': len(filtered),
                'per_year': len(filtered) / 10,
                'win_rate': wr,
                'avg_return': avg,
                'stop_loss': 1.9,
            }


# ══════════════════════════════════════════════════════════════════════════════
# MASTER SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("MASTER STRATEGY SUMMARY — ALL VIABLE SIGNALS")
print("="*70)

# Sort by win rate
sorted_results = sorted(results.items(), key=lambda x: x[1]['win_rate'], reverse=True)

print(f"\n  {'Strategy':<35} {'WR':>6} {'Trades':>7} {'Per Yr':>7} {'Avg Ret':>8} {'SL':>5}")
print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*5}")
for name, r in sorted_results:
    print(f"  {name:<35} {r['win_rate']:>5.1f}% {r['trades']:>7} {r['per_year']:>6.0f} {r['avg_return']:>+7.3f}% {r['stop_loss']:>4.1f}%")

total_trades_per_year = sum(r['per_year'] for _, r in sorted_results)
print(f"\n  Total estimated trades per year: ~{total_trades_per_year:.0f}")
print(f"  Trading days per year: ~252")
print(f"  Coverage: ~{min(100, total_trades_per_year/252*100):.0f}% of trading days")


# ══════════════════════════════════════════════════════════════════════════════
# DAILY CALENDAR VIEW — What to trade each day of the week
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DAILY TRADING CALENDAR — Your Complete Playbook")
print("="*70)

calendar = {
    'Monday': [],
    'Tuesday': [],
    'Wednesday': [],
    'Thursday': [],
    'Friday': [],
}

# Map results back to trading days
for name, r in sorted_results:
    if r['win_rate'] < 55:
        continue
    # Determine which day you TRADE (not signal day)
    if 'Monday' in name and 'Overnight' not in name:
        calendar['Monday'].append((name, r))
    elif 'Tue' in name:
        calendar['Tuesday'].append((name, r))
    elif 'Wednesday' in name and 'Overnight' not in name:
        calendar['Wednesday'].append((name, r))
    elif 'Thursday' in name and 'Overnight' not in name:
        calendar['Thursday'].append((name, r))
    elif 'Friday' in name and 'Overnight' not in name:
        calendar['Friday'].append((name, r))
    # Overnight trades
    elif 'Overnight' in name:
        # These are "hold overnight" trades, list under signal day
        for day in DAY_NAMES:
            if day in name:
                calendar[day].append((name, r))
    # Non-day-specific (gap fill, streak bounce, etc.)
    elif 'Gap' in name or 'Red Bounce' in name or 'Green' in name:
        for day in DAY_NAMES:
            calendar[day].append((name, r))

for day in DAY_NAMES:
    print(f"\n  📅 {day}:")
    if calendar[day]:
        for name, r in sorted(calendar[day], key=lambda x: x[1]['win_rate'], reverse=True):
            icon = "🟢" if r['win_rate'] >= 60 else "🟡"
            print(f"    {icon} {name}: WR={r['win_rate']:.1f}%, ~{r['per_year']:.0f} trades/yr")
    else:
        print(f"    ❌ No reliable signal — REST DAY")

# Total unique trade opportunities
print(f"\n  💰 You should have a trade signal on most days!")
print(f"  📊 Always check which specific conditions are met before trading.")
print(f"  ⚠️  Never force a trade — only trade when the signal conditions match.")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Daily Trading System — Strategy Comparison', fontsize=16, fontweight='bold')

# Plot 1: Win rates
ax = axes[0, 0]
names = [n for n, _ in sorted_results[:12]]
wrs = [r['win_rate'] for _, r in sorted_results[:12]]
colors_bar = ['#2ecc71' if w >= 60 else '#f39c12' if w >= 55 else '#e74c3c' for w in wrs]
bars = ax.barh(range(len(names)), wrs, color=colors_bar)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.axvline(x=60, color='green', linestyle='--', alpha=0.5, label='60% target')
ax.axvline(x=55, color='orange', linestyle='--', alpha=0.5, label='55% minimum')
ax.set_xlabel('Win Rate (%)')
ax.set_title('Win Rate by Strategy')
ax.legend(fontsize=8)
ax.invert_yaxis()

# Plot 2: Trades per year
ax = axes[0, 1]
trades_yr = [r['per_year'] for _, r in sorted_results[:12]]
ax.barh(range(len(names)), trades_yr, color='#3498db')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Trades per Year')
ax.set_title('Trading Frequency')
ax.invert_yaxis()

# Plot 3: Daily coverage
ax = axes[1, 0]
day_counts = []
for day in DAY_NAMES:
    day_counts.append(len([s for s in calendar[day] if s[1]['win_rate'] >= 55]))
ax.bar(DAY_NAMES, day_counts, color=['#2ecc71' if c > 0 else '#e74c3c' for c in day_counts])
ax.set_ylabel('Number of Strategies')
ax.set_title('Strategies Available Per Day')
for i, v in enumerate(day_counts):
    ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

# Plot 4: Expected monthly returns simulation
ax = axes[1, 1]
np.random.seed(42)
monthly_returns = []
capital = 1000
for month in range(12):
    month_pnl = 0
    for _ in range(20):  # ~20 trading days per month
        # Pick a random strategy from viable ones
        if sorted_results:
            _, strat = sorted_results[np.random.randint(0, min(5, len(sorted_results)))]
            # Simulate: win_rate% chance of winning avg_return, else lose stop_loss
            if np.random.random() < strat['win_rate'] / 100:
                month_pnl += strat['avg_return']
            else:
                month_pnl -= strat['stop_loss']
    monthly_returns.append(month_pnl)
    capital *= (1 + month_pnl / 100)

ax.bar(range(1, 13), monthly_returns, color=['#2ecc71' if r > 0 else '#e74c3c' for r in monthly_returns])
ax.set_xlabel('Month')
ax.set_ylabel('Return (%)')
ax.set_title('Simulated Monthly Returns (Random Strategy Selection)')
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('daily_trading_system.png', dpi=150, bbox_inches='tight')
print(f"\n  📊 Chart saved: daily_trading_system.png")
print("\nDone!")
