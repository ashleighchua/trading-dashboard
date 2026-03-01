"""
SPY Every-Day Trading Analysis
================================
What if you buy SPY at open and sell at close EVERY day?
And what's the best filter for each day of the week?

Tests:
1. Blind buy every day (no filter)
2. Buy only if previous day was red
3. Buy only if previous day was green
4. Best combined daily strategy
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Download SPY data ────────────────────────────────────────────────────────
print("Downloading SPY historical data (5 years)...")
spy = yf.download("SPY", period="5y", auto_adjust=True)
spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy

spy['DayOfWeek'] = spy.index.dayofweek
spy['DayName'] = spy.index.day_name()
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')
spy['Prev_Color'] = spy['Color'].shift(1)
spy['Prev_Return'] = spy['Return'].shift(1)
spy = spy.dropna(subset=['Prev_Color'])

print(f"Data: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)\n")

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
SL = 1.9

# ══════════════════════════════════════════════════════════════════════════════
# 1. BASELINE: Buy every day — no filter
# ══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("1. BASELINE: Buy at open, sell at close — EVERY DAY")
print("="*80)

print(f"\n  {'Day':<12} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'CumRet':>10} {'Best Filter':>20}")
print("  " + "-"*70)

day_best_filter = {}

for dow in range(5):
    day_data = spy[spy['DayOfWeek'] == dow]
    n = len(day_data)
    wr = (day_data['Return'] > 0).mean() * 100
    avg = day_data['Return'].mean()
    cum = ((1 + day_data['Return'] / 100).prod() - 1) * 100

    # Test filters
    prev_red = day_data[day_data['Prev_Color'] == 'Red']
    prev_green = day_data[day_data['Prev_Color'] == 'Green']

    filters = {
        'No filter': day_data,
        'Prev Red': prev_red,
        'Prev Green': prev_green,
    }

    best_name = 'No filter'
    best_wr = wr

    for fname, fdata in filters.items():
        if len(fdata) > 10:
            f_wr = (fdata['Return'] > 0).mean() * 100
            if f_wr > best_wr:
                best_wr = f_wr
                best_name = fname

    day_best_filter[dow] = best_name
    print(f"  {day_names[dow]:<12} {n:>7} {wr:>6.1f}% {avg:>+8.3f}% {cum:>+9.2f}% {best_name:>20} ({best_wr:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DETAILED DAY-BY-DAY BREAKDOWN WITH ALL FILTERS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("2. EVERY DAY — DETAILED FILTER ANALYSIS")
print("="*80)

all_day_strategies = []

for dow in range(5):
    day_data = spy[spy['DayOfWeek'] == dow]
    print(f"\n  ── {day_names[dow].upper()} ──")

    filters = {
        'No filter (buy every day)': day_data,
        'Only if prev day RED': day_data[day_data['Prev_Color'] == 'Red'],
        'Only if prev day GREEN': day_data[day_data['Prev_Color'] == 'Green'],
        'Only if prev day RED > -0.5%': day_data[(day_data['Prev_Color'] == 'Red') & (day_data['Prev_Return'] >= -0.5)],
        'Only if prev day RED < -0.5%': day_data[(day_data['Prev_Color'] == 'Red') & (day_data['Prev_Return'] < -0.5)],
        'Only if prev day GREEN > +0.5%': day_data[(day_data['Prev_Color'] == 'Green') & (day_data['Prev_Return'] > 0.5)],
    }

    print(f"  {'Filter':<32} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'CumRet':>10} {'PF':>7}")
    print("  " + "-"*75)

    best_score = -999
    best_filter_name = None

    for fname, fdata in filters.items():
        if len(fdata) < 10:
            continue
        n = len(fdata)
        wr = (fdata['Return'] > 0).mean() * 100
        avg = fdata['Return'].mean()
        cum = ((1 + fdata['Return'] / 100).prod() - 1) * 100
        gp = fdata[fdata['Return'] > 0]['Return'].sum()
        gl = abs(fdata[fdata['Return'] < 0]['Return'].sum())
        pf = gp / gl if gl > 0 else float('inf')

        # Score: balance win rate, return, and trade frequency
        score = avg * np.sqrt(n)  # reward higher avg return AND more trades

        marker = ""
        if score > best_score and fname != 'No filter (buy every day)':
            best_score = score
            best_filter_name = fname
            marker = " ◀"

        print(f"  {fname:<32} {n:>7} {wr:>6.1f}% {avg:>+8.3f}% {cum:>+9.2f}% {pf:>6.2f}{marker}")

    if best_filter_name:
        all_day_strategies.append({
            'day': dow,
            'day_name': day_names[dow],
            'filter': best_filter_name,
        })

# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST: Trade every day with optimal filter per day
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("3. BACKTEST COMPARISON: 4 Approaches")
print("="*80)

def backtest(trades_df, label, sl_pct=SL):
    """Run backtest on a set of trades with stop loss."""
    capital = 10000
    initial = capital
    log = []

    for _, row in trades_df.iterrows():
        stop_price = row['Open'] * (1 - sl_pct / 100)
        if row['Low'] <= stop_price:
            exit_price = stop_price
        else:
            exit_price = row['Close']

        ret = (exit_price - row['Open']) / row['Open']
        pnl = capital * ret
        capital += pnl
        log.append({'Date': row.name, 'Return': ret * 100, 'PnL': pnl, 'Win': ret > 0, 'Capital': capital})

    tl = pd.DataFrame(log)
    if len(tl) == 0:
        return None

    total_ret = (capital - initial) / initial * 100
    years = (trades_df.index[-1] - trades_df.index[0]).days / 365.25
    annual = ((capital / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    wr = tl['Win'].mean() * 100
    tl['Peak'] = tl['Capital'].cummax()
    tl['DD'] = (tl['Capital'] - tl['Peak']) / tl['Peak'] * 100
    mdd = tl['DD'].min()
    sharpe = (tl['Return'].mean() / tl['Return'].std()) * np.sqrt(len(tl) / years) if tl['Return'].std() > 0 and years > 0 else 0
    tpy = len(tl) / years if years > 0 else 0

    return {
        'Label': label,
        'Capital': capital,
        'TotalRet': total_ret,
        'Annual': annual,
        'WinRate': wr,
        'MaxDD': mdd,
        'Sharpe': sharpe,
        'Trades': len(tl),
        'TradesPerYear': tpy,
        'equity': tl[['Date', 'Capital']],
    }


# Approach A: Buy every single day
result_a = backtest(spy, "A: Buy every day (no filter)")

# Approach B: Buy every day, but only if previous day was red
prev_red_all = spy[spy['Prev_Color'] == 'Red']
result_b = backtest(prev_red_all, "B: Buy only after RED days")

# Approach C: Use the 6 best signals from previous analysis (signal-only days)
# Rebuild the signal-filtered trades
signal_rules = {
    0: 'Red',    # Red Monday → Buy Wednesday (dow=2)
    1: 'Red',    # Red Tuesday → Buy Friday (dow=4)
    2: 'Red',    # Red Wednesday → Buy Friday (dow=4) + Monday (dow=0)
    3: 'Red',    # Red Thursday → Buy Friday (dow=4)
    4: 'Red',    # Red Friday → Buy Monday (dow=0)
}
# Translate: which trade days are triggered and by what?
# Trade Monday if: prev Fri red OR prev Wed red
# Trade Wednesday if: prev Mon red
# Trade Friday if: prev Tue/Wed/Thu red

def get_filtered_trades():
    trades = []
    for i in range(len(spy)):
        row = spy.iloc[i]
        dow = row['DayOfWeek']

        should_trade = False

        if dow == 0:  # Monday: trade if prev Friday was red OR prev Wednesday was red
            # Check previous Friday
            prev_days = spy.iloc[:i]
            prev_fri = prev_days[prev_days['DayOfWeek'] == 4]
            prev_wed = prev_days[prev_days['DayOfWeek'] == 2]
            if len(prev_fri) > 0 and (spy.index[i] - prev_fri.index[-1]).days <= 5:
                if prev_fri.iloc[-1]['Color'] == 'Red':
                    should_trade = True
            if len(prev_wed) > 0 and (spy.index[i] - prev_wed.index[-1]).days <= 7:
                if prev_wed.iloc[-1]['Color'] == 'Red':
                    should_trade = True

        elif dow == 2:  # Wednesday: trade if Monday was red
            if row['Prev_Color'] == 'Red':  # prev trading day
                prev_days = spy.iloc[:i]
                prev_mon = prev_days[prev_days['DayOfWeek'] == 0]
                if len(prev_mon) > 0 and (spy.index[i] - prev_mon.index[-1]).days <= 3:
                    if prev_mon.iloc[-1]['Color'] == 'Red':
                        should_trade = True

        elif dow == 4:  # Friday: trade if Tue/Wed/Thu was red
            prev_days = spy.iloc[max(0,i-4):i]
            for _, pd_row in prev_days.iterrows():
                if pd_row['DayOfWeek'] in [1, 2, 3] and pd_row['Color'] == 'Red':
                    should_trade = True
                    break

        if should_trade:
            trades.append(spy.index[i])

    return spy.loc[trades]

print("\n  Building signal-filtered trades (this may take a moment)...")
signal_trades = get_filtered_trades()
result_c = backtest(signal_trades, "C: Signal days only (Mon/Wed/Fri)")

# Approach D: Optimal filter per day — trade every day with the best filter
def get_optimal_daily_trades():
    """For each day, apply the best filter we found."""
    trades_idx = []

    for i in range(len(spy)):
        row = spy.iloc[i]
        dow = row['DayOfWeek']

        # Best filter per day based on our analysis:
        # Monday: best after red prev day (Friday)
        # Tuesday: generally weak — skip or use green prev day
        # Wednesday: best after red prev day (Tuesday)
        # Thursday: best after green prev day (Wednesday)
        # Friday: best after red prev day (any)

        if dow == 0:  # Monday — buy if prev day (Friday) was red
            if row['Prev_Color'] == 'Red':
                trades_idx.append(spy.index[i])
        elif dow == 1:  # Tuesday — buy if prev day (Monday) was green
            if row['Prev_Color'] == 'Green':
                trades_idx.append(spy.index[i])
        elif dow == 2:  # Wednesday — buy if prev day (Tuesday) was red
            if row['Prev_Color'] == 'Red':
                trades_idx.append(spy.index[i])
        elif dow == 3:  # Thursday — buy if prev day (Wednesday) was green
            if row['Prev_Color'] == 'Green':
                trades_idx.append(spy.index[i])
        elif dow == 4:  # Friday — buy if prev day (Thursday) was red
            if row['Prev_Color'] == 'Red':
                trades_idx.append(spy.index[i])

    return spy.loc[trades_idx]

optimal_trades = get_optimal_daily_trades()
result_d = backtest(optimal_trades, "D: Every day + best filter")

# Print comparison
results = [r for r in [result_a, result_b, result_c, result_d] if r is not None]

print(f"\n  {'Strategy':<36} {'Trades':>7} {'T/Yr':>6} {'Win%':>7} {'Annual':>8} {'TotRet':>9} {'MaxDD':>8} {'Sharpe':>7}")
print("  " + "-"*90)
for r in results:
    marker = " ★" if r['Sharpe'] == max(x['Sharpe'] for x in results) else ""
    print(f"  {r['Label']:<36} {r['Trades']:>7} {r['TradesPerYear']:>5.0f} {r['WinRate']:>6.1f}% {r['Annual']:>+7.2f}% {r['TotalRet']:>+8.2f}% {r['MaxDD']:>+7.2f}% {r['Sharpe']:>6.2f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PRACTICAL EVERY-DAY TRADING PLAN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("4. YOUR EVERY-DAY TRADING PLAN")
print("="*80)

# Recalculate the best action for each day
print(f"\n  For each day, showing: buy blind vs buy with filter vs skip\n")

plan = []
for dow in range(5):
    day_data = spy[spy['DayOfWeek'] == dow]

    options = {}

    # No filter
    n = len(day_data)
    wr = (day_data['Return'] > 0).mean() * 100
    avg = day_data['Return'].mean()
    options['Buy (no filter)'] = {'wr': wr, 'avg': avg, 'n': n}

    # Prev red
    prev_red = day_data[day_data['Prev_Color'] == 'Red']
    if len(prev_red) >= 10:
        wr_r = (prev_red['Return'] > 0).mean() * 100
        avg_r = prev_red['Return'].mean()
        options['Buy if prev RED'] = {'wr': wr_r, 'avg': avg_r, 'n': len(prev_red)}

    # Prev green
    prev_green = day_data[day_data['Prev_Color'] == 'Green']
    if len(prev_green) >= 10:
        wr_g = (prev_green['Return'] > 0).mean() * 100
        avg_g = prev_green['Return'].mean()
        options['Buy if prev GREEN'] = {'wr': wr_g, 'avg': avg_g, 'n': len(prev_green)}

    print(f"  {day_names[dow].upper()}")
    best_opt = None
    best_wr = 0
    for opt_name, opt_data in options.items():
        marker = ""
        if opt_data['wr'] > best_wr:
            best_wr = opt_data['wr']
            best_opt = opt_name
        print(f"    {opt_name:<24} WR: {opt_data['wr']:>5.1f}%  Avg: {opt_data['avg']:>+6.3f}%  (n={opt_data['n']})")

    recommendation = best_opt
    rec_data = options[best_opt]

    if rec_data['wr'] < 52:
        action = f"SKIP (best WR only {rec_data['wr']:.1f}%)"
    elif rec_data['wr'] < 56:
        action = f"OPTIONAL — {best_opt} (WR: {rec_data['wr']:.1f}%)"
    else:
        action = f"BUY — {best_opt} (WR: {rec_data['wr']:.1f}%)"

    plan.append({
        'day': day_names[dow],
        'action': action,
        'filter': best_opt,
        'wr': rec_data['wr'],
        'avg': rec_data['avg'],
    })
    print(f"    → RECOMMENDATION: {action}\n")

print(f"\n  ┌{'─'*64}┐")
print(f"  │  YOUR WEEKLY TRADING SCHEDULE{' '*33}│")
print(f"  ├{'─'*64}┤")
for p in plan:
    line = f"  │  {p['day']:<11} {p['action']:<51}│"
    print(line)
print(f"  ├{'─'*64}┤")
print(f"  │  RULES: Buy at OPEN, SL: 1.9%, Sell at CLOSE{' '*17}│")
print(f"  └{'─'*64}┘")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 16))
fig.suptitle('SPY Every-Day Trading Analysis', fontsize=15, fontweight='bold', y=0.98)

# 1. Equity curves comparison
ax1 = fig.add_subplot(3, 1, 1)
colors_line = ['#9e9e9e', '#ff9800', '#1976d2', '#26a69a']
for r, c in zip(results, colors_line):
    eq = r['equity']
    ax1.plot(eq['Date'], eq['Capital'], color=c, linewidth=2 if c != '#9e9e9e' else 1.5,
             alpha=0.9 if c != '#9e9e9e' else 0.5, label=r['Label'])
ax1.axhline(y=10000, color='gray', linestyle=':', alpha=0.3)
ax1.set_title('Equity Curves: 4 Strategies Compared', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Win rate by day of week with filter overlay
ax2 = fig.add_subplot(3, 2, 3)
x = np.arange(5)
width = 0.25

wr_blind = [(spy[spy['DayOfWeek'] == d]['Return'] > 0).mean() * 100 for d in range(5)]
wr_prev_red = [(spy[(spy['DayOfWeek'] == d) & (spy['Prev_Color'] == 'Red')]['Return'] > 0).mean() * 100 for d in range(5)]
wr_prev_green = [(spy[(spy['DayOfWeek'] == d) & (spy['Prev_Color'] == 'Green')]['Return'] > 0).mean() * 100 for d in range(5)]

bars1 = ax2.bar(x - width, wr_blind, width, label='No filter', color='#9e9e9e', alpha=0.7)
bars2 = ax2.bar(x, wr_prev_red, width, label='After RED day', color='#ef5350', alpha=0.7)
bars3 = ax2.bar(x + width, wr_prev_green, width, label='After GREEN day', color='#26a69a', alpha=0.7)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Win Rate by Day + Previous Day Filter', fontsize=12)
ax2.set_ylabel('Win Rate %')
ax2.set_xticks(x)
ax2.set_xticklabels(day_names)
ax2.set_ylim(40, 75)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.0f}', ha='center', fontsize=7)

# 3. Average return by day with filter overlay
ax3 = fig.add_subplot(3, 2, 4)

avg_blind = [spy[spy['DayOfWeek'] == d]['Return'].mean() for d in range(5)]
avg_prev_red = [spy[(spy['DayOfWeek'] == d) & (spy['Prev_Color'] == 'Red')]['Return'].mean() for d in range(5)]
avg_prev_green = [spy[(spy['DayOfWeek'] == d) & (spy['Prev_Color'] == 'Green')]['Return'].mean() for d in range(5)]

ax3.bar(x - width, avg_blind, width, label='No filter', color='#9e9e9e', alpha=0.7)
ax3.bar(x, avg_prev_red, width, label='After RED day', color='#ef5350', alpha=0.7)
ax3.bar(x + width, avg_prev_green, width, label='After GREEN day', color='#26a69a', alpha=0.7)
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax3.set_title('Avg Return by Day + Previous Day Filter', fontsize=12)
ax3.set_ylabel('Avg Return %')
ax3.set_xticks(x)
ax3.set_xticklabels(day_names)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Trade frequency per strategy
ax4 = fig.add_subplot(3, 2, 5)
strat_names = [r['Label'].replace('A: ','').replace('B: ','').replace('C: ','').replace('D: ','') for r in results]
tpy = [r['TradesPerYear'] for r in results]
annual_rets = [r['Annual'] for r in results]
colors_bar = ['#9e9e9e', '#ff9800', '#1976d2', '#26a69a']
ax4.bar(strat_names, tpy, color=colors_bar, alpha=0.8)
ax4.set_title('Trades per Year', fontsize=12)
ax4.set_ylabel('Trades/Year')
for j, (v, ar) in enumerate(zip(tpy, annual_rets)):
    ax4.text(j, v + 2, f'{v:.0f}\n({ar:+.1f}%/yr)', ha='center', fontsize=9)
ax4.tick_params(axis='x', rotation=15)
ax4.grid(True, alpha=0.3, axis='y')

# 5. Sharpe ratio comparison
ax5 = fig.add_subplot(3, 2, 6)
sharpes = [r['Sharpe'] for r in results]
ax5.bar(strat_names, sharpes, color=colors_bar, alpha=0.8)
ax5.set_title('Sharpe Ratio Comparison', fontsize=12)
ax5.set_ylabel('Sharpe Ratio')
for j, v in enumerate(sharpes):
    ax5.text(j, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
ax5.tick_params(axis='x', rotation=15)
ax5.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/ashleighchua/trading analyses/spy_everyday_analysis.png', dpi=150, bbox_inches='tight')
print("\nCharts saved to: spy_everyday_analysis.png")
print("\n✅ Analysis complete!")
