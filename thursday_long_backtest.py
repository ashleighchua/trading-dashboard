"""
Thursday Long Strategy Backtest
=================================
Hypothesis: If BOTH Tuesday AND Wednesday close green by 0.5%+ each,
Thursday has momentum continuation — go LONG at Thursday open.

Found from thursday_flip_backtest.py:
  Both >= 0.5% each → Thursday green 67.7%
  Both >= 0.7% each → Thursday green 81.2%
  Both >= 1.0% each → Thursday green 100%
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Download SPY data ─────────────────────────────────────────────────────────
print("Downloading SPY data (10 years)...")
spy = yf.download("SPY", period="10y", auto_adjust=True)
spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy

spy['DayOfWeek'] = spy.index.dayofweek
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

print(f"Data: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)")

# ── Build Tue→Wed→Thu triplets ────────────────────────────────────────────────
tuesdays   = spy[spy['DayOfWeek'] == 1]
wednesdays = spy[spy['DayOfWeek'] == 2]
thursdays  = spy[spy['DayOfWeek'] == 3]

triplets = []
for tue_date, tue_row in tuesdays.iterrows():
    next_weds = wednesdays[wednesdays.index > tue_date]
    if len(next_weds) == 0:
        continue
    wed = next_weds.iloc[0]
    if (wed.name - tue_date).days > 3:
        continue
    next_thus = thursdays[thursdays.index > wed.name]
    if len(next_thus) == 0:
        continue
    thu = next_thus.iloc[0]
    if (thu.name - wed.name).days > 3:
        continue
    triplets.append({
        'Tue_Date':    tue_date,
        'Tue_Return':  tue_row['Return'],
        'Tue_Color':   tue_row['Color'],
        'Wed_Date':    wed.name,
        'Wed_Return':  wed['Return'],
        'Wed_Color':   wed['Color'],
        'Thu_Date':    thu.name,
        'Thu_Return':  thu['Return'],
        'Thu_Color':   thu['Color'],
        'Thu_Open':    thu['Open'],
        'Thu_High':    thu['High'],
        'Thu_Low':     thu['Low'],
        'Thu_Close':   thu['Close'],
    })

df = pd.DataFrame(triplets)
print(f"Total Tue→Wed→Thu triplets: {len(df)}")

# ── THRESHOLD ANALYSIS ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("WIN RATE BY MINIMUM GREEN THRESHOLD (Both Tue AND Wed)")
print("="*70)

baseline_green = (df['Thu_Color'] == 'Green').sum() / len(df) * 100
print(f"\n  Baseline Thursday green rate: {baseline_green:.1f}%")

thresholds = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
print(f"\n  {'Min each day':>13} {'Count':>6} {'Thu Green%':>11} {'Thu Red%':>9} {'Avg Thu Ret':>12} {'Edge':>8}")
print("  " + "-"*63)
for thresh in thresholds:
    mask = (df['Tue_Return'] >= thresh) & (df['Wed_Return'] >= thresh)
    sub = df[mask]
    if len(sub) == 0:
        continue
    g = (sub['Thu_Color'] == 'Green').sum() / len(sub) * 100
    r = 100 - g
    avg = sub['Thu_Return'].mean()
    edge = g - baseline_green
    print(f"  Both >= {thresh:.1f}%      {len(sub):>6} {g:>10.1f}% {r:>8.1f}% {avg:>+11.3f}% {edge:>+7.1f}%")

# ── STOP LOSS OPTIMIZATION (0.5% threshold) ───────────────────────────────────
print("\n" + "="*70)
print("STOP LOSS OPTIMIZATION (Signal: Both Tue+Wed >= 0.5% each)")
print("="*70)

signal_df = df[(df['Tue_Return'] >= 0.5) & (df['Wed_Return'] >= 0.5)].copy()
print(f"\nSignals: {len(signal_df)} trades over 10 years (~{len(signal_df)/10:.0f}/year)")

stop_levels = np.arange(0.1, 3.05, 0.1)
results = []

for sl in stop_levels:
    trades = []
    for _, row in signal_df.iterrows():
        entry = row['Thu_Open']
        stop_price = entry * (1 - sl / 100)

        if row['Thu_Low'] <= stop_price:
            exit_price = stop_price
            stopped = True
        else:
            exit_price = row['Thu_Close']
            stopped = False

        ret = (exit_price - entry) / entry * 100
        trades.append({'return': ret, 'stopped': stopped, 'win': ret > 0})

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wins = trades_df['win'].sum()
    stopped_count = trades_df['stopped'].sum()
    total_ret = (1 + trades_df['return'] / 100).prod()
    cumulative_ret = (total_ret - 1) * 100
    avg_ret = trades_df['return'].mean()
    gross_profit = trades_df[trades_df['return'] > 0]['return'].sum()
    gross_loss = abs(trades_df[trades_df['return'] < 0]['return'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    years = 10.0
    sharpe = (avg_ret / trades_df['return'].std()) * np.sqrt(n / years) if trades_df['return'].std() > 0 else 0

    results.append({
        'Stop_Loss': sl,
        'Win_Rate': wins / n * 100,
        'Avg_Return': avg_ret,
        'Cumulative_Return': cumulative_ret,
        'Profit_Factor': pf,
        'Sharpe': sharpe,
        'Times_Stopped': stopped_count,
    })

res_df = pd.DataFrame(results)

no_sl_trades = (signal_df['Thu_Close'] - signal_df['Thu_Open']) / signal_df['Thu_Open'] * 100
no_sl_cum = ((1 + no_sl_trades / 100).prod() - 1) * 100
no_sl_wr = (no_sl_trades > 0).sum() / len(no_sl_trades) * 100
no_sl_gp = no_sl_trades[no_sl_trades > 0].sum()
no_sl_gl = abs(no_sl_trades[no_sl_trades < 0].sum())
no_sl_pf = no_sl_gp / no_sl_gl if no_sl_gl > 0 else float('inf')
no_sl_sharpe = (no_sl_trades.mean() / no_sl_trades.std()) * np.sqrt(len(signal_df) / 10.0) if no_sl_trades.std() > 0 else 0

print(f"\n  {'SL %':>5} {'Win%':>6} {'AvgRet':>8} {'CumRet':>9} {'PF':>6} {'Sharpe':>7} {'Stopped':>8}")
print("  " + "-"*60)
for _, r in res_df.iterrows():
    marker = " ◀ BEST" if r['Sharpe'] == res_df['Sharpe'].max() else ""
    print(f"  {r['Stop_Loss']:>4.1f}% {r['Win_Rate']:>5.1f}% {r['Avg_Return']:>+7.3f}% {r['Cumulative_Return']:>+8.2f}% {r['Profit_Factor']:>5.2f} {r['Sharpe']:>6.2f}  ({r['Times_Stopped']:.0f}x){marker}")
print(f"  {'None':>5} {no_sl_wr:>5.1f}% {no_sl_trades.mean():>+7.3f}% {no_sl_cum:>+8.2f}% {no_sl_pf:>5.2f} {no_sl_sharpe:>6.2f}  (0x)")

best = res_df.loc[res_df['Sharpe'].idxmax()]
optimal_sl = best['Stop_Loss']
print(f"\n  ✅ Best stop loss (by Sharpe): {optimal_sl:.1f}%")

# ── FULL BACKTEST ─────────────────────────────────────────────────────────────
print(f"\n" + "="*70)
print(f"FULL BACKTEST — Long Thursday, {optimal_sl:.1f}% Stop Loss")
print("="*70)

capital = 10000
initial = capital
equity = []
all_trades = []

for _, row in signal_df.iterrows():
    entry = row['Thu_Open']
    stop_price = entry * (1 - optimal_sl / 100)

    if row['Thu_Low'] <= stop_price:
        exit_price = stop_price
        exit_type = 'STOP'
    else:
        exit_price = row['Thu_Close']
        exit_type = 'CLOSE'

    ret = (exit_price - entry) / entry
    pnl = capital * ret
    capital += pnl

    all_trades.append({
        'Date':       row['Thu_Date'],
        'Tue_Ret':    row['Tue_Return'],
        'Wed_Ret':    row['Wed_Return'],
        'Entry':      entry,
        'Exit':       exit_price,
        'Exit_Type':  exit_type,
        'Return_Pct': ret * 100,
        'PnL':        pnl,
        'Capital':    capital,
        'Win':        ret > 0,
    })
    equity.append({'Date': row['Thu_Date'], 'Capital': capital})

bt = pd.DataFrame(all_trades)
eq = pd.DataFrame(equity)

total_return = (capital - initial) / initial * 100
years = (signal_df['Thu_Date'].iloc[-1] - signal_df['Thu_Date'].iloc[0]).days / 365.25
annual_ret = ((capital / initial) ** (1 / years) - 1) * 100
win_rate = bt['Win'].mean() * 100
avg_win = bt[bt['Win']]['Return_Pct'].mean()
avg_loss = bt[~bt['Win']]['Return_Pct'].mean()
gross_profit = bt[bt['Win']]['PnL'].sum()
gross_loss = abs(bt[~bt['Win']]['PnL'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
eq['Peak'] = eq['Capital'].cummax()
eq['DD'] = (eq['Capital'] - eq['Peak']) / eq['Peak'] * 100
max_dd = eq['DD'].min()
sharpe = (bt['Return_Pct'].mean() / bt['Return_Pct'].std()) * np.sqrt(len(bt) / years) if bt['Return_Pct'].std() > 0 else 0
stops_hit = (bt['Exit_Type'] == 'STOP').sum()

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  THURSDAY LONG — Momentum Continuation               │
  ├──────────────────────────────────────────────────────┤
  │  Signal:      Tue >= 0.5% green AND Wed >= 0.5% green│
  │  Entry:       LONG at Thursday OPEN                  │
  │  Stop Loss:   {optimal_sl:.1f}% below entry                    │
  │  Exit:        Thursday CLOSE                         │
  ├──────────────────────────────────────────────────────┤
  │  PERFORMANCE ({years:.1f} years)                         │
  ├──────────────────────────────────────────────────────┤
  │  Starting Capital:     ${initial:>10,.2f}             │
  │  Ending Capital:       ${capital:>10,.2f}             │
  │  Total Return:          {total_return:>+9.2f}%             │
  │  Annualized Return:     {annual_ret:>+9.2f}%             │
  │  Max Drawdown:          {max_dd:>+9.2f}%             │
  │  Sharpe Ratio:           {sharpe:>8.2f}              │
  │  Profit Factor:          {profit_factor:>8.2f}              │
  ├──────────────────────────────────────────────────────┤
  │  TRADE STATS                                         │
  ├──────────────────────────────────────────────────────┤
  │  Total Trades:           {len(bt):>8}              │
  │  Trades/Year:            {len(bt)/years:>8.1f}              │
  │  Win Rate:               {win_rate:>7.1f}%              │
  │  Avg Winner:            {avg_win:>+9.3f}%             │
  │  Avg Loser:             {avg_loss:>+9.3f}%             │
  │  Times Stopped Out:  {stops_hit:>4} ({stops_hit/len(bt)*100:.1f}%)              │
  └──────────────────────────────────────────────────────┘
""")

print("  YEARLY BREAKDOWN:")
print(f"  {'Year':<6} {'Trades':>7} {'Wins':>5} {'Stops':>6} {'WinRate':>8} {'Return':>9} {'PnL':>12}")
print("  " + "-"*58)
bt['Year'] = bt['Date'].dt.year
for year, g in bt.groupby('Year'):
    print(f"  {year:<6} {len(g):>7} {g['Win'].sum():>5} {(g['Exit_Type']=='STOP').sum():>6} {g['Win'].mean()*100:>7.1f}% {g['Return_Pct'].sum():>+8.2f}% ${g['PnL'].sum():>+10,.2f}")

# ── FULL PLAYBOOK SUMMARY ─────────────────────────────────────────────────────
print(f"\n" + "="*70)
print("FULL PLAYBOOK SUMMARY — All Strategies")
print("="*70)
print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  STRATEGY 1: Red Friday → Long Monday                           │
  │  Signal:   SPY Friday red by >= 0.5%                           │
  │  Entry:    Buy SPY at Monday open  (9:30 PM Thailand Sun)       │
  │  Exit:     Monday close            (4:00 AM Thailand Tue)       │
  │  Stop:     1.9% below entry                                     │
  │  Win Rate: ~66% (81% if Friday red by 2%+)                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  STRATEGY 2: Tue+Wed Both Green → Long Thursday                 │
  │  Signal:   Tuesday >= 0.5% green AND Wednesday >= 0.5% green   │
  │  Entry:    Buy SPY at Thursday open (9:30 PM Thailand Wed)      │
  │  Exit:     Thursday close           (4:00 AM Thailand Thu)      │
  │  Stop:     {optimal_sl:.1f}% below entry                                │
  │  Win Rate: ~68% (81% if both days >= 0.7%)                      │
  ├─────────────────────────────────────────────────────────────────┤
  │  COMBINED: Up to ~{(len(signal_df)/10 + 19):.0f} trades/year across both strategies    │
  └─────────────────────────────────────────────────────────────────┘
""")

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 16))
fig.suptitle('Thursday Long: Buy SPY when Tue+Wed Both Green (>=0.5% each)',
             fontsize=14, fontweight='bold')

# 1. Equity curve
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(eq['Date'], eq['Capital'], color='#26a69a', linewidth=2)
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                 where=eq['Capital'] >= initial, alpha=0.15, color='#26a69a')
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                 where=eq['Capital'] < initial, alpha=0.15, color='#ef5350')
ax1.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax1.set_title(f'Equity Curve ($10K start → ${capital:,.0f})', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)')
ax1.grid(True, alpha=0.3)

# 2. Win rate by threshold
ax2 = fig.add_subplot(3, 2, 3)
thresh_labels = ['0.0%', '0.2%', '0.3%', '0.5%', '0.7%', '1.0%']
thresh_green = []
thresh_counts = []
for thresh in thresholds:
    mask = (df['Tue_Return'] >= thresh) & (df['Wed_Return'] >= thresh)
    sub = df[mask]
    if len(sub) > 0:
        thresh_green.append((sub['Thu_Color'] == 'Green').sum() / len(sub) * 100)
        thresh_counts.append(len(sub))
    else:
        thresh_green.append(0)
        thresh_counts.append(0)
colors_t = ['#26a69a' if g >= baseline_green else '#ef5350' for g in thresh_green]
bars2 = ax2.bar(thresh_labels, thresh_green, color=colors_t, width=0.5)
ax2.axhline(y=baseline_green, color='gray', linestyle='--', alpha=0.7, label=f'Baseline: {baseline_green:.1f}%')
ax2.set_xlabel('Min green % required from each day')
ax2.set_ylabel('Thursday Green %')
ax2.set_title('Thursday Win Rate by Signal Threshold')
ax2.legend(fontsize=9)
ax2.set_ylim(0, 110)
for bar, g, c in zip(bars2, thresh_green, thresh_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, g + 1,
             f'{g:.1f}%\n(n={c})', ha='center', va='bottom', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Stop loss optimization
ax3 = fig.add_subplot(3, 2, 4)
ax3.plot(res_df['Stop_Loss'], res_df['Sharpe'], 'o-', color='#1976d2', linewidth=2, markersize=5)
ax3.axvline(x=optimal_sl, color='#26a69a', linestyle='--', alpha=0.7, label=f'Best: {optimal_sl:.1f}%')
ax3.axhline(y=no_sl_sharpe, color='#9e9e9e', linestyle=':', alpha=0.7, label=f'No SL: {no_sl_sharpe:.2f}')
ax3.set_title('Sharpe Ratio by Stop Loss Level', fontsize=12)
ax3.set_xlabel('Stop Loss %')
ax3.set_ylabel('Sharpe Ratio')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Individual trades
ax4 = fig.add_subplot(3, 2, 5)
colors_pnl = ['#26a69a' if r > 0 else '#ef5350' for r in bt['Return_Pct']]
ax4.bar(range(len(bt)), bt['Return_Pct'], color=colors_pnl, width=1.0)
ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax4.axhline(y=-optimal_sl, color='#ef5350', linestyle='--', alpha=0.5, label=f'Stop: -{optimal_sl:.1f}%')
ax4.set_title('Individual Trade Returns', fontsize=12)
ax4.set_xlabel('Trade #')
ax4.set_ylabel('Return %')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Drawdown
ax5 = fig.add_subplot(3, 2, 6)
ax5.fill_between(eq['Date'], eq['DD'], 0, color='#ef5350', alpha=0.4)
ax5.plot(eq['Date'], eq['DD'], color='#ef5350', linewidth=1)
ax5.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontsize=12)
ax5.set_ylabel('Drawdown %')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/ashleighchua/trading analyses/thursday_long_backtest.png', dpi=150, bbox_inches='tight')
print("Chart saved to: thursday_long_backtest.png")
print("✅ Analysis complete!")
