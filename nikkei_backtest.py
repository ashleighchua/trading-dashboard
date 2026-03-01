"""
Nikkei 225 Day Trading Backtest: Red Friday → Buy Monday Open
=============================================================
Strategy:
  - If Friday closes red (close < open), buy Nikkei at Monday open
  - Hold until Monday close
  - Stop loss triggers intraday (using Monday's low as proxy)
  - No take profit — let winners run to close

Ticker: ^N225 (Nikkei 225 via Yahoo Finance)
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Download Nikkei data ──────────────────────────────────────────────────────
print("Downloading Nikkei 225 historical data (5 years)...")
nikkei = yf.download("^N225", period="5y", auto_adjust=True)
nikkei = nikkei.droplevel('Ticker', axis=1) if isinstance(nikkei.columns, pd.MultiIndex) else nikkei

nikkei['DayOfWeek'] = nikkei.index.dayofweek
nikkei['Return'] = (nikkei['Close'] - nikkei['Open']) / nikkei['Open'] * 100
nikkei['Color'] = nikkei['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

print(f"Data: {nikkei.index[0].date()} → {nikkei.index[-1].date()} ({len(nikkei)} days)")

# ── Build Friday→Monday pairs ─────────────────────────────────────────────────
fridays = nikkei[nikkei['DayOfWeek'] == 4]
mondays = nikkei[nikkei['DayOfWeek'] == 0]

pairs = []
for fri_date, fri_row in fridays.iterrows():
    next_mons = mondays[mondays.index > fri_date]
    if len(next_mons) == 0:
        continue
    mon = next_mons.iloc[0]
    if (mon.name - fri_date).days > 5:
        continue
    pairs.append({
        'Friday_Date':   fri_date,
        'Friday_Return': fri_row['Return'],
        'Friday_Color':  fri_row['Color'],
        'Monday_Date':   mon.name,
        'Monday_Open':   mon['Open'],
        'Monday_High':   mon['High'],
        'Monday_Low':    mon['Low'],
        'Monday_Close':  mon['Close'],
    })

df = pd.DataFrame(pairs)
signals = df[df['Friday_Color'] == 'Red'].copy()

print(f"Total Friday→Monday pairs: {len(df)}")
print(f"Red Friday signals: {len(signals)}")
print(f"Green Friday (comparison): {len(df[df['Friday_Color'] == 'Green'])}")

# ── Baseline comparison ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("BASELINE: Does Red Friday predict Green Monday?")
print("="*70)

red_fri = df[df['Friday_Color'] == 'Red'].copy()
red_fri['Monday_Color'] = (red_fri['Monday_Close'] > red_fri['Monday_Open']).map({True: 'Green', False: 'Red'})
red_green = (red_fri['Monday_Color'] == 'Green').sum()
red_wr = red_green / len(red_fri) * 100

green_fri = df[df['Friday_Color'] == 'Green'].copy()
green_fri['Monday_Color'] = (green_fri['Monday_Close'] > green_fri['Monday_Open']).map({True: 'Green', False: 'Red'})
green_green = (green_fri['Monday_Color'] == 'Green').sum()
green_wr = green_green / len(green_fri) * 100

all_mon_wr = ((df['Monday_Close'] > df['Monday_Open']).sum() / len(df)) * 100

print(f"\n  Baseline (all Mondays green):        {all_mon_wr:.1f}%")
print(f"  After Red Friday → Monday green:     {red_wr:.1f}%  ({red_green}/{len(red_fri)})")
print(f"  After Green Friday → Monday green:   {green_wr:.1f}%  ({green_green}/{len(green_fri)})")

if red_wr > all_mon_wr:
    print(f"\n  ✅ Edge exists: Red Friday boosts Monday win rate by +{red_wr - all_mon_wr:.1f}%")
else:
    print(f"\n  ❌ No edge: Red Friday does NOT reliably predict Green Monday on Nikkei")

# ── Stop Loss Optimization ────────────────────────────────────────────────────
print("\n" + "="*70)
print("STOP LOSS OPTIMIZATION")
print("="*70)

signals['Max_Drawdown_Pct'] = (signals['Monday_Low'] - signals['Monday_Open']) / signals['Monday_Open'] * 100
signals['Max_Runup_Pct'] = (signals['Monday_High'] - signals['Monday_Open']) / signals['Monday_Open'] * 100
signals['Close_Return_Pct'] = (signals['Monday_Close'] - signals['Monday_Open']) / signals['Monday_Open'] * 100

print(f"\n  MAX ADVERSE EXCURSION (intraday drop from Monday open):")
print(f"    Mean:   {signals['Max_Drawdown_Pct'].mean():.3f}%")
print(f"    Median: {signals['Max_Drawdown_Pct'].median():.3f}%")
print(f"    Worst:  {signals['Max_Drawdown_Pct'].min():.3f}%")
print(f"    25th pct: {signals['Max_Drawdown_Pct'].quantile(0.25):.3f}%")
print(f"    10th pct: {signals['Max_Drawdown_Pct'].quantile(0.10):.3f}%")
print(f"    5th pct:  {signals['Max_Drawdown_Pct'].quantile(0.05):.3f}%")

stop_levels = np.arange(0.1, 3.05, 0.1)
results = []

for sl in stop_levels:
    trades = []
    for _, row in signals.iterrows():
        entry = row['Monday_Open']
        stop_price = entry * (1 - sl / 100)

        if row['Monday_Low'] <= stop_price:
            exit_price = stop_price
            stopped = True
        else:
            exit_price = row['Monday_Close']
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

    years = 5.0
    trades_per_year = n / years
    sharpe = (avg_ret / trades_df['return'].std()) * np.sqrt(trades_per_year) if trades_df['return'].std() > 0 else 0

    results.append({
        'Stop_Loss': sl,
        'Win_Rate': wins / n * 100,
        'Avg_Return': avg_ret,
        'Cumulative_Return': cumulative_ret,
        'Profit_Factor': pf,
        'Sharpe': sharpe,
        'Times_Stopped': stopped_count,
        'Stop_Pct': stopped_count / n * 100,
    })

res_df = pd.DataFrame(results)

no_sl_trades = signals['Close_Return_Pct']
no_sl_cum = ((1 + no_sl_trades / 100).prod() - 1) * 100
no_sl_wr = (no_sl_trades > 0).sum() / len(no_sl_trades) * 100
no_sl_gp = no_sl_trades[no_sl_trades > 0].sum()
no_sl_gl = abs(no_sl_trades[no_sl_trades < 0].sum())
no_sl_pf = no_sl_gp / no_sl_gl if no_sl_gl > 0 else float('inf')
no_sl_sharpe = (no_sl_trades.mean() / no_sl_trades.std()) * np.sqrt(len(signals) / 5.0) if no_sl_trades.std() > 0 else 0

print(f"\n  {'SL %':>5} {'Win%':>6} {'AvgRet':>8} {'CumRet':>9} {'PF':>6} {'Sharpe':>7} {'Stopped':>8}")
print("  " + "-"*60)
for _, r in res_df.iterrows():
    marker = " ◀ BEST" if r['Sharpe'] == res_df['Sharpe'].max() else ""
    print(f"  {r['Stop_Loss']:>4.1f}% {r['Win_Rate']:>5.1f}% {r['Avg_Return']:>+7.3f}% {r['Cumulative_Return']:>+8.2f}% {r['Profit_Factor']:>5.2f} {r['Sharpe']:>6.2f}  ({r['Times_Stopped']:.0f}x){marker}")
print(f"  {'None':>5} {no_sl_wr:>5.1f}% {no_sl_trades.mean():>+7.3f}% {no_sl_cum:>+8.2f}% {no_sl_pf:>5.2f} {no_sl_sharpe:>6.2f}  (0x)")

best = res_df.loc[res_df['Sharpe'].idxmax()]
print(f"\n  ✅ Best stop loss (by Sharpe): {best['Stop_Loss']:.1f}%")

# ── Full Backtest ─────────────────────────────────────────────────────────────
optimal_sl = best['Stop_Loss']

print(f"\n" + "="*70)
print(f"FULL BACKTEST — {optimal_sl:.1f}% Stop Loss")
print("="*70)

capital = 10000
initial = capital
equity = []
all_trades = []

for _, row in signals.iterrows():
    entry = row['Monday_Open']
    stop_price = entry * (1 - optimal_sl / 100)

    if row['Monday_Low'] <= stop_price:
        exit_price = stop_price
        exit_type = 'STOP'
    else:
        exit_price = row['Monday_Close']
        exit_type = 'CLOSE'

    ret = (exit_price - entry) / entry
    pnl = capital * ret
    capital += pnl

    all_trades.append({
        'Date':       row['Monday_Date'],
        'Entry':      entry,
        'Exit':       exit_price,
        'Exit_Type':  exit_type,
        'Return_Pct': ret * 100,
        'PnL':        pnl,
        'Capital':    capital,
        'Win':        ret > 0,
    })
    equity.append({'Date': row['Monday_Date'], 'Capital': capital})

bt = pd.DataFrame(all_trades)
eq = pd.DataFrame(equity)

total_return = (capital - initial) / initial * 100
years = (signals['Monday_Date'].iloc[-1] - signals['Monday_Date'].iloc[0]).days / 365.25
annual_ret = ((capital / initial) ** (1 / years) - 1) * 100
win_rate = bt['Win'].mean() * 100
avg_win = bt[bt['Win']]['Return_Pct'].mean()
avg_loss = bt[~bt['Win']]['Return_Pct'].mean()
max_win = bt['Return_Pct'].max()
max_loss = bt['Return_Pct'].min()
gross_profit = bt[bt['Win']]['PnL'].sum()
gross_loss = abs(bt[~bt['Win']]['PnL'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

eq['Peak'] = eq['Capital'].cummax()
eq['DD'] = (eq['Capital'] - eq['Peak']) / eq['Peak'] * 100
max_dd = eq['DD'].min()

sharpe = (bt['Return_Pct'].mean() / bt['Return_Pct'].std()) * np.sqrt(len(bt) / years) if bt['Return_Pct'].std() > 0 else 0
stops_hit = (bt['Exit_Type'] == 'STOP').sum()
stops_pct = stops_hit / len(bt) * 100

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  NIKKEI 225 — Red Friday → Buy Monday Strategy       │
  ├──────────────────────────────────────────────────────┤
  │  Signal:      SPY Friday closes RED (open > close)   │
  │  Entry:       Monday OPEN                            │
  │  Stop Loss:   {optimal_sl:.1f}% below entry                    │
  │  Take Profit: NONE (hold to Monday close)            │
  ├──────────────────────────────────────────────────────┤
  │  PERFORMANCE ({years:.1f} years)                          │
  ├──────────────────────────────────────────────────────┤
  │  Starting Capital:    ¥{initial:>10,.0f}              │
  │  Ending Capital:      ¥{capital:>10,.0f}              │
  │  Total Return:         {total_return:>+9.2f}%              │
  │  Annualized Return:    {annual_ret:>+9.2f}%              │
  │  Max Drawdown:         {max_dd:>+9.2f}%              │
  │  Sharpe Ratio:          {sharpe:>8.2f}               │
  │  Profit Factor:         {profit_factor:>8.2f}               │
  ├──────────────────────────────────────────────────────┤
  │  TRADE STATS                                         │
  ├──────────────────────────────────────────────────────┤
  │  Total Trades:          {len(bt):>8}               │
  │  Win Rate:              {win_rate:>7.1f}%               │
  │  Avg Winner:           {avg_win:>+9.3f}%              │
  │  Avg Loser:            {avg_loss:>+9.3f}%              │
  │  Best Trade:           {max_win:>+9.3f}%              │
  │  Worst Trade:          {max_loss:>+9.3f}%              │
  │  Times Stopped Out: {stops_hit:>4} ({stops_pct:.1f}%)              │
  └──────────────────────────────────────────────────────┘
""")

# ── Yearly Breakdown ──────────────────────────────────────────────────────────
print("  YEARLY BREAKDOWN:")
print(f"  {'Year':<6} {'Trades':>7} {'Wins':>5} {'Stops':>6} {'WinRate':>8} {'Return':>9} {'PnL':>12}")
print("  " + "-"*58)
bt['Year'] = bt['Date'].dt.year
for year, g in bt.groupby('Year'):
    print(f"  {year:<6} {len(g):>7} {g['Win'].sum():>5} {(g['Exit_Type']=='STOP').sum():>6} {g['Win'].mean()*100:>7.1f}% {g['Return_Pct'].sum():>+8.2f}% ¥{g['PnL'].sum():>+10,.0f}")

# ── SPY vs Nikkei Comparison ──────────────────────────────────────────────────
print(f"\n" + "="*70)
print("COMPARISON: SPY vs NIKKEI for Red Friday → Monday Strategy")
print("="*70)
print(f"""
  The same Red Friday signal is used for both.
  SPY signal day  = US Friday
  Nikkei trade day = Japan Monday (same calendar day as US Monday)

  Key difference:
  • SPY trades during US hours  (9:30 PM – 4:00 AM Thailand time)
  • Nikkei trades during Asia hours (8:00 AM – 3:00 PM Thailand time)

  If the Nikkei win rate is similar to SPY (~60%+), you can trade
  this strategy during normal waking hours in Thailand.
""")

# ── Charts ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18))
fig.suptitle(f'Nikkei 225 Backtest: Red Friday → Buy Monday (SL = {optimal_sl:.1f}%)',
             fontsize=15, fontweight='bold', y=0.98)

# 1. Equity curve
ax1 = fig.add_subplot(4, 2, (1, 2))
ax1.plot(eq['Date'], eq['Capital'], color='#26a69a', linewidth=2)
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                 where=eq['Capital'] >= initial, alpha=0.15, color='#26a69a')
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                 where=eq['Capital'] < initial, alpha=0.15, color='#ef5350')
ax1.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax1.set_title(f'Equity Curve (¥10K start → ¥{capital:,.0f})', fontsize=12)
ax1.set_ylabel('Portfolio Value (¥)')
ax1.grid(True, alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(4, 2, (3, 4))
ax2.fill_between(eq['Date'], eq['DD'], 0, color='#ef5350', alpha=0.4)
ax2.plot(eq['Date'], eq['DD'], color='#ef5350', linewidth=1)
ax2.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontsize=12)
ax2.set_ylabel('Drawdown %')
ax2.grid(True, alpha=0.3)

# 3. Sharpe by stop loss
ax3 = fig.add_subplot(4, 2, 5)
ax3.plot(res_df['Stop_Loss'], res_df['Sharpe'], 'o-', color='#1976d2', linewidth=2, markersize=5)
ax3.axvline(x=optimal_sl, color='#26a69a', linestyle='--', alpha=0.7, label=f'Best: {optimal_sl:.1f}%')
ax3.axhline(y=no_sl_sharpe, color='#9e9e9e', linestyle=':', alpha=0.7, label=f'No SL: {no_sl_sharpe:.2f}')
ax3.set_title('Sharpe Ratio by Stop Loss Level', fontsize=12)
ax3.set_xlabel('Stop Loss %')
ax3.set_ylabel('Sharpe Ratio')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Cumulative return by stop loss
ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(res_df['Stop_Loss'], res_df['Cumulative_Return'], 'o-', color='#ff9800', linewidth=2, markersize=5)
ax4.axvline(x=optimal_sl, color='#26a69a', linestyle='--', alpha=0.7, label=f'Best: {optimal_sl:.1f}%')
ax4.axhline(y=no_sl_cum, color='#9e9e9e', linestyle=':', alpha=0.7, label=f'No SL: {no_sl_cum:.1f}%')
ax4.set_title('Cumulative Return by Stop Loss Level', fontsize=12)
ax4.set_xlabel('Stop Loss %')
ax4.set_ylabel('Cumulative Return %')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Individual trade returns
ax5 = fig.add_subplot(4, 2, 7)
colors_pnl = ['#26a69a' if r > 0 else '#ef5350' for r in bt['Return_Pct']]
ax5.bar(range(len(bt)), bt['Return_Pct'], color=colors_pnl, width=1.0)
ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax5.axhline(y=-optimal_sl, color='#ef5350', linestyle='--', alpha=0.5, label=f'Stop: -{optimal_sl:.1f}%')
ax5.set_title('Individual Trade Returns', fontsize=12)
ax5.set_xlabel('Trade #')
ax5.set_ylabel('Return %')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Max adverse excursion
ax6 = fig.add_subplot(4, 2, 8)
ax6.scatter(signals['Max_Drawdown_Pct'], signals['Close_Return_Pct'],
            c=['#26a69a' if r > 0 else '#ef5350' for r in signals['Close_Return_Pct']],
            alpha=0.6, s=40, edgecolors='white', linewidth=0.5)
ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax6.axvline(x=-optimal_sl, color='#ef5350', linestyle='--', alpha=0.7, label=f'Stop: -{optimal_sl:.1f}%')
ax6.set_title('Max Adverse Excursion vs Final Return', fontsize=12)
ax6.set_xlabel('Max Intraday Drop from Open (%)')
ax6.set_ylabel('Close Return (%)')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/ashleighchua/trading analyses/nikkei_backtest.png', dpi=150, bbox_inches='tight')
print("\nChart saved to: nikkei_backtest.png")
print("✅ Analysis complete!")
