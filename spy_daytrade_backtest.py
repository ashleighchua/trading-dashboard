"""
SPY Day Trading Backtest: Red Friday → Buy Monday Open
=======================================================
Strategy:
  - If Friday closes red, buy SPY at Monday open
  - Hold until Monday close (end of day)
  - Stop loss triggers intraday (using Monday's low as proxy)
  - No take profit limit — let winners run to close

Uses intraday High/Low to simulate stop loss hits.
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Download SPY data ────────────────────────────────────────────────────────
print("Downloading SPY historical data (5 years)...")
spy = yf.download("SPY", period="5y", auto_adjust=True)
spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy

spy['DayOfWeek'] = spy.index.dayofweek
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

print(f"Data: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)")

# ── Build Friday→Monday pairs with OHLC ─────────────────────────────────────
fridays = spy[spy['DayOfWeek'] == 4]
mondays = spy[spy['DayOfWeek'] == 0]

pairs = []
for fri_date, fri_row in fridays.iterrows():
    next_mons = mondays[mondays.index > fri_date]
    if len(next_mons) == 0:
        continue
    mon = next_mons.iloc[0]
    if (mon.name - fri_date).days > 5:
        continue
    pairs.append({
        'Friday_Date': fri_date,
        'Friday_Return': fri_row['Return'],
        'Friday_Color': fri_row['Color'],
        'Monday_Date': mon.name,
        'Monday_Open': mon['Open'],
        'Monday_High': mon['High'],
        'Monday_Low': mon['Low'],
        'Monday_Close': mon['Close'],
    })

df = pd.DataFrame(pairs)
signals = df[df['Friday_Color'] == 'Red'].copy()

print(f"Total red Friday signals: {len(signals)}")

# ══════════════════════════════════════════════════════════════════════════════
# STOP LOSS OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STOP LOSS OPTIMIZATION")
print("="*70)

# For each trade, calculate: how far did price drop from open (max adverse excursion)?
signals['Max_Drawdown_Pct'] = (signals['Monday_Low'] - signals['Monday_Open']) / signals['Monday_Open'] * 100
signals['Max_Runup_Pct'] = (signals['Monday_High'] - signals['Monday_Open']) / signals['Monday_Open'] * 100
signals['Close_Return_Pct'] = (signals['Monday_Close'] - signals['Monday_Open']) / signals['Monday_Open'] * 100

print(f"\n  MAX ADVERSE EXCURSION (how far price drops intraday from your entry):")
print(f"    Mean max drawdown from open: {signals['Max_Drawdown_Pct'].mean():.3f}%")
print(f"    Median max drawdown from open: {signals['Max_Drawdown_Pct'].median():.3f}%")
print(f"    Worst intraday drop from open: {signals['Max_Drawdown_Pct'].min():.3f}%")
print(f"    25th percentile: {signals['Max_Drawdown_Pct'].quantile(0.25):.3f}%")
print(f"    10th percentile: {signals['Max_Drawdown_Pct'].quantile(0.10):.3f}%")
print(f"    5th percentile:  {signals['Max_Drawdown_Pct'].quantile(0.05):.3f}%")

# Test stop loss levels from 0.1% to 2.0%
stop_levels = np.arange(0.1, 2.05, 0.1)
results = []

for sl in stop_levels:
    trades = []
    for _, row in signals.iterrows():
        entry = row['Monday_Open']
        stop_price = entry * (1 - sl / 100)

        # Did the low hit the stop loss?
        if row['Monday_Low'] <= stop_price:
            # Stop loss triggered — exit at stop price
            exit_price = stop_price
            stopped = True
        else:
            # No stop hit — exit at close
            exit_price = row['Monday_Close']
            stopped = False

        ret = (exit_price - entry) / entry * 100
        trades.append({
            'return': ret,
            'stopped': stopped,
            'win': ret > 0,
        })

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wins = trades_df['win'].sum()
    stopped_count = trades_df['stopped'].sum()
    total_ret = (1 + trades_df['return'] / 100).prod()
    cumulative_ret = (total_ret - 1) * 100
    avg_ret = trades_df['return'].mean()

    # Profit factor
    gross_profit = trades_df[trades_df['return'] > 0]['return'].sum()
    gross_loss = abs(trades_df[trades_df['return'] < 0]['return'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe (annualized, ~19 trades/year)
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

# Also add "no stop loss" as baseline
no_sl_trades = signals['Close_Return_Pct']
no_sl_cum = ((1 + no_sl_trades / 100).prod() - 1) * 100
no_sl_wr = (no_sl_trades > 0).sum() / len(no_sl_trades) * 100
no_sl_gp = no_sl_trades[no_sl_trades > 0].sum()
no_sl_gl = abs(no_sl_trades[no_sl_trades < 0].sum())
no_sl_pf = no_sl_gp / no_sl_gl if no_sl_gl > 0 else float('inf')
no_sl_sharpe = (no_sl_trades.mean() / no_sl_trades.std()) * np.sqrt(len(signals) / 5.0) if no_sl_trades.std() > 0 else 0

print(f"\n  {'SL %':>5} {'Win%':>6} {'AvgRet':>8} {'CumRet':>9} {'PF':>6} {'Sharpe':>7} {'Stopped':>8}")
print("  " + "-"*55)
for _, r in res_df.iterrows():
    marker = " ◀" if r['Sharpe'] == res_df['Sharpe'].max() else ""
    print(f"  {r['Stop_Loss']:>4.1f}% {r['Win_Rate']:>5.1f}% {r['Avg_Return']:>+7.3f}% {r['Cumulative_Return']:>+8.2f}% {r['Profit_Factor']:>5.2f} {r['Sharpe']:>6.2f}  ({r['Times_Stopped']:.0f}x){marker}")
print(f"  {'None':>5} {no_sl_wr:>5.1f}% {no_sl_trades.mean():>+7.3f}% {no_sl_cum:>+8.2f}% {no_sl_pf:>5.2f} {no_sl_sharpe:>6.2f}  (0x)")

best = res_df.loc[res_df['Sharpe'].idxmax()]
print(f"\n  ✅ BEST STOP LOSS (by Sharpe): {best['Stop_Loss']:.1f}%")
print(f"     Sharpe: {best['Sharpe']:.2f} | Win Rate: {best['Win_Rate']:.1f}% | Cum Return: {best['Cumulative_Return']:+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# FULL BACKTEST WITH OPTIMAL STOP LOSS
# ══════════════════════════════════════════════════════════════════════════════
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
        'Date': row['Monday_Date'],
        'Entry': entry,
        'Exit': exit_price,
        'Exit_Type': exit_type,
        'Return_Pct': ret * 100,
        'PnL': pnl,
        'Capital': capital,
        'Win': ret > 0,
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
  ┌──────────────────────────────────────────────┐
  │  STRATEGY RULES                              │
  │  Signal:     Buy if Friday closed RED        │
  │  Entry:      Monday OPEN                     │
  │  Stop Loss:  {optimal_sl:.1f}% below entry{' '*18}│
  │  Take Profit: NONE (hold to close)           │
  │  Exit:       Monday CLOSE (if stop not hit)  │
  ├──────────────────────────────────────────────┤
  │  PERFORMANCE ({years:.1f} years)                    │
  ├──────────────────────────────────────────────┤
  │  Starting Capital:     ${initial:>10,.2f}        │
  │  Ending Capital:       ${capital:>10,.2f}        │
  │  Total Return:          {total_return:>+9.2f}%        │
  │  Annualized Return:     {annual_ret:>+9.2f}%        │
  │  Max Drawdown:          {max_dd:>+9.2f}%        │
  │  Sharpe Ratio:           {sharpe:>8.2f}         │
  │  Profit Factor:          {profit_factor:>8.2f}         │
  ├──────────────────────────────────────────────┤
  │  TRADE STATS                                 │
  ├──────────────────────────────────────────────┤
  │  Total Trades:           {len(bt):>8}         │
  │  Win Rate:               {win_rate:>7.1f}%         │
  │  Avg Winner:            {avg_win:>+9.3f}%        │
  │  Avg Loser:             {avg_loss:>+9.3f}%        │
  │  Best Trade:            {max_win:>+9.3f}%        │
  │  Worst Trade:           {max_loss:>+9.3f}%        │
  │  Times Stopped Out:  {stops_hit:>4} ({stops_pct:.1f}%)         │
  └──────────────────────────────────────────────┘
""")

# Yearly breakdown
print("  YEARLY BREAKDOWN:")
print(f"  {'Year':<6} {'Trades':>7} {'Wins':>5} {'Stops':>6} {'WinRate':>8} {'Return':>9} {'PnL':>12}")
print("  " + "-"*58)
bt['Year'] = bt['Date'].dt.year
for year, g in bt.groupby('Year'):
    print(f"  {year:<6} {len(g):>7} {g['Win'].sum():>5} {(g['Exit_Type']=='STOP').sum():>6} {g['Win'].mean()*100:>7.1f}% {g['Return_Pct'].sum():>+8.2f}% ${g['PnL'].sum():>+10,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# PRACTICAL TRADING PLAN
# ══════════════════════════════════════════════════════════════════════════════
# Calculate position sizing for common account sizes
print(f"\n" + "="*70)
print("PRACTICAL TRADING PLAN")
print("="*70)

print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  YOUR DAY TRADING CHECKLIST                          │
  │                                                      │
  │  WHEN: Every Friday at market close (4PM ET)         │
  │                                                      │
  │  CHECK: Did SPY close RED today?                     │
  │    • Compare Close vs Open (NOT vs yesterday)        │
  │    • If YES → prepare Monday trade                   │
  │    • If NO  → no trade, enjoy your weekend           │
  │                                                      │
  │  MONDAY MORNING:                                     │
  │    1. Buy SPY (or SPY calls) at MARKET OPEN          │
  │    2. Immediately set stop loss at -{optimal_sl:.1f}%          │
  │       from your entry price                          │
  │    3. DO NOT set a take profit                       │
  │    4. Walk away — let it ride                        │
  │                                                      │
  │  MONDAY CLOSE (3:55-4:00 PM ET):                     │
  │    5. If still in trade, SELL at market close        │
  │    6. Done. Log the trade.                           │
  └──────────────────────────────────────────────────────┘

  POSITION SIZING (1% risk per trade):
""")

for acct_size in [1000, 5000, 10000, 25000, 50000]:
    risk_amount = acct_size * 0.01  # 1% risk
    # With a stop loss of optimal_sl%, position size = risk / stop_pct
    position_size = risk_amount / (optimal_sl / 100)
    shares = int(position_size / signals['Monday_Open'].iloc[-1])  # latest price
    latest_price = signals['Monday_Open'].iloc[-1]
    print(f"    ${acct_size:>6,} account → Position: ${position_size:>8,.0f} ({shares} shares @ ~${latest_price:.0f})")
    print(f"    {'':>19}   Risk per trade: ${risk_amount:>6,.0f} ({optimal_sl:.1f}% SL)")

print(f"""
  ⚠️  IMPORTANT NOTES:
  • This strategy only trades ~19 times per year
  • You're in the market for only ~1 day per trade
  • Most of the time you're sitting in CASH
  • Consider combining with other strategies for more action
  • Past performance does not guarantee future results
  • Start paper trading first before using real money
""")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 20))
fig.suptitle(f'SPY Day Trade Backtest: Red Friday → Buy Monday (SL = {optimal_sl:.1f}%)',
             fontsize=15, fontweight='bold', y=0.98)

# 1. Equity curve
ax1 = fig.add_subplot(4, 2, (1, 2))
ax1.plot(eq['Date'], eq['Capital'], color='#26a69a', linewidth=2)
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                  where=eq['Capital'] >= initial, alpha=0.15, color='#26a69a')
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                  where=eq['Capital'] < initial, alpha=0.15, color='#ef5350')
ax1.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax1.set_title(f'Equity Curve ($10K start → ${capital:,.0f})', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)')
ax1.grid(True, alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(4, 2, (3, 4))
ax2.fill_between(eq['Date'], eq['DD'], 0, color='#ef5350', alpha=0.4)
ax2.plot(eq['Date'], eq['DD'], color='#ef5350', linewidth=1)
ax2.set_title(f'Drawdown (Max: {max_dd:.2f}%)', fontsize=12)
ax2.set_ylabel('Drawdown %')
ax2.grid(True, alpha=0.3)

# 3. Stop loss optimization chart
ax3 = fig.add_subplot(4, 2, 5)
ax3.plot(res_df['Stop_Loss'], res_df['Sharpe'], 'o-', color='#1976d2', linewidth=2, markersize=5)
ax3.axvline(x=optimal_sl, color='#26a69a', linestyle='--', alpha=0.7, label=f'Best: {optimal_sl:.1f}%')
ax3.axhline(y=no_sl_sharpe, color='#9e9e9e', linestyle=':', alpha=0.7, label=f'No SL: {no_sl_sharpe:.2f}')
ax3.set_title('Sharpe Ratio by Stop Loss Level', fontsize=12)
ax3.set_xlabel('Stop Loss %')
ax3.set_ylabel('Sharpe Ratio')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Cumulative return by SL
ax4 = fig.add_subplot(4, 2, 6)
ax4.plot(res_df['Stop_Loss'], res_df['Cumulative_Return'], 'o-', color='#ff9800', linewidth=2, markersize=5)
ax4.axvline(x=optimal_sl, color='#26a69a', linestyle='--', alpha=0.7, label=f'Best: {optimal_sl:.1f}%')
ax4.axhline(y=no_sl_cum, color='#9e9e9e', linestyle=':', alpha=0.7, label=f'No SL: {no_sl_cum:.1f}%')
ax4.set_title('Cumulative Return by Stop Loss Level', fontsize=12)
ax4.set_xlabel('Stop Loss %')
ax4.set_ylabel('Cumulative Return %')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. Trade PnL waterfall
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
ax6.set_xlabel('Max Intraday Drawdown from Open (%)')
ax6.set_ylabel('Close-to-Open Return (%)')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/ashleighchua/trading analyses/spy_daytrade_backtest.png', dpi=150, bbox_inches='tight')
print("Charts saved to: spy_daytrade_backtest.png")
print("\n✅ Analysis complete!")
