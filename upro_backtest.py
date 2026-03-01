"""
UPRO (3x Leveraged SPY) Day Trading Backtest
==============================================
Same signals as SPY strategy, but trading UPRO instead.
Uses SPY for signal detection, UPRO for execution.
Tests multiple stop loss levels (since 3x leverage needs wider stops).
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Download data ────────────────────────────────────────────────────────────
print("Downloading SPY + UPRO historical data...")
spy = yf.download("SPY", period="5y", auto_adjust=True)
upro = yf.download("UPRO", period="5y", auto_adjust=True)

spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy
upro = upro.droplevel('Ticker', axis=1) if isinstance(upro.columns, pd.MultiIndex) else upro

# Align dates
common_dates = spy.index.intersection(upro.index)
spy = spy.loc[common_dates]
upro = upro.loc[common_dates]

spy['DayOfWeek'] = spy.index.dayofweek
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

upro['DayOfWeek'] = upro.index.dayofweek
upro['Return'] = (upro['Close'] - upro['Open']) / upro['Open'] * 100

print(f"SPY:  {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)")
print(f"UPRO: {upro.index[0].date()} → {upro.index[-1].date()} ({len(upro)} days)")
print(f"UPRO latest price: ${upro['Close'].iloc[-1]:.2f}")

# ── Build signal trades ─────────────────────────────────────────────────────
# Signal map: if SPY is red on signal day → trade UPRO on trade day
SIGNALS = {
    0: [2],       # Red Monday → Buy Wednesday
    1: [4],       # Red Tuesday → Buy Friday
    2: [4, 0],    # Red Wednesday → Buy Friday + Monday
    3: [4],       # Red Thursday → Buy Friday
    4: [0],       # Red Friday → Buy Monday
}

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Collect all trade days
trade_dates = set()
trade_info = {}  # date → reason

for sig_dow, trade_dows in SIGNALS.items():
    sig_days = spy[(spy['DayOfWeek'] == sig_dow) & (spy['Color'] == 'Red')]
    for sig_date, sig_row in sig_days.iterrows():
        for trade_dow in trade_dows:
            # Find next occurrence of trade_dow
            future = upro[(upro.index > sig_date) & (upro['DayOfWeek'] == trade_dow)]
            if len(future) == 0:
                continue
            trade_date = future.index[0]
            if (trade_date - sig_date).days <= 7:
                trade_dates.add(trade_date)
                trade_info[trade_date] = f"Red {day_names[sig_dow]}"

trade_dates = sorted(trade_dates)
print(f"\nTotal signal trade days: {len(trade_dates)}")

# ══════════════════════════════════════════════════════════════════════════════
# STOP LOSS OPTIMIZATION FOR UPRO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STOP LOSS OPTIMIZATION — UPRO (3x leveraged)")
print("="*70)

# Max adverse excursion for UPRO
upro_trades = upro.loc[upro.index.isin(trade_dates)]
mae = (upro_trades['Low'] - upro_trades['Open']) / upro_trades['Open'] * 100

print(f"\n  UPRO Max Adverse Excursion on signal days:")
print(f"    Mean max drop from open: {mae.mean():.3f}%")
print(f"    Median:    {mae.median():.3f}%")
print(f"    Worst:     {mae.min():.3f}%")
print(f"    10th pct:  {mae.quantile(0.10):.3f}%")
print(f"    5th pct:   {mae.quantile(0.05):.3f}%")

# Test stop losses from 1% to 8%
stop_levels = np.arange(1.0, 8.5, 0.5)
results = []

for sl in stop_levels:
    capital = 1000
    initial = capital
    log = []

    for td in trade_dates:
        if td not in upro.index:
            continue
        row = upro.loc[td]
        entry = row['Open']
        stop_price = entry * (1 - sl / 100)

        if row['Low'] <= stop_price:
            exit_price = stop_price
        else:
            exit_price = row['Close']

        ret = (exit_price - entry) / entry
        pnl = capital * ret
        capital += pnl
        log.append({'return': ret * 100, 'win': ret > 0, 'pnl': pnl})

    tl = pd.DataFrame(log)
    if len(tl) == 0:
        continue

    n = len(tl)
    wr = tl['win'].mean() * 100
    avg = tl['return'].mean()
    cum = (capital - initial) / initial * 100
    gp = tl[tl['return'] > 0]['return'].sum()
    gl = abs(tl[tl['return'] < 0]['return'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    years = (trade_dates[-1] - trade_dates[0]).days / 365.25
    sharpe = (avg / tl['return'].std()) * np.sqrt(n / years) if tl['return'].std() > 0 else 0
    stopped = (tl['return'] <= -sl + 0.01).sum()

    results.append({
        'SL': sl, 'WinRate': wr, 'AvgRet': avg, 'CumRet': cum,
        'PF': pf, 'Sharpe': sharpe, 'Capital': capital, 'Stopped': stopped,
    })

# No stop loss baseline
capital_nosl = 1000
for td in trade_dates:
    if td not in upro.index:
        continue
    row = upro.loc[td]
    ret = (row['Close'] - row['Open']) / row['Open']
    capital_nosl += capital_nosl * ret

nosl_cum = (capital_nosl - 1000) / 1000 * 100

res_df = pd.DataFrame(results)

print(f"\n  {'SL %':>5} {'Win%':>7} {'AvgRet':>8} {'CumRet':>10} {'$1K→':>8} {'PF':>6} {'Sharpe':>7} {'Stopped':>8}")
print("  " + "-"*68)
for _, r in res_df.iterrows():
    marker = " ◀" if r['Sharpe'] == res_df['Sharpe'].max() else ""
    print(f"  {r['SL']:>4.1f}% {r['WinRate']:>6.1f}% {r['AvgRet']:>+7.3f}% {r['CumRet']:>+9.1f}% ${r['Capital']:>7,.0f} {r['PF']:>5.2f} {r['Sharpe']:>6.2f}  ({r['Stopped']:.0f}x){marker}")
print(f"  {'None':>5} {'':>7} {'':>8} {nosl_cum:>+9.1f}% ${capital_nosl:>7,.0f}")

best = res_df.loc[res_df['Sharpe'].idxmax()]
OPTIMAL_SL = best['SL']

# ══════════════════════════════════════════════════════════════════════════════
# FULL BACKTEST WITH OPTIMAL SL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"FULL BACKTEST — UPRO with {OPTIMAL_SL:.1f}% Stop Loss ($1,000 start)")
print("="*70)

capital = 1000
initial = capital
equity = []
all_trades = []

for td in trade_dates:
    if td not in upro.index:
        continue
    row = upro.loc[td]
    entry = row['Open']
    stop_price = entry * (1 - OPTIMAL_SL / 100)

    if row['Low'] <= stop_price:
        exit_price = stop_price
        exit_type = 'STOP'
    else:
        exit_price = row['Close']
        exit_type = 'CLOSE'

    ret = (exit_price - entry) / entry
    pnl = capital * ret
    capital += pnl

    all_trades.append({
        'Date': td,
        'Day': day_names[td.dayofweek],
        'Signal': trade_info.get(td, ''),
        'Entry': entry,
        'Exit': exit_price,
        'Exit_Type': exit_type,
        'Return': ret * 100,
        'PnL': pnl,
        'Capital': capital,
        'Win': ret > 0,
    })
    equity.append({'Date': td, 'Capital': capital})

bt = pd.DataFrame(all_trades)
eq = pd.DataFrame(equity)

total_ret = (capital - initial) / initial * 100
years = (trade_dates[-1] - trade_dates[0]).days / 365.25
annual = ((capital / initial) ** (1 / years) - 1) * 100
wr = bt['Win'].mean() * 100
avg_win = bt[bt['Win']]['Return'].mean()
avg_loss = bt[~bt['Win']]['Return'].mean()
gp = bt[bt['Win']]['PnL'].sum()
gl = abs(bt[~bt['Win']]['PnL'].sum())
pf = gp / gl if gl > 0 else float('inf')
eq['Peak'] = eq['Capital'].cummax()
eq['DD'] = (eq['Capital'] - eq['Peak']) / eq['Peak'] * 100
mdd = eq['DD'].min()
sharpe = (bt['Return'].mean() / bt['Return'].std()) * np.sqrt(len(bt) / years) if bt['Return'].std() > 0 else 0
stops = (bt['Exit_Type'] == 'STOP').sum()
tpy = len(bt) / years

# Also backtest SPY with same signals for comparison
capital_spy = 1000
spy_equity = []
for td in trade_dates:
    if td not in spy.index:
        continue
    row = spy.loc[td]
    entry = row['Open']
    stop_price = entry * (1 - 1.9 / 100)  # SPY uses 1.9% SL
    if row['Low'] <= stop_price:
        exit_price = stop_price
    else:
        exit_price = row['Close']
    ret = (exit_price - entry) / entry
    capital_spy += capital_spy * ret
    spy_equity.append({'Date': td, 'Capital': capital_spy})

spy_eq = pd.DataFrame(spy_equity)

print(f"""
  ┌───────────────────────────────────────────────────────┐
  │  UPRO (3x SPY) vs SPY — Same Signals, $1,000 Start   │
  ├───────────────────────────────────────────────────────┤
  │                         UPRO            SPY           │
  │  Ending Capital:    ${capital:>9,.2f}     ${capital_spy:>9,.2f}      │
  │  Total Return:      {total_ret:>+9.1f}%     {(capital_spy-1000)/10:>+9.1f}%      │
  │  Annualized:        {annual:>+9.1f}%     {((capital_spy/1000)**(1/years)-1)*100:>+9.1f}%      │
  │  Max Drawdown:      {mdd:>+9.1f}%                       │
  │  Sharpe Ratio:       {sharpe:>8.2f}                        │
  │  Profit Factor:      {pf:>8.2f}                        │
  ├───────────────────────────────────────────────────────┤
  │  TRADE STATS (UPRO)                                   │
  ├───────────────────────────────────────────────────────┤
  │  Total Trades:        {len(bt):>8}                        │
  │  Trades/Year:         {tpy:>8.1f}                        │
  │  Win Rate:            {wr:>7.1f}%                        │
  │  Avg Winner:         {avg_win:>+8.2f}%                        │
  │  Avg Loser:          {avg_loss:>+8.2f}%                        │
  │  Best Trade:         {bt['Return'].max():>+8.2f}%                        │
  │  Worst Trade:        {bt['Return'].min():>+8.2f}%                        │
  │  Stop Loss:           {OPTIMAL_SL:>7.1f}%                        │
  │  Times Stopped:    {stops:>4} ({stops/len(bt)*100:.1f}%)                        │
  └───────────────────────────────────────────────────────┘
""")

# Yearly breakdown
print("  YEARLY BREAKDOWN (UPRO):")
print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'Return':>9} {'PnL':>10} {'Capital':>10}")
print("  " + "-"*55)
bt['Year'] = bt['Date'].dt.year
running = initial
for year, g in bt.groupby('Year'):
    running += g['PnL'].sum()
    print(f"  {year:<6} {len(g):>7} {g['Win'].mean()*100:>6.1f}% {g['Return'].sum():>+8.2f}% ${g['PnL'].sum():>+8,.0f} ${running:>9,.0f}")

# Position sizing for $1K
print(f"\n  POSITION SIZING for $1,000 account:")
latest_upro_price = upro['Close'].iloc[-1]
shares = int(1000 / latest_upro_price)
position = shares * latest_upro_price
risk = position * OPTIMAL_SL / 100
print(f"    UPRO price: ~${latest_upro_price:.2f}")
print(f"    You can buy: {shares} shares (${position:.0f})")
print(f"    Risk per trade at {OPTIMAL_SL:.1f}% SL: ${risk:.0f}")
print(f"    That's {risk/1000*100:.1f}% of your $1,000 account")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 16))
fig.suptitle(f'UPRO (3x SPY) Day Trading Backtest — $1,000 Start', fontsize=15, fontweight='bold', y=0.98)

# 1. Equity curve: UPRO vs SPY
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(eq['Date'], eq['Capital'], color='#26a69a', linewidth=2, label=f'UPRO → ${capital:,.0f}')
ax1.plot(spy_eq['Date'], spy_eq['Capital'], color='#9e9e9e', linewidth=1.5, linestyle='--',
         label=f'SPY → ${capital_spy:,.0f}', alpha=0.7)
ax1.axhline(y=initial, color='gray', linestyle=':', alpha=0.3)
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                  where=eq['Capital'] >= initial, alpha=0.15, color='#26a69a')
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                  where=eq['Capital'] < initial, alpha=0.15, color='#ef5350')
ax1.set_title(f'$1,000 → UPRO ${capital:,.0f} vs SPY ${capital_spy:,.0f}', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(3, 2, 3)
ax2.fill_between(eq['Date'], eq['DD'], 0, color='#ef5350', alpha=0.4)
ax2.plot(eq['Date'], eq['DD'], color='#ef5350', linewidth=1)
ax2.set_title(f'UPRO Drawdown (Max: {mdd:.1f}%)', fontsize=12)
ax2.set_ylabel('Drawdown %')
ax2.grid(True, alpha=0.3)

# 3. Stop loss optimization
ax3 = fig.add_subplot(3, 2, 4)
ax3.plot(res_df['SL'], res_df['Sharpe'], 'o-', color='#1976d2', linewidth=2, markersize=5)
ax3.axvline(x=OPTIMAL_SL, color='#26a69a', linestyle='--', alpha=0.7, label=f'Best: {OPTIMAL_SL:.1f}%')
ax3.set_title('Sharpe Ratio by Stop Loss (UPRO)', fontsize=12)
ax3.set_xlabel('Stop Loss %')
ax3.set_ylabel('Sharpe Ratio')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Trade returns
ax4 = fig.add_subplot(3, 2, 5)
colors_pnl = ['#26a69a' if r > 0 else '#ef5350' for r in bt['Return']]
ax4.bar(range(len(bt)), bt['Return'], color=colors_pnl, width=1.0)
ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax4.set_title('Individual UPRO Trade Returns', fontsize=12)
ax4.set_xlabel('Trade #')
ax4.set_ylabel('Return %')
ax4.grid(True, alpha=0.3)

# 5. Monthly PnL
bt['Month'] = bt['Date'].dt.to_period('M')
monthly = bt.groupby('Month')['PnL'].sum()
ax5 = fig.add_subplot(3, 2, 6)
colors_m = ['#26a69a' if v > 0 else '#ef5350' for v in monthly.values]
ax5.bar(range(len(monthly)), monthly.values, color=colors_m, width=0.8)
ax5.set_title('Monthly PnL ($)', fontsize=12)
ax5.set_ylabel('PnL ($)')
ax5.set_xlabel('Month')
# Label every 6th month
tick_positions = list(range(0, len(monthly), 6))
tick_labels = [str(monthly.index[i]) for i in tick_positions]
ax5.set_xticks(tick_positions)
ax5.set_xticklabels(tick_labels, rotation=45, fontsize=8)
ax5.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/ashleighchua/trading analyses/upro_backtest.png', dpi=150, bbox_inches='tight')
print("\nCharts saved to: upro_backtest.png")
print("\nDone!")
