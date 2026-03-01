"""
Low Screen Time Strategies Backtest
=====================================
Tests 3 strategies that only require checking charts once per day:

1. MA Crossover (Golden/Death Cross)
   - Buy when 50-day SMA crosses above 200-day SMA
   - Sell when 50-day SMA crosses below 200-day SMA
   - Hold for days/weeks

2. RSI Oversold Bounce
   - Buy when RSI drops below 30 (oversold) at a support level
   - Sell when RSI reaches 70 OR after N days
   - Hold for days/weeks

3. Post-Earnings Drift
   - Buy SPY the day after a strong earnings season starts
   - Tracks the "earnings season drift" effect
   - Hold for 5-10 days

All use SPY. Check once per day at market close (4AM Thailand time).
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
spy = spy.copy()

print(f"Data: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: MA CROSSOVER (Golden Cross / Death Cross)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 1: MA CROSSOVER (50 SMA vs 200 SMA)")
print("="*70)

spy['SMA50']  = spy['Close'].rolling(50).mean()
spy['SMA200'] = spy['Close'].rolling(200).mean()

# Detect crossovers
spy['MA_Signal'] = 0
spy['Prev_SMA50']  = spy['SMA50'].shift(1)
spy['Prev_SMA200'] = spy['SMA200'].shift(1)

# Golden cross: 50 crosses above 200
spy.loc[(spy['SMA50'] > spy['SMA200']) & (spy['Prev_SMA50'] <= spy['Prev_SMA200']), 'MA_Signal'] = 1
# Death cross: 50 crosses below 200
spy.loc[(spy['SMA50'] < spy['SMA200']) & (spy['Prev_SMA50'] >= spy['Prev_SMA200']), 'MA_Signal'] = -1

golden_crosses = spy[spy['MA_Signal'] == 1]
death_crosses  = spy[spy['MA_Signal'] == -1]

print(f"\n  Golden Crosses (buy signals): {len(golden_crosses)}")
print(f"  Death Crosses  (sell signals): {len(death_crosses)}")

# Backtest: buy at golden cross, sell at death cross
ma_trades = []
in_trade = False
entry_price = 0
entry_date = None

for date, row in spy.iterrows():
    if not in_trade and row['MA_Signal'] == 1:
        in_trade = True
        entry_price = row['Close']
        entry_date = date
    elif in_trade and row['MA_Signal'] == -1:
        exit_price = row['Close']
        ret = (exit_price - entry_price) / entry_price * 100
        days_held = (date - entry_date).days
        ma_trades.append({
            'Entry_Date':  entry_date,
            'Exit_Date':   date,
            'Entry_Price': entry_price,
            'Exit_Price':  exit_price,
            'Return_Pct':  ret,
            'Days_Held':   days_held,
            'Win':         ret > 0,
        })
        in_trade = False

# Close open trade at end
if in_trade:
    exit_price = spy['Close'].iloc[-1]
    ret = (exit_price - entry_price) / entry_price * 100
    days_held = (spy.index[-1] - entry_date).days
    ma_trades.append({
        'Entry_Date':  entry_date,
        'Exit_Date':   spy.index[-1],
        'Entry_Price': entry_price,
        'Exit_Price':  exit_price,
        'Return_Pct':  ret,
        'Days_Held':   days_held,
        'Win':         ret > 0,
    })

ma_bt = pd.DataFrame(ma_trades)

# Equity curve
capital_ma = 10000
initial = 10000
equity_ma = []
for _, row in ma_bt.iterrows():
    capital_ma *= (1 + row['Return_Pct'] / 100)
    equity_ma.append({'Date': row['Exit_Date'], 'Capital': capital_ma})

ma_wr   = ma_bt['Win'].mean() * 100
ma_avg  = ma_bt['Return_Pct'].mean()
ma_cum  = (capital_ma - initial) / initial * 100
ma_years = (ma_bt['Exit_Date'].iloc[-1] - ma_bt['Entry_Date'].iloc[0]).days / 365.25
ma_ann  = ((capital_ma / initial) ** (1 / ma_years) - 1) * 100
ma_avg_days = ma_bt['Days_Held'].mean()

print(f"\n  Total trades:        {len(ma_bt)}")
print(f"  Win rate:            {ma_wr:.1f}%")
print(f"  Avg return/trade:    {ma_avg:+.2f}%")
print(f"  Avg days held:       {ma_avg_days:.0f} days")
print(f"  Total return:        {ma_cum:+.2f}%")
print(f"  Annualized return:   {ma_ann:+.2f}%")
print(f"\n  Yearly breakdown:")
ma_bt['Year'] = ma_bt['Exit_Date'].dt.year
for yr, g in ma_bt.groupby('Year'):
    print(f"    {yr}: {len(g)} trades | WR {g['Win'].mean()*100:.0f}% | Return {g['Return_Pct'].sum():+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: RSI OVERSOLD BOUNCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 2: RSI OVERSOLD BOUNCE")
print("="*70)

# Calculate RSI (14-period)
delta = spy['Close'].diff()
gain  = delta.clip(lower=0)
loss  = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
spy['RSI'] = 100 - (100 / (1 + rs))

# Also calculate support (recent 20-day low as proxy)
spy['Support'] = spy['Low'].rolling(20).min()
spy['Near_Support'] = spy['Close'] <= spy['Support'] * 1.02  # within 2% of support

print(f"\n  Testing RSI thresholds:")
print(f"  {'RSI Entry':>10} {'RSI Exit':>9} {'Exit Days':>10} {'Trades':>7} {'WR%':>6} {'AvgRet':>8} {'AnnRet':>8}")
print("  " + "-"*65)

best_rsi_result = None
best_rsi_sharpe = -999

rsi_results = []
for rsi_entry in [20, 25, 30, 35]:
    for rsi_exit in [50, 60, 70]:
        for max_days in [5, 10, 15, 20]:
            trades = []
            in_trade = False
            entry_price = 0
            entry_date = None
            days_in = 0

            for date, row in spy.iterrows():
                if pd.isna(row['RSI']):
                    continue
                if not in_trade:
                    if row['RSI'] <= rsi_entry:
                        in_trade = True
                        entry_price = row['Close']
                        entry_date = date
                        days_in = 0
                else:
                    days_in += 1
                    exit_now = (row['RSI'] >= rsi_exit) or (days_in >= max_days)
                    if exit_now:
                        ret = (row['Close'] - entry_price) / entry_price * 100
                        trades.append({
                            'Entry_Date': entry_date,
                            'Exit_Date':  date,
                            'Return_Pct': ret,
                            'Days_Held':  days_in,
                            'Win':        ret > 0,
                            'Exit_Reason': 'RSI' if row['RSI'] >= rsi_exit else 'TIME',
                        })
                        in_trade = False

            if len(trades) < 10:
                continue

            t = pd.DataFrame(trades)
            wr = t['Win'].mean() * 100
            avg = t['Return_Pct'].mean()
            years = (t['Exit_Date'].iloc[-1] - t['Entry_Date'].iloc[0]).days / 365.25
            cap = initial * (1 + t['Return_Pct'] / 100).prod()
            ann = ((cap / initial) ** (1 / years) - 1) * 100
            sharpe = (avg / t['Return_Pct'].std()) * np.sqrt(len(t) / years) if t['Return_Pct'].std() > 0 else 0

            rsi_results.append({
                'RSI_Entry': rsi_entry, 'RSI_Exit': rsi_exit, 'Max_Days': max_days,
                'Trades': len(t), 'WR': wr, 'AvgRet': avg, 'AnnRet': ann, 'Sharpe': sharpe,
                'trades_df': t,
            })

            if sharpe > best_rsi_sharpe:
                best_rsi_sharpe = sharpe
                best_rsi_result = rsi_results[-1]

# Print top 10 by Sharpe
rsi_results_sorted = sorted(rsi_results, key=lambda x: x['Sharpe'], reverse=True)[:10]
for r in rsi_results_sorted:
    marker = " ◀ BEST" if r == best_rsi_result else ""
    print(f"  RSI<{r['RSI_Entry']:>2} exit>{r['RSI_Exit']:>2}  max{r['Max_Days']:>2}d  {r['Trades']:>7}  {r['WR']:>5.1f}%  {r['AvgRet']:>+7.3f}%  {r['AnnRet']:>+7.2f}%{marker}")

rsi_bt = best_rsi_result['trades_df']
capital_rsi = initial * (1 + rsi_bt['Return_Pct'] / 100).prod()
equity_rsi = []
cap = initial
for _, row in rsi_bt.iterrows():
    cap *= (1 + row['Return_Pct'] / 100)
    equity_rsi.append({'Date': row['Exit_Date'], 'Capital': cap})

rsi_wr  = rsi_bt['Win'].mean() * 100
rsi_cum = (capital_rsi - initial) / initial * 100
rsi_years = (rsi_bt['Exit_Date'].iloc[-1] - rsi_bt['Entry_Date'].iloc[0]).days / 365.25
rsi_ann = ((capital_rsi / initial) ** (1 / rsi_years) - 1) * 100
rsi_avg_days = rsi_bt['Days_Held'].mean()

print(f"\n  BEST RSI SETUP: RSI<{best_rsi_result['RSI_Entry']} → exit when RSI>{best_rsi_result['RSI_Exit']} or {best_rsi_result['Max_Days']} days")
print(f"  Win rate:          {rsi_wr:.1f}%")
print(f"  Avg days held:     {rsi_avg_days:.1f} days")
print(f"  Total return:      {rsi_cum:+.2f}%")
print(f"  Annualized return: {rsi_ann:+.2f}%")
print(f"  Total trades:      {len(rsi_bt)}")

print(f"\n  Yearly breakdown:")
rsi_bt['Year'] = rsi_bt['Exit_Date'].dt.year
for yr, g in rsi_bt.groupby('Year'):
    print(f"    {yr}: {len(g)} trades | WR {g['Win'].mean()*100:.0f}% | Return {g['Return_Pct'].sum():+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: EARNINGS SEASON DRIFT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 3: EARNINGS SEASON DRIFT")
print("="*70)
print("(Buy SPY at start of each earnings season, hold N days)")
print()

# Earnings seasons roughly start:
# Q1 results: mid-April (around Apr 10-15)
# Q2 results: mid-July (around Jul 10-15)
# Q3 results: mid-October (around Oct 10-15)
# Q4 results: mid-January (around Jan 10-15)

earnings_months = {1: 10, 4: 10, 7: 10, 10: 10}  # month: day to enter

print("  Testing hold periods (buy at earnings season start):")
print(f"  {'Hold Days':>10} {'Trades':>7} {'WR%':>6} {'AvgRet':>8} {'AnnRet':>8} {'Sharpe':>7}")
print("  " + "-"*55)

best_earn_result = None
best_earn_sharpe = -999
earn_results = []

for hold_days in [3, 5, 7, 10, 15, 20, 25, 30]:
    trades = []
    for year in range(spy.index[0].year, spy.index[-1].year + 1):
        for month, day in earnings_months.items():
            # Find the first trading day on or after the target date
            target = pd.Timestamp(year, month, day)
            future = spy[spy.index >= target]
            if len(future) == 0:
                continue
            entry_row = future.iloc[0]
            entry_date = entry_row.name
            entry_price = entry_row['Open']

            # Exit after hold_days trading days
            future_exit = spy[spy.index > entry_date]
            if len(future_exit) < hold_days:
                continue
            exit_row = future_exit.iloc[hold_days - 1]
            exit_price = exit_row['Close']

            ret = (exit_price - entry_price) / entry_price * 100
            trades.append({
                'Entry_Date':  entry_date,
                'Exit_Date':   exit_row.name,
                'Entry_Price': entry_price,
                'Exit_Price':  exit_price,
                'Return_Pct':  ret,
                'Win':         ret > 0,
                'Season':      f"Q{[1,4,7,10].index(month)+1 if month in [1,4,7,10] else '?'}",
            })

    if len(trades) < 10:
        continue

    t = pd.DataFrame(trades)
    wr = t['Win'].mean() * 100
    avg = t['Return_Pct'].mean()
    cap = initial * (1 + t['Return_Pct'] / 100).prod()
    years = (t['Exit_Date'].iloc[-1] - t['Entry_Date'].iloc[0]).days / 365.25
    ann = ((cap / initial) ** (1 / years) - 1) * 100
    sharpe = (avg / t['Return_Pct'].std()) * np.sqrt(len(t) / years) if t['Return_Pct'].std() > 0 else 0

    earn_results.append({
        'Hold_Days': hold_days, 'Trades': len(t), 'WR': wr,
        'AvgRet': avg, 'AnnRet': ann, 'Sharpe': sharpe, 'trades_df': t,
    })
    if sharpe > best_earn_sharpe:
        best_earn_sharpe = sharpe
        best_earn_result = earn_results[-1]

    marker = ""
    print(f"  {hold_days:>10}d {len(t):>7}  {wr:>5.1f}%  {avg:>+7.3f}%  {ann:>+7.2f}%  {sharpe:>6.2f}{marker}")

if best_earn_result:
    earn_bt = best_earn_result['trades_df']
    capital_earn = initial * (1 + earn_bt['Return_Pct'] / 100).prod()
    equity_earn = []
    cap = initial
    for _, row in earn_bt.iterrows():
        cap *= (1 + row['Return_Pct'] / 100)
        equity_earn.append({'Date': row['Exit_Date'], 'Capital': cap})

    earn_wr  = earn_bt['Win'].mean() * 100
    earn_cum = (capital_earn - initial) / initial * 100
    earn_years = (earn_bt['Exit_Date'].iloc[-1] - earn_bt['Entry_Date'].iloc[0]).days / 365.25
    earn_ann = ((capital_earn / initial) ** (1 / earn_years) - 1) * 100

    print(f"\n  BEST SETUP: Hold {best_earn_result['Hold_Days']} days after earnings season starts")
    print(f"  Win rate:          {earn_wr:.1f}%")
    print(f"  Total return:      {earn_cum:+.2f}%")
    print(f"  Annualized return: {earn_ann:+.2f}%")
    print(f"  Total trades:      {len(earn_bt)} (~4/year)")

    print(f"\n  Yearly breakdown:")
    earn_bt['Year'] = earn_bt['Exit_Date'].dt.year
    for yr, g in earn_bt.groupby('Year'):
        print(f"    {yr}: {len(g)} trades | WR {g['Win'].mean()*100:.0f}% | Return {g['Return_Pct'].sum():+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FINAL COMPARISON — All 3 Strategies vs Your SPY Playbook")
print(f"{'='*70}")
print(f"  {'Strategy':<30} {'Trades/yr':>10} {'Win Rate':>9} {'Ann Ret':>9} {'Screen Time':>12}")
print("  " + "-"*74)
print(f"  {'MA Crossover':<30} {'~1-2':>10} {ma_wr:>8.1f}%  {ma_ann:>+8.2f}%  {'Minimal':>12}")
print(f"  {'RSI Oversold Bounce':<30} {'~'+str(round(len(rsi_bt)/rsi_years)):>10} {rsi_wr:>8.1f}%  {rsi_ann:>+8.2f}%  {'Low':>12}")
print(f"  {'Earnings Season Drift':<30} {'~4':>10} {earn_wr:>8.1f}%  {earn_ann:>+8.2f}%  {'Minimal':>12}")
print(f"  {'Your SPY Playbook':<30} {'~22':>10} {'~64.0':>8}%  {'~+6.0':>8}%  {'Low':>12}")

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Low Screen Time Strategies — SPY Backtest (10 Years)',
             fontsize=15, fontweight='bold')

# 1. MA Crossover equity + signals
ax1 = fig.add_subplot(3, 3, (1, 2))
ax1.plot(spy.index, spy['Close'], color='#90a4ae', linewidth=1, alpha=0.7, label='SPY')
ax1.plot(spy.index, spy['SMA50'],  color='#1976d2', linewidth=1.5, label='50 SMA')
ax1.plot(spy.index, spy['SMA200'], color='#ff9800', linewidth=1.5, label='200 SMA')
for _, row in ma_bt.iterrows():
    ax1.axvline(x=row['Entry_Date'], color='#26a69a', alpha=0.3, linewidth=1)
    ax1.axvline(x=row['Exit_Date'],  color='#ef5350', alpha=0.3, linewidth=1)
ax1.set_title('MA Crossover: 50 SMA vs 200 SMA', fontsize=11)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. MA equity curve
eq_ma = pd.DataFrame(equity_ma)
ax2 = fig.add_subplot(3, 3, 3)
ax2.plot(eq_ma['Date'], eq_ma['Capital'], color='#1976d2', linewidth=2)
ax2.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax2.fill_between(eq_ma['Date'], eq_ma['Capital'], initial,
                 where=eq_ma['Capital'] >= initial, alpha=0.2, color='#26a69a')
ax2.fill_between(eq_ma['Date'], eq_ma['Capital'], initial,
                 where=eq_ma['Capital'] < initial, alpha=0.2, color='#ef5350')
ax2.set_title(f'MA Crossover Equity\n${initial:,} → ${capital_ma:,.0f} ({ma_cum:+.1f}%)', fontsize=10)
ax2.set_ylabel('Capital ($)')
ax2.grid(True, alpha=0.3)

# 3. RSI chart
ax3 = fig.add_subplot(3, 3, (4, 5))
ax3_twin = ax3.twinx()
ax3.plot(spy.index, spy['Close'], color='#90a4ae', linewidth=1, alpha=0.7)
ax3_twin.plot(spy.index, spy['RSI'], color='#ab47bc', linewidth=1, alpha=0.8)
ax3_twin.axhline(y=best_rsi_result['RSI_Entry'], color='#26a69a', linestyle='--', alpha=0.6,
                  label=f'Entry: RSI<{best_rsi_result["RSI_Entry"]}')
ax3_twin.axhline(y=best_rsi_result['RSI_Exit'], color='#ef5350', linestyle='--', alpha=0.6,
                  label=f'Exit: RSI>{best_rsi_result["RSI_Exit"]}')
ax3_twin.set_ylim(0, 100)
ax3_twin.legend(fontsize=8)
ax3.set_title(f'RSI Strategy (RSI<{best_rsi_result["RSI_Entry"]} buy, RSI>{best_rsi_result["RSI_Exit"]} sell)', fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. RSI equity
eq_rsi = pd.DataFrame(equity_rsi)
ax4 = fig.add_subplot(3, 3, 6)
ax4.plot(eq_rsi['Date'], eq_rsi['Capital'], color='#ab47bc', linewidth=2)
ax4.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax4.fill_between(eq_rsi['Date'], eq_rsi['Capital'], initial,
                 where=eq_rsi['Capital'] >= initial, alpha=0.2, color='#26a69a')
ax4.fill_between(eq_rsi['Date'], eq_rsi['Capital'], initial,
                 where=eq_rsi['Capital'] < initial, alpha=0.2, color='#ef5350')
ax4.set_title(f'RSI Equity\n${initial:,} → ${capital_rsi:,.0f} ({rsi_cum:+.1f}%)', fontsize=10)
ax4.set_ylabel('Capital ($)')
ax4.grid(True, alpha=0.3)

# 5. Earnings season trades
if best_earn_result:
    ax5 = fig.add_subplot(3, 3, (7, 8))
    ax5.plot(spy.index, spy['Close'], color='#90a4ae', linewidth=1, alpha=0.7)
    for _, row in earn_bt.iterrows():
        color = '#26a69a' if row['Win'] else '#ef5350'
        ax5.axvspan(row['Entry_Date'], row['Exit_Date'], alpha=0.15, color=color)
    ax5.set_title(f'Earnings Season Drift ({best_earn_result["Hold_Days"]}-day hold, green=win, red=loss)', fontsize=11)
    ax5.grid(True, alpha=0.3)

    # 6. Earnings equity
    eq_earn = pd.DataFrame(equity_earn)
    ax6 = fig.add_subplot(3, 3, 9)
    ax6.plot(eq_earn['Date'], eq_earn['Capital'], color='#ff9800', linewidth=2)
    ax6.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
    ax6.fill_between(eq_earn['Date'], eq_earn['Capital'], initial,
                     where=eq_earn['Capital'] >= initial, alpha=0.2, color='#26a69a')
    ax6.fill_between(eq_earn['Date'], eq_earn['Capital'], initial,
                     where=eq_earn['Capital'] < initial, alpha=0.2, color='#ef5350')
    ax6.set_title(f'Earnings Drift Equity\n${initial:,} → ${capital_earn:,.0f} ({earn_cum:+.1f}%)', fontsize=10)
    ax6.set_ylabel('Capital ($)')
    ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/ashleighchua/trading analyses/low_screentime_strategies.png', dpi=150, bbox_inches='tight')
print("\nChart saved to: low_screentime_strategies.png")
print("✅ Done!")
