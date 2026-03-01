"""
Thursday Flip Strategy Backtest
================================
Hypothesis: If Tuesday AND Wednesday are both green (especially big green),
Thursday is likely to flip red — mean reversion after 2 consecutive up days.

Tests:
  1. Tue green + Wed green → does Thursday flip red?
  2. Does magnitude matter? (bigger green = stronger flip?)
  3. Backtest shorting Thursday open, covering at Thursday close
  4. Combined filter: Tue + Wed both green AND above certain % threshold
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
    # Find same-week Wednesday
    next_weds = wednesdays[(wednesdays.index > tue_date)]
    if len(next_weds) == 0:
        continue
    wed = next_weds.iloc[0]
    if (wed.name - tue_date).days > 3:
        continue

    # Find same-week Thursday
    next_thus = thursdays[(thursdays.index > wed.name)]
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
        'Combined_Return': tue_row['Return'] + wed['Return'],
    })

df = pd.DataFrame(triplets)
print(f"Total Tue→Wed→Thu triplets: {len(df)}")

# ── BASELINE ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BASELINE: Thursday Color Distribution")
print("="*70)

baseline_red = (df['Thu_Color'] == 'Red').sum() / len(df) * 100
baseline_green = 100 - baseline_red
print(f"\n  All Thursdays:  Green {baseline_green:.1f}%  |  Red {baseline_red:.1f}%")

# ── CONDITION 1: Both Tue + Wed green ────────────────────────────────────────
print("\n" + "="*70)
print("CONDITION 1: Tue + Wed Color Combinations → Thursday")
print("="*70)

combos = [
    ('Green', 'Green', 'Both green'),
    ('Green', 'Red',   'Tue green, Wed red'),
    ('Red',   'Green', 'Tue red, Wed green'),
    ('Red',   'Red',   'Both red'),
]

print(f"\n  {'Combination':<25} {'Count':>6} {'Thu Green%':>11} {'Thu Red%':>9} {'Avg Thu Ret':>12}")
print("  " + "-"*66)
for tc, wc, label in combos:
    sub = df[(df['Tue_Color'] == tc) & (df['Wed_Color'] == wc)]
    if len(sub) == 0:
        continue
    g = (sub['Thu_Color'] == 'Green').sum() / len(sub) * 100
    r = 100 - g
    avg = sub['Thu_Return'].mean()
    marker = " ◀" if tc == 'Green' and wc == 'Green' else ""
    print(f"  {label:<25} {len(sub):>6} {g:>10.1f}% {r:>8.1f}% {avg:>+11.3f}%{marker}")

# ── CONDITION 2: Magnitude of Tue+Wed combined gain ──────────────────────────
print("\n" + "="*70)
print("CONDITION 2: Combined Tue+Wed Return Magnitude → Thursday")
print("="*70)
print("(Only when BOTH Tue AND Wed are green)")

both_green = df[(df['Tue_Color'] == 'Green') & (df['Wed_Color'] == 'Green')].copy()

buckets = [
    (0,   0.5, 'Small green (0–0.5% combined)'),
    (0.5, 1.0, 'Mild green (0.5–1.0%)'),
    (1.0, 2.0, 'Moderate green (1–2%)'),
    (2.0, 99,  'Big green (2%+)'),
]

print(f"\n  {'Combined Tue+Wed Return':<32} {'Count':>6} {'Thu Green%':>11} {'Thu Red%':>9} {'Avg Thu Ret':>12}")
print("  " + "-"*73)
for lo, hi, label in buckets:
    mask = (both_green['Combined_Return'] >= lo) & (both_green['Combined_Return'] < hi)
    sub = both_green[mask]
    if len(sub) == 0:
        continue
    g = (sub['Thu_Color'] == 'Green').sum() / len(sub) * 100
    r = 100 - g
    avg = sub['Thu_Return'].mean()
    marker = " ◀ STRONGEST FLIP" if r == max([(100-(both_green[(both_green['Combined_Return'] >= l) & (both_green['Combined_Return'] < h)]['Thu_Color'] == 'Green').sum() / max(len(both_green[(both_green['Combined_Return'] >= l) & (both_green['Combined_Return'] < h)]),1) * 100) for l,h,_ in buckets]) else ""
    print(f"  {label:<32} {len(sub):>6} {g:>10.1f}% {r:>8.1f}% {avg:>+11.3f}%{marker}")

# ── CONDITION 3: Individual day magnitude ─────────────────────────────────────
print("\n" + "="*70)
print("CONDITION 3: Both days must be green by at least X%")
print("="*70)

thresholds = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
print(f"\n  {'Min each day':>13} {'Count':>6} {'Thu Green%':>11} {'Thu Red%':>9} {'Avg Thu Ret':>12}")
print("  " + "-"*55)
for thresh in thresholds:
    mask = (df['Tue_Return'] >= thresh) & (df['Wed_Return'] >= thresh)
    sub = df[mask]
    if len(sub) == 0:
        continue
    g = (sub['Thu_Color'] == 'Green').sum() / len(sub) * 100
    r = 100 - g
    avg = sub['Thu_Return'].mean()
    print(f"  Both >= {thresh:.1f}%      {len(sub):>6} {g:>10.1f}% {r:>8.1f}% {avg:>+11.3f}%")

# ── FULL BACKTEST: Short Thursday when Tue+Wed both green ────────────────────
print("\n" + "="*70)
print("FULL BACKTEST: Short SPY at Thursday Open, Cover at Thursday Close")
print("(Signal: Tue green + Wed green, combined return >= 0.5%)")
print("="*70)

# Use 0.5% combined as the filter based on findings above
signal_df = both_green[both_green['Combined_Return'] >= 0.5].copy()
print(f"\nSignals (Tue+Wed combined >= 0.5%): {len(signal_df)}")
print(f"Trades per year: ~{len(signal_df)/10:.0f}")

# Stop loss optimization for SHORT trades
stop_levels = np.arange(0.1, 3.05, 0.1)
results = []

for sl in stop_levels:
    trades = []
    for _, row in signal_df.iterrows():
        entry = row['Thu_Open']
        stop_price = entry * (1 + sl / 100)  # stop is ABOVE entry for shorts

        # Did price hit stop (Thu High >= stop)?
        if row['Thu_High'] >= stop_price:
            exit_price = stop_price
            stopped = True
        else:
            exit_price = row['Thu_Close']
            stopped = False

        # Short trade: profit when price goes DOWN
        ret = (entry - exit_price) / entry * 100
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
    })

res_df = pd.DataFrame(results)

# No stop baseline (short)
no_sl_trades = (signal_df['Thu_Open'] - signal_df['Thu_Close']) / signal_df['Thu_Open'] * 100
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

# ── Full backtest with optimal SL ────────────────────────────────────────────
capital = 10000
initial = capital
equity = []
all_trades = []

for _, row in signal_df.iterrows():
    entry = row['Thu_Open']
    stop_price = entry * (1 + optimal_sl / 100)

    if row['Thu_High'] >= stop_price:
        exit_price = stop_price
        exit_type = 'STOP'
    else:
        exit_price = row['Thu_Close']
        exit_type = 'CLOSE'

    ret = (entry - exit_price) / entry
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
  │  THURSDAY FLIP — Short Strategy                      │
  ├──────────────────────────────────────────────────────┤
  │  Signal:      Tue green + Wed green (combined≥0.5%)  │
  │  Entry:       SHORT at Thursday OPEN                 │
  │  Stop Loss:   {optimal_sl:.1f}% above entry                    │
  │  Exit:        Cover at Thursday CLOSE                │
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

# Yearly breakdown
print("  YEARLY BREAKDOWN:")
print(f"  {'Year':<6} {'Trades':>7} {'Wins':>5} {'Stops':>6} {'WinRate':>8} {'Return':>9} {'PnL':>12}")
print("  " + "-"*58)
bt['Year'] = bt['Date'].dt.year
for year, g in bt.groupby('Year'):
    print(f"  {year:<6} {len(g):>7} {g['Win'].sum():>5} {(g['Exit_Type']=='STOP').sum():>6} {g['Win'].mean()*100:>7.1f}% {g['Return_Pct'].sum():>+8.2f}% ${g['PnL'].sum():>+10,.2f}")

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 16))
fig.suptitle('Thursday Flip: Short SPY when Tue+Wed Both Green', fontsize=15, fontweight='bold')

# 1. Equity curve
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(eq['Date'], eq['Capital'], color='#ab47bc', linewidth=2)
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                 where=eq['Capital'] >= initial, alpha=0.15, color='#26a69a')
ax1.fill_between(eq['Date'], eq['Capital'], initial,
                 where=eq['Capital'] < initial, alpha=0.15, color='#ef5350')
ax1.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax1.set_title(f'Equity Curve ($10K start → ${capital:,.0f})', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)')
ax1.grid(True, alpha=0.3)

# 2. Thu red rate by combo
ax2 = fig.add_subplot(3, 2, 3)
combo_labels = ['Both\nGreen', 'Tue G\nWed R', 'Tue R\nWed G', 'Both\nRed']
combo_red_rates = []
for tc, wc, label in combos:
    sub = df[(df['Tue_Color'] == tc) & (df['Wed_Color'] == wc)]
    if len(sub) > 0:
        combo_red_rates.append((sub['Thu_Color'] == 'Red').sum() / len(sub) * 100)
    else:
        combo_red_rates.append(0)
colors_bar = ['#ef5350' if r > baseline_red else '#26a69a' for r in combo_red_rates]
bars = ax2.bar(combo_labels, combo_red_rates, color=colors_bar, width=0.5)
ax2.axhline(y=baseline_red, color='gray', linestyle='--', alpha=0.7, label=f'Baseline: {baseline_red:.1f}%')
ax2.set_ylabel('Thursday Red %')
ax2.set_title('Thursday Red Rate by Tue+Wed Combo')
ax2.legend(fontsize=9)
for bar, r in zip(bars, combo_red_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, r + 0.5, f'{r:.1f}%', ha='center', va='bottom', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Combined magnitude → Thu red rate
ax3 = fig.add_subplot(3, 2, 4)
mag_labels = ['0–0.5%', '0.5–1%', '1–2%', '2%+']
mag_red_rates = []
mag_counts = []
for lo, hi, label in buckets:
    mask = (both_green['Combined_Return'] >= lo) & (both_green['Combined_Return'] < hi)
    sub = both_green[mask]
    if len(sub) > 0:
        mag_red_rates.append((sub['Thu_Color'] == 'Red').sum() / len(sub) * 100)
        mag_counts.append(len(sub))
    else:
        mag_red_rates.append(0)
        mag_counts.append(0)
colors_mag = ['#ef5350' if r > baseline_red else '#26a69a' for r in mag_red_rates]
bars3 = ax3.bar(mag_labels, mag_red_rates, color=colors_mag, width=0.5)
ax3.axhline(y=baseline_red, color='gray', linestyle='--', alpha=0.7, label=f'Baseline: {baseline_red:.1f}%')
ax3.set_ylabel('Thursday Red %')
ax3.set_title('Thursday Red Rate by Tue+Wed Combined Size\n(Both Green only)')
ax3.legend(fontsize=9)
for bar, r, c in zip(bars3, mag_red_rates, mag_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, r + 0.5, f'{r:.1f}%\n(n={c})', ha='center', va='bottom', fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Individual trade returns
ax4 = fig.add_subplot(3, 2, 5)
colors_pnl = ['#26a69a' if r > 0 else '#ef5350' for r in bt['Return_Pct']]
ax4.bar(range(len(bt)), bt['Return_Pct'], color=colors_pnl, width=1.0)
ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax4.axhline(y=-optimal_sl, color='#ef5350', linestyle='--', alpha=0.5, label=f'Stop: +{optimal_sl:.1f}%')
ax4.set_title('Individual Trade Returns (Short)', fontsize=12)
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
plt.savefig('/Users/ashleighchua/trading analyses/thursday_flip_backtest.png', dpi=150, bbox_inches='tight')
print("\nChart saved to: thursday_flip_backtest.png")
print("✅ Analysis complete!")
