"""
SPY All Day-of-Week Signals Backtest
=====================================
Tests EVERY combination:
  - If [Day X] is red/green → buy [Day Y] at open, sell at close
  - With stop loss optimization for each
  - Find which day pairs have a real edge

Also tests: "next trading day" signals (red Monday → buy Tuesday, etc.)
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

print(f"Data: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)\n")

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# ══════════════════════════════════════════════════════════════════════════════
# 1. NEXT-DAY REVERSAL: If Day X is RED → buy next trading day
# ══════════════════════════════════════════════════════════════════════════════
print("="*80)
print("SIGNAL: If [Signal Day] is RED → Buy NEXT trading day at open, sell at close")
print("="*80)

next_day_results = []

for signal_dow in range(5):
    signal_days = spy[spy['DayOfWeek'] == signal_dow]

    pairs = []
    for sig_date, sig_row in signal_days.iterrows():
        if sig_row['Color'] != 'Red':
            continue
        # Find next trading day
        future = spy[spy.index > sig_date]
        if len(future) == 0:
            continue
        nxt = future.iloc[0]
        days_gap = (nxt.name - sig_date).days
        if days_gap > 5:
            continue
        pairs.append({
            'Signal_Date': sig_date,
            'Signal_Return': sig_row['Return'],
            'Trade_Date': nxt.name,
            'Trade_Day': nxt['DayName'],
            'Trade_Open': nxt['Open'],
            'Trade_High': nxt['High'],
            'Trade_Low': nxt['Low'],
            'Trade_Close': nxt['Close'],
            'Trade_Return': (nxt['Close'] - nxt['Open']) / nxt['Open'] * 100,
        })

    if len(pairs) == 0:
        continue

    pdf = pd.DataFrame(pairs)
    n = len(pdf)
    green = (pdf['Trade_Return'] > 0).sum()
    wr = green / n * 100
    avg_ret = pdf['Trade_Return'].mean()
    cum_ret = ((1 + pdf['Trade_Return'] / 100).prod() - 1) * 100
    gp = pdf[pdf['Trade_Return'] > 0]['Trade_Return'].sum()
    gl = abs(pdf[pdf['Trade_Return'] < 0]['Trade_Return'].sum())
    pf = gp / gl if gl > 0 else float('inf')

    trade_day_name = pdf['Trade_Day'].mode()[0] if len(pdf) > 0 else '?'

    next_day_results.append({
        'Signal': f"Red {day_names[signal_dow]}",
        'Trade': f"Buy {trade_day_name}",
        'Trades': n,
        'WinRate': wr,
        'AvgReturn': avg_ret,
        'CumReturn': cum_ret,
        'ProfitFactor': pf,
    })

ndr = pd.DataFrame(next_day_results)
print(f"\n{'Signal':<18} {'Trade Day':<16} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'CumRet':>10} {'PF':>7}")
print("-"*80)
for _, r in ndr.sort_values('CumReturn', ascending=False).iterrows():
    marker = " ★" if r['CumReturn'] == ndr['CumReturn'].max() else ""
    print(f"{r['Signal']:<18} {r['Trade']:<16} {r['Trades']:>7} {r['WinRate']:>6.1f}% {r['AvgReturn']:>+8.3f}% {r['CumReturn']:>+9.2f}% {r['ProfitFactor']:>6.2f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SAME-DIRECTION: If Day X is GREEN → buy next trading day
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("SIGNAL: If [Signal Day] is GREEN → Buy NEXT trading day at open, sell at close")
print("="*80)

green_day_results = []

for signal_dow in range(5):
    signal_days = spy[spy['DayOfWeek'] == signal_dow]

    pairs = []
    for sig_date, sig_row in signal_days.iterrows():
        if sig_row['Color'] != 'Green':
            continue
        future = spy[spy.index > sig_date]
        if len(future) == 0:
            continue
        nxt = future.iloc[0]
        days_gap = (nxt.name - sig_date).days
        if days_gap > 5:
            continue
        pairs.append({
            'Trade_Date': nxt.name,
            'Trade_Day': nxt['DayName'],
            'Trade_Return': (nxt['Close'] - nxt['Open']) / nxt['Open'] * 100,
        })

    if len(pairs) == 0:
        continue

    pdf = pd.DataFrame(pairs)
    n = len(pdf)
    green = (pdf['Trade_Return'] > 0).sum()
    wr = green / n * 100
    avg_ret = pdf['Trade_Return'].mean()
    cum_ret = ((1 + pdf['Trade_Return'] / 100).prod() - 1) * 100
    gp = pdf[pdf['Trade_Return'] > 0]['Trade_Return'].sum()
    gl = abs(pdf[pdf['Trade_Return'] < 0]['Trade_Return'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    trade_day_name = pdf['Trade_Day'].mode()[0]

    green_day_results.append({
        'Signal': f"Green {day_names[signal_dow]}",
        'Trade': f"Buy {trade_day_name}",
        'Trades': n,
        'WinRate': wr,
        'AvgReturn': avg_ret,
        'CumReturn': cum_ret,
        'ProfitFactor': pf,
    })

gdr = pd.DataFrame(green_day_results)
print(f"\n{'Signal':<18} {'Trade Day':<16} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'CumRet':>10} {'PF':>7}")
print("-"*80)
for _, r in gdr.sort_values('CumReturn', ascending=False).iterrows():
    marker = " ★" if r['CumReturn'] == gdr['CumReturn'].max() else ""
    print(f"{r['Signal']:<18} {r['Trade']:<16} {r['Trades']:>7} {r['WinRate']:>6.1f}% {r['AvgReturn']:>+8.3f}% {r['CumReturn']:>+9.2f}% {r['ProfitFactor']:>6.2f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ALL SAME-WEEK DAY PAIRS (not just next day)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("ALL DAY-PAIR SIGNALS: If [Day X] is RED → Buy [Day Y] same/next week")
print("="*80)

all_pair_results = []

for sig_dow in range(5):
    for trade_dow in range(5):
        if sig_dow == trade_dow:
            continue

        sig_days = spy[(spy['DayOfWeek'] == sig_dow) & (spy['Color'] == 'Red')]
        trade_days = spy[spy['DayOfWeek'] == trade_dow]

        pairs = []
        for sig_date, sig_row in sig_days.iterrows():
            # Find next occurrence of trade_dow after signal
            future_trade = trade_days[trade_days.index > sig_date]
            if len(future_trade) == 0:
                continue
            nxt = future_trade.iloc[0]
            # Must be within same or next week (max 7 calendar days)
            days_gap = (nxt.name - sig_date).days
            if days_gap > 7:
                continue
            pairs.append({
                'Trade_Return': (nxt['Close'] - nxt['Open']) / nxt['Open'] * 100,
                'Trade_Low': nxt['Low'],
                'Trade_Open': nxt['Open'],
            })

        if len(pairs) < 10:
            continue

        pdf = pd.DataFrame(pairs)
        n = len(pdf)
        green = (pdf['Trade_Return'] > 0).sum()
        wr = green / n * 100
        avg_ret = pdf['Trade_Return'].mean()
        cum_ret = ((1 + pdf['Trade_Return'] / 100).prod() - 1) * 100
        gp = pdf[pdf['Trade_Return'] > 0]['Trade_Return'].sum()
        gl = abs(pdf[pdf['Trade_Return'] < 0]['Trade_Return'].sum())
        pf = gp / gl if gl > 0 else float('inf')

        all_pair_results.append({
            'Signal': f"Red {day_names[sig_dow]}",
            'Trade': f"Buy {day_names[trade_dow]}",
            'Sig_DOW': sig_dow,
            'Trade_DOW': trade_dow,
            'Trades': n,
            'WinRate': wr,
            'AvgReturn': avg_ret,
            'CumReturn': cum_ret,
            'ProfitFactor': pf,
        })

apr = pd.DataFrame(all_pair_results)
print(f"\n{'Signal':<18} {'Trade Day':<16} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'CumRet':>10} {'PF':>7}")
print("-"*80)
for _, r in apr.sort_values('CumReturn', ascending=False).iterrows():
    marker = " ★" if r['WinRate'] >= 60 and r['CumReturn'] > 5 else ""
    print(f"{r['Signal']:<18} {r['Trade']:<16} {r['Trades']:>7} {r['WinRate']:>6.1f}% {r['AvgReturn']:>+8.3f}% {r['CumReturn']:>+9.2f}% {r['ProfitFactor']:>6.2f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. COMBINED STRATEGY BACKTEST — Trade ALL profitable signals
# ══════════════════════════════════════════════════════════════════════════════
# Pick signals with win rate >= 58% and positive cumulative return
good_signals = apr[(apr['WinRate'] >= 58) & (apr['CumReturn'] > 0)]

print(f"\n{'='*80}")
print(f"COMBINED STRATEGY — Using {len(good_signals)} profitable signals together")
print("="*80)

if len(good_signals) > 0:
    print("\n  Signals included:")
    for _, gs in good_signals.iterrows():
        print(f"    • {gs['Signal']} → {gs['Trade']} (WR: {gs['WinRate']:.1f}%, Cum: {gs['CumReturn']:+.1f}%)")

    # Build all trades chronologically
    all_trades = []
    for _, gs in good_signals.iterrows():
        sig_dow = gs['Sig_DOW']
        trade_dow = gs['Trade_DOW']

        sig_days = spy[(spy['DayOfWeek'] == sig_dow) & (spy['Color'] == 'Red')]
        trade_days = spy[spy['DayOfWeek'] == trade_dow]

        for sig_date, sig_row in sig_days.iterrows():
            future_trade = trade_days[trade_days.index > sig_date]
            if len(future_trade) == 0:
                continue
            nxt = future_trade.iloc[0]
            if (nxt.name - sig_date).days > 7:
                continue

            all_trades.append({
                'Signal': f"Red {day_names[sig_dow]}",
                'Trade_Date': nxt.name,
                'Trade_Day': day_names[trade_dow],
                'Open': nxt['Open'],
                'High': nxt['High'],
                'Low': nxt['Low'],
                'Close': nxt['Close'],
                'Return': (nxt['Close'] - nxt['Open']) / nxt['Open'] * 100,
            })

    trades_all = pd.DataFrame(all_trades).sort_values('Trade_Date')

    # Remove duplicate trade dates (if multiple signals point to same day, only trade once)
    trades_all = trades_all.drop_duplicates(subset='Trade_Date', keep='first')

    # Backtest with 1.9% stop loss
    SL = 1.9
    capital = 10000
    initial = capital
    equity = [{'Date': trades_all.iloc[0]['Trade_Date'], 'Capital': capital}]
    trade_log = []

    for _, t in trades_all.iterrows():
        stop_price = t['Open'] * (1 - SL / 100)
        if t['Low'] <= stop_price:
            exit_price = stop_price
            exit_type = 'STOP'
        else:
            exit_price = t['Close']
            exit_type = 'CLOSE'

        ret = (exit_price - t['Open']) / t['Open']
        pnl = capital * ret
        capital += pnl
        trade_log.append({
            'Date': t['Trade_Date'],
            'Signal': t['Signal'],
            'Day': t['Trade_Day'],
            'Return': ret * 100,
            'PnL': pnl,
            'Exit': exit_type,
            'Win': ret > 0,
            'Capital': capital,
        })
        equity.append({'Date': t['Trade_Date'], 'Capital': capital})

    tl = pd.DataFrame(trade_log)
    eq = pd.DataFrame(equity)

    total_ret = (capital - initial) / initial * 100
    years = (trades_all['Trade_Date'].iloc[-1] - trades_all['Trade_Date'].iloc[0]).days / 365.25
    annual_ret = ((capital / initial) ** (1 / years) - 1) * 100
    wr = tl['Win'].mean() * 100
    avg_win = tl[tl['Win']]['Return'].mean()
    avg_loss = tl[~tl['Win']]['Return'].mean()
    gp = tl[tl['Win']]['PnL'].sum()
    gl = abs(tl[~tl['Win']]['PnL'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    eq['Peak'] = eq['Capital'].cummax()
    eq['DD'] = (eq['Capital'] - eq['Peak']) / eq['Peak'] * 100
    max_dd = eq['DD'].min()
    sharpe = (tl['Return'].mean() / tl['Return'].std()) * np.sqrt(len(tl) / years) if tl['Return'].std() > 0 else 0
    trades_per_year = len(tl) / years

    print(f"""
  ┌──────────────────────────────────────────────────────┐
  │  COMBINED STRATEGY — {len(good_signals)} signals, {SL}% stop loss{' '*11}│
  ├──────────────────────────────────────────────────────┤
  │  Starting Capital:      ${initial:>10,.2f}           │
  │  Ending Capital:        ${capital:>10,.2f}           │
  │  Total Return:           {total_ret:>+9.2f}%           │
  │  Annualized Return:      {annual_ret:>+9.2f}%           │
  │  Max Drawdown:           {max_dd:>+9.2f}%           │
  │  Sharpe Ratio:            {sharpe:>8.2f}            │
  │  Profit Factor:           {pf:>8.2f}            │
  ├──────────────────────────────────────────────────────┤
  │  Total Trades:            {len(tl):>8}            │
  │  Trades per Year:         {trades_per_year:>8.1f}            │
  │  Win Rate:                {wr:>7.1f}%            │
  │  Avg Winner:             {avg_win:>+9.3f}%           │
  │  Avg Loser:              {avg_loss:>+9.3f}%           │
  └──────────────────────────────────────────────────────┘
""")

    # Breakdown by signal
    print("  BY SIGNAL:")
    print(f"  {'Signal':<20} {'Trades':>7} {'Win%':>7} {'AvgRet':>9} {'PnL':>12}")
    print("  " + "-"*58)
    for sig, g in tl.groupby('Signal'):
        print(f"  {sig:<20} {len(g):>7} {g['Win'].mean()*100:>6.1f}% {g['Return'].mean():>+8.3f}% ${g['PnL'].sum():>+10,.2f}")

    # Yearly
    print(f"\n  YEARLY BREAKDOWN:")
    print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'Return':>9} {'PnL':>12}")
    print("  " + "-"*45)
    tl['Year'] = tl['Date'].dt.year
    for year, g in tl.groupby('Year'):
        print(f"  {year:<6} {len(g):>7} {g['Win'].mean()*100:>6.1f}% {g['Return'].sum():>+8.2f}% ${g['PnL'].sum():>+10,.2f}")

    # ── CHARTS ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(f'SPY All Day-of-Week Signals Analysis', fontsize=15, fontweight='bold', y=0.98)

    # 1. Heatmap: Win rate for "Red Day X → Buy Day Y"
    ax1 = fig.add_subplot(3, 2, (1, 2))
    heatmap_wr = np.full((5, 5), np.nan)
    heatmap_cr = np.full((5, 5), np.nan)
    for _, r in apr.iterrows():
        heatmap_wr[int(r['Sig_DOW']), int(r['Trade_DOW'])] = r['WinRate']
        heatmap_cr[int(r['Sig_DOW']), int(r['Trade_DOW'])] = r['CumReturn']

    im = ax1.imshow(heatmap_wr, cmap='RdYlGn', vmin=40, vmax=70, aspect='auto')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(day_names)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(day_names)
    ax1.set_xlabel('Trade Day (Buy Open → Sell Close)')
    ax1.set_ylabel('Signal Day (must be RED)')
    ax1.set_title('Win Rate: "If [Signal Day] is RED → Buy [Trade Day]"', fontsize=12)
    for i in range(5):
        for j in range(5):
            if not np.isnan(heatmap_wr[i, j]):
                color = 'white' if heatmap_wr[i, j] < 48 or heatmap_wr[i, j] > 65 else 'black'
                ax1.text(j, i, f'{heatmap_wr[i,j]:.1f}%\n({heatmap_cr[i,j]:+.1f}%)',
                        ha='center', va='center', fontsize=9, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Win Rate %')

    # 2. Combined equity curve
    ax2 = fig.add_subplot(3, 2, (3, 4))
    ax2.plot(eq['Date'], eq['Capital'], color='#26a69a', linewidth=2, label='Combined Strategy')
    ax2.fill_between(eq['Date'], eq['Capital'], initial,
                      where=eq['Capital'] >= initial, alpha=0.15, color='#26a69a')
    ax2.fill_between(eq['Date'], eq['Capital'], initial,
                      where=eq['Capital'] < initial, alpha=0.15, color='#ef5350')
    ax2.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title(f'Combined Strategy Equity Curve (${initial:,.0f} → ${capital:,.0f})', fontsize=12)
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Bar chart: Cumulative return per signal
    ax3 = fig.add_subplot(3, 2, 5)
    apr_sorted = apr.sort_values('CumReturn', ascending=True)
    colors = ['#26a69a' if cr > 0 else '#ef5350' for cr in apr_sorted['CumReturn']]
    labels = [f"{r['Signal']}→{r['Trade'].replace('Buy ', '')}" for _, r in apr_sorted.iterrows()]
    ax3.barh(range(len(apr_sorted)), apr_sorted['CumReturn'], color=colors)
    ax3.set_yticks(range(len(apr_sorted)))
    ax3.set_yticklabels(labels, fontsize=7)
    ax3.set_xlabel('Cumulative Return %')
    ax3.set_title('Cumulative Return: All Red Day → Trade Day Pairs', fontsize=12)
    ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Win rate bar chart for next-day signals
    ax4 = fig.add_subplot(3, 2, 6)
    ndr_sorted = ndr.sort_values('WinRate', ascending=True)
    colors = ['#26a69a' if wr > 55 else '#ffa726' if wr > 50 else '#ef5350' for wr in ndr_sorted['WinRate']]
    ax4.barh(range(len(ndr_sorted)),  ndr_sorted['WinRate'], color=colors)
    ax4.set_yticks(range(len(ndr_sorted)))
    ax4.set_yticklabels([f"{r['Signal']}→{r['Trade'].replace('Buy ','')}" for _, r in ndr_sorted.iterrows()], fontsize=9)
    ax4.set_xlabel('Win Rate %')
    ax4.set_title('Win Rate: Red Day → Buy Next Day', fontsize=12)
    ax4.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    for i, (_, r) in enumerate(ndr_sorted.iterrows()):
        ax4.text(r['WinRate'] + 0.5, i, f"{r['WinRate']:.1f}%", va='center', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/Users/ashleighchua/trading analyses/spy_all_days_analysis.png', dpi=150, bbox_inches='tight')
    print("\nCharts saved to: spy_all_days_analysis.png")

print("\n✅ Analysis complete!")
