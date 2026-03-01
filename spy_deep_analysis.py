"""
SPY Deep Analysis: Red Friday → Green Monday Strategy
======================================================
1. Market regime analysis (bull vs bear vs sideways)
2. Statistical significance tests
3. Full backtest with equity curve
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime

# ── Download SPY data ────────────────────────────────────────────────────────
print("Downloading SPY historical data...")
spy = yf.download("SPY", period="5y", auto_adjust=True)
spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy

spy['DayOfWeek'] = spy.index.dayofweek
spy['DayName'] = spy.index.day_name()
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Close_Return'] = spy['Close'].pct_change() * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

# 50-day SMA for regime detection
spy['SMA50'] = spy['Close'].rolling(50).mean()
spy['SMA200'] = spy['Close'].rolling(200).mean()
spy['Regime'] = 'Sideways'
spy.loc[spy['Close'] > spy['SMA50'], 'Regime'] = 'Bull'
spy.loc[spy['Close'] < spy['SMA50'], 'Regime'] = 'Bear'

print(f"Data range: {spy.index[0].date()} to {spy.index[-1].date()}")
print(f"Total trading days: {len(spy)}")

# ── Build Friday→Monday pairs ────────────────────────────────────────────────
fridays = spy[spy['DayOfWeek'] == 4].copy()
mondays = spy[spy['DayOfWeek'] == 0].copy()

pairs = []
for fri_date, fri_row in fridays.iterrows():
    next_mondays = mondays[mondays.index > fri_date]
    if len(next_mondays) == 0:
        continue
    next_mon = next_mondays.iloc[0]
    days_diff = (next_mon.name - fri_date).days
    if days_diff <= 5:
        pairs.append({
            'Friday_Date': fri_date,
            'Friday_Color': fri_row['Color'],
            'Friday_Return': fri_row['Return'],
            'Friday_Close': fri_row['Close'],
            'Monday_Date': next_mon.name,
            'Monday_Color': next_mon['Color'],
            'Monday_Return': next_mon['Return'],
            'Monday_Open': next_mon['Open'],
            'Monday_Close': next_mon['Close'],
            'Regime': fri_row['Regime'],
        })

df = pd.DataFrame(pairs)
red_fri = df[df['Friday_Color'] == 'Red']
green_fri = df[df['Friday_Color'] == 'Green']

# ══════════════════════════════════════════════════════════════════════════════
# 1. MARKET REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. MARKET REGIME ANALYSIS (Red Friday → Monday)")
print("   Regime = Price above/below 50-day SMA")
print("="*70)

for regime in ['Bull', 'Bear']:
    regime_data = red_fri[red_fri['Regime'] == regime]
    if len(regime_data) == 0:
        continue
    green_mon = regime_data[regime_data['Monday_Color'] == 'Green']
    pct = len(green_mon) / len(regime_data) * 100
    avg_ret = regime_data['Monday_Return'].mean()

    print(f"\n  {regime.upper()} market (price {'above' if regime == 'Bull' else 'below'} 50-SMA):")
    print(f"    Red Fridays in {regime} regime: {len(regime_data)}")
    print(f"    → Monday Green: {len(green_mon)}/{len(regime_data)} ({pct:.1f}%)")
    print(f"    → Avg Monday Return: {avg_ret:+.3f}%")

# Also show severity breakdown
print("\n" + "-"*50)
print("  BY FRIDAY DROP SEVERITY:")
print("-"*50)

for label, condition in [
    ("Small red (0% to -0.5%)", (red_fri['Friday_Return'] >= -0.5) & (red_fri['Friday_Return'] < 0)),
    ("Medium red (-0.5% to -1%)", (red_fri['Friday_Return'] >= -1.0) & (red_fri['Friday_Return'] < -0.5)),
    ("Large red (< -1%)", red_fri['Friday_Return'] < -1.0),
]:
    subset = red_fri[condition]
    if len(subset) == 0:
        continue
    green_mon = subset[subset['Monday_Color'] == 'Green']
    pct = len(green_mon) / len(subset) * 100
    avg_ret = subset['Monday_Return'].mean()
    print(f"\n  {label}:")
    print(f"    Count: {len(subset)}")
    print(f"    → Monday Green: {len(green_mon)}/{len(subset)} ({pct:.1f}%)")
    print(f"    → Avg Monday Return: {avg_ret:+.3f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 2. STATISTICAL SIGNIFICANCE TESTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. STATISTICAL SIGNIFICANCE TESTS")
print("="*70)

# Test 1: Binomial test — is 68.1% green rate significantly > 50%?
n_red_fri = len(red_fri)
n_green_mon = len(red_fri[red_fri['Monday_Color'] == 'Green'])
binom_result = stats.binomtest(n_green_mon, n_red_fri, 0.5, alternative='greater')
binom_p = binom_result.pvalue

print(f"\n  Test A: Binomial test (Is green rate after red Friday > 50%?)")
print(f"    Successes: {n_green_mon}/{n_red_fri} = {n_green_mon/n_red_fri*100:.1f}%")
print(f"    p-value: {binom_p:.4f}")
print(f"    {'✅ Statistically significant (p < 0.05)' if binom_p < 0.05 else '❌ Not statistically significant'}")

# Test 2: Binomial test — is it significantly > baseline Monday green rate?
baseline_green_rate = (df['Monday_Color'] == 'Green').mean()
binom_result2 = stats.binomtest(n_green_mon, n_red_fri, baseline_green_rate, alternative='greater')
binom_p2 = binom_result2.pvalue

print(f"\n  Test B: Binomial test (Is green rate after red Friday > baseline {baseline_green_rate*100:.1f}%?)")
print(f"    Successes: {n_green_mon}/{n_red_fri} = {n_green_mon/n_red_fri*100:.1f}%")
print(f"    p-value: {binom_p2:.4f}")
print(f"    {'✅ Statistically significant (p < 0.05)' if binom_p2 < 0.05 else '❌ Not statistically significant'}")

# Test 3: Two-sample t-test — Monday returns after red Friday vs green Friday
t_stat, t_pval = stats.ttest_ind(
    red_fri['Monday_Return'].values,
    green_fri['Monday_Return'].values,
    equal_var=False
)

print(f"\n  Test C: Two-sample t-test (Monday returns: after red Friday vs green Friday)")
print(f"    After Red Friday:   mean = {red_fri['Monday_Return'].mean():+.3f}%, std = {red_fri['Monday_Return'].std():.3f}%")
print(f"    After Green Friday: mean = {green_fri['Monday_Return'].mean():+.3f}%, std = {green_fri['Monday_Return'].std():.3f}%")
print(f"    t-statistic: {t_stat:.3f}")
print(f"    p-value: {t_pval:.4f}")
print(f"    {'✅ Statistically significant (p < 0.05)' if t_pval < 0.05 else '❌ Not statistically significant'}")

# Test 4: Chi-squared test for independence
contingency = pd.crosstab(df['Friday_Color'], df['Monday_Color'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)

print(f"\n  Test D: Chi-squared test (Is Friday color independent of Monday color?)")
print(f"    Contingency table:")
print(f"    {contingency.to_string().replace(chr(10), chr(10) + '    ')}")
print(f"    Chi² = {chi2:.3f}, p-value = {chi_p:.4f}")
print(f"    {'✅ Colors are NOT independent (relationship exists)' if chi_p < 0.05 else '❌ Colors appear independent (no significant relationship)'}")

# Effect size (Cohen's h for proportions)
p1 = n_green_mon / n_red_fri
p2 = baseline_green_rate
cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
print(f"\n  Effect Size (Cohen's h): {cohens_h:.3f}")
print(f"    {'Small' if abs(cohens_h) < 0.3 else 'Medium' if abs(cohens_h) < 0.5 else 'Large'} effect size")
print(f"    (0.2 = small, 0.5 = medium, 0.8 = large)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BACKTEST: Red Friday → Buy Monday Open, Sell Monday Close
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. BACKTEST: Red Friday → Buy Monday Open, Sell Monday Close")
print("="*70)

initial_capital = 10000
capital = initial_capital
buy_and_hold_capital = initial_capital

trades = []
equity_curve = [{'Date': df.iloc[0]['Monday_Date'], 'Equity': capital, 'BuyHold': buy_and_hold_capital}]

# Track buy & hold from first to last Monday in our dataset
bh_start_price = df.iloc[0]['Monday_Open']

for _, row in df.iterrows():
    bh_price = row['Monday_Close']
    bh_equity = initial_capital * (bh_price / bh_start_price)

    if row['Friday_Color'] == 'Red':
        # Buy at Monday open, sell at Monday close
        ret = (row['Monday_Close'] - row['Monday_Open']) / row['Monday_Open']
        pnl = capital * ret
        capital += pnl
        trades.append({
            'Date': row['Monday_Date'],
            'Entry': row['Monday_Open'],
            'Exit': row['Monday_Close'],
            'Return': ret * 100,
            'PnL': pnl,
            'Equity': capital,
            'Win': ret > 0,
        })

    equity_curve.append({
        'Date': row['Monday_Date'],
        'Equity': capital,
        'BuyHold': bh_equity,
    })

eq_df = pd.DataFrame(equity_curve)
trades_df = pd.DataFrame(trades)

total_return = (capital - initial_capital) / initial_capital * 100
n_trades = len(trades_df)
n_wins = trades_df['Win'].sum()
win_rate = n_wins / n_trades * 100
avg_win = trades_df[trades_df['Win']]['Return'].mean()
avg_loss = trades_df[~trades_df['Win']]['Return'].mean()
max_trade = trades_df['Return'].max()
min_trade = trades_df['Return'].min()
profit_factor = abs(trades_df[trades_df['Win']]['PnL'].sum() / trades_df[~trades_df['Win']]['PnL'].sum()) if trades_df[~trades_df['Win']]['PnL'].sum() != 0 else float('inf')

# Max drawdown
eq_df['Peak'] = eq_df['Equity'].cummax()
eq_df['Drawdown'] = (eq_df['Equity'] - eq_df['Peak']) / eq_df['Peak'] * 100
max_drawdown = eq_df['Drawdown'].min()

# Annualized return (approximate)
years = (df['Monday_Date'].iloc[-1] - df['Monday_Date'].iloc[0]).days / 365.25
annual_return = ((capital / initial_capital) ** (1 / years) - 1) * 100

# Sharpe ratio (annualized, assuming ~48 trades/year)
if trades_df['Return'].std() > 0:
    trades_per_year = n_trades / years
    sharpe = (trades_df['Return'].mean() / trades_df['Return'].std()) * np.sqrt(trades_per_year)
else:
    sharpe = 0

print(f"""
  Strategy: Buy SPY at Monday open if prior Friday was red, sell Monday close
  Period: {df['Monday_Date'].iloc[0].date()} to {df['Monday_Date'].iloc[-1].date()} ({years:.1f} years)

  ┌─────────────────────────────────────────┐
  │  PERFORMANCE SUMMARY                    │
  ├─────────────────────────────────────────┤
  │  Starting Capital:    ${initial_capital:>10,.2f}       │
  │  Ending Capital:      ${capital:>10,.2f}       │
  │  Total Return:         {total_return:>+9.2f}%       │
  │  Annualized Return:    {annual_return:>+9.2f}%       │
  │  Max Drawdown:         {max_drawdown:>+9.2f}%       │
  │  Sharpe Ratio:          {sharpe:>8.2f}        │
  ├─────────────────────────────────────────┤
  │  TRADE STATISTICS                       │
  ├─────────────────────────────────────────┤
  │  Total Trades:          {n_trades:>8}        │
  │  Win Rate:              {win_rate:>7.1f}%        │
  │  Avg Winning Trade:    {avg_win:>+9.3f}%       │
  │  Avg Losing Trade:     {avg_loss:>+9.3f}%       │
  │  Best Trade:           {max_trade:>+9.3f}%       │
  │  Worst Trade:          {min_trade:>+9.3f}%       │
  │  Profit Factor:         {profit_factor:>8.2f}        │
  └─────────────────────────────────────────┘
""")

# Yearly breakdown
print("  YEARLY BREAKDOWN:")
print(f"  {'Year':<6} {'Trades':>7} {'Win Rate':>9} {'Return':>9} {'PnL':>12}")
print("  " + "-"*48)
trades_df['Year'] = trades_df['Date'].dt.year
for year, group in trades_df.groupby('Year'):
    yr_trades = len(group)
    yr_wins = group['Win'].sum()
    yr_wr = yr_wins / yr_trades * 100 if yr_trades > 0 else 0
    yr_pnl = group['PnL'].sum()
    yr_ret = group['Return'].sum()
    print(f"  {year:<6} {yr_trades:>7} {yr_wr:>8.1f}% {yr_ret:>+8.2f}% ${yr_pnl:>+10,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 18))
fig.suptitle('SPY "Red Friday → Green Monday" Deep Analysis', fontsize=15, fontweight='bold', y=0.98)

# Chart 1: Equity Curve
ax1 = fig.add_subplot(3, 2, (1, 2))
ax1.plot(eq_df['Date'], eq_df['Equity'], color='#26a69a', linewidth=2, label='Strategy')
ax1.plot(eq_df['Date'], eq_df['BuyHold'], color='#9e9e9e', linewidth=1.5, alpha=0.7, linestyle='--', label='Buy & Hold SPY')
ax1.fill_between(eq_df['Date'], eq_df['Equity'], initial_capital,
                  where=eq_df['Equity'] >= initial_capital, alpha=0.15, color='#26a69a')
ax1.fill_between(eq_df['Date'], eq_df['Equity'], initial_capital,
                  where=eq_df['Equity'] < initial_capital, alpha=0.15, color='#ef5350')
ax1.axhline(y=initial_capital, color='gray', linestyle=':', alpha=0.5)
ax1.set_title('Equity Curve: $10,000 Starting Capital', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Chart 2: Trade Returns Distribution
ax2 = fig.add_subplot(3, 2, 3)
wins = trades_df[trades_df['Win']]['Return']
losses = trades_df[~trades_df['Win']]['Return']
ax2.hist(wins, bins=20, alpha=0.7, color='#26a69a', label=f'Wins ({len(wins)})')
ax2.hist(losses, bins=20, alpha=0.7, color='#ef5350', label=f'Losses ({len(losses)})')
ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
ax2.axvline(x=trades_df['Return'].mean(), color='blue', linestyle='--', alpha=0.7,
            label=f'Mean: {trades_df["Return"].mean():+.3f}%')
ax2.set_title('Trade Returns Distribution', fontsize=12)
ax2.set_xlabel('Return (%)')
ax2.set_ylabel('Count')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Chart 3: By Market Regime
ax3 = fig.add_subplot(3, 2, 4)
regime_stats = []
for regime in ['Bull', 'Bear']:
    r_data = red_fri[red_fri['Regime'] == regime]
    if len(r_data) > 0:
        regime_stats.append({
            'Regime': regime,
            'Green%': (r_data['Monday_Color'] == 'Green').mean() * 100,
            'AvgRet': r_data['Monday_Return'].mean(),
            'Count': len(r_data)
        })
rs_df = pd.DataFrame(regime_stats)
bars = ax3.bar(rs_df['Regime'], rs_df['Green%'],
               color=['#26a69a', '#ef5350'], alpha=0.8, width=0.5)
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='50% baseline')
ax3.set_title('Monday Green % After Red Friday\nby Market Regime', fontsize=12)
ax3.set_ylabel('Green Monday %')
ax3.set_ylim(0, 100)
for bar, row in zip(bars, rs_df.itertuples()):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{row._2:.1f}%\n(n={row.Count})', ha='center', fontsize=10)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Chart 4: By Friday Drop Severity
ax4 = fig.add_subplot(3, 2, 5)
severity_labels = []
severity_pcts = []
severity_counts = []
for label, cond in [
    ("Small\n(0 to -0.5%)", (red_fri['Friday_Return'] >= -0.5) & (red_fri['Friday_Return'] < 0)),
    ("Medium\n(-0.5 to -1%)", (red_fri['Friday_Return'] >= -1.0) & (red_fri['Friday_Return'] < -0.5)),
    ("Large\n(< -1%)", red_fri['Friday_Return'] < -1.0),
]:
    subset = red_fri[cond]
    if len(subset) > 0:
        severity_labels.append(label)
        severity_pcts.append((subset['Monday_Color'] == 'Green').mean() * 100)
        severity_counts.append(len(subset))

sev_colors = ['#66bb6a', '#ffa726', '#ef5350']
bars = ax4.bar(severity_labels, severity_pcts, color=sev_colors, alpha=0.8, width=0.5)
ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax4.set_title('Monday Green % by Friday Drop Size', fontsize=12)
ax4.set_ylabel('Green Monday %')
ax4.set_ylim(0, 100)
for bar, pct, cnt in zip(bars, severity_pcts, severity_counts):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{pct:.1f}%\n(n={cnt})', ha='center', fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# Chart 5: Yearly Win Rate
ax5 = fig.add_subplot(3, 2, 6)
yearly = trades_df.groupby('Year').agg(
    WinRate=('Win', lambda x: x.mean() * 100),
    Count=('Win', 'count'),
    TotalReturn=('Return', 'sum')
).reset_index()
bars = ax5.bar(yearly['Year'].astype(str), yearly['WinRate'],
               color=['#26a69a' if wr >= 50 else '#ef5350' for wr in yearly['WinRate']],
               alpha=0.8)
ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
ax5.set_title('Win Rate by Year', fontsize=12)
ax5.set_ylabel('Win Rate %')
ax5.set_ylim(0, 100)
for bar, row in zip(bars, yearly.itertuples()):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{row.WinRate:.0f}%\n({row.Count}t)', ha='center', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/Users/ashleighchua/trading analyses/spy_deep_analysis.png', dpi=150, bbox_inches='tight')
print("\nCharts saved to: spy_deep_analysis.png")
print("Analysis complete!")
