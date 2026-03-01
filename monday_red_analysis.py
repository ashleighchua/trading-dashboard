"""
When Does Monday Print Red?
============================
Flipping the strategy — instead of finding when Monday is green,
find what conditions predict a RED Monday.

If we know when Monday is likely red, we know:
  1. When NOT to take the Red Friday → Buy Monday trade
  2. Potentially when to go SHORT on Monday
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Download SPY data ─────────────────────────────────────────────────────────
print("Downloading SPY data (10 years for larger sample)...")
spy = yf.download("SPY", period="10y", auto_adjust=True)
spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy

spy['DayOfWeek'] = spy.index.dayofweek
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

print(f"Data: {spy.index[0].date()} → {spy.index[-1].date()} ({len(spy)} days)")

# ── Build Friday→Monday pairs ─────────────────────────────────────────────────
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
        'Friday_Date':    fri_date,
        'Friday_Return':  fri_row['Return'],
        'Friday_Color':   fri_row['Color'],
        'Friday_Open':    fri_row['Open'],
        'Friday_Close':   fri_row['Close'],
        'Monday_Date':    mon.name,
        'Monday_Return':  mon['Return'],
        'Monday_Color':   mon['Color'],
        'Monday_Open':    mon['Open'],
        'Monday_Close':   mon['Close'],
        'Monday_High':    mon['High'],
        'Monday_Low':     mon['Low'],
    })

df = pd.DataFrame(pairs)

print(f"\nTotal Friday→Monday pairs: {len(df)}")

# ── BASELINE ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("BASELINE: Overall Monday Green/Red Rate")
print("="*70)

all_green = (df['Monday_Color'] == 'Green').sum()
all_red = (df['Monday_Color'] == 'Red').sum()
baseline_green = all_green / len(df) * 100
baseline_red = all_red / len(df) * 100

print(f"\n  All Mondays:   Green {baseline_green:.1f}%  |  Red {baseline_red:.1f}%")
print(f"  Sample size:   {len(df)} Mondays over 10 years")

# ── FRIDAY COLOR EFFECT ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("CONDITION 1: Friday Color")
print("="*70)

for color in ['Red', 'Green']:
    subset = df[df['Friday_Color'] == color]
    green_pct = (subset['Monday_Color'] == 'Green').sum() / len(subset) * 100
    red_pct = 100 - green_pct
    avg_ret = subset['Monday_Return'].mean()
    print(f"\n  Friday {color} ({len(subset)} cases):")
    print(f"    Monday Green: {green_pct:.1f}%  |  Monday Red: {red_pct:.1f}%")
    print(f"    Avg Monday return: {avg_ret:+.3f}%")

# ── FRIDAY RETURN MAGNITUDE ───────────────────────────────────────────────────
print("\n" + "="*70)
print("CONDITION 2: How RED was Friday? (magnitude matters)")
print("="*70)

red_fridays = df[df['Friday_Color'] == 'Red'].copy()
red_fridays['Friday_Return_Abs'] = red_fridays['Friday_Return'].abs()

# Bucket by severity
buckets = [
    (0, 0.5,  'Barely red (0–0.5%)'),
    (0.5, 1.0, 'Mildly red (0.5–1.0%)'),
    (1.0, 2.0, 'Moderately red (1–2%)'),
    (2.0, 99,  'Very red (2%+)'),
]

print(f"\n  {'Severity':<28} {'Count':>6} {'Mon Green%':>11} {'Mon Red%':>9} {'Avg Mon Ret':>12}")
print("  " + "-"*70)
for lo, hi, label in buckets:
    mask = (red_fridays['Friday_Return_Abs'] >= lo) & (red_fridays['Friday_Return_Abs'] < hi)
    sub = red_fridays[mask]
    if len(sub) == 0:
        continue
    g = (sub['Monday_Color'] == 'Green').sum() / len(sub) * 100
    r = 100 - g
    avg = sub['Monday_Return'].mean()
    print(f"  {label:<28} {len(sub):>6} {g:>10.1f}% {r:>8.1f}% {avg:>+11.3f}%")

# ── TWO CONSECUTIVE RED FRIDAYS ───────────────────────────────────────────────
print("\n" + "="*70)
print("CONDITION 3: Two Consecutive Red Fridays")
print("="*70)

df_sorted = df.sort_values('Friday_Date').reset_index(drop=True)
df_sorted['Prev_Friday_Color'] = df_sorted['Friday_Color'].shift(1)

two_red = df_sorted[(df_sorted['Friday_Color'] == 'Red') & (df_sorted['Prev_Friday_Color'] == 'Red')]
one_red = df_sorted[(df_sorted['Friday_Color'] == 'Red') & (df_sorted['Prev_Friday_Color'] == 'Green')]

two_red_green = (two_red['Monday_Color'] == 'Green').sum() / len(two_red) * 100 if len(two_red) > 0 else 0
one_red_green = (one_red['Monday_Color'] == 'Green').sum() / len(one_red) * 100 if len(one_red) > 0 else 0

print(f"\n  Single red Friday (prev was green): {one_red_green:.1f}% Monday green  ({len(one_red)} cases)")
print(f"  Two consecutive red Fridays:        {two_red_green:.1f}% Monday green  ({len(two_red)} cases)")
if two_red_green < one_red_green:
    print(f"\n  ⚠️  Two consecutive red Fridays WEAKENS the edge by {one_red_green - two_red_green:.1f}%")
else:
    print(f"\n  ✅ Two consecutive red Fridays doesn't hurt the edge")

# ── MARKET REGIME ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("CONDITION 4: Market Regime (Bull vs Bear)")
print("="*70)

spy['SMA50'] = spy['Close'].rolling(50).mean()
spy['SMA200'] = spy['Close'].rolling(200).mean()

df['SMA50_Friday'] = df['Friday_Date'].map(spy['SMA50'])
df['SMA200_Friday'] = df['Friday_Date'].map(spy['SMA200'])
df['Close_Friday'] = df['Friday_Date'].map(spy['Close'])

df['Regime'] = 'Sideways'
df.loc[df['Close_Friday'] > df['SMA50_Friday'], 'Regime'] = 'Bull'
df.loc[df['Close_Friday'] < df['SMA50_Friday'], 'Regime'] = 'Bear'

red_df = df[df['Friday_Color'] == 'Red']

print(f"\n  After Red Friday, Monday green rate by market regime:")
print(f"  {'Regime':<12} {'Count':>6} {'Mon Green%':>11} {'Mon Red%':>9} {'Avg Mon Ret':>12}")
print("  " + "-"*50)
for regime in ['Bull', 'Bear', 'Sideways']:
    sub = red_df[red_df['Regime'] == regime]
    if len(sub) == 0:
        continue
    g = (sub['Monday_Color'] == 'Green').sum() / len(sub) * 100
    r = 100 - g
    avg = sub['Monday_Return'].mean()
    print(f"  {regime:<12} {len(sub):>6} {g:>10.1f}% {r:>8.1f}% {avg:>+11.3f}%")

# ── FRIDAY CLOSING POSITION ───────────────────────────────────────────────────
print("\n" + "="*70)
print("CONDITION 5: Where did Friday close relative to its range?")
print("="*70)
print("(Did it close near the lows = more panic, or recover toward highs?)")

spy_fri = spy[spy['DayOfWeek'] == 4].copy()
spy_fri['Close_Position'] = (spy_fri['Close'] - spy_fri['Low']) / (spy_fri['High'] - spy_fri['Low']) * 100

df['Friday_Close_Position'] = df['Friday_Date'].map(spy_fri['Close_Position'])

red_df2 = df[df['Friday_Color'] == 'Red'].copy().dropna(subset=['Friday_Close_Position'])

buckets2 = [
    (0, 25,  'Closed near LOWS (0–25% of range)'),
    (25, 50, 'Closed lower half (25–50%)'),
    (50, 75, 'Closed upper half (50–75%)'),
    (75, 100,'Closed near HIGHS (75–100%)'),
]

print(f"\n  {'Where Friday closed':<35} {'Count':>6} {'Mon Green%':>11} {'Mon Red%':>9} {'Avg Mon Ret':>12}")
print("  " + "-"*76)
for lo, hi, label in buckets2:
    mask = (red_df2['Friday_Close_Position'] >= lo) & (red_df2['Friday_Close_Position'] < hi)
    sub = red_df2[mask]
    if len(sub) == 0:
        continue
    g = (sub['Monday_Color'] == 'Green').sum() / len(sub) * 100
    r = 100 - g
    avg = sub['Monday_Return'].mean()
    print(f"  {label:<35} {len(sub):>6} {g:>10.1f}% {r:>8.1f}% {avg:>+11.3f}%")

# ── SUMMARY: WHEN IS MONDAY MOST LIKELY RED? ─────────────────────────────────
print("\n" + "="*70)
print("SUMMARY: When is Monday MOST LIKELY to print RED?")
print("="*70)

print(f"""
  Conditions that increase Monday red probability:

  1. Friday GREEN (not red)
     → Monday green only {(df[df['Friday_Color']=='Green']['Monday_Color']=='Green').sum()/len(df[df['Friday_Color']=='Green'])*100:.1f}%
       vs {(df[df['Friday_Color']=='Red']['Monday_Color']=='Green').sum()/len(df[df['Friday_Color']=='Red'])*100:.1f}% after red Friday

  2. Friday was VERY red (2%+) — panic selling can continue into Monday

  3. Market is in BEAR regime (price below 50-day SMA)
     → Mean reversion is weaker in downtrends

  4. Friday closed near its LOWS (no intraday recovery)
     → Suggests sustained selling pressure, not just a dip
""")

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SPY: When Does Monday Print Red?', fontsize=15, fontweight='bold')

# 1. Friday color → Monday outcome
ax1 = axes[0, 0]
categories = ['Red Friday', 'Green Friday', 'All Fridays']
green_rates = [
    (df[df['Friday_Color']=='Red']['Monday_Color']=='Green').mean()*100,
    (df[df['Friday_Color']=='Green']['Monday_Color']=='Green').mean()*100,
    baseline_green,
]
red_rates = [100 - g for g in green_rates]
x = np.arange(len(categories))
bars1 = ax1.bar(x, green_rates, color='#26a69a', label='Monday Green', width=0.4)
bars2 = ax1.bar(x, red_rates, bottom=green_rates, color='#ef5350', label='Monday Red', width=0.4)
ax1.set_xticks(x)
ax1.set_xticklabels(categories, fontsize=9)
ax1.set_ylabel('Probability %')
ax1.set_title('Monday Outcome by Friday Color')
ax1.legend(fontsize=9)
ax1.set_ylim(0, 100)
for i, (g, r) in enumerate(zip(green_rates, red_rates)):
    ax1.text(i, g/2, f'{g:.1f}%', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax1.text(i, g + r/2, f'{r:.1f}%', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Friday red magnitude → Monday green rate
ax2 = axes[0, 1]
labels2 = ['0–0.5%\n(barely red)', '0.5–1%\n(mild)', '1–2%\n(moderate)', '2%+\n(very red)']
green_rates2 = []
counts2 = []
for lo, hi, label in buckets:
    mask = (red_fridays['Friday_Return_Abs'] >= lo) & (red_fridays['Friday_Return_Abs'] < hi)
    sub = red_fridays[mask]
    if len(sub) > 0:
        green_rates2.append((sub['Monday_Color'] == 'Green').sum() / len(sub) * 100)
        counts2.append(len(sub))
    else:
        green_rates2.append(0)
        counts2.append(0)
colors2 = ['#26a69a' if g >= baseline_green else '#ef5350' for g in green_rates2]
bars = ax2.bar(labels2, green_rates2, color=colors2, width=0.5)
ax2.axhline(y=baseline_green, color='gray', linestyle='--', alpha=0.7, label=f'Baseline: {baseline_green:.1f}%')
ax2.set_ylabel('Monday Green %')
ax2.set_title('Monday Green Rate by Friday Red Severity')
ax2.legend(fontsize=9)
ax2.set_ylim(0, 100)
for bar, g, c in zip(bars, green_rates2, counts2):
    ax2.text(bar.get_x() + bar.get_width()/2, g + 1, f'{g:.1f}%\n(n={c})',
             ha='center', va='bottom', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Market regime
ax3 = axes[1, 0]
regimes = ['Bull', 'Bear', 'Sideways']
regime_green = []
regime_counts = []
for r in regimes:
    sub = red_df[red_df['Regime'] == r]
    if len(sub) > 0:
        regime_green.append((sub['Monday_Color'] == 'Green').sum() / len(sub) * 100)
        regime_counts.append(len(sub))
    else:
        regime_green.append(0)
        regime_counts.append(0)
colors3 = ['#26a69a' if g >= baseline_green else '#ef5350' for g in regime_green]
bars3 = ax3.bar(regimes, regime_green, color=colors3, width=0.4)
ax3.axhline(y=baseline_green, color='gray', linestyle='--', alpha=0.7, label=f'Baseline: {baseline_green:.1f}%')
ax3.set_ylabel('Monday Green %')
ax3.set_title('Monday Green Rate by Market Regime\n(After Red Friday)')
ax3.legend(fontsize=9)
ax3.set_ylim(0, 100)
for bar, g, c in zip(bars3, regime_green, regime_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, g + 1, f'{g:.1f}%\n(n={c})',
             ha='center', va='bottom', fontsize=8)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Friday close position
ax4 = axes[1, 1]
labels4 = ['Near Lows\n(0–25%)', 'Lower Half\n(25–50%)', 'Upper Half\n(50–75%)', 'Near Highs\n(75–100%)']
green_rates4 = []
counts4 = []
for lo, hi, label in buckets2:
    mask = (red_df2['Friday_Close_Position'] >= lo) & (red_df2['Friday_Close_Position'] < hi)
    sub = red_df2[mask]
    if len(sub) > 0:
        green_rates4.append((sub['Monday_Color'] == 'Green').sum() / len(sub) * 100)
        counts4.append(len(sub))
    else:
        green_rates4.append(0)
        counts4.append(0)
colors4 = ['#26a69a' if g >= baseline_green else '#ef5350' for g in green_rates4]
bars4 = ax4.bar(labels4, green_rates4, color=colors4, width=0.5)
ax4.axhline(y=baseline_green, color='gray', linestyle='--', alpha=0.7, label=f'Baseline: {baseline_green:.1f}%')
ax4.set_ylabel('Monday Green %')
ax4.set_title('Monday Green Rate by Where Friday Closed\n(After Red Friday)')
ax4.legend(fontsize=9)
ax4.set_ylim(0, 100)
for bar, g, c in zip(bars4, green_rates4, counts4):
    ax4.text(bar.get_x() + bar.get_width()/2, g + 1, f'{g:.1f}%\n(n={c})',
             ha='center', va='bottom', fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/ashleighchua/trading analyses/monday_red_analysis.png', dpi=150, bbox_inches='tight')
print("\nChart saved to: monday_red_analysis.png")
print("✅ Analysis complete!")
