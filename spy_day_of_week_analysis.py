"""
SPY Day-of-Week Trading Strategy Analysis
==========================================
Tests two hypotheses:
1. If Friday is red, Monday is likely green
2. Thursday follows Tuesday's color (green/red)

Uses historical SPY data to validate these patterns.
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ── Download SPY data (5 years) ──────────────────────────────────────────────
print("Downloading SPY historical data...")
spy = yf.download("SPY", period="5y", auto_adjust=True)
spy = spy.droplevel('Ticker', axis=1) if isinstance(spy.columns, pd.MultiIndex) else spy

# ── Add day-of-week and color columns ────────────────────────────────────────
spy['DayOfWeek'] = spy.index.dayofweek  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
spy['DayName'] = spy.index.day_name()
spy['Return'] = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color'] = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

print(f"\nData range: {spy.index[0].date()} to {spy.index[-1].date()}")
print(f"Total trading days: {len(spy)}")

# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 1: If Friday is red → Monday is likely green
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("HYPOTHESIS 1: If Friday is RED → Monday is GREEN")
print("="*70)

fridays = spy[spy['DayOfWeek'] == 4].copy()
mondays = spy[spy['DayOfWeek'] == 0].copy()

# Match each Friday with the following Monday
results_h1 = []
for fri_date, fri_row in fridays.iterrows():
    # Find next Monday (should be ~3 days later)
    next_mondays = mondays[mondays.index > fri_date]
    if len(next_mondays) == 0:
        continue
    next_mon = next_mondays.iloc[0]
    days_diff = (next_mon.name - fri_date).days
    if days_diff <= 5:  # Allow for holidays (up to 5 calendar days)
        results_h1.append({
            'Friday_Date': fri_date,
            'Friday_Color': fri_row['Color'],
            'Friday_Return': fri_row['Return'],
            'Monday_Date': next_mon.name,
            'Monday_Color': next_mon['Color'],
            'Monday_Return': next_mon['Return'],
        })

df_h1 = pd.DataFrame(results_h1)

# All Fridays → Monday analysis
print(f"\nTotal Friday→Monday pairs: {len(df_h1)}")

# Red Friday → Monday
red_fri = df_h1[df_h1['Friday_Color'] == 'Red']
red_fri_green_mon = red_fri[red_fri['Monday_Color'] == 'Green']
red_fri_pct = len(red_fri_green_mon) / len(red_fri) * 100 if len(red_fri) > 0 else 0

print(f"\nRed Fridays: {len(red_fri)}")
print(f"  → Monday Green: {len(red_fri_green_mon)} ({red_fri_pct:.1f}%)")
print(f"  → Monday Red:   {len(red_fri) - len(red_fri_green_mon)} ({100 - red_fri_pct:.1f}%)")
print(f"  → Avg Monday Return after Red Friday: {red_fri['Monday_Return'].mean():.3f}%")

# For comparison: Green Friday → Monday
green_fri = df_h1[df_h1['Friday_Color'] == 'Green']
green_fri_green_mon = green_fri[green_fri['Monday_Color'] == 'Green']
green_fri_pct = len(green_fri_green_mon) / len(green_fri) * 100 if len(green_fri) > 0 else 0

print(f"\nGreen Fridays (for comparison): {len(green_fri)}")
print(f"  → Monday Green: {len(green_fri_green_mon)} ({green_fri_pct:.1f}%)")
print(f"  → Monday Red:   {len(green_fri) - len(green_fri_green_mon)} ({100 - green_fri_pct:.1f}%)")
print(f"  → Avg Monday Return after Green Friday: {green_fri['Monday_Return'].mean():.3f}%")

# Baseline Monday
mon_green_pct = len(mondays[mondays.index.isin(df_h1['Monday_Date'])][spy['Color'] == 'Green']) / len(df_h1) * 100
all_mon_green = (df_h1['Monday_Color'] == 'Green').sum()
all_mon_pct = all_mon_green / len(df_h1) * 100
print(f"\nBaseline: All Mondays green rate: {all_mon_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS 2: Thursday follows Tuesday's color
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("HYPOTHESIS 2: Thursday follows Tuesday's color")
print("="*70)

tuesdays = spy[spy['DayOfWeek'] == 1].copy()
thursdays = spy[spy['DayOfWeek'] == 3].copy()

results_h2 = []
for tue_date, tue_row in tuesdays.iterrows():
    # Find same-week Thursday (should be 2 days later)
    next_thurs = thursdays[thursdays.index > tue_date]
    if len(next_thurs) == 0:
        continue
    next_thu = next_thurs.iloc[0]
    days_diff = (next_thu.name - tue_date).days
    if days_diff <= 4:  # Same week
        results_h2.append({
            'Tuesday_Date': tue_date,
            'Tuesday_Color': tue_row['Color'],
            'Tuesday_Return': tue_row['Return'],
            'Thursday_Date': next_thu.name,
            'Thursday_Color': next_thu['Color'],
            'Thursday_Return': next_thu['Return'],
            'Same_Color': tue_row['Color'] == next_thu['Color'],
        })

df_h2 = pd.DataFrame(results_h2)

print(f"\nTotal Tuesday→Thursday pairs: {len(df_h2)}")

same_color_count = df_h2['Same_Color'].sum()
same_color_pct = same_color_count / len(df_h2) * 100

print(f"\nThursday MATCHES Tuesday's color: {same_color_count} ({same_color_pct:.1f}%)")
print(f"Thursday DIFFERS from Tuesday:    {len(df_h2) - same_color_count} ({100 - same_color_pct:.1f}%)")

# Break down by Tuesday color
green_tue = df_h2[df_h2['Tuesday_Color'] == 'Green']
green_tue_match = green_tue[green_tue['Same_Color']]
green_tue_pct = len(green_tue_match) / len(green_tue) * 100 if len(green_tue) > 0 else 0

red_tue = df_h2[df_h2['Tuesday_Color'] == 'Red']
red_tue_match = red_tue[red_tue['Same_Color']]
red_tue_pct = len(red_tue_match) / len(red_tue) * 100 if len(red_tue) > 0 else 0

print(f"\nGreen Tuesday → Green Thursday: {len(green_tue_match)}/{len(green_tue)} ({green_tue_pct:.1f}%)")
print(f"Red Tuesday   → Red Thursday:   {len(red_tue_match)}/{len(red_tue)} ({red_tue_pct:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# GENERAL DAY-OF-WEEK STATS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("GENERAL DAY-OF-WEEK STATISTICS")
print("="*70)

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(f"\n{'Day':<12} {'Avg Return':>10} {'Green %':>10} {'Avg Green':>10} {'Avg Red':>10} {'Count':>7}")
print("-" * 62)
for i, day in enumerate(day_names):
    day_data = spy[spy['DayOfWeek'] == i]
    green_pct = (day_data['Color'] == 'Green').sum() / len(day_data) * 100
    avg_ret = day_data['Return'].mean()
    avg_green = day_data[day_data['Color'] == 'Green']['Return'].mean()
    avg_red = day_data[day_data['Color'] == 'Red']['Return'].mean()
    print(f"{day:<12} {avg_ret:>+9.3f}% {green_pct:>9.1f}% {avg_green:>+9.3f}% {avg_red:>+9.3f}% {len(day_data):>7}")

# ══════════════════════════════════════════════════════════════════════════════
# CONSECUTIVE DAY PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FULL DAY-TO-DAY TRANSITION MATRIX")
print("="*70)
print("(Probability that next trading day is GREEN given current day's color)\n")

print(f"{'Current Day':<15} {'If GREEN → Next GREEN':>22} {'If RED → Next GREEN':>22}")
print("-" * 62)

for i, day in enumerate(day_names):
    day_data = spy[spy['DayOfWeek'] == i].copy()

    green_days = day_data[day_data['Color'] == 'Green']
    red_days = day_data[day_data['Color'] == 'Red']

    # Find next trading day for each
    next_green_count = 0
    next_green_total = 0
    for idx in green_days.index:
        next_days = spy[spy.index > idx]
        if len(next_days) > 0:
            next_day = next_days.iloc[0]
            next_green_total += 1
            if next_day['Color'] == 'Green':
                next_green_count += 1

    next_red_green_count = 0
    next_red_total = 0
    for idx in red_days.index:
        next_days = spy[spy.index > idx]
        if len(next_days) > 0:
            next_day = next_days.iloc[0]
            next_red_total += 1
            if next_day['Color'] == 'Green':
                next_red_green_count += 1

    g2g_pct = next_green_count / next_green_total * 100 if next_green_total > 0 else 0
    r2g_pct = next_red_green_count / next_red_total * 100 if next_red_total > 0 else 0

    print(f"{day:<15} {g2g_pct:>18.1f}% ({next_green_count}/{next_green_total})   {r2g_pct:>14.1f}% ({next_red_green_count}/{next_red_total})")

# ══════════════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VERDICT")
print("="*70)

print(f"""
Hypothesis 1 — "Red Friday → Green Monday":
  Result: {red_fri_pct:.1f}% of the time Monday is green after a red Friday
  Baseline (any Monday green): {all_mon_pct:.1f}%
  Edge over baseline: {red_fri_pct - all_mon_pct:+.1f} percentage points
  {"✅ SUPPORTED" if red_fri_pct > 55 else "⚠️  WEAK/NOT SUPPORTED"} — {"meaningful" if abs(red_fri_pct - all_mon_pct) > 5 else "marginal"} edge vs baseline

Hypothesis 2 — "Thursday follows Tuesday's color":
  Result: {same_color_pct:.1f}% of the time they match
  Random chance would be ~50%
  Edge over random: {same_color_pct - 50:+.1f} percentage points
  {"✅ SUPPORTED" if same_color_pct > 55 else "⚠️  WEAK/NOT SUPPORTED"} — {"meaningful" if abs(same_color_pct - 50) > 5 else "marginal"} edge vs random
""")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SPY Day-of-Week Trading Strategy Analysis (5Y)', fontsize=14, fontweight='bold')

# Chart 1: Hypothesis 1 results
ax1 = axes[0, 0]
cats = ['Red Fri→\nGreen Mon', 'Red Fri→\nRed Mon', 'Green Fri→\nGreen Mon', 'Green Fri→\nRed Mon']
vals = [
    len(red_fri_green_mon), len(red_fri) - len(red_fri_green_mon),
    len(green_fri_green_mon), len(green_fri) - len(green_fri_green_mon)
]
colors_bar = ['#26a69a', '#ef5350', '#26a69a', '#ef5350']
ax1.bar(cats, vals, color=colors_bar)
ax1.set_title('H1: Friday→Monday Color Transitions')
ax1.set_ylabel('Count')
for j, v in enumerate(vals):
    ax1.text(j, v + 1, str(v), ha='center', fontsize=10)

# Chart 2: Hypothesis 2 results
ax2 = axes[0, 1]
h2_cats = ['Green Tue→\nGreen Thu', 'Green Tue→\nRed Thu', 'Red Tue→\nRed Thu', 'Red Tue→\nGreen Thu']
h2_vals = [
    len(green_tue_match), len(green_tue) - len(green_tue_match),
    len(red_tue_match), len(red_tue) - len(red_tue_match)
]
h2_colors = ['#26a69a', '#ef5350', '#ef5350', '#26a69a']
ax2.bar(h2_cats, h2_vals, color=h2_colors)
ax2.set_title('H2: Tuesday→Thursday Color Match')
ax2.set_ylabel('Count')
for j, v in enumerate(h2_vals):
    ax2.text(j, v + 1, str(v), ha='center', fontsize=10)

# Chart 3: Average return by day of week
ax3 = axes[1, 0]
avg_returns = [spy[spy['DayOfWeek'] == i]['Return'].mean() for i in range(5)]
bar_colors = ['#26a69a' if r >= 0 else '#ef5350' for r in avg_returns]
ax3.bar(day_names, avg_returns, color=bar_colors)
ax3.set_title('Average Return by Day of Week')
ax3.set_ylabel('Average Return (%)')
ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for j, v in enumerate(avg_returns):
    ax3.text(j, v + 0.002, f'{v:+.3f}%', ha='center', fontsize=9)

# Chart 4: Green day percentage by day of week
ax4 = axes[1, 1]
green_pcts = [(spy[spy['DayOfWeek'] == i]['Color'] == 'Green').mean() * 100 for i in range(5)]
ax4.bar(day_names, green_pcts, color='#26a69a')
ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% baseline')
ax4.set_title('Green Day Probability by Day of Week')
ax4.set_ylabel('Green Day %')
ax4.set_ylim(40, 60)
ax4.legend()
for j, v in enumerate(green_pcts):
    ax4.text(j, v + 0.3, f'{v:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/ashleighchua/trading analyses/spy_strategy_analysis.png', dpi=150, bbox_inches='tight')
print("\nChart saved to: spy_strategy_analysis.png")
print("Analysis complete!")
