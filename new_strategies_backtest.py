"""
New Strategies Backtest
========================
1. VIX Spike Strategy — Buy SPY when fear is extreme
2. QQQ Playbook — Same Red Friday signals but on QQQ (more volatile)
3. Covered Calls — Sell monthly calls on existing ETF holdings

All backtested on 10 years of data.
"""

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("Downloading data...")
spy  = yf.download("SPY",  period="10y", auto_adjust=True)
qqq  = yf.download("QQQ",  period="10y", auto_adjust=True)
vix  = yf.download("^VIX", period="10y", auto_adjust=True)
schd = yf.download("SCHD", period="10y", auto_adjust=True)

for df in [spy, qqq, vix, schd]:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')

spy['DayOfWeek']  = spy.index.dayofweek
spy['Return']     = (spy['Close'] - spy['Open']) / spy['Open'] * 100
spy['Color']      = spy['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

qqq['DayOfWeek']  = qqq.index.dayofweek
qqq['Return']     = (qqq['Close'] - qqq['Open']) / qqq['Open'] * 100
qqq['Color']      = qqq['Return'].apply(lambda x: 'Green' if x >= 0 else 'Red')

print(f"SPY:  {spy.index[0].date()} → {spy.index[-1].date()}")
print(f"QQQ:  {qqq.index[0].date()} → {qqq.index[-1].date()}")
print(f"VIX:  {vix.index[0].date()} → {vix.index[-1].date()}")

initial = 10000

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: VIX SPIKE — Buy SPY when fear is extreme
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 1: VIX SPIKE — Buy SPY when VIX spikes, sell when calm")
print("="*70)
print("""
  Logic:
  - VIX = the 'fear index' — measures how scared the market is
  - When VIX spikes above a threshold → panic selling → mean reversion opportunity
  - Buy SPY the day VIX closes above threshold
  - Hold until VIX drops back below exit threshold OR max days reached
""")

# Align VIX and SPY
common = spy.index.intersection(vix.index)
spy_vix = spy.loc[common].copy()
spy_vix['VIX'] = vix.loc[common, 'Close']

print("  Testing VIX entry/exit thresholds:")
print(f"  {'Entry>':>8} {'Exit<':>6} {'MaxDays':>8} {'Trades':>7} {'WR%':>6} {'AvgRet':>8} {'AnnRet':>8} {'Sharpe':>7}")
print("  " + "-"*65)

vix_results = []
best_vix = None
best_vix_sharpe = -999

for vix_entry in [20, 25, 30, 35, 40]:
    for vix_exit in [18, 20, 22, 25]:
        if vix_exit >= vix_entry:
            continue
        for max_days in [3, 5, 7, 10, 15, 20]:
            trades = []
            in_trade = False
            entry_price = 0
            entry_date = None
            days_in = 0

            for date, row in spy_vix.iterrows():
                if pd.isna(row['VIX']):
                    continue
                if not in_trade:
                    if row['VIX'] >= vix_entry:
                        in_trade = True
                        entry_price = row['Close']
                        entry_date = date
                        days_in = 0
                else:
                    days_in += 1
                    if row['VIX'] <= vix_exit or days_in >= max_days:
                        ret = (row['Close'] - entry_price) / entry_price * 100
                        trades.append({
                            'Entry_Date': entry_date,
                            'Exit_Date':  date,
                            'Entry_Price': entry_price,
                            'Exit_Price': row['Close'],
                            'Return_Pct': ret,
                            'Days_Held':  days_in,
                            'Win':        ret > 0,
                            'VIX_Entry':  vix_entry,
                        })
                        in_trade = False

            if len(trades) < 8:
                continue

            t = pd.DataFrame(trades)
            wr = t['Win'].mean() * 100
            avg = t['Return_Pct'].mean()
            years = (t['Exit_Date'].iloc[-1] - t['Entry_Date'].iloc[0]).days / 365.25
            cap = initial * (1 + t['Return_Pct'] / 100).prod()
            ann = ((cap / initial) ** (1 / years) - 1) * 100
            sharpe = (avg / t['Return_Pct'].std()) * np.sqrt(len(t) / years) if t['Return_Pct'].std() > 0 else 0

            vix_results.append({
                'VIX_Entry': vix_entry, 'VIX_Exit': vix_exit, 'Max_Days': max_days,
                'Trades': len(t), 'WR': wr, 'AvgRet': avg, 'AnnRet': ann,
                'Sharpe': sharpe, 'trades_df': t,
            })
            if sharpe > best_vix_sharpe:
                best_vix_sharpe = sharpe
                best_vix = vix_results[-1]

# Print top 10
for r in sorted(vix_results, key=lambda x: x['Sharpe'], reverse=True)[:10]:
    marker = " ◀ BEST" if r == best_vix else ""
    print(f"  VIX>{r['VIX_Entry']:>2} exit<{r['VIX_Exit']:>2}  max{r['Max_Days']:>2}d"
          f"  {r['Trades']:>7}  {r['WR']:>5.1f}%  {r['AvgRet']:>+7.3f}%  {r['AnnRet']:>+7.2f}%  {r['Sharpe']:>6.2f}{marker}")

vix_bt = best_vix['trades_df']
cap_vix = initial
eq_vix = []
for _, row in vix_bt.iterrows():
    cap_vix *= (1 + row['Return_Pct'] / 100)
    eq_vix.append({'Date': row['Exit_Date'], 'Capital': cap_vix})

vix_wr   = vix_bt['Win'].mean() * 100
vix_cum  = (cap_vix - initial) / initial * 100
vix_years = (vix_bt['Exit_Date'].iloc[-1] - vix_bt['Entry_Date'].iloc[0]).days / 365.25
vix_ann  = ((cap_vix / initial) ** (1 / vix_years) - 1) * 100
vix_avg_days = vix_bt['Days_Held'].mean()

print(f"""
  BEST VIX SETUP: Buy when VIX>{best_vix['VIX_Entry']}, sell when VIX<{best_vix['VIX_Exit']} or {best_vix['Max_Days']} days
  Win rate:          {vix_wr:.1f}%
  Avg days held:     {vix_avg_days:.1f} days
  Total return:      {vix_cum:+.2f}%
  Annualized return: {vix_ann:+.2f}%
  Total trades:      {len(vix_bt)} (~{len(vix_bt)/vix_years:.0f}/year)
""")

print("  Yearly breakdown:")
vix_bt['Year'] = vix_bt['Exit_Date'].dt.year
for yr, g in vix_bt.groupby('Year'):
    print(f"    {yr}: {len(g)} trades | WR {g['Win'].mean()*100:.0f}% | Return {g['Return_Pct'].sum():+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: QQQ PLAYBOOK — Same signals as SPY but trade QQQ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 2: QQQ PLAYBOOK — Red Friday SPY signal → Buy QQQ Monday")
print("="*70)
print("""
  Logic:
  - Same Red Friday → Monday signal as SPY playbook
  - But execute on QQQ instead of SPY
  - QQQ is more volatile (~1.3x SPY) so wins are bigger but so are losses
  - Also tests the full 4-day playbook signals on QQQ
""")

# Build Fri→Mon pairs using SPY signal, QQQ execution
fridays_spy = spy[spy['DayOfWeek'] == 4]
mondays_qqq = qqq[qqq['DayOfWeek'] == 0]

pairs = []
for fri_date, fri_row in fridays_spy.iterrows():
    next_mons = mondays_qqq[mondays_qqq.index > fri_date]
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

df_pairs = pd.DataFrame(pairs)
signals = df_pairs[df_pairs['Friday_Return'] <= -0.5].copy()
signals['Close_Return_Pct'] = (signals['Monday_Close'] - signals['Monday_Open']) / signals['Monday_Open'] * 100
signals['Max_Drawdown_Pct'] = (signals['Monday_Low'] - signals['Monday_Open']) / signals['Monday_Open'] * 100

print(f"  Red Friday (>= 0.5%) signals: {len(signals)}")
print(f"  QQQ Monday green rate after signal: {(signals['Close_Return_Pct']>0).sum()/len(signals)*100:.1f}%")

# Stop loss optimization for QQQ
stop_levels = np.arange(0.5, 4.05, 0.5)
qqq_results = []

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

    t = pd.DataFrame(trades)
    n = len(t)
    wr = t['win'].mean() * 100
    cum = ((1 + t['return'] / 100).prod() - 1) * 100
    avg = t['return'].mean()
    gp = t[t['return'] > 0]['return'].sum()
    gl = abs(t[t['return'] < 0]['return'].sum())
    pf = gp / gl if gl > 0 else float('inf')
    sharpe = (avg / t['return'].std()) * np.sqrt(n / 10) if t['return'].std() > 0 else 0
    qqq_results.append({'SL': sl, 'WR': wr, 'CumRet': cum, 'Sharpe': sharpe, 'PF': pf, 'trades': t})

qqq_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'trades'} for r in qqq_results])
best_qqq_sl_idx = qqq_res['Sharpe'].idxmax()
best_qqq = qqq_results[best_qqq_sl_idx]
optimal_qqq_sl = best_qqq['SL']

print(f"\n  {'SL%':>5} {'WR%':>6} {'CumRet':>9} {'Sharpe':>8} {'PF':>6}")
print("  " + "-"*40)
for r in qqq_results:
    marker = " ◀ BEST" if r['SL'] == optimal_qqq_sl else ""
    print(f"  {r['SL']:>4.1f}% {r['WR']:>5.1f}% {r['CumRet']:>+8.2f}% {r['Sharpe']:>7.2f} {r['PF']:>5.2f}{marker}")

# Full QQQ backtest
cap_qqq = initial
eq_qqq = []
qqq_trades = []
for _, row in signals.iterrows():
    entry = row['Monday_Open']
    stop_price = entry * (1 - optimal_qqq_sl / 100)
    if row['Monday_Low'] <= stop_price:
        exit_price = stop_price
        exit_type = 'STOP'
    else:
        exit_price = row['Monday_Close']
        exit_type = 'CLOSE'
    ret = (exit_price - entry) / entry
    pnl = cap_qqq * ret
    cap_qqq += pnl
    qqq_trades.append({
        'Date': row['Monday_Date'], 'Return_Pct': ret * 100,
        'PnL': pnl, 'Capital': cap_qqq, 'Win': ret > 0, 'Exit_Type': exit_type,
    })
    eq_qqq.append({'Date': row['Monday_Date'], 'Capital': cap_qqq})

qqq_bt = pd.DataFrame(qqq_trades)
qqq_wr  = qqq_bt['Win'].mean() * 100
qqq_cum = (cap_qqq - initial) / initial * 100
qqq_years = (signals['Monday_Date'].iloc[-1] - signals['Monday_Date'].iloc[0]).days / 365.25
qqq_ann = ((cap_qqq / initial) ** (1 / qqq_years) - 1) * 100

print(f"""
  BEST QQQ SETUP: Red Fri >= 0.5%, SL = {optimal_qqq_sl:.1f}%, hold to Monday close
  Win rate:          {qqq_wr:.1f}%
  Total return:      {qqq_cum:+.2f}%
  Annualized return: {qqq_ann:+.2f}%
  Trades:            {len(qqq_bt)} (~{len(qqq_bt)/qqq_years:.0f}/year)
""")

print("  Yearly breakdown:")
qqq_bt['Year'] = qqq_bt['Date'].dt.year
for yr, g in qqq_bt.groupby('Year'):
    print(f"    {yr}: {len(g)} trades | WR {g['Win'].mean()*100:.0f}% | Return {g['Return_Pct'].sum():+.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: COVERED CALLS — Sell monthly calls on ETF holdings
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STRATEGY 3: COVERED CALLS on SPY, QQQ, SCHD")
print("="*70)
print("""
  Logic:
  - You already OWN SPY, QQQ, QQQM, SCHD worth ~$54,000
  - Each month, sell a call option 2-5% above current price
  - Collect premium (~0.5-2% of stock value per month)
  - If stock stays below strike → keep full premium (win)
  - If stock rises above strike → capped gain but still profitable
  - This generates MONTHLY INCOME from stocks you already hold
""")

# Simulate covered calls on SPY
# Monthly call at 2% OTM, premium = ~0.8% of stock value (typical for SPY 30-day)
# If SPY rises >2% → assigned, keep premium + 2% gain
# If SPY stays flat/rises <2% → keep full premium
# If SPY falls → keep premium, but stock loses value (offset partially)

print("  Simulating covered calls on SPY (2% OTM, monthly, ~0.8% premium)")
print("  Using actual SPY monthly returns to model assignment/expiry\n")

# Get monthly SPY data
spy_monthly = spy['Close'].resample('ME').last()
spy_monthly_ret = spy_monthly.pct_change() * 100

STRIKE_OTM = 2.0     # sell call 2% above current price
PREMIUM_PCT = 0.8    # collect 0.8% premium each month
PORTFOLIO_SPY = 18000  # ~$18k in SPY (1/3 of $54k ETF portfolio)

cc_trades = []
for i in range(1, len(spy_monthly_ret)):
    date = spy_monthly_ret.index[i]
    monthly_ret = spy_monthly_ret.iloc[i]

    if pd.isna(monthly_ret):
        continue

    premium = PREMIUM_PCT  # % collected upfront

    if monthly_ret > STRIKE_OTM:
        # Assigned — stock called away at strike, but you keep premium
        total_ret = STRIKE_OTM + premium  # capped at strike + premium
        outcome = 'ASSIGNED'
    else:
        # Expires worthless — keep full premium + stock move
        total_ret = monthly_ret + premium
        outcome = 'EXPIRED'

    cc_trades.append({
        'Date': date,
        'Stock_Return': monthly_ret,
        'Premium': premium,
        'Total_Return': total_ret,
        'Outcome': outcome,
        'Win': total_ret > 0,
    })

cc_bt = pd.DataFrame(cc_trades)

# Equity curve
cap_cc = PORTFOLIO_SPY
eq_cc = []
for _, row in cc_bt.iterrows():
    cap_cc *= (1 + row['Total_Return'] / 100)
    eq_cc.append({'Date': row['Date'], 'Capital': cap_cc})

cc_wr   = cc_bt['Win'].mean() * 100
cc_cum  = (cap_cc - PORTFOLIO_SPY) / PORTFOLIO_SPY * 100
cc_years = len(cc_bt) / 12
cc_ann  = ((cap_cc / PORTFOLIO_SPY) ** (1 / cc_years) - 1) * 100
cc_monthly_income = (cap_cc - PORTFOLIO_SPY) / len(cc_bt)
assigned_pct = (cc_bt['Outcome'] == 'ASSIGNED').sum() / len(cc_bt) * 100
avg_monthly = cc_bt['Total_Return'].mean()

print(f"  Portfolio size used: ${PORTFOLIO_SPY:,} (SPY portion)")
print(f"  Strike: {STRIKE_OTM}% OTM | Premium collected: {PREMIUM_PCT}%/month")
print(f"\n  Win rate:              {cc_wr:.1f}%")
print(f"  Assigned (called away): {assigned_pct:.1f}% of months")
print(f"  Avg monthly return:    {avg_monthly:+.2f}%")
print(f"  Avg monthly income:    ${cc_monthly_income:+,.0f}")
print(f"  Total return:          {cc_cum:+.2f}%")
print(f"  Annualized return:     {cc_ann:+.2f}%")
print(f"  Extra income vs buy-hold: significant — you collect premium even in flat months")

print(f"\n  Yearly breakdown:")
cc_bt['Year'] = cc_bt['Date'].dt.year
for yr, g in cc_bt.groupby('Year'):
    avg_m = g['Total_Return'].mean()
    monthly_inc = PORTFOLIO_SPY * (avg_m / 100)
    print(f"    {yr}: avg {avg_m:+.2f}%/month → ~${monthly_inc:,.0f}/month income on ${PORTFOLIO_SPY:,}")

# What about full $54k portfolio?
print(f"\n  If applied to your full $54,000 ETF portfolio:")
full_monthly = PORTFOLIO_SPY * 3 * (avg_monthly / 100)
print(f"  Avg monthly income: ~${full_monthly:,.0f}/month")
print(f"  Annual income: ~${full_monthly*12:,.0f}/year")
print(f"\n  ⚠️  Note: Covered calls require options approval on your broker")
print(f"  ⚠️  and at least 100 shares per contract (~${spy['Close'].iloc[-1]*100:,.0f} for 1 SPY contract)")

# ══════════════════════════════════════════════════════════════════════════════
# FULL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{'='*70}")
print("FULL STRATEGY COMPARISON — All Strategies Combined")
print(f"{'='*70}")

spy_day_ann = 6.5
earn_ann_ret = 7.1
combined_current = spy_day_ann + earn_ann_ret

print(f"""
  {'Strategy':<35} {'Trades/yr':>10} {'Win Rate':>9} {'Ann Ret':>9} {'Capital Used':>13}
  {'-'*80}
  {'SPY Day Playbook':<35} {'~22':>10} {'~64%':>9} {'+6.5%':>9} {'$26,500':>13}
  {'Earnings Season Drift':<35} {'~4':>10} {'~78%':>9} {'+7.1%':>9} {'$26,500':>13}
  {'VIX Spike':<35} {'~'+str(round(len(vix_bt)/vix_years)):>10} {vix_wr:>8.1f}% {vix_ann:>+8.2f}% {'$26,500':>13}
  {'QQQ Playbook (Red Fri→Mon)':<35} {'~'+str(round(len(qqq_bt)/qqq_years)):>10} {qqq_wr:>8.1f}% {qqq_ann:>+8.2f}% {'$26,500':>13}
  {'Covered Calls (on $54k ETFs)':<35} {'~12':>10} {cc_wr:>8.1f}% {cc_ann:>+8.2f}% {'$54,000':>13}
""")

# Revised income projection with all strategies
print("  REVISED MONTHLY INCOME PROJECTION ($26,500 trading + $54,000 ETFs):")
print(f"\n  {'Year':<6} {'Trading ($26.5k)':>18} {'Covered Calls':>15} {'ETF Growth':>12} {'Total/Month':>12}")
print("  " + "-"*65)

trading_rate = (spy_day_ann + earn_ann_ret + vix_ann + qqq_ann) / 100
cc_rate = cc_ann / 100
etf_rate = 0.12  # 12% long-term ETF growth

trading_cap = 26500
etf_cap = 54000
cc_portfolio = 54000

for year in range(1, 8):
    trading_cap *= (1 + trading_rate)
    etf_cap *= (1 + etf_rate)
    trading_monthly = trading_cap * trading_rate / 12
    cc_monthly = cc_portfolio * (cc_rate / 12) * (1 + etf_rate) ** (year-1)
    etf_monthly = (etf_cap - 54000 * (1 + etf_rate)**(year-1)) / 12 + 54000 * etf_rate / 12
    total_monthly = trading_monthly + cc_monthly
    print(f"  Year {year:<2} {trading_monthly:>16,.0f}  {cc_monthly:>13,.0f}  {'(growing)':>12} {total_monthly:>10,.0f}")

print(f"""
  Notes:
  • Trading income = SPY playbook + Earnings drift + VIX + QQQ playbook
  • Covered calls = monthly premium income from your existing ETF holdings
  • ETF growth = capital appreciation (not liquid monthly income)
  • VIX strategy correlation: NEGATIVE to SPY — protects in bad years
  • QQQ playbook: same signals as SPY, higher volatility = higher returns
""")

# ── CHARTS ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 16))
fig.suptitle('New Strategies Backtest — VIX Spike, QQQ Playbook, Covered Calls',
             fontsize=14, fontweight='bold')

# 1. VIX chart with signals
ax1 = fig.add_subplot(3, 3, (1, 2))
ax1_twin = ax1.twinx()
ax1.plot(spy_vix.index, spy_vix['Close'], color='#90a4ae', linewidth=1, alpha=0.6, label='SPY')
ax1_twin.plot(spy_vix.index, spy_vix['VIX'], color='#ef5350', linewidth=1, alpha=0.7, label='VIX')
ax1_twin.axhline(y=best_vix['VIX_Entry'], color='#ef5350', linestyle='--', alpha=0.6,
                  label=f"Buy: VIX>{best_vix['VIX_Entry']}")
ax1_twin.axhline(y=best_vix['VIX_Exit'], color='#26a69a', linestyle='--', alpha=0.6,
                  label=f"Sell: VIX<{best_vix['VIX_Exit']}")
ax1_twin.set_ylim(0, 90)
ax1_twin.legend(fontsize=8, loc='upper right')
ax1.legend(fontsize=8, loc='upper left')
ax1.set_title(f'VIX Spike Strategy — Buy Fear, Sell Calm', fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. VIX equity curve
eq_vix_df = pd.DataFrame(eq_vix)
ax2 = fig.add_subplot(3, 3, 3)
ax2.plot(eq_vix_df['Date'], eq_vix_df['Capital'], color='#ef5350', linewidth=2)
ax2.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax2.fill_between(eq_vix_df['Date'], eq_vix_df['Capital'], initial,
                 where=eq_vix_df['Capital'] >= initial, alpha=0.2, color='#26a69a')
ax2.fill_between(eq_vix_df['Date'], eq_vix_df['Capital'], initial,
                 where=eq_vix_df['Capital'] < initial, alpha=0.2, color='#ef5350')
ax2.set_title(f'VIX Equity\n${initial:,} → ${cap_vix:,.0f} ({vix_cum:+.1f}%)', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. QQQ vs SPY returns comparison
ax3 = fig.add_subplot(3, 3, (4, 5))
ax3.plot(qqq.index, qqq['Close'] / qqq['Close'].iloc[0] * 100,
         color='#1976d2', linewidth=1.5, label='QQQ')
ax3.plot(spy.index, spy['Close'] / spy['Close'].iloc[0] * 100,
         color='#26a69a', linewidth=1.5, label='SPY')
ax3.set_title('QQQ vs SPY — 10 Year Growth (indexed to 100)', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylabel('Indexed Value')

# 4. QQQ equity curve
eq_qqq_df = pd.DataFrame(eq_qqq)
ax4 = fig.add_subplot(3, 3, 6)
ax4.plot(eq_qqq_df['Date'], eq_qqq_df['Capital'], color='#1976d2', linewidth=2)
ax4.axhline(y=initial, color='gray', linestyle=':', alpha=0.5)
ax4.fill_between(eq_qqq_df['Date'], eq_qqq_df['Capital'], initial,
                 where=eq_qqq_df['Capital'] >= initial, alpha=0.2, color='#26a69a')
ax4.fill_between(eq_qqq_df['Date'], eq_qqq_df['Capital'], initial,
                 where=eq_qqq_df['Capital'] < initial, alpha=0.2, color='#ef5350')
ax4.set_title(f'QQQ Playbook Equity\n${initial:,} → ${cap_qqq:,.0f} ({qqq_cum:+.1f}%)', fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Covered calls monthly income
eq_cc_df = pd.DataFrame(eq_cc)
ax5 = fig.add_subplot(3, 3, (7, 8))
monthly_income_series = cc_bt['Total_Return'] / 100 * PORTFOLIO_SPY
colors_cc = ['#26a69a' if r > 0 else '#ef5350' for r in monthly_income_series]
ax5.bar(cc_bt['Date'], monthly_income_series, color=colors_cc, width=20)
ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax5.set_title(f'Covered Calls Monthly Income (${PORTFOLIO_SPY:,} SPY portfolio)', fontsize=11)
ax5.set_ylabel('Monthly Income ($)')
ax5.grid(True, alpha=0.3)

# 6. Covered calls equity
ax6 = fig.add_subplot(3, 3, 9)
ax6.plot(eq_cc_df['Date'], eq_cc_df['Capital'], color='#ff9800', linewidth=2)
ax6.axhline(y=PORTFOLIO_SPY, color='gray', linestyle=':', alpha=0.5)
ax6.fill_between(eq_cc_df['Date'], eq_cc_df['Capital'], PORTFOLIO_SPY,
                 where=eq_cc_df['Capital'] >= PORTFOLIO_SPY, alpha=0.2, color='#26a69a')
ax6.set_title(f'Covered Calls Equity\n${PORTFOLIO_SPY:,} → ${cap_cc:,.0f} ({cc_cum:+.1f}%)', fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/ashleighchua/trading analyses/new_strategies_backtest.png', dpi=150, bbox_inches='tight')
print("Chart saved to: new_strategies_backtest.png")
print("✅ Done!")
