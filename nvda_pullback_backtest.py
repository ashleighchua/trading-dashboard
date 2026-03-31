"""
NVDA Trend Pullback Long — Deep Backtest
==========================================
NVDA-only backtest adding RSI filter to trend pullback long strategy.

Parameter sweep:
  EMA combo:      (20,50) vs (50,200)
  Pullback depth: 2%, 3%, 4%, 5%
  RSI filter:     None (no filter), RSI<45, RSI<40, RSI<35
  Trailing stop:  1.5%, 2%, 3%
  Max hold:       5, 10, 15 days

Period: 2020-01-01 to 2026-03-28
Starting equity: $1,000
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings("ignore")

START          = "2019-06-01"
BACKTEST_START = "2020-01-01"
END            = "2026-03-28"
INIT_EQ        = 1_000.0

EMA_COMBOS    = [(20, 50), (50, 200)]
PULLBACK_PCTS = [2.0, 3.0, 4.0, 5.0]
RSI_FILTERS   = [None, 45, 40, 35]
TRAIL_PCTS    = [0.015, 0.020, 0.030]
MAX_HOLDS     = [5, 10, 15]

print("Downloading NVDA 2019-06-01→2026-03-28...")
df_raw = yf.download("NVDA", start=START, end=END, auto_adjust=True, progress=False)
# Flatten MultiIndex columns if present (yfinance ≥0.2 returns ticker-level MultiIndex)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)
df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
df_raw = df_raw.dropna()
print(f"Done. {len(df_raw)} bars.\n")


def add_indicators(df, ema_fast, ema_slow):
    df = df.copy()
    df["ema_fast"]     = df["Close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"]     = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["high10"]       = df["High"].rolling(10).max()
    df["pullback_pct"] = (df["high10"] - df["Close"]) / df["high10"] * 100
    # RSI-14
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=13, adjust=False).mean()
    avg_l  = loss.ewm(com=13, adjust=False).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    return df


def is_entry(df, i, pullback_threshold, rsi_max):
    if i < 200:
        return False
    row = df.iloc[i]
    if row["ema_fast"] <= row["ema_slow"]:
        return False
    window = df.iloc[max(0, i - 2):i + 1]
    if window["pullback_pct"].max() < pullback_threshold:
        return False
    if rsi_max is not None:
        if pd.isna(row["rsi14"]) or row["rsi14"] >= rsi_max:
            return False
    return True


def simulate_long(df, entry_idx, entry_price, trail_pct, max_hold):
    stop = entry_price * (1 - trail_pct)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        high  = df.iloc[j]["High"]
        low_  = df.iloc[j]["Low"]
        close = df.iloc[j]["Close"]
        new_stop = close * (1 - trail_pct)
        if new_stop > stop:
            stop = new_stop
        if low_ <= stop:
            pnl_pct = (stop - entry_price) / entry_price * 100
            return pnl_pct, j, "stop"
    exit_price = df.iloc[min(entry_idx + max_hold, len(df) - 1)]["Close"]
    pnl_pct = (exit_price - entry_price) / entry_price * 100
    return pnl_pct, min(entry_idx + max_hold, len(df) - 1), "maxhold"


def run(ema_fast, ema_slow, pullback_pct, rsi_max, trail_pct, max_hold):
    df     = add_indicators(df_raw, ema_fast, ema_slow)
    trades = []
    equity = INIT_EQ
    in_trade_until = -1

    all_dates = df.index
    start_idx = all_dates.searchsorted(pd.Timestamp(BACKTEST_START))

    for i in range(start_idx, len(df)):
        if i <= in_trade_until:
            continue
        if not is_entry(df, i, pullback_pct, rsi_max):
            continue
        if i + 1 >= len(df):
            continue
        entry_price = df.iloc[i + 1]["Open"]
        if entry_price <= 0:
            continue

        trade_size = equity * 0.50
        pnl_pct, exit_idx, reason = simulate_long(df, i + 1, entry_price, trail_pct, max_hold)
        pnl_dollar = pnl_pct / 100 * trade_size
        equity += pnl_dollar
        in_trade_until = exit_idx

        trades.append({
            "date":    all_dates[i],
            "entry":   round(entry_price, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_$":   round(pnl_dollar, 2),
            "reason":  reason,
            "win":     pnl_pct > 0,
            "equity":  round(equity, 2),
            "year":    all_dates[i].year,
        })

    return pd.DataFrame(trades)


def stats(df):
    if df.empty:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
    n    = len(df)
    wins = df["win"].sum()
    wr   = wins / n * 100
    gp   = df.loc[df["win"],  "pnl_pct"].sum()
    gl   = abs(df.loc[~df["win"], "pnl_pct"].sum())
    pf   = gp / gl if gl > 0 else float("inf")
    return {"n": n, "wr": round(wr, 1), "pf": round(pf, 2), "pnl": round(df["pnl_$"].sum(), 2)}


# ── Sweep ──────────────────────────────────────────────────────────────────────
print("Sweeping parameters...\n")
results = []
for (ef, es), pb, rsi, trail, hold in product(EMA_COMBOS, PULLBACK_PCTS, RSI_FILTERS, TRAIL_PCTS, MAX_HOLDS):
    trades = run(ef, es, pb, rsi, trail, hold)
    s = stats(trades)
    rsi_label = f"RSI<{rsi}" if rsi else "no RSI"
    s.update({"ema_fast": ef, "ema_slow": es, "pullback": pb,
               "rsi_max": rsi, "rsi_label": rsi_label, "trail": trail, "hold": hold})
    results.append(s)

results_df = pd.DataFrame(results).sort_values("pf", ascending=False)

print("=" * 72)
print("  TOP 20 COMBOS BY PROFIT FACTOR  (min 10 trades — single ticker)")
print("=" * 72)
top = results_df[results_df["n"] >= 10].head(20)
for _, row in top.iterrows():
    flag = "✅" if row["wr"] >= 55 and row["pf"] >= 1.5 else "⚠️ "
    print(f"  {flag} EMA({int(row['ema_fast'])},{int(row['ema_slow'])}) "
          f"pb={row['pullback']:.0f}% {row['rsi_label']:<8} "
          f"stop={row['trail']*100:.1f}% hold={int(row['hold'])}d "
          f"| {int(row['n']):3d} trades  {row['wr']:.1f}% WR  PF {row['pf']:.2f}  ${row['pnl']:+.0f}")

# ── Best combo year-by-year ────────────────────────────────────────────────────
valid = results_df[results_df["n"] >= 10]
if not valid.empty:
    best = valid.iloc[0]
    rsi_label = f"RSI<{int(best['rsi_max'])}" if best["rsi_max"] else "no RSI"
    print(f"\n{'=' * 72}")
    print(f"  BEST: EMA({int(best['ema_fast'])},{int(best['ema_slow'])}) pb={best['pullback']:.0f}% "
          f"{rsi_label} stop={best['trail']*100:.1f}% hold={int(best['hold'])}d")
    print(f"{'=' * 72}")

    best_trades = run(int(best["ema_fast"]), int(best["ema_slow"]),
                      best["pullback"], best["rsi_max"], best["trail"], int(best["hold"]))

    if not best_trades.empty:
        yearly = best_trades.groupby("year").agg(
            n   =("pnl_pct", "count"),
            wr  =("win",     lambda x: x.mean() * 100),
            pnl =("pnl_$",  "sum"),
        )
        for yr, row in yearly.iterrows():
            bar  = "█" * max(0, int(row["pnl"] / 10)) if row["pnl"] > 0 else "▒" * max(0, int(abs(row["pnl"]) / 10))
            flag = "✓" if row["wr"] >= 55 else "✗"
            print(f"    {yr}: {int(row['n']):2d} trades  {row['wr']:4.1f}% WR  ${row['pnl']:+6.1f}  {flag}  {bar}")

# ── RSI impact summary ─────────────────────────────────────────────────────────
print(f"\n{'=' * 72}")
print("  RSI FILTER IMPACT  (best stop+hold per RSI tier, ≥10 trades)")
print(f"{'=' * 72}")
for rsi_val in RSI_FILTERS:
    subset = results_df[(results_df["rsi_max"] == rsi_val) & (results_df["n"] >= 10)]
    if subset.empty:
        continue
    best_row = subset.iloc[0]
    label = f"RSI<{int(rsi_val)}" if rsi_val else "No RSI "
    flag = "✅" if best_row["wr"] >= 55 and best_row["pf"] >= 1.5 else "⚠️ "
    print(f"  {flag} {label}  best: {int(best_row['n']):3d} trades  "
          f"{best_row['wr']:.1f}% WR  PF {best_row['pf']:.2f}  ${best_row['pnl']:+.0f}")
