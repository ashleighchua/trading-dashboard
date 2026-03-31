"""
BTC 200-Day MA Trend Following — Backtest
==========================================
Hold BTC when price > 200-day SMA. Exit to cash when price < 200-day SMA.
Signal checked weekly (Mondays). Trade executes next open.

Period: 2020-01-01 to 2026-03-28
Starting equity: $1,000
Position sizes tested: 20% and 100% of equity
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

START          = "2018-01-01"   # warmup for 200-day MA
BACKTEST_START = "2020-01-01"
END            = "2026-03-28"
INIT_EQ        = 1_000.0

print("Downloading BTC-USD daily bars...")
df_raw = yf.download("BTC-USD", start=START, end=END, auto_adjust=True, progress=False)
if isinstance(df_raw.columns, pd.MultiIndex):
    df_raw.columns = df_raw.columns.get_level_values(0)
df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
df_raw = df_raw.dropna()
print("Done. {} bars.\n".format(len(df_raw)))


def run(size_pct):
    df = df_raw.copy()
    df["ma200"]   = df["Close"].rolling(200).mean()
    df["above_ma"] = df["Close"] > df["ma200"]
    df["weekday"] = df.index.weekday  # 0 = Monday

    trades  = []
    equity  = INIT_EQ
    in_pos  = False
    entry_price = None
    entry_date  = None
    entry_size  = None

    start_idx = df.index.searchsorted(pd.Timestamp(BACKTEST_START))

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]

        # Only check signal on Mondays
        if row["weekday"] != 0:
            continue
        if pd.isna(row["ma200"]):
            continue

        above = row["above_ma"]
        next_open = float(df.iloc[i + 1]["Open"])

        if not in_pos and above:
            # Enter long
            entry_price = next_open
            entry_date  = df.index[i + 1]
            entry_size  = equity * size_pct
            in_pos = True

        elif in_pos and not above:
            # Exit long
            exit_price = next_open
            pnl_pct    = (exit_price - entry_price) / entry_price * 100
            pnl_dollar = pnl_pct / 100 * entry_size
            equity    += pnl_dollar
            trades.append({
                "entry_date": entry_date,
                "exit_date":  df.index[i + 1],
                "entry":      round(entry_price, 2),
                "exit":       round(exit_price, 2),
                "pnl_pct":    round(pnl_pct, 2),
                "pnl_$":      round(pnl_dollar, 2),
                "win":        pnl_pct > 0,
                "equity":     round(equity, 2),
                "year":       entry_date.year,
            })
            in_pos = False

    # Close any open position at last price
    if in_pos:
        exit_price = float(df.iloc[-1]["Close"])
        pnl_pct    = (exit_price - entry_price) / entry_price * 100
        pnl_dollar = pnl_pct / 100 * entry_size
        equity    += pnl_dollar
        trades.append({
            "entry_date": entry_date,
            "exit_date":  df.index[-1],
            "entry":      round(entry_price, 2),
            "exit":       round(exit_price, 2),
            "pnl_pct":    round(pnl_pct, 2),
            "pnl_$":      round(pnl_dollar, 2),
            "win":        pnl_pct > 0,
            "equity":     round(equity, 2),
            "year":       entry_date.year,
        })

    return pd.DataFrame(trades), equity


def stats(df, final_equity):
    if df.empty:
        return
    n    = len(df)
    wins = df["win"].sum()
    wr   = wins / n * 100
    gp   = df.loc[df["win"],  "pnl_pct"].sum()
    gl   = abs(df.loc[~df["win"], "pnl_pct"].sum())
    pf   = gp / gl if gl > 0 else float("inf")
    print("  Trades: {}  WR: {:.1f}%  PF: {:.2f}  Final equity: ${:.0f}".format(
        n, wr, pf, final_equity))
    flag = "✅" if wr >= 55 and pf >= 1.5 and n >= 5 else "⚠️ "
    print("  {} {}".format(flag, "PASSES" if wr >= 55 and pf >= 1.5 else "does not pass acceptance bar"))


for size_label, size_pct in [("20% position", 0.20), ("100% all-in", 1.00)]:
    print("=" * 62)
    print("  {} size".format(size_label))
    print("=" * 62)
    trades, final_eq = run(size_pct)
    stats(trades, final_eq)

    if not trades.empty:
        print("\n  Year-by-year (entry year):")
        yearly = trades.groupby("year").agg(
            n   =("pnl_pct", "count"),
            wr  =("win",     lambda x: x.mean() * 100),
            pnl =("pnl_$",  "sum"),
        )
        for yr, row in yearly.iterrows():
            bar  = "█" * max(0, int(row["pnl"] / 20)) if row["pnl"] > 0 else "▒" * max(0, int(abs(row["pnl"]) / 20))
            flag = "✓" if row["wr"] >= 55 else "✗"
            print("    {}: {:2d} trades  {:4.1f}% WR  ${:+6.1f}  {}  {}".format(
                yr, int(row["n"]), row["wr"], row["pnl"], flag, bar))

        print("\n  Individual trades:")
        for _, t in trades.iterrows():
            days = (t["exit_date"] - t["entry_date"]).days
            w = "W" if t["win"] else "L"
            print("    {} {} entry=${:.0f} exit=${:.0f} {:+.1f}% ${:+.0f} ({} days)".format(
                t["entry_date"].strftime("%Y-%m-%d"), w,
                t["entry"], t["exit"], t["pnl_pct"], t["pnl_$"], days))
    print()

# Buy and hold comparison
bh_start = float(df_raw[df_raw.index >= pd.Timestamp(BACKTEST_START)].iloc[0]["Close"])
bh_end   = float(df_raw.iloc[-1]["Close"])
bh_return = (bh_end - bh_start) / bh_start * 100
print("=" * 62)
print("  Buy & Hold BTC (2020–2026): {:.1f}% return".format(bh_return))
print("  (starting ${:.0f} → ending ${:.0f})".format(bh_start, bh_end))
print("=" * 62)
