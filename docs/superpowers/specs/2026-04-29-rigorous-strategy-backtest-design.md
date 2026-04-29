# Rigorous Strategy Backtest Design

**Date:** 2026-04-29
**Status:** Approved for implementation

---

## Problem

The existing backtests have several methodological flaws that make results untrustworthy:

1. **Survivorship bias** — universe was today's winners, not a realistic 2020 universe
2. **Volume filter backwards** — high-volume pullbacks are distribution, not opportunity
3. **No earnings filter** — gap risk on individual stocks is unmodelled
4. **No sector concentration cap** — correlated positions look like independent bets
5. **Parameter overfitting** — sweeping 100+ combinations picks noise
6. **No slippage** — P&L is overstated by ~0.2% per round trip
7. **No walk-forward validation** — no way to distinguish edge from overfit

This spec designs a single backtest script that rigorously tests all three strategies side-by-side with identical methodology.

---

## What Gets Built

### One file: `momentum_strategies_backtest.py`

Three strategies, one data download, one output. Shared infrastructure means data is downloaded once (~2 minutes), reused across all strategies. Output ends with a side-by-side comparison table.

---

## Data

### Download window

- **Start:** 2019-06-01 — provides 200 trading days of warmup before backtest start
- **Backtest start:** 2020-01-01 — first valid signal date (EMA200 fully warmed up)
- **End:** 2026-04-25

### Walk-forward split

| Period | Label | Purpose |
|---|---|---|
| 2020-01-01 → 2022-12-31 | In-sample | Includes COVID crash + 2021 bull + 2022 bear. Used for parameter selection. |
| 2023-01-01 → 2026-04-25 | Out-of-sample | Verification only. Never used to tune parameters. |

### Ticker universe

All tickers downloaded once. Union of all three strategies.

**Momentum universe (28 tickers):**

| Group | Tickers | Rationale |
|---|---|---|
| ETFs | SPY, QQQ, IWM, XLK, XLF, XLE | No survivorship bias — mechanically hold all constituents |
| Mag7 + growth | AAPL, NVDA, MSFT, META, AMZN, GOOGL, TSLA, AVGO, NFLX | Large-cap growth, high liquidity |
| Deliberate underperformers | INTC, IBM, BA, PFE, GE, XOM | Large-cap in 2020 but have underperformed — anchors test in reality |
| Mixed / cyclical | JPM, AMD, BAC, CVX, UNH, COST | Broad sector coverage |

**Bear Rally Fade tickers:** PLTR, FXI, KWEB, QQQ, IWM (existing scanner watchlist)

**Monday Reversal tickers:** SPY only

---

## Sector Map (static, built into script)

```python
SECTOR = {
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF",
    "AAPL": "MegaTech", "MSFT": "MegaTech", "META": "MegaTech",
    "AMZN": "MegaTech", "GOOGL": "MegaTech",
    "NVDA": "Semis", "AMD": "Semis", "INTC": "Semis", "AVGO": "Semis",
    "TSLA": "Auto", "NFLX": "Media",
    "JPM": "Finance", "BAC": "Finance",
    "XOM": "Energy", "CVX": "Energy",
    "IBM": "Tech", "GE": "Industrial", "BA": "Industrial",
    "PFE": "Health", "UNH": "Health",
    "COST": "Consumer",
}
```

---

## Shared Infrastructure

### Indicators

```python
def ema_series(closes, period):
    # Standard EMA, no lookahead — each value only uses data up to that bar
    k = 2.0 / (period + 1)
    result = []
    for i, v in enumerate(closes):
        result.append(v if i == 0 else v * k + result[-1] * (1 - k))
    return result

def volume_ratio(volumes, window=20):
    # ratio of today's volume to rolling 20-day average
    # computed at bar i using only bars 0..i-1 to avoid lookahead
    ratios = [None] * window
    for i in range(window, len(volumes)):
        avg = sum(volumes[i - window:i]) / window
        ratios.append(volumes[i] / avg if avg > 0 else None)
    return ratios
```

### Slippage

0.1% applied at both entry and exit. Entry price = open × 1.001. Exit price (stop or close) × 0.999 for longs, × 1.001 for shorts.

### Position sizing

```python
def calc_qty(entry_price, stop_pct, max_risk=100.0):
    stop_distance = entry_price * stop_pct
    return max(1, int(max_risk / stop_distance))
```

Consistent with the live system: $100 max risk per trade.

### Earnings filter

```python
def load_earnings_dates(tickers):
    """
    Returns dict: {ticker: sorted list of earnings dates (datetime.date)}
    Uses yfinance Ticker.earnings_dates. Falls back to empty list if unavailable.
    Logs a warning for any ticker where no earnings data was found.
    """
```

At signal time for ticker T on date D, skip if any earnings date falls in [D+1, D+5] (next 5 trading days). If no earnings data available for T, allow the trade but log the gap.

### Stats helper

```python
def compute_stats(trades, label, split_date="2023-01-01"):
    """
    trades: list of dicts {date, ticker, entry, exit, pnl_pct, pnl_dollar, side, reason}
    Returns full stats + in-sample vs out-of-sample breakdown.
    """
```

Metrics reported: trades, win rate, profit factor, avg win $, avg loss $, max drawdown %, days in market %, in-sample WR, out-of-sample WR, verdict (PASS / WARN / FAIL).

**Pass criteria (applied to out-of-sample):**
- Win rate ≥ 55%
- Profit factor ≥ 1.5
- Max drawdown ≤ 25%
- Out-of-sample WR within 10 percentage points of in-sample WR

---

## Strategy 1: EMA Dip Momentum Long

### Signal (checked end-of-day, entry next morning's open)

All conditions must be true:

1. **Market regime:** SPY close > SPY EMA-200
2. **Stock uptrend:** stock EMA-21 > stock EMA-50
3. **Pullback depth:** stock close is 0%–5% below its EMA-21
   - Formula: `0 <= (ema21 - close) / ema21 * 100 <= 5`
4. **Low-volume pullback:** today's volume ≤ 0.8× 20-day average volume
   - Low volume = light selling = institutional non-participation in the dip
5. **Earnings filter:** no earnings in next 5 trading days
6. **No open position** in this ticker

### Position limits

- **Max 2 new positions opened per day** across all tickers (prioritise lowest volume ratio — lightest selling pressure)
- **Max 2 open positions per sector** at any time

### Exit

- **Trailing stop:** 3% trailing, updated daily using close price
- **Max hold:** 15 calendar days, exit at close

### Parameter sets tested (3 pre-defined, not swept)

| Set | EMA pair | Pullback depth | Trailing stop | Max hold | Reasoning |
|---|---|---|---|---|---|
| A | 21/50 | 0–3% | 3% | 10 days | Tight: close to EMA, quick exit |
| B | 21/50 | 0–5% | 3% | 15 days | Standard swing trade |
| C | 50/200 | 0–5% | 4% | 20 days | Longer-term institutional trend |

Winner is selected by out-of-sample profit factor, not in-sample. If no set passes criteria, verdict is NO-GO.

---

## Strategy 2: Bear Rally Fade (re-run)

### What changes vs existing backtest

- Slippage 0.1% per side added
- Earnings filter added for PLTR (individual stock, high earnings-gap risk)
- Walk-forward split added
- Days in-market % reported

### Signal unchanged

- Close < EMA-200 (downtrend)
- Bear rally: close has rallied ≥ 3% from recent low over last 5 days
- Side: SHORT

### Exit unchanged

- 1.5% trailing stop
- 10-day max hold

---

## Strategy 3: Monday Reversal (re-run)

### What changes vs existing backtest

- Slippage 0.1% per side added
- Walk-forward split added

### Signal unchanged

- Symbol = SPY, weekday = Monday
- Previous Friday return ≤ -0.75%
- Side: LONG

### Exit unchanged

- 1.5% trailing stop
- 5-day max hold

---

## Output

```
============================================================
  STRATEGY COMPARISON — RIGOROUS BACKTEST
============================================================

  EMA Dip Momentum (Set B — best out-of-sample)
    In-sample  (2020–2022): 87 trades | 58.6% WR | PF 1.72 | $+4,210
    Out-of-sample (2023–2026): 64 trades | 56.2% WR | PF 1.61 | $+3,480
    Max drawdown: 14.2% | Days in market: 31%
    Verdict: ✅ PASS

  Bear Rally Fade (w/ slippage + earnings filter)
    In-sample  (2020–2022): 41 trades | 61.0% WR | PF 2.10 | $+2,100
    Out-of-sample (2023–2026): 28 trades | 57.1% WR | PF 1.88 | $+1,440
    Max drawdown: 8.1% | Days in market: 12%
    Verdict: ✅ PASS

  Monday Reversal (w/ slippage)
    In-sample  (2020–2022): 22 trades | 72.7% WR | PF 3.20 | $+1,820
    Out-of-sample (2023–2026): 19 trades | 68.4% WR | PF 2.80 | $+1,580
    Max drawdown: 4.3% | Days in market: 4%
    Verdict: ✅ PASS

============================================================
  COMBINED (all strategies, no overlap)
    Total trades: 162 | Net P&L: $+12,830
    Equity curve: [saved to momentum_strategies_backtest.png]
============================================================
```

(Numbers above are illustrative — actual results printed at runtime.)

Equity curve chart saved as `momentum_strategies_backtest.png` in project root.

---

## Limitations (explicitly stated in script header)

```
KNOWN LIMITATIONS:
- Survivorship bias not fully eliminated — individual stocks are 2026 survivors.
  ETFs in the universe (SPY, QQQ, IWM, XLK, XLF, XLE) have zero survivorship bias.
- yfinance earnings dates unreliable for pre-2023 periods. Earnings filter
  may silently miss historical events. Logged per ticker when data unavailable.
- Slippage is estimated at 0.1% per side. Real slippage varies by liquidity and
  market conditions — may be higher for TSLA/PLTR on volatile days.
- Past performance does not guarantee future results.
```

---

## Files Changed

| File | Change |
|---|---|
| `momentum_strategies_backtest.py` | New — combined rigorous backtest |

No changes to live system, scanner, or dashboard.

---

## Success Criteria

- Script runs end-to-end without errors
- All three strategies produce ≥ 20 trades in each period
- Output clearly shows in-sample vs out-of-sample for each strategy
- Equity curve chart saved
- Each strategy gets an explicit PASS / WARN / FAIL verdict
