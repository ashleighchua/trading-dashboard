"""
Rigorous Strategy Backtest
===========================
Three strategies tested with identical methodology:
  1. EMA Dip Momentum Long   — 28-ticker universe, uptrend + EMA pullback
  2. Bear Rally Fade (Short) — existing strategy, re-run with slippage + earnings filter
  3. Monday Reversal (Long)  — existing strategy, re-run with slippage

KNOWN LIMITATIONS:
- Survivorship bias not fully eliminated — individual stocks are 2026 survivors.
  ETFs (SPY, QQQ, IWM, XLK, XLF, XLE) in the universe have zero survivorship bias.
- yfinance earnings dates unreliable for pre-2023 periods. Earnings filter may
  silently miss historical events. Logged per ticker when data unavailable.
- Slippage estimated at 0.1% per side. Real slippage varies by liquidity.
- Past performance does not guarantee future results.

Period:         2019-06-01 → 2026-04-25 (download)
Backtest start: 2020-01-01 (EMA-200 fully warmed up)
In-sample:      2020-01-01 → 2022-12-31
Out-of-sample:  2023-01-01 → 2026-04-25
Position size:  $100 max risk per trade (qty = 100 / (entry * stop_pct))
Slippage:       0.1% per side
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import sys
from datetime import date, timedelta, datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ── Date constants ────────────────────────────────────────────────────────────
DOWNLOAD_START  = "2019-06-01"   # warmup for EMA-200
BACKTEST_START  = "2020-01-01"   # first valid signal date
BACKTEST_END    = "2026-04-25"
SPLIT_DATE      = "2023-01-01"   # in-sample / out-of-sample boundary

# ── Sizing & slippage ─────────────────────────────────────────────────────────
MAX_RISK        = 100.0          # max dollar risk per trade
SLIP            = 0.001          # 0.1% per side

# ── Momentum universe ─────────────────────────────────────────────────────────
MOMENTUM_TICKERS = [
    # ETFs — zero survivorship bias
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE",
    # Mag7 + growth
    "AAPL", "NVDA", "MSFT", "META", "AMZN", "GOOGL", "TSLA", "AVGO", "NFLX",
    # Deliberate underperformers — anchors test in reality
    "INTC", "IBM", "BA", "PFE", "GE", "XOM",
    # Mixed / cyclical
    "JPM", "AMD", "BAC", "CVX", "UNH", "COST",
]

# ── Bear rally fade universe ──────────────────────────────────────────────────
FADE_TICKERS = ["PLTR", "FXI", "KWEB", "QQQ", "IWM"]

# ── Monday reversal universe ──────────────────────────────────────────────────
REVERSAL_TICKERS = ["SPY"]

# ── All tickers (downloaded once) ────────────────────────────────────────────
ALL_TICKERS = sorted(set(MOMENTUM_TICKERS + FADE_TICKERS + REVERSAL_TICKERS))

# ── Sector map ────────────────────────────────────────────────────────────────
SECTOR = {
    "SPY": "ETF",   "QQQ": "ETF",   "IWM": "ETF",
    "XLK": "ETF",   "XLF": "ETF",   "XLE": "ETF",
    "AAPL": "MegaTech", "MSFT": "MegaTech", "META": "MegaTech",
    "AMZN": "MegaTech", "GOOGL": "MegaTech",
    "NVDA": "Semis", "AMD": "Semis", "INTC": "Semis", "AVGO": "Semis",
    "TSLA": "Auto",  "NFLX": "Media",
    "JPM": "Finance", "BAC": "Finance",
    "XOM": "Energy",  "CVX": "Energy",
    "IBM": "Tech",   "GE": "Industrial", "BA": "Industrial",
    "PFE": "Health", "UNH": "Health",
    "COST": "Consumer",
    # Fade tickers not in momentum universe
    "PLTR": "Tech",  "FXI": "ETF",  "KWEB": "ETF",
}

# ── Parameter sets for momentum (pre-defined, not swept) ─────────────────────
MOMENTUM_PARAM_SETS = [
    {"label": "A", "ema_fast": 21, "ema_slow": 50, "pullback_max": 3.0, "trail": 0.03, "max_hold": 10},
    {"label": "B", "ema_fast": 21, "ema_slow": 50, "pullback_max": 5.0, "trail": 0.03, "max_hold": 15},
    {"label": "C", "ema_fast": 50, "ema_slow": 200, "pullback_max": 5.0, "trail": 0.04, "max_hold": 20},
]

# ── Parameter sets for breakout (pre-defined, not swept) ─────────────────────
BREAKOUT_PARAM_SETS = [
    # Set A: standard O'Neil breakout — 6% trail, 15-day hold
    {"label": "A", "lookback": 50, "vol_min": 1.5, "consol_15": 5.0, "consol_5": 3.0, "trail": 0.06, "max_hold": 15},
    # Set B: more room for retest — 7% trail, 20-day hold
    {"label": "B", "lookback": 50, "vol_min": 1.5, "consol_15": 5.0, "consol_5": 3.0, "trail": 0.07, "max_hold": 20},
    # Set C: institutional conviction — tighter VCP, 2× volume, 7% trail, 25-day hold
    {"label": "C", "lookback": 50, "vol_min": 2.0, "consol_15": 4.0, "consol_5": 2.0, "trail": 0.07, "max_hold": 25},
]

# ── Data download ─────────────────────────────────────────────────────────────

def download_data():
    logging.info(f"Downloading {len(ALL_TICKERS)} tickers {DOWNLOAD_START}→{BACKTEST_END}...")
    raw = yf.download(
        ALL_TICKERS,
        start=DOWNLOAD_START,
        end=BACKTEST_END,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    logging.info("Download complete.")
    return raw


def get_df(raw, ticker):
    """Extract a clean OHLCV DataFrame for one ticker. Returns None if data missing."""
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            df = pd.DataFrame({
                "Open":   raw["Open"][ticker],
                "High":   raw["High"][ticker],
                "Low":    raw["Low"][ticker],
                "Close":  raw["Close"][ticker],
                "Volume": raw["Volume"][ticker],
            }).dropna()
        else:
            df = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[df.index >= DOWNLOAD_START]
        if len(df) < 210:
            logging.warning(f"{ticker}: only {len(df)} rows — skipping")
            return None
        return df
    except Exception as e:
        logging.warning(f"{ticker}: data extraction failed — {e}")
        return None


def _test_get_df_shape(raw):
    """Sanity check: SPY should have data."""
    df = get_df(raw, "SPY")
    assert df is not None, "SPY data missing"
    assert len(df) > 1000, f"SPY too few rows: {len(df)}"
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert df.index[0] < pd.Timestamp("2020-01-01"), "Warmup data missing"
    logging.info("✅ get_df sanity check passed")


# ── Indicators ────────────────────────────────────────────────────────────────

def ema_series(values, period):
    """
    Standard EMA. No lookahead — each value uses only data up to that bar.
    Returns list same length as input.
    """
    k = 2.0 / (period + 1)
    result = []
    for i, v in enumerate(values):
        if i == 0:
            result.append(float(v))
        else:
            result.append(float(v) * k + result[-1] * (1 - k))
    return result


def volume_ratio_series(volumes, window=20):
    """
    Ratio of bar[i] volume to mean of bar[i-window : i] (prior window bars only).
    Returns list same length as input; first `window` entries are None (warmup).
    No lookahead — bar[i] average does NOT include bar[i] itself.
    """
    vols = list(volumes)
    result = [None] * window
    for i in range(window, len(vols)):
        avg = sum(vols[i - window:i]) / window
        result.append(vols[i] / avg if avg > 0 else None)
    return result


def add_indicators(df, ema_fast, ema_slow):
    """Add EMA columns and volume ratio to a copy of df."""
    df = df.copy()
    closes = df["Close"].tolist()
    vols   = df["Volume"].tolist()
    df[f"ema{ema_fast}"] = ema_series(closes, ema_fast)
    df[f"ema{ema_slow}"] = ema_series(closes, ema_slow)
    df["vol_ratio"]      = volume_ratio_series(vols, 20)
    return df


def _test_indicators():
    # EMA: first value equals first input
    e = ema_series([10.0, 12.0, 11.0], 3)
    assert e[0] == 10.0, f"EMA[0] should be 10.0, got {e[0]}"
    # EMA period=1 is the value itself
    e1 = ema_series([5.0, 8.0, 3.0], 1)
    assert e1 == [5.0, 8.0, 3.0], f"EMA(1) should equal input, got {e1}"
    # EMA length preserved
    assert len(e) == 3

    # volume_ratio: warmup is None
    vr = volume_ratio_series([100] * 25, 20)
    assert vr[:20] == [None] * 20, "Warmup should be None"
    assert abs(vr[20] - 1.0) < 1e-9, f"Flat volume ratio should be 1.0, got {vr[20]}"
    # Low volume bar
    vols = [100] * 20 + [50]   # bar 20 is half the average
    vr2 = volume_ratio_series(vols, 20)
    assert abs(vr2[20] - 0.5) < 1e-9, f"Half-volume ratio should be 0.5, got {vr2[20]}"

    logging.info("✅ indicator tests passed")

_test_indicators()

# ── Sizing & slippage helpers ─────────────────────────────────────────────────

def calc_qty(entry_price, stop_pct, max_risk=MAX_RISK):
    """Shares = max_risk / (entry_price * stop_pct). Minimum 1."""
    stop_distance = entry_price * stop_pct
    return max(1, int(max_risk / stop_distance))


def long_entry_price(open_price):
    """Apply 0.1% slippage on long entry (buy slightly above open)."""
    return open_price * (1 + SLIP)


def long_exit_price(price):
    """Apply 0.1% slippage on long exit (sell slightly below)."""
    return price * (1 - SLIP)


def short_entry_price(open_price):
    """Apply 0.1% slippage on short entry (sell slightly below open)."""
    return open_price * (1 - SLIP)


def short_exit_price(price):
    """Apply 0.1% slippage on short exit (buy slightly above)."""
    return price * (1 + SLIP)


# ── Earnings filter ───────────────────────────────────────────────────────────

def load_earnings_dates(tickers):
    """
    Returns {ticker: sorted list of datetime.date}.
    Falls back to [] with a warning if yfinance has no data.
    ETFs are skipped (no earnings).
    """
    ETF_SET = {"SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "FXI", "KWEB"}
    result = {}
    for t in tickers:
        if t in ETF_SET:
            result[t] = []
            continue
        try:
            ed = yf.Ticker(t).earnings_dates
            if ed is None or ed.empty:
                logging.warning(f"Earnings filter: no data for {t} — filter disabled for this ticker")
                result[t] = []
            else:
                result[t] = sorted(
                    idx.date() if hasattr(idx, "date") else idx
                    for idx in ed.index
                )
        except Exception as e:
            logging.warning(f"Earnings filter: {t} fetch failed ({e}) — filter disabled")
            result[t] = []
    return result


def near_earnings(signal_date, ticker, earnings_map, window_calendar_days=10):
    """
    True if any earnings date falls in (signal_date, signal_date + window_calendar_days].
    window_calendar_days=10 ≈ 5 trading days (conservative).
    """
    dates = earnings_map.get(ticker, [])
    if not dates:
        return False
    end = signal_date + timedelta(days=window_calendar_days)
    return any(signal_date < d <= end for d in dates)


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(trades, label, trading_days_is, trading_days_oos):
    """
    trades: list of dicts with keys:
        date (datetime.date), ticker (str), entry (float), exit (float),
        pnl_dollar (float), side (str), reason (str), hold_days (int)

    trading_days_is:  count of trading days in in-sample period
    trading_days_oos: count of trading days in out-of-sample period

    Returns dict with full stats and a verdict string.
    """
    if not trades:
        return {
            "label": label, "n": 0, "verdict": "❌ FAIL",
            "is": {}, "oos": {}, "all_pnls": [],
        }

    split = pd.Timestamp(SPLIT_DATE)
    is_trades  = [t for t in trades if pd.Timestamp(t["date"]) <  split]
    oos_trades = [t for t in trades if pd.Timestamp(t["date"]) >= split]

    def _stats_block(group, total_days):
        if not group:
            return {"n": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0,
                    "avg_win": 0.0, "avg_loss": 0.0, "max_dd_pct": 0.0, "days_in_mkt_pct": 0.0}
        pnls   = [t["pnl_dollar"] for t in group]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr     = len(wins) / len(pnls) * 100
        gp     = sum(wins) if wins else 0.0
        gl     = abs(sum(losses)) if losses else 0.0
        pf     = gp / gl if gl > 0 else float("inf")
        avg_w  = gp / len(wins) if wins else 0.0
        avg_l  = gl / len(losses) if losses else 0.0

        # Max drawdown: peak-to-trough dollar loss, expressed as % of $10k notional
        # ($10k = ~100x $100-per-trade risk; avoids division-by-zero when equity starts at 0)
        NOTIONAL = 10_000.0
        cum, peak, max_dd_dollar = 0.0, 0.0, 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            max_dd_dollar = min(max_dd_dollar, cum - peak)
        max_dd_pct = abs(max_dd_dollar) / NOTIONAL * 100

        # Days in market: fraction of trading days with ≥1 open position
        # Cap each trade's hold at total_days to avoid >100% from concurrent positions
        hold_days = sum(min(t.get("hold_days", 1), total_days) for t in group)
        dim = min(hold_days / total_days * 100, 100.0) if total_days > 0 else 0.0

        return {
            "n": len(group),
            "wr": round(wr, 1),
            "pf": round(pf, 2) if pf != float("inf") else 999.0,
            "pnl": round(sum(pnls), 2),
            "avg_win": round(avg_w, 2),
            "avg_loss": round(avg_l, 2),
            "max_dd_pct": round(max_dd_pct, 1),
            "days_in_mkt_pct": round(dim, 1),
        }

    is_s  = _stats_block(is_trades,  trading_days_is)
    oos_s = _stats_block(oos_trades, trading_days_oos)

    # Verdict: based on out-of-sample performance
    oos_wr_ok  = oos_s["n"] >= 10 and oos_s["wr"] >= 55.0
    oos_pf_ok  = oos_s["pf"] >= 1.5
    oos_dd_ok  = oos_s["max_dd_pct"] <= 25.0
    drift_ok   = is_s["n"] > 0 and oos_s["n"] > 0 and abs(oos_s["wr"] - is_s["wr"]) <= 10.0

    if oos_wr_ok and oos_pf_ok and oos_dd_ok and drift_ok:
        verdict = "✅ PASS"
    elif oos_wr_ok and oos_pf_ok and (not oos_dd_ok or not drift_ok):
        verdict = "⚠️  WARN"
    else:
        verdict = "❌ FAIL"

    return {"label": label, "is": is_s, "oos": oos_s, "verdict": verdict,
            "all_pnls": sorted([(t["date"], t["pnl_dollar"]) for t in trades], key=lambda x: x[0])}


def _test_helpers():
    # calc_qty: $100 risk, $50 stock, 2% stop → stop_distance=$1 → qty=100
    assert calc_qty(50.0, 0.02) == 100, f"Expected 100, got {calc_qty(50.0, 0.02)}"
    # calc_qty: minimum 1 (stop_distance=$200 > MAX_RISK=$100 → floor to 0 → clamp to 1)
    assert calc_qty(10000.0, 0.02) == 1, f"Expected 1, got {calc_qty(10000.0, 0.02)}"
    # Slippage on entry/exit
    assert abs(long_entry_price(100.0) - 100.1) < 1e-9
    assert abs(long_exit_price(100.0) - 99.9) < 1e-9
    assert abs(short_entry_price(100.0) - 99.9) < 1e-9
    assert abs(short_exit_price(100.0) - 100.1) < 1e-9
    # near_earnings: no dates → False
    assert not near_earnings(date(2024, 1, 10), "AAPL", {"AAPL": []})
    # near_earnings: earnings 5 days later → True
    assert near_earnings(date(2024, 1, 10), "AAPL", {"AAPL": [date(2024, 1, 15)]})
    # near_earnings: earnings 15 days later → False (outside 10-day window)
    assert not near_earnings(date(2024, 1, 10), "AAPL", {"AAPL": [date(2024, 1, 25)]})
    # compute_stats: empty trades → FAIL verdict
    result = compute_stats([], "test", 100, 100)
    assert result["verdict"] == "❌ FAIL"
    assert result["n"] == 0
    logging.info("✅ helper tests passed")

_test_helpers()

# ── Breakout helpers ──────────────────────────────────────────────────────────

def consolidation_pct(df, idx, window):
    """
    High-to-low range over the prior `window` bars as % of close at idx.
    Uses only bars BEFORE idx (no lookahead). Returns None if insufficient data.
    """
    if idx < window:
        return None
    slice_ = df.iloc[idx - window:idx]
    high   = slice_["High"].max()
    low    = slice_["Low"].min()
    close  = df.iloc[idx]["Close"]
    if close <= 0:
        return None
    return (high - low) / close * 100


def stage2_confirmed(df, idx):
    """
    Weinstein Stage 2: close > 150-day MA AND 150-day MA is rising.
    Rising = MA computed at idx > MA computed at idx-5.
    Returns False if insufficient data (need idx >= 155).
    """
    if idx < 155:
        return False
    ma_now  = df["Close"].iloc[idx - 150:idx].mean()
    ma_prev = df["Close"].iloc[idx - 155:idx - 5].mean()
    close   = df.iloc[idx]["Close"]
    return bool(close > ma_now and ma_now > ma_prev)


def relative_strength_vs_spy(df, spy_df, idx, window=63):
    """
    O'Neil RS: stock's window-bar return > SPY's window-bar return.
    window=63 ≈ 3 months of trading days.
    Aligns by bar index (valid — both DFs use the same yfinance trading calendar).
    Returns False if insufficient data.
    """
    if idx < window or len(spy_df) <= idx:
        return False
    stock_base = df.iloc[idx - window]["Close"]
    if stock_base <= 0:
        return False
    stock_ret = (df.iloc[idx]["Close"] - stock_base) / stock_base
    spy_idx   = min(idx, len(spy_df) - 1)
    if spy_idx < window:
        return False
    spy_base = spy_df.iloc[spy_idx - window]["Close"]
    if spy_base <= 0:
        return False
    spy_ret = (spy_df.iloc[spy_idx]["Close"] - spy_base) / spy_base
    return bool(stock_ret > spy_ret)


def _test_breakout_helpers():
    """Inline tests for consolidation_pct, stage2_confirmed, relative_strength_vs_spy."""

    # ── consolidation_pct ────────────────────────────────────────────────────
    n = 20
    df_flat = pd.DataFrame({
        "High":   [110.0] * n,
        "Low":    [90.0]  * n,
        "Close":  [100.0] * n,
        "Open":   [100.0] * n,
        "Volume": [1e6]   * n,
    })
    result = consolidation_pct(df_flat, 15, 10)
    assert result is not None and abs(result - 20.0) < 1e-6, f"Expected 20.0, got {result}"
    assert consolidation_pct(df_flat, 5, 10) is None  # insufficient data

    # ── stage2_confirmed ─────────────────────────────────────────────────────
    n2 = 200
    prices_rising = [100.0 + i * 0.5 for i in range(n2)]
    df_rising = pd.DataFrame({
        "High": prices_rising, "Low": prices_rising, "Close": prices_rising,
        "Open": prices_rising, "Volume": [1e6] * n2,
    })
    assert stage2_confirmed(df_rising, 160) is True

    prices_flat = [100.0] * n2
    df_flat2 = pd.DataFrame({
        "High": prices_flat, "Low": prices_flat, "Close": prices_flat,
        "Open": prices_flat, "Volume": [1e6] * n2,
    })
    assert stage2_confirmed(df_flat2, 160) is False
    assert stage2_confirmed(df_rising, 100) is False  # insufficient data

    # ── relative_strength_vs_spy ─────────────────────────────────────────────
    n3 = 100
    stock_prices = [100.0 + (20.0 / 63) * i for i in range(n3)]
    spy_prices   = [100.0 + (10.0 / 63) * i for i in range(n3)]
    df_stock = pd.DataFrame({"Close": stock_prices, "High": stock_prices,
                              "Low": stock_prices, "Open": stock_prices, "Volume": [1e6]*n3})
    df_spy   = pd.DataFrame({"Close": spy_prices,  "High": spy_prices,
                              "Low": spy_prices,  "Open": spy_prices,  "Volume": [1e6]*n3})
    assert relative_strength_vs_spy(df_stock, df_spy, 90) is True

    df_stock_flat = pd.DataFrame({"Close": [100.0]*n3, "High": [100.0]*n3,
                                   "Low": [100.0]*n3, "Open": [100.0]*n3, "Volume": [1e6]*n3})
    assert relative_strength_vs_spy(df_stock_flat, df_spy, 90) is False
    assert relative_strength_vs_spy(df_stock, df_spy, 10) is False  # insufficient data

    logging.info("✅ breakout helper tests passed")


_test_breakout_helpers()

# ── Strategy 1: EMA Dip Momentum ─────────────────────────────────────────────

def simulate_long_trade(df, entry_idx, entry_price, trail_pct, max_hold):
    """
    Simulate a long trade starting at entry_idx with entry_price already slippage-adjusted.
    Trail stop uses daily close. Exit at stop (apply exit slippage) or max_hold close.
    Returns (exit_price_after_slippage, exit_idx, reason, hold_days).
    """
    stop = entry_price * (1 - trail_pct)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        low   = df.iloc[j]["Low"]
        close = df.iloc[j]["Close"]
        # Raise stop if close is higher
        new_stop = close * (1 - trail_pct)
        if new_stop > stop:
            stop = new_stop
        if low <= stop:
            return long_exit_price(stop), j, "stop", j - entry_idx
    # Max hold: exit at close of last bar
    exit_idx = min(entry_idx + max_hold, len(df) - 1)
    return long_exit_price(df.iloc[exit_idx]["Close"]), exit_idx, "maxhold", exit_idx - entry_idx


def run_momentum(raw, earnings_map, params):
    """
    Run EMA Dip Momentum for one parameter set.
    Returns list of trade dicts.

    Signal (end-of-day, entry next morning):
      1. SPY close > SPY EMA-200 (market regime)
      2. Stock EMA-fast > EMA-slow (individual uptrend)
      3. 0% <= (ema_fast - close) / ema_fast * 100 <= pullback_max (pullback depth)
      4. vol_ratio <= 0.8 (low-volume pullback = light selling)
      5. No earnings in next 10 calendar days
      6. No open position in this ticker

    Position limits:
      - Max 2 new positions opened per day (prioritised by lowest vol_ratio)
      - Max 2 open positions per sector at any time
    """
    ema_fast     = params["ema_fast"]
    ema_slow     = params["ema_slow"]
    pullback_max = params["pullback_max"]
    trail        = params["trail"]
    max_hold     = params["max_hold"]

    # Prepare all DataFrames with indicators
    dfs = {}
    for t in MOMENTUM_TICKERS:
        df = get_df(raw, t)
        if df is not None:
            dfs[t] = add_indicators(df, ema_fast, ema_slow)

    spy_df = dfs.get("SPY")
    if spy_df is None:
        raise RuntimeError("SPY data missing — cannot determine market regime")

    # Build SPY EMA-200 series keyed by date
    spy_ema200 = ema_series(spy_df["Close"].tolist(), 200)
    spy_ema200_by_date = {
        spy_df.index[i].date(): spy_ema200[i]
        for i in range(len(spy_df))
    }

    backtest_start = pd.Timestamp(BACKTEST_START)
    trades = []
    open_positions = {}   # {ticker: exit_idx in that ticker's df}
    open_sectors   = {}   # {sector: count of currently open positions}

    # Iterate over all trading dates from SPY starting at backtest start
    all_dates = spy_df[spy_df.index >= backtest_start].index

    for date_ts in all_dates:
        signal_date = date_ts.date()

        # ── First: close out positions whose exit has passed ─────────────────
        for t in list(open_positions.keys()):
            t_df = dfs.get(t)
            if t_df is None:
                s = SECTOR.get(t, "Other")
                open_sectors[s] = max(0, open_sectors.get(s, 0) - 1)
                del open_positions[t]
                continue
            if date_ts not in t_df.index:
                continue
            cur_idx = t_df.index.get_loc(date_ts)
            if open_positions[t] <= cur_idx:
                s = SECTOR.get(t, "Other")
                open_sectors[s] = max(0, open_sectors.get(s, 0) - 1)
                del open_positions[t]

        # ── Collect all valid signals for today ───────────────────────────────
        new_today   = 0
        day_signals = []

        for ticker in MOMENTUM_TICKERS:
            if ticker not in dfs:
                continue
            df = dfs[ticker]
            if date_ts not in df.index:
                continue
            idx = df.index.get_loc(date_ts)

            # Warmup: need at least ema_slow bars of data
            if idx < ema_slow:
                continue

            # Skip if already in a position in this ticker
            if ticker in open_positions and open_positions[ticker] > idx:
                continue

            row   = df.iloc[idx]
            ema_f = row[f"ema{ema_fast}"]
            ema_s = row[f"ema{ema_slow}"]
            close = row["Close"]
            vr    = row["vol_ratio"]

            # 1. Market regime: SPY close > SPY EMA-200
            spy_ema = spy_ema200_by_date.get(signal_date)
            if spy_ema is None or spy_df.loc[date_ts, "Close"] <= spy_ema:
                continue

            # 2. Individual uptrend: ema_fast > ema_slow
            if ema_f <= ema_s:
                continue

            # 3. Pullback depth: 0% to pullback_max% below ema_fast
            if ema_f <= 0:
                continue
            pullback_pct = (ema_f - close) / ema_f * 100
            if not (0 <= pullback_pct <= pullback_max):
                continue

            # 4. Low-volume pullback: vol_ratio <= 0.8
            if vr is None or vr > 0.8:
                continue

            # 5. Earnings filter
            if near_earnings(signal_date, ticker, earnings_map):
                continue

            day_signals.append({
                "ticker":    ticker,
                "vol_ratio": vr,
                "idx":       idx,
            })

        # ── Sort by lowest vol_ratio (lightest selling pressure) ──────────────
        day_signals.sort(key=lambda s: s["vol_ratio"])

        # ── Enter positions (up to 2 per day, up to 2 per sector) ────────────
        for sig in day_signals:
            if new_today >= 2:
                break
            ticker = sig["ticker"]
            idx    = sig["idx"]
            df     = dfs[ticker]
            sector = SECTOR.get(ticker, "Other")

            # Sector cap: max 2 open per sector
            if open_sectors.get(sector, 0) >= 2:
                continue

            # Need a next bar for entry
            if idx + 1 >= len(df):
                continue

            entry_open  = df.iloc[idx + 1]["Open"]
            entry_price = long_entry_price(entry_open)
            qty         = calc_qty(entry_price, trail)

            exit_price, exit_idx, reason, hold_days = simulate_long_trade(
                df, idx + 1, entry_price, trail, max_hold
            )
            pnl_dollar = (exit_price - entry_price) * qty

            trades.append({
                "date":       signal_date,
                "ticker":     ticker,
                "entry":      round(entry_price, 4),
                "exit":       round(exit_price, 4),
                "qty":        qty,
                "pnl_dollar": round(pnl_dollar, 2),
                "side":       "long",
                "reason":     reason,
                "hold_days":  hold_days,
            })

            open_positions[ticker] = exit_idx
            open_sectors[sector]   = open_sectors.get(sector, 0) + 1
            new_today += 1

    return trades

# ── Strategy 2: Bear Rally Fade ───────────────────────────────────────────────

def simulate_short_trade(df, entry_idx, entry_price, trail_pct, max_hold):
    """
    Simulate a short trade. Trail stop uses daily close.
    For shorts: stop starts ABOVE entry and trails DOWN as price falls.
    entry_price is already slippage-adjusted (below open).
    Returns (exit_price_after_slippage, exit_idx, reason, hold_days).
    """
    stop = entry_price * (1 + trail_pct)
    for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        high  = df.iloc[j]["High"]
        close = df.iloc[j]["Close"]
        # Lower stop as price falls (trail down)
        new_stop = close * (1 + trail_pct)
        if new_stop < stop:
            stop = new_stop
        if high >= stop:
            return short_exit_price(stop), j, "stop", j - entry_idx
    exit_idx = min(entry_idx + max_hold, len(df) - 1)
    return short_exit_price(df.iloc[exit_idx]["Close"]), exit_idx, "maxhold", exit_idx - entry_idx


def run_bear_rally_fade(raw, earnings_map):
    """
    Bear Rally Fade (short) — PLTR, FXI, KWEB, QQQ, IWM.
    Signal: close < EMA-200 AND rallied >= 3% from 5-day low.
    Exit: 1.5% trailing stop OR 10-day max hold.
    Earnings filter applied to PLTR only (ETFs have no earnings).
    Slippage 0.1% per side.
    One position at a time across all tickers.
    """
    TRAIL            = 0.015
    MAX_HOLD         = 10
    RALLY_THRESHOLD  = 3.0   # percent rally from 5-day low

    dfs = {}
    for t in FADE_TICKERS:
        df = get_df(raw, t)
        if df is not None:
            df = df.copy()
            df["ema200"] = ema_series(df["Close"].tolist(), 200)
            dfs[t] = df

    backtest_start = pd.Timestamp(BACKTEST_START)
    trades = []
    in_position  = None    # ticker currently held short
    exit_idx_map = {}      # {ticker: exit_idx in that ticker's df}

    spy_df    = get_df(raw, "SPY")
    all_dates = spy_df[spy_df.index >= backtest_start].index

    for date_ts in all_dates:
        signal_date = date_ts.date()

        # Clear position if it has closed
        if in_position and in_position in exit_idx_map:
            t_df = dfs.get(in_position)
            if t_df is not None and date_ts in t_df.index:
                cur_idx = t_df.index.get_loc(date_ts)
                if exit_idx_map[in_position] <= cur_idx:
                    in_position = None

        if in_position:
            continue

        for ticker in FADE_TICKERS:
            if ticker not in dfs:
                continue
            df = dfs[ticker]
            if date_ts not in df.index:
                continue
            idx = df.index.get_loc(date_ts)
            if idx < 205:    # EMA-200 warmup
                continue

            row    = df.iloc[idx]
            close  = row["Close"]
            ema200 = row["ema200"]

            # 1. Downtrend: close < EMA-200
            if close >= ema200:
                continue

            # 2. Bear rally: close is >= 3% above 5-day low
            low5      = df.iloc[idx - 4:idx + 1]["Low"].min()
            rally_pct = (close - low5) / low5 * 100
            if rally_pct < RALLY_THRESHOLD:
                continue

            # 3. Earnings filter (PLTR only — ETFs have no earnings)
            if ticker == "PLTR" and near_earnings(signal_date, ticker, earnings_map):
                continue

            # Need next bar for entry
            if idx + 1 >= len(df):
                continue

            entry_open  = df.iloc[idx + 1]["Open"]
            entry_price = short_entry_price(entry_open)
            qty         = calc_qty(entry_price, TRAIL)

            exit_price, ex_idx, reason, hold_days = simulate_short_trade(
                df, idx + 1, entry_price, TRAIL, MAX_HOLD
            )
            pnl_dollar = (entry_price - exit_price) * qty   # short P&L

            trades.append({
                "date":       signal_date,
                "ticker":     ticker,
                "entry":      round(entry_price, 4),
                "exit":       round(exit_price, 4),
                "qty":        qty,
                "pnl_dollar": round(pnl_dollar, 2),
                "side":       "short",
                "reason":     reason,
                "hold_days":  hold_days,
            })

            in_position = ticker
            exit_idx_map[ticker] = ex_idx
            break    # one position at a time

    return trades


# ── Strategy 3: Monday Reversal ───────────────────────────────────────────────

def run_monday_reversal(raw):
    """
    Monday Reversal (long SPY).
    Signal: weekday == Monday AND previous Friday close return <= -0.75%.
    Exit: 1.5% trailing stop OR 5-day max hold.
    Slippage 0.1% per side. One position at a time.
    """
    TRAIL      = 0.015
    MAX_HOLD   = 5
    FRI_THRESH = -0.75    # percent

    df = get_df(raw, "SPY")
    if df is None:
        raise RuntimeError("SPY data missing for Monday Reversal")

    backtest_start     = pd.Timestamp(BACKTEST_START)
    trades             = []
    in_position_until  = -1    # row index until which we're occupied

    for i in range(1, len(df)):
        date_ts = df.index[i]
        if date_ts < backtest_start:
            continue
        if i <= in_position_until:
            continue

        # Must be Monday (weekday 0)
        if date_ts.weekday() != 0:
            continue

        # Find the most recent Friday
        fri_idx = i - 1
        while fri_idx >= 0 and df.index[fri_idx].weekday() != 4:
            fri_idx -= 1
        if fri_idx < 1:
            continue

        fri_close = df.iloc[fri_idx]["Close"]
        thu_close = df.iloc[fri_idx - 1]["Close"]
        if thu_close <= 0:
            continue
        fri_return = (fri_close - thu_close) / thu_close * 100

        if fri_return > FRI_THRESH:
            continue

        # Signal fires — entry at Monday's open with slippage
        entry_open  = df.iloc[i]["Open"]
        entry_price = long_entry_price(entry_open)
        qty         = calc_qty(entry_price, TRAIL)

        exit_price, exit_idx, reason, hold_days = simulate_long_trade(
            df, i, entry_price, TRAIL, MAX_HOLD
        )
        pnl_dollar = (exit_price - entry_price) * qty

        trades.append({
            "date":       date_ts.date(),
            "ticker":     "SPY",
            "entry":      round(entry_price, 4),
            "exit":       round(exit_price, 4),
            "qty":        qty,
            "pnl_dollar": round(pnl_dollar, 2),
            "side":       "long",
            "reason":     reason,
            "hold_days":  hold_days,
        })

        in_position_until = exit_idx

    return trades

# ── Strategy 4: 50-Day Breakout Momentum ─────────────────────────────────────

def run_breakout(raw, earnings_map, params):
    """
    Run 50-Day Breakout Momentum for one parameter set.
    Returns list of trade dicts.

    Signal (end-of-day, entry next morning):
      1. SPY close > SPY EMA-200 (market uptrend)
      2. Stock close > highest close of prior `lookback` bars (50-day high)
      3. Today's volume >= vol_min * 20-day avg volume (institutional surge)
      4. 15-day high-to-low range <= consol_15% of close (VCP outer band)
      5. 5-day high-to-low range  <= consol_5%  of close (VCP inner band)
      6. Stage 2: close > 150-day MA AND MA rising (Weinstein)
      7. Relative strength: stock 3-month return > SPY 3-month return (O'Neil)
      8. No earnings within 10 calendar days
      9. No open position in this ticker

    Position limits:
      - Max 2 new positions per day (prioritised by highest vol_ratio — most conviction)
      - Max 2 open positions per sector at any time
    """
    lookback  = params["lookback"]
    vol_min   = params["vol_min"]
    consol_15 = params["consol_15"]
    consol_5  = params["consol_5"]
    trail     = params["trail"]
    max_hold  = params["max_hold"]

    # Prepare DataFrames with volume ratio column
    dfs = {}
    for t in MOMENTUM_TICKERS:
        df = get_df(raw, t)
        if df is not None:
            vr = volume_ratio_series(df["Volume"].tolist(), window=20)
            df = df.copy()
            df["vol_ratio"] = vr
            dfs[t] = df

    spy_df = dfs.get("SPY")
    if spy_df is None:
        raise RuntimeError("SPY data missing — cannot determine market regime")

    # SPY EMA-200 by date for market regime check
    spy_ema200 = ema_series(spy_df["Close"].tolist(), 200)
    spy_ema200_by_date = {
        spy_df.index[i].date(): spy_ema200[i]
        for i in range(len(spy_df))
    }

    backtest_start = pd.Timestamp(BACKTEST_START)
    trades         = []
    open_positions = {}   # {ticker: exit_idx in that ticker's df}
    open_sectors   = {}   # {sector: count of currently open positions}

    all_dates = spy_df[spy_df.index >= backtest_start].index

    for date_ts in all_dates:
        signal_date = date_ts.date()

        # ── Close out expired positions ───────────────────────────────────────
        for t in list(open_positions.keys()):
            t_df = dfs.get(t)
            if t_df is None:
                s = SECTOR.get(t, "Other")
                open_sectors[s] = max(0, open_sectors.get(s, 0) - 1)
                del open_positions[t]
                continue
            if date_ts not in t_df.index:
                continue
            cur_idx = t_df.index.get_loc(date_ts)
            if open_positions[t] <= cur_idx:
                s = SECTOR.get(t, "Other")
                open_sectors[s] = max(0, open_sectors.get(s, 0) - 1)
                del open_positions[t]

        # ── Collect all valid signals for today ───────────────────────────────
        day_signals = []

        for ticker in MOMENTUM_TICKERS:
            if ticker not in dfs:
                continue
            df = dfs[ticker]
            if date_ts not in df.index:
                continue
            idx = df.index.get_loc(date_ts)

            # Need enough warmup for 150-day MA + 63-day RS + 50-day lookback
            if idx < 155:
                continue

            # Skip if already in a position in this ticker
            if ticker in open_positions and open_positions[ticker] > idx:
                continue

            close = df.iloc[idx]["Close"]
            vr    = df.iloc[idx]["vol_ratio"]

            # 1. Market regime: SPY close > SPY EMA-200
            spy_ema = spy_ema200_by_date.get(signal_date)
            if spy_ema is None or spy_df.loc[date_ts, "Close"] <= spy_ema:
                continue

            # 2. 50-day closing high: close > max of prior lookback closes
            prior_high = df["Close"].iloc[idx - lookback:idx].max()
            if close <= prior_high:
                continue

            # 3. Volume surge: vol_ratio >= vol_min
            if vr is None or vr < vol_min:
                continue

            # 4. VCP outer band: 15-day consolidation
            c15 = consolidation_pct(df, idx, 15)
            if c15 is None or c15 > consol_15:
                continue

            # 5. VCP inner band: 5-day consolidation
            c5 = consolidation_pct(df, idx, 5)
            if c5 is None or c5 > consol_5:
                continue

            # 6. Stage 2 confirmation (Weinstein)
            if not stage2_confirmed(df, idx):
                continue

            # 7. Relative strength vs SPY (O'Neil)
            if not relative_strength_vs_spy(df, spy_df, idx):
                continue

            # 8. Earnings filter
            if near_earnings(signal_date, ticker, earnings_map):
                continue

            day_signals.append({
                "ticker":    ticker,
                "vol_ratio": vr,
                "idx":       idx,
            })

        # ── Sort by highest vol_ratio (strongest conviction breakout) ─────────
        day_signals.sort(key=lambda s: s["vol_ratio"], reverse=True)

        # ── Enter positions (up to 2 per day, up to 2 per sector) ────────────
        new_today = 0
        for sig in day_signals:
            if new_today >= 2:
                break
            ticker = sig["ticker"]
            idx    = sig["idx"]
            df     = dfs[ticker]
            sector = SECTOR.get(ticker, "Other")

            # Sector cap
            if open_sectors.get(sector, 0) >= 2:
                continue

            # Need a next bar for entry
            if idx + 1 >= len(df):
                continue

            entry_open  = df.iloc[idx + 1]["Open"]
            entry_price = long_entry_price(entry_open)
            qty         = calc_qty(entry_price, trail)

            exit_price, exit_idx, reason, hold_days = simulate_long_trade(
                df, idx + 1, entry_price, trail, max_hold
            )
            pnl_dollar = (exit_price - entry_price) * qty

            trades.append({
                "date":       signal_date,
                "ticker":     ticker,
                "entry":      round(entry_price, 4),
                "exit":       round(exit_price, 4),
                "qty":        qty,
                "pnl_dollar": round(pnl_dollar, 2),
                "side":       "long",
                "reason":     reason,
                "hold_days":  hold_days,
            })

            open_positions[ticker] = exit_idx
            open_sectors[sector]   = open_sectors.get(sector, 0) + 1
            new_today += 1

    return trades

# ---------------------------------------------------------------------------
# Task 8 — Output formatter, equity curve, main
# ---------------------------------------------------------------------------

def print_strategy_result(result):
    """Print a formatted summary for one compute_stats result dict."""
    label   = result["label"]
    is_     = result["is"]
    oos     = result["oos"]
    verdict = result["verdict"]

    print(f"  {label}")
    print(
        f"    In-sample  (2020–2022): {is_['n']} trades | "
        f"{is_['wr']:.1f}% WR | PF {is_['pf']:.2f} | ${is_['pnl']:+,.0f}"
    )
    print(
        f"    Out-of-sample (2023–2026): {oos['n']} trades | "
        f"{oos['wr']:.1f}% WR | PF {oos['pf']:.2f} | ${oos['pnl']:+,.0f}"
    )
    print(
        f"    Max drawdown: IS {is_['max_dd_pct']:.1f}% / OOS {oos['max_dd_pct']:.1f}%"
        f" | Days in market: IS {is_['days_in_mkt_pct']:.0f}% / OOS {oos['days_in_mkt_pct']:.0f}%"
    )
    print(f"    Verdict: {verdict}")


def save_equity_curve(results, output_path="momentum_strategies_backtest.png"):
    """
    Plot individual and combined equity curves for all strategy results.
    Each result dict must contain an 'all_pnls' key: list of (date, pnl_dollar) tuples.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    # --- Top panel: combined equity curve ---
    ax_combined = axes[0]
    all_entries = []
    for r in results:
        all_entries.extend(r["all_pnls"])
    all_entries.sort(key=lambda x: x[0])

    if all_entries:
        dates_combined  = [e[0] for e in all_entries]
        cumsum_combined = list(pd.Series([e[1] for e in all_entries]).cumsum())
        ax_combined.plot(dates_combined, cumsum_combined, color="black", linewidth=2, label="Combined")
        ax_combined.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_combined.fill_between(dates_combined, cumsum_combined, 0,
                                 where=[v >= 0 for v in cumsum_combined],
                                 alpha=0.15, color="green")
        ax_combined.fill_between(dates_combined, cumsum_combined, 0,
                                 where=[v < 0 for v in cumsum_combined],
                                 alpha=0.15, color="red")

    # Mark IS/OOS split
    split_ts = pd.Timestamp(SPLIT_DATE)
    ax_combined.axvline(split_ts, color="steelblue", linewidth=1.2, linestyle=":", label=f"IS/OOS split ({SPLIT_DATE})")
    ax_combined.set_title("Combined Equity Curve (all strategies)", fontsize=12, fontweight="bold")
    ax_combined.set_ylabel("Cumulative P&L ($)")
    ax_combined.legend(fontsize=9)
    ax_combined.grid(True, alpha=0.3)

    # --- Bottom panel: per-strategy equity curves ---
    ax_per = axes[1]
    colors = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]
    for idx, r in enumerate(results):
        pnls = r["all_pnls"]
        if not pnls:
            continue
        dates  = [e[0] for e in pnls]
        cumsum = list(pd.Series([e[1] for e in pnls]).cumsum())
        short_label = r["label"].split("(")[0].strip()
        ax_per.plot(dates, cumsum, color=colors[idx % len(colors)],
                    linewidth=1.5, label=short_label)

    ax_per.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_per.axvline(split_ts, color="steelblue", linewidth=1.2, linestyle=":")
    ax_per.set_title("Per-Strategy Equity Curves", fontsize=12, fontweight="bold")
    ax_per.set_ylabel("Cumulative P&L ($)")
    ax_per.legend(fontsize=9)
    ax_per.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logging.info("Equity curve saved to %s", output_path)


def main():
    logging.info("Downloading data (%s → %s)...", DOWNLOAD_START, BACKTEST_END)
    raw = download_data()

    logging.info("Loading earnings dates...")
    earnings_map = load_earnings_dates(ALL_TICKERS)

    # Count trading days in each period (use SPY as calendar)
    spy_df = get_df(raw, "SPY")
    spy_df = spy_df[spy_df.index >= pd.Timestamp(BACKTEST_START)]
    split_ts = pd.Timestamp(SPLIT_DATE)
    trading_days_is  = int((spy_df.index < split_ts).sum())
    trading_days_oos = int((spy_df.index >= split_ts).sum())

    # Run EMA Dip Momentum — all param sets, pick best OOS profit factor
    logging.info("Running EMA Dip Momentum (%d param sets)...", len(MOMENTUM_PARAM_SETS))
    momentum_results = []   # list of (result_dict, trades_list)
    for params in MOMENTUM_PARAM_SETS:
        trades = run_momentum(raw, earnings_map, params)
        label = (
            f"EMA Dip Momentum Set {params['label']} "
            f"(ema{params['ema_fast']}/{params['ema_slow']}, "
            f"pullback≤{params['pullback_max']}%, "
            f"trail {int(params['trail'] * 100)}%, "
            f"hold {params['max_hold']}d)"
        )
        result = compute_stats(trades, label, trading_days_is, trading_days_oos)
        momentum_results.append((result, trades))
        logging.info(
            "  Set %s: OOS PF=%.2f verdict=%s",
            params["label"], result["oos"]["pf"], result["verdict"]
        )

    # Select best param set by OOS profit factor
    best_momentum, best_momentum_trades = max(momentum_results, key=lambda x: x[0]["oos"]["pf"])

    # Run Bear Rally Fade
    logging.info("Running Bear Rally Fade...")
    fade_trades = run_bear_rally_fade(raw, earnings_map)
    fade_result = compute_stats(
        fade_trades,
        "Bear Rally Fade (Short, w/ slippage + earnings filter)",
        trading_days_is, trading_days_oos,
    )

    # Run Monday Reversal
    logging.info("Running Monday Reversal...")
    rev_trades = run_monday_reversal(raw)
    rev_result = compute_stats(
        rev_trades,
        "Monday Reversal (Long SPY, w/ slippage)",
        trading_days_is, trading_days_oos,
    )

    # Print results
    separator = "=" * 60
    print(f"\n{separator}")
    print("  STRATEGY COMPARISON — RIGOROUS BACKTEST")
    print(separator)
    print()
    print_strategy_result(best_momentum)
    print()
    print_strategy_result(fade_result)
    print()
    print_strategy_result(rev_result)
    print()

    # Combined stats (no re-running — reuse already-computed trades)
    all_trades = best_momentum_trades + fade_trades + rev_trades
    total_pnl  = sum(t["pnl_dollar"] for t in all_trades)
    print(separator)
    print("  COMBINED (all strategies, no overlap in SPY positions)")
    print(f"    Total trades: {len(all_trades)} | Net P&L: ${total_pnl:+,.0f}")
    print("    Equity curve: [saved to momentum_strategies_backtest.png]")
    print(separator)
    print()

    save_equity_curve([best_momentum, fade_result, rev_result])


if __name__ == "__main__":
    main()
