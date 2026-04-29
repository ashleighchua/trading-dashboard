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

        # Max drawdown on cumulative P&L series
        cum, peak, max_dd = 0.0, 0.0, 0.0
        for p in pnls:
            cum += p
            peak = max(peak, cum)
            dd = (cum - peak) / (abs(peak) + 1e-9) * 100
            max_dd = min(max_dd, dd)

        # Days in market: sum of hold durations / total trading days
        hold_days = sum(t.get("hold_days", 1) for t in group)
        dim = hold_days / total_days * 100 if total_days > 0 else 0.0

        return {
            "n": len(group),
            "wr": round(wr, 1),
            "pf": round(pf, 2) if pf != float("inf") else 999.0,
            "pnl": round(sum(pnls), 2),
            "avg_win": round(avg_w, 2),
            "avg_loss": round(avg_l, 2),
            "max_dd_pct": round(abs(max_dd), 1),
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
            "all_pnls": [t["pnl_dollar"] for t in trades]}


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
