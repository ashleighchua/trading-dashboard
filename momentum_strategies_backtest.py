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
