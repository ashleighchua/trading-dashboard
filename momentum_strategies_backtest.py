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
