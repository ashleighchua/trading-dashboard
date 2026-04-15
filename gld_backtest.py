#!/usr/bin/env python3
"""
GLD (SPDR Gold ETF) Backtest
==============================
Tests two strategy families side-by-side:
  Part 1 — Day-of-week patterns (buy open, sell close)
  Part 2 — Trend pullback long (EMA uptrend + RSI oversold + pullback from high)

Data source: Alpaca → Tiingo → yfinance (via dashboard/data_provider.py)
Period: 2020-01-01 to present
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load .env so data_provider can find Alpaca/Tiingo keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Use the same data provider as the live scanner
sys.path.insert(0, str(Path(__file__).parent / "dashboard"))
from data_provider import download

SYMBOL         = "GLD"
START          = "2019-06-01"   # extra history for EMA warmup
BACKTEST_START = "2020-01-01"
SL_PCT         = 1.5            # 1.5% stop loss — GLD daily range ~0.5-1%
INIT_EQ        = 10_000.0

print(f"Downloading {SYMBOL} data ({START} → today)...")
# data_provider.download() takes period strings like "5y", "max"
# For backtests we need a specific start date, so use the Alpaca client directly
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

key    = os.environ.get("APCA_API_KEY_ID", "")
secret = os.environ.get("APCA_API_SECRET_KEY", "")
client = StockHistoricalDataClient(key, secret)

req = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=datetime.strptime(START, "%Y-%m-%d"),
)
bars = client.get_stock_bars(req)
df_raw = bars.df

# Alpaca returns MultiIndex (symbol, timestamp) — flatten to single index
if isinstance(df_raw.index, pd.MultiIndex):
    df_raw = df_raw.xs(SYMBOL, level="symbol")
df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
df_raw = df_raw.rename(columns={
    "open": "Open", "high": "High", "low": "Low",
    "close": "Close", "volume": "Volume"
})
df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
print(f"Done. {len(df_raw)} bars ({df_raw.index[0].date()} → {df_raw.index[-1].date()})\n")
