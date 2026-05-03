"""
Unified Data Provider
======================
Fetches market data from:
  - Alpaca: US tickers (SPY, QQQ, etc.) — primary
  - Tiingo: US tickers + indices (^VIX) — fast, reliable fallback
  - AkShare: Asian tickers (*.HK, *.T, *.SI via mapped symbols)
  - yfinance: last-resort fallback only

Setup:
  export APCA_API_KEY_ID=your_key
  export APCA_API_SECRET_KEY=your_secret
  export TIINGO_API_KEY=your_tiingo_key  (free at tiingo.com)

Free Alpaca account gives unlimited historical data for US stocks.
Tiingo free tier: 500 calls/day, 5 years history.
"""

import os
import json
import logging
import urllib.request
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# ── Alpaca client (lazy init) ───────────────────────────────────────────────

_alpaca_client = None


def _get_alpaca():
    global _alpaca_client
    if _alpaca_client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        key = os.environ.get("APCA_API_KEY_ID", "")
        secret = os.environ.get("APCA_API_SECRET_KEY", "")
        if not key or not secret:
            raise RuntimeError(
                "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars. "
                "Get free keys at https://app.alpaca.markets/signup"
            )
        _alpaca_client = StockHistoricalDataClient(key, secret)
    return _alpaca_client


def _is_us_ticker(symbol: str) -> bool:
    """Return True if the symbol is a US-listed ticker (no exchange suffix)."""
    return "." not in symbol and not symbol.startswith("^")


def _is_index(symbol: str) -> bool:
    """Return True if the symbol is a market index (^VIX, ^GSPC, etc.)."""
    return symbol.startswith("^")


def _is_akshare_ticker(symbol: str) -> bool:
    """Return True if AkShare can handle this ticker (HK, Japan)."""
    return symbol.endswith(".HK") or symbol.endswith(".T")


# AkShare symbol mapping: yfinance format → akshare format
_AKSHARE_HK_MAP = {
    # yfinance "2800.HK" → akshare "02800" (5-digit zero-padded)
}

def _to_akshare_hk(symbol: str) -> str:
    """Convert '2800.HK' → '02800' for akshare."""
    code = symbol.replace(".HK", "")
    return code.zfill(5)

def _to_akshare_jp(symbol: str) -> str:
    """Convert '1306.T' → '1306' for akshare."""
    return symbol.replace(".T", "")


# ── Period string → date range ──────────────────────────────────────────────

_PERIOD_MAP = {
    "5d": 5, "10d": 10, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
}


def _period_to_dates(period: str):
    days = _PERIOD_MAP.get(period, 10)
    # Add buffer for weekends/holidays
    end = datetime.now()
    start = end - timedelta(days=int(days * 1.6) + 5)
    return start, end


# ── Public API ──────────────────────────────────────────────────────────────

def download(symbol: str, period: str = "10d") -> pd.DataFrame:
    """
    Download OHLCV data.  Returns a DataFrame with columns:
      Open, High, Low, Close, Volume
    indexed by date (tz-naive).

    Priority: Alpaca (US) → Tiingo (US + indices) → AkShare (Asian) → yfinance (last resort)
    """
    if _is_us_ticker(symbol):
        try:
            return _download_alpaca(symbol, period)
        except Exception as e:
            logger.warning("Alpaca failed for %s (%s), trying Tiingo", symbol, e)
            return _download_tiingo(symbol, period)
    elif _is_index(symbol):
        # Tiingo doesn't carry index data (^VIX etc) — use yfinance directly
        return _download_yfinance(symbol, period)
    elif _is_akshare_ticker(symbol):
        return _download_akshare(symbol, period)
    else:
        return _download_tiingo(symbol, period)


def download_multi(symbols, period="10d"):
    """Download data for multiple symbols. Returns {symbol: df}."""
    results = {}
    # Split US vs non-US
    us = [s for s in symbols if _is_us_ticker(s)]
    non_us = [s for s in symbols if not _is_us_ticker(s)]

    # Batch US tickers via Alpaca
    if us:
        try:
            batch = _download_alpaca_multi(us, period)
            results.update(batch)
        except Exception as e:
            logger.warning("Alpaca batch failed, falling back to Tiingo: %s", e)
            for s in us:
                try:
                    results[s] = _download_tiingo(s, period)
                except Exception:
                    pass

    # Non-US: AkShare for HK/JP, Tiingo for indices, yfinance last resort
    for s in non_us:
        try:
            if _is_akshare_ticker(s):
                results[s] = _download_akshare(s, period)
            else:
                results[s] = _download_tiingo(s, period)
        except Exception:
            try:
                results[s] = _download_yfinance(s, period)
            except Exception:
                pass

    return results


# ── Alpaca implementation ───────────────────────────────────────────────────

def _download_alpaca(symbol: str, period: str) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_alpaca()
    start, end = _period_to_dates(period)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex",  # Free tier — use IEX feed instead of SIP
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        return pd.DataFrame()

    # Alpaca returns MultiIndex (symbol, timestamp) — drop symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.droplevel("symbol")

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    # Keep only the columns we need, drop trade_count/vwap
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Trim to requested period
    days = _PERIOD_MAP.get(period, 10)
    df = df.tail(days)

    return df


def _download_alpaca_multi(symbols, period):
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = _get_alpaca()
    start, end = _period_to_dates(period)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex",  # Free tier — use IEX feed instead of SIP
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df.empty:
        return {}

    results = {}
    days = _PERIOD_MAP.get(period, 10)

    for sym in symbols:
        try:
            sdf = df.xs(sym, level="symbol")
            sdf.index = pd.to_datetime(sdf.index).tz_localize(None)
            sdf = sdf.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            sdf = sdf[["Open", "High", "Low", "Close", "Volume"]]
            sdf = sdf.tail(days)
            if not sdf.empty:
                results[sym] = sdf
        except KeyError:
            pass

    return results


# ── AkShare implementation (Asian markets) ─────────────────────────────────

def _download_akshare(symbol: str, period: str) -> pd.DataFrame:
    """Download HK or JP stock data via AkShare, fall back to yfinance on error."""
    try:
        import akshare as ak

        days = _PERIOD_MAP.get(period, 10)
        end = datetime.now()
        start = end - timedelta(days=int(days * 1.6) + 5)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        if symbol.endswith(".HK"):
            code = _to_akshare_hk(symbol)
            df = ak.stock_hk_hist(
                symbol=code,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="qfq",
            )
            df = df.rename(columns={
                "日期": "Date", "开盘": "Open", "最高": "High",
                "最低": "Low", "收盘": "Close", "成交量": "Volume",
            })

        elif symbol.endswith(".T"):
            code = _to_akshare_jp(symbol)
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="qfq",
            )
            df = df.rename(columns={
                "日期": "Date", "开盘": "Open", "最高": "High",
                "最低": "Low", "收盘": "Close", "成交量": "Volume",
            })
        else:
            return _download_yfinance(symbol, period)

        if df.empty:
            logger.warning("AkShare returned empty for %s, falling back to yfinance", symbol)
            return _download_yfinance(symbol, period)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.tail(days)

        logger.info("AkShare: fetched %d rows for %s", len(df), symbol)
        return df

    except Exception as e:
        logger.warning("AkShare failed for %s (%s), falling back to yfinance", symbol, e)
        return _download_yfinance(symbol, period)


# ── Tiingo implementation ───────────────────────────────────────────────────

def _tiingo_symbol(symbol: str) -> str:
    """Convert yfinance-style symbol to Tiingo format. ^VIX → vix"""
    return symbol.lstrip("^").lower()


def _download_tiingo(symbol: str, period: str) -> pd.DataFrame:
    """Fetch daily OHLCV from Tiingo. Fast, reliable, free tier 500 calls/day."""
    token = os.environ.get("TIINGO_API_KEY", "")
    if not token:
        logger.warning("TIINGO_API_KEY not set, falling back to yfinance")
        return _download_yfinance(symbol, period)

    start, end = _period_to_dates(period)
    ticker = _tiingo_symbol(symbol)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    url = (
        "https://api.tiingo.com/tiingo/daily/{}/prices"
        "?startDate={}&endDate={}&token={}"
    ).format(ticker, start_str, end_str, token)

    try:
        req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        if not data:
            logger.warning("Tiingo returned empty for %s", symbol)
            return _download_yfinance(symbol, period)

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date")
        df = df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })

        # Use adjClose if available for accurate prices
        if "adjClose" in df.columns:
            df["Close"] = df["adjClose"]

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        days = _PERIOD_MAP.get(period, 10)
        df = df.tail(days)

        logger.info("Tiingo: fetched %d rows for %s", len(df), symbol)
        return df

    except Exception as e:
        logger.warning("Tiingo failed for %s (%s), falling back to yfinance", symbol, e)
        return _download_yfinance(symbol, period)


# ── yfinance — last resort only ─────────────────────────────────────────────

def _download_yfinance(symbol: str, period: str) -> pd.DataFrame:
    """Last resort fallback. Avoid — slow, unreliable, causes dashboard timeouts."""
    import yfinance as yf
    logger.warning("Using yfinance for %s — consider adding Tiingo support", symbol)
    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance ≥0.2.38: columns are (Price, Ticker) — drop the Ticker level (level 1)
        ticker_level = df.columns.names.index("Ticker")
        df.columns = df.columns.droplevel(ticker_level)
    return df
