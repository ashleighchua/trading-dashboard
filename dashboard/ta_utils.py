"""
Technical Analysis Utilities
=============================
Pure pandas/numpy implementations of common indicators.
No extra dependencies — works with bar data from Alpaca.

Usage:
    from dashboard.ta_utils import rsi, macd, trend_context, full_analysis

    bars = [{"c": 251.85, "o": 251.0, "h": 253.0, "l": 249.5, "v": 900000, "t": "..."}, ...]
    analysis = full_analysis(bars)
"""

import numpy as np


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def rsi(closes, period=14):
    """
    Compute RSI for a list of closing prices.
    Returns a list of RSI values (first `period` values are None).
    """
    closes = list(closes)
    if len(closes) < period + 1:
        return [None] * len(closes)

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsi_values = [None] * (period + 1)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi_values


def rsi_latest(closes, period=14):
    """Return the single most recent RSI value."""
    values = rsi(closes, period)
    for v in reversed(values):
        if v is not None:
            return round(v, 1)
    return None


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def ema(values, period):
    """Exponential moving average."""
    result = []
    k = 2.0 / (period + 1)
    for i, v in enumerate(values):
        if i == 0:
            result.append(v)
        else:
            result.append(v * k + result[-1] * (1 - k))
    return result


def macd(closes, fast=12, slow=26, signal=9):
    """
    Returns dict with keys: macd_line, signal_line, histogram (all lists).
    Values at the start may be unreliable until enough data exists.
    """
    closes = list(closes)
    fast_ema = ema(closes, fast)
    slow_ema = ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }


def macd_latest(closes):
    """Return latest MACD values as a compact dict."""
    m = macd(closes)
    return {
        "macd": round(m["macd_line"][-1], 3),
        "signal": round(m["signal_line"][-1], 3),
        "histogram": round(m["histogram"][-1], 3),
        "bullish_cross": m["histogram"][-1] > 0 and m["histogram"][-2] <= 0 if len(m["histogram"]) >= 2 else False,
        "bearish_cross": m["histogram"][-1] < 0 and m["histogram"][-2] >= 0 if len(m["histogram"]) >= 2 else False,
    }


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------

def trend_context(bars, short_window=5, long_window=20):
    """
    Assess trend from daily bars.

    Returns dict:
      trend: "uptrend" | "downtrend" | "choppy"
      distance_from_high_pct: how far price is from 20-day high (negative = below)
      distance_from_low_pct: how far price is from 20-day low (positive = above)
      bear_rally: True if price bounced >4% from recent low but trend is down
      recent_bounce_pct: % move from recent low to current price
    """
    if len(bars) < long_window:
        return {"trend": "insufficient data"}

    closes = [b["c"] for b in bars]
    recent = closes[-long_window:]
    current = closes[-1]
    high_20 = max(b["h"] for b in bars[-long_window:])
    low_20 = min(b["l"] for b in bars[-long_window:])

    # Short vs long MA
    short_ma = sum(closes[-short_window:]) / short_window
    long_ma = sum(recent) / long_window

    dist_from_high = (current - high_20) / high_20 * 100
    dist_from_low = (current - low_20) / low_20 * 100

    if short_ma > long_ma * 1.005:
        trend = "uptrend"
    elif short_ma < long_ma * 0.995:
        trend = "downtrend"
    else:
        trend = "choppy"

    # Bear rally: in a downtrend but bounced >2% off recent low
    recent_low = min(b["l"] for b in bars[-5:])
    recent_bounce_pct = (current - recent_low) / recent_low * 100
    bear_rally = trend == "downtrend" and recent_bounce_pct >= 4.0

    return {
        "trend": trend,
        "short_ma": round(short_ma, 2),
        "long_ma": round(long_ma, 2),
        "distance_from_high_pct": round(dist_from_high, 2),
        "distance_from_low_pct": round(dist_from_low, 2),
        "bear_rally": bear_rally,
        "recent_bounce_pct": round(recent_bounce_pct, 2),
        "high_20": round(high_20, 2),
        "low_20": round(low_20, 2),
    }


def trend_pullback(bars, ema_fast=50, ema_slow=200, pullback_threshold=3.0, rsi_max=35):
    """
    Detect a pullback entry in an uptrending ticker — mirror of bear rally detection.

    Validated parameters (NVDA backtest 2020-2026, 15 trades, 60% WR, PF 4.82):
      ema_fast=50, ema_slow=200, pullback_threshold=3.0, rsi_max=35

    Uptrend:  EMA(ema_fast) > EMA(ema_slow)
    Signal:   price pulled back >= pullback_threshold% from 10-day high
              within last 3 bars AND RSI-14 < rsi_max

    Returns dict:
      trend:               "uptrend" | "downtrend" | "choppy"
      pullback_long:       True if all conditions met
      recent_pullback_pct: % below 10-day high right now
      rsi_14:              current RSI-14 value
      high_10:             10-day high price
      ema_fast:            latest fast EMA value
      ema_slow:            latest slow EMA value
    """
    min_bars = max(ema_slow + 1, 20)
    if len(bars) < min_bars:
        return {"trend": "insufficient data", "pullback_long": False,
                "recent_pullback_pct": 0.0, "rsi_14": None,
                "high_10": 0.0, "ema_fast": 0.0, "ema_slow": 0.0}

    closes = [b["c"] for b in bars]

    # Use existing ema() helper
    ema_fast_vals = ema(closes, ema_fast)
    ema_slow_vals = ema(closes, ema_slow)

    latest_fast = ema_fast_vals[-1]
    latest_slow = ema_slow_vals[-1]

    if latest_fast > latest_slow * 1.005:
        trend = "uptrend"
    elif latest_fast < latest_slow * 0.995:
        trend = "downtrend"
    else:
        trend = "choppy"

    # 10-day high and pullback from it
    high_10 = max(b["h"] for b in bars[-10:])
    current = closes[-1]
    recent_pullback_pct = (high_10 - current) / high_10 * 100

    # Check if pullback threshold was reached within last 3 bars
    pullback_3d = max(
        (high_10 - b["c"]) / high_10 * 100
        for b in bars[-3:]
    )

    # RSI-14 (reuse existing rsi_latest helper)
    rsi_val = rsi_latest(closes, 14)

    pullback_long = (
        trend == "uptrend"
        and pullback_3d >= pullback_threshold
        and rsi_val is not None
        and rsi_val < rsi_max
    )

    return {
        "trend":               trend,
        "pullback_long":       pullback_long,
        "recent_pullback_pct": round(recent_pullback_pct, 2),
        "rsi_14":              rsi_val,
        "high_10":             round(high_10, 2),
        "ema_fast":            round(latest_fast, 2),
        "ema_slow":            round(latest_slow, 2),
    }


# ---------------------------------------------------------------------------
# Volume analysis
# ---------------------------------------------------------------------------

def volume_analysis(bars, lookback=10):
    """
    Compare today's volume to the recent average.
    Returns: ratio (today / avg), and a label.
    """
    if len(bars) < lookback + 1:
        return {"ratio": None, "label": "insufficient data"}

    avg_vol = sum(b["v"] for b in bars[-(lookback + 1):-1]) / lookback
    today_vol = bars[-1]["v"]

    if avg_vol == 0:
        return {"ratio": None, "label": "no volume"}

    ratio = today_vol / avg_vol
    if ratio >= 2.0:
        label = "very high volume"
    elif ratio >= 1.3:
        label = "above average volume"
    elif ratio >= 0.7:
        label = "normal volume"
    else:
        label = "low volume"

    return {"ratio": round(ratio, 2), "label": label, "avg_volume": int(avg_vol)}


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def full_analysis(bars):
    """
    Run all indicators on a list of daily bars.
    Each bar should have keys: o, h, l, c, v, t

    Returns a dict with all indicator values ready for trade decision-making.
    """
    if not bars or len(bars) < 5:
        return {"error": "insufficient bars"}

    closes = [b["c"] for b in bars]

    result = {
        "current_price": closes[-1],
        "prev_close": closes[-2] if len(closes) >= 2 else None,
        "change_pct": round((closes[-1] - closes[-2]) / closes[-2] * 100, 2) if len(closes) >= 2 else None,
        "rsi_14": rsi_latest(closes, 14),
        "macd": macd_latest(closes) if len(closes) >= 30 else None,
        "trend": trend_context(bars),
        "volume": volume_analysis(bars),
    }

    # Signal summary
    rsi_val = result["rsi_14"]
    trend = result["trend"].get("trend", "choppy")
    bear_rally = result["trend"].get("bear_rally", False)

    signals = []
    if bear_rally:
        signals.append("BEAR RALLY FADE candidate")
    if rsi_val and rsi_val > 65 and trend == "downtrend":
        signals.append("RSI overbought in downtrend — fade setup")
    if rsi_val and rsi_val < 35:
        signals.append("RSI oversold — potential bounce")
    if result["macd"] and result["macd"].get("bearish_cross"):
        signals.append("MACD bearish crossover")
    if result["macd"] and result["macd"].get("bullish_cross"):
        signals.append("MACD bullish crossover")

    result["signals"] = signals
    result["conviction"] = len(signals)  # 0-4, higher = more aligned signals

    return result
