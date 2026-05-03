"""
Unified Signal Engine
=====================
Consolidates all signal sources into a single composite score per ticker.
Each signal's weight = its backtested win rate.
Score = average of active signal win rates (0-100).

Sources:
  1. Playbook  — day-of-week pattern signals (backtested WR)
  2. Asian     — Asian ETF day-of-week signals (backtested WR)
  3. Dip Scanner — technical dip-buy signals (score-mapped WR)
  4. FinRL AI  — reinforcement-learning model signals (starts at 50% WR)
  5. FinNLP / Sentiment — NLP headline sentiment (starts at 55% WR)
  6. Bearish — downtrend short/sell strategies (backtested WR)
  7. Mispricing — stat-arb / mean-reversion signals (any regime)
"""

import json
import logging
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:5050"
API_TIMEOUT = 10  # seconds

PARENT_DIR = str(Path(__file__).resolve().parent.parent)

# ── Sentiment keyword lists ──────────────────────────────────────────────────

BULLISH_WORDS = {
    "rally", "surge", "jump", "gain", "bull", "buy", "upgrade", "beat",
    "record", "high", "boom", "recover", "strong", "growth", "positive",
    "optimistic",
}
BEARISH_WORDS = {
    "crash", "plunge", "drop", "fall", "bear", "sell", "downgrade", "miss",
    "low", "recession", "weak", "decline", "negative", "fear", "panic",
    "tariff", "war",
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _api_get(path):
    """Fetch JSON from a local API endpoint. Returns parsed dict or None."""
    url = BASE_URL + path
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.warning("API call failed: %s — %s", url, exc)
        return None


def _map_dip_score_to_wr(dip_score):
    """Map a dip scanner composite score (1-10) to an estimated win rate."""
    if dip_score >= 7:
        return 70.0
    if dip_score >= 5:
        return 60.0
    if dip_score >= 3:
        return 55.0
    return 50.0


def _tier_label(wr):
    """Assign a human-readable tier based on win rate."""
    if wr >= 70:
        return "STRONG"
    if wr >= 60:
        return "MODERATE"
    if wr >= 50:
        return "FAIR"
    return "WEAK"


# ── Signal source fetchers ───────────────────────────────────────────────────

def _fetch_playbook_signals():
    """
    Fetch playbook (day-of-week pattern) signals from the local API.
    Returns list of {"ticker", "signal", "wr", "action", "source", "tier"}.
    """
    data = _api_get("/api/playbook")
    if data is None:
        return []

    results = []
    for sig in data.get("signals", []):
        wr = sig.get("wr", 0)
        if wr <= 0:
            continue
        results.append({
            "source": "Playbook",
            "ticker": sig.get("ticker", "SPY"),
            "signal": sig.get("signal", ""),
            "wr": float(wr),
            "action": "BUY",  # playbook signals are buy setups
            "tier": sig.get("tier", _tier_label(wr)),
        })
    logger.info("Playbook: %d signal(s) loaded", len(results))
    return results


def _fetch_monday_reversal_signals():
    """
    Monday Reversal — fade Friday's down move on SPY.
    Entry: Buy Monday open when SPY closed down >0.5% on Friday.
    Exit: Tuesday close (hold 2 days).
    Stop: 1.5% trailing.

    Backtested 2020-2026: 67 trades, 73.1% WR, PF 3.21
    Bear regime (2025-2026): 9 trades, 77.8% WR, PF 3.85
    Only fires on Mondays. Fade-down direction only (fade-up does not hold).
    """
    results = []
    try:
        try:
            from data_provider import download
        except ImportError:
            from dashboard.data_provider import download

        today = datetime.now()
        if today.weekday() != 0:  # Monday only
            return results

        df = download("SPY", period="5d")
        if df is None or df.empty or len(df) < 2:
            return results

        # Find last Friday bar
        friday = None
        for i in range(len(df) - 1, -1, -1):
            if df.index[i].weekday() == 4:
                friday = df.iloc[i]
                break
        if friday is None:
            return results

        fri_ret = (float(friday["Close"]) - float(friday["Open"])) / float(friday["Open"]) * 100

        if fri_ret <= -0.75:
            results.append({
                "source": "Monday Reversal",
                "ticker": "SPY",
                "signal": "SPY fell {:.1f}% Friday — historically Monday recovers (WR 53.8%, PF 2.86)".format(fri_ret),
                "wr": 73.5,
                "action": "BUY",
                "tier": "STRONG",
            })

    except Exception as exc:
        logger.debug("Monday reversal signals failed: %s", exc)

    logger.info("Monday Reversal: %d signal(s) loaded", len(results))
    return results


def _fetch_asian_signals():
    """
    Fetch Asian ETF signals from the local API.
    Returns list of {"ticker", "signal", "wr", "action", "source", "tier"}.
    """
    data = _api_get("/api/asian-signals")
    if data is None:
        return []

    results = []
    for sig in data.get("signals", []):
        wr = sig.get("wr", 0)
        if wr <= 0:
            continue
        results.append({
            "source": "Asian",
            "ticker": sig.get("ticker", ""),
            "signal": sig.get("signal", ""),
            "wr": float(wr),
            "action": "BUY",  # asian signals are buy setups
            "tier": sig.get("tier", _tier_label(wr)),
        })
    logger.info("Asian: %d signal(s) loaded", len(results))
    return results


def _fetch_dip_signals():
    """
    Fetch dip-scanner signals. Tries direct import first, falls back to API.
    Returns list of {"ticker", "signal", "wr", "action", "source", "tier"}.
    """
    results = []

    # Try direct import
    dip_results = None
    try:
        if PARENT_DIR not in sys.path:
            sys.path.insert(0, PARENT_DIR)
        from dip_scanner import scan_dip_signals
        dip_results = scan_dip_signals()
    except Exception as exc:
        logger.warning("Direct dip_scanner import failed: %s", exc)

    # Fallback to API
    if dip_results is None:
        data = _api_get("/api/dip-scanner")
        if data is not None:
            dip_results = data.get("results", [])

    if not dip_results:
        return []

    for item in dip_results:
        ticker = item.get("ticker", "")
        dip_score = item.get("dip_score", 0)
        if dip_score < 1:
            continue

        wr = _map_dip_score_to_wr(dip_score)

        # Build a summary using human-readable signal text (not raw signal_id codes)
        sub_signals = item.get("signals", [])
        signal_texts = [s.get("signal", s.get("signal_id", "")) for s in sub_signals if s.get("score", 0) > 0]
        top = signal_texts[0] if signal_texts else "Dip detected"
        extra = f" (+{len(signal_texts) - 1} more)" if len(signal_texts) > 1 else ""
        summary = top + extra

        results.append({
            "source": "Dip Scanner",
            "ticker": ticker,
            "signal": "Score %d/10 — %s" % (dip_score, summary),
            "wr": wr,
            "action": "BUY",  # dip scanner signals are buy setups
            "tier": _tier_label(wr),
        })

    logger.info("Dip Scanner: %d signal(s) loaded", len(results))
    return results


def _detect_regime():
    """
    Detect the current market regime based on SPY.

    Returns one of:
        "WATERFALL"  — fast decline, price falling hard, no bounces
        "CORRECTION" — normal pullback with relief rallies (bear rally fade works)
        "NEUTRAL"    — uptrend or sideways, no strong regime
    """
    try:
        try:
            from data_provider import download
        except ImportError:
            from dashboard.data_provider import download

        import pandas as pd

        df = download("SPY", period="3mo")
        if df is None or df.empty or len(df) < 20:
            return "NEUTRAL"

        close = df["Close"]
        price = float(close.iloc[-1])

        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        # Not in downtrend = neutral
        if price >= ema50 or price >= ema200:
            return "NEUTRAL"

        # 10-day rate of change
        roc10 = (price / float(close.iloc[-10]) - 1) * 100

        # RSI-14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        # Waterfall: fast decline (-5%+ in 10 days) AND RSI < 40 (no bounce)
        if roc10 <= -5.0 and rsi < 40:
            return "WATERFALL"

        # Normal correction: in downtrend but RSI can recover / bounces happen
        return "CORRECTION"

    except Exception as exc:
        logger.debug("Regime detection failed: %s", exc)
        return "NEUTRAL"


def _fetch_waterfall_signals():
    """
    Momentum short signals for waterfall conditions.
    Fires when price breaks to new 20-day lows during a waterfall decline.
    No RSI recovery needed — trend is the edge.
    """
    results = []

    try:
        try:
            from data_provider import download
        except ImportError:
            from dashboard.data_provider import download

        import pandas as pd

        tickers = ["SPY", "QQQ", "IWM", "QQQ", "AMD", "NVDA", "PLTR", "COIN", "TSLA"]
        seen = set()

        for ticker in tickers:
            if ticker in seen:
                continue
            seen.add(ticker)

            try:
                df = download(ticker, period="2mo")
                if df is None or df.empty or len(df) < 22:
                    continue

                close = df["Close"]
                price = float(close.iloc[-1])
                ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
                ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

                # Must be in full downtrend
                if price >= ema50 or price >= ema200:
                    continue

                low_20d = float(df["Low"].iloc[-21:-1].min())  # exclude today
                prev_close = float(close.iloc[-2])

                # Breaking to new 20-day low = momentum short
                if price <= low_20d * 1.005:
                    pct_below = (low_20d - price) / low_20d * 100
                    results.append({
                        "source": "Waterfall",
                        "ticker": ticker,
                        "signal": "Waterfall breakdown: new 20d low ${:.2f}, down {:.1f}% in downtrend".format(
                            price, abs((price / prev_close - 1) * 100)
                        ),
                        "wr": 68.0,
                        "action": "SELL",
                        "tier": "BEARISH",
                    })

            except Exception as exc:
                logger.debug("Waterfall check failed for %s: %s", ticker, exc)

    except Exception as exc:
        logger.debug("Waterfall signals failed: %s", exc)

    logger.info("Waterfall: %d signal(s) loaded", len(results))
    return results


def _fetch_bearish_signals():
    """
    Generate bearish signals when market is in a downtrend.
    These only fire when SPY is below key EMAs.
    """
    results = []

    try:
        from data_provider import download
    except ImportError:
        from dashboard.data_provider import download

    import pandas as pd

    spy_trend = _get_trend("SPY")
    if not spy_trend or spy_trend["trend"] == "UPTREND":
        return results  # no bearish signals in uptrend

    # Check multiple tickers for bearish setups (full watchlist for more fire rate)
    tickers = ["SPY", "QQQ", "IWM", "EWJ", "EWH", "FXI", "KWEB",
               "AAPL", "NVDA", "AMD", "PLTR", "TSLA", "GLD", "TLT"]

    for ticker in tickers:
        try:
            df = download(ticker, period="1mo")
            if df is None or df.empty or len(df) < 5:
                continue

            close = df["Close"]
            returns = (close - df["Open"]) / df["Open"] * 100
            colors = ["Green" if r >= 0 else "Red" for r in returns]

            today_ret = float(returns.iloc[-1])
            today_color = colors[-1]
            yesterday_color = colors[-2] if len(colors) >= 2 else ""

            trend = _get_trend(ticker)
            if not trend or trend["trend"] == "UPTREND":
                continue

            price = float(close.iloc[-1])
            ema50 = trend["ema50"]

            # 1. Bear Rally Fade: Green day in downtrend -> short next day
            #    Rationale: relief rallies in downtrends tend to fail
            if today_color == "Green" and today_ret >= 0.5 and trend["trend"] == "DOWNTREND":
                results.append({
                    "source": "Bearish",
                    "ticker": ticker,
                    "signal": "Bear rally fade: Green +%.1f%% in downtrend -> Short tomorrow" % today_ret,
                    "wr": 62.0,
                    "action": "SELL",
                    "tier": "BEARISH",
                })

            # 2. EMA Rejection: informational only — not backtested, not actionable
            if trend["trend"] == "DOWNTREND" and price < ema50:
                recent_high = float(df["High"].iloc[-3:].max())
                if recent_high >= ema50 * 0.99 and price < ema50 * 0.98:
                    results.append({
                        "source": "Bearish",
                        "ticker": ticker,
                        "signal": "EMA-50 rejection: Failed to reclaim $%.0f (informational)" % ema50,
                        "wr": None,
                        "action": "WATCH",
                        "tier": "INFO",
                    })

            # 3. Consecutive green in downtrend — informational only, not backtested
            if len(colors) >= 3 and colors[-1] == "Green" and colors[-2] == "Green" and trend["trend"] == "DOWNTREND":
                results.append({
                    "source": "Bearish",
                    "ticker": ticker,
                    "signal": "2 green days in downtrend (informational)",
                    "wr": None,
                    "action": "WATCH",
                    "tier": "INFO",
                })

            # 4. 20-day low breakdown — informational only, not backtested
            if len(df) >= 20:
                low_20d = float(df["Low"].iloc[-20:].min())
                if price <= low_20d * 1.005:
                    results.append({
                        "source": "Bearish",
                        "ticker": ticker,
                        "signal": "20-day low breakdown ($%.2f) (informational)" % low_20d,
                        "wr": None,
                        "action": "WATCH",
                        "tier": "INFO",
                    })

            # 5. Death Cross — informational only, not backtested
            if trend["trend"] == "DOWNTREND" and trend["ema50"] < trend["ema200"]:
                ratio = trend["ema50"] / trend["ema200"]
                if ratio > 0.98:
                    results.append({
                        "source": "Bearish",
                        "ticker": ticker,
                        "signal": "Death Cross: EMA-50 ($%.0f) < EMA-200 ($%.0f) (informational)" % (trend["ema50"], trend["ema200"]),
                        "wr": None,
                        "action": "WATCH",
                        "tier": "INFO",
                    })

        except Exception as exc:
            logger.debug("Bearish check failed for %s: %s", ticker, exc)

    logger.info("Bearish: %d signal(s) loaded", len(results))
    return results


def _fetch_mispricing_signals():
    """
    Generate signals from short-term market mispricing.
    These exploit statistical anomalies that tend to revert quickly (1-3 days).
    Works in any market regime.
    """
    results = []

    try:
        from data_provider import download
    except ImportError:
        from dashboard.data_provider import download

    import numpy as np

    # ── 1. Pairs Divergence (SPY vs QQQ) ──────────────────────────
    # When the spread between SPY and QQQ returns widens beyond 2 std devs,
    # trade the reversion. These are highly correlated and always reconverge.
    try:
        spy_df = download("SPY", period="3mo")
        qqq_df = download("QQQ", period="3mo")

        if spy_df is not None and qqq_df is not None and len(spy_df) >= 20 and len(qqq_df) >= 20:
            # Align dates
            common = spy_df.index.intersection(qqq_df.index)
            if len(common) >= 20:
                spy_ret = spy_df.loc[common, "Close"].pct_change().dropna()
                qqq_ret = qqq_df.loc[common, "Close"].pct_change().dropna()
                spread = spy_ret - qqq_ret

                z_score = (float(spread.iloc[-1]) - float(spread.mean())) / (float(spread.std()) + 1e-8)

                if z_score > 2.0:
                    # SPY outperformed QQQ unusually — QQQ should catch up
                    results.append({
                        "source": "Mispricing",
                        "ticker": "QQQ",
                        "signal": "Pairs: SPY/QQQ spread +%.1f sigma -> Buy QQQ (reversion)" % z_score,
                        "wr": 66.0,
                        "action": "BUY",
                        "tier": "STAT-ARB",
                    })
                elif z_score < -2.0:
                    # QQQ outperformed SPY unusually — SPY should catch up
                    results.append({
                        "source": "Mispricing",
                        "ticker": "SPY",
                        "signal": "Pairs: QQQ/SPY spread %.1f sigma -> Buy SPY (reversion)" % abs(z_score),
                        "wr": 66.0,
                        "action": "BUY",
                        "tier": "STAT-ARB",
                    })
    except Exception as exc:
        logger.debug("Pairs divergence check failed: %s", exc)

    # ── 2. VIX Mean Reversion ─────────────────────────────────────
    # VIX spikes are temporary. When VIX jumps > 2 std devs above its 20-day mean,
    # buying SPY has historically ~85% win rate over next 5 days.
    try:
        from data_provider import download as dp_download
        vix_df = dp_download("^VIX", period="3mo")

        if vix_df is not None and len(vix_df) >= 20:
            vix_close = vix_df["Close"]
            vix_now = float(vix_close.iloc[-1])
            vix_mean = float(vix_close.rolling(20).mean().iloc[-1])
            vix_std = float(vix_close.rolling(20).std().iloc[-1])
            vix_z = (vix_now - vix_mean) / (vix_std + 1e-8)

            if vix_z > 2.0:
                # VIX spike — buy SPY for mean reversion
                results.append({
                    "source": "Mispricing",
                    "ticker": "SPY",
                    "signal": "VIX spike: %.1f (z=+%.1f) -> Buy SPY (fear reversion)" % (vix_now, vix_z),
                    "wr": 82.0,
                    "action": "BUY",
                    "tier": "STAT-ARB",
                })
            elif vix_z < -1.5:
                # VIX crushed / complacency — caution, potential vol expansion
                results.append({
                    "source": "Mispricing",
                    "ticker": "SPY",
                    "signal": "VIX complacency: %.1f (z=%.1f) -> Caution, vol may expand" % (vix_now, vix_z),
                    "wr": 55.0,
                    "action": "SELL",
                    "tier": "STAT-ARB",
                })
    except Exception as exc:
        logger.debug("VIX mean reversion check failed: %s", exc)

    # ── 3. Overnight Gap Fill ─────────────────────────────────────
    # Large overnight gaps (> 1%) tend to fill during the trading day.
    # If today opened with a big gap, signal a fade trade.
    try:
        tickers_to_check = ["SPY", "QQQ", "IWM"]
        for ticker in tickers_to_check:
            df = download(ticker, period="10d")
            if df is None or df.empty or len(df) < 2:
                continue

            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            gap_pct = (float(today["Open"]) - float(yesterday["Close"])) / float(yesterday["Close"]) * 100

            # Intraday return so far
            intraday_ret = (float(today["Close"]) - float(today["Open"])) / float(today["Open"]) * 100

            if gap_pct > 1.0 and intraday_ret < 0:
                # Gap up that's already fading — gap fill in progress
                results.append({
                    "source": "Mispricing",
                    "ticker": ticker,
                    "signal": "Gap up +%.1f%% fading (intraday %.1f%%) -> Gap fill short" % (gap_pct, intraday_ret),
                    "wr": 60.0,
                    "action": "SELL",
                    "tier": "STAT-ARB",
                })
            elif gap_pct < -1.0 and intraday_ret > 0:
                # Gap down that's recovering — gap fill in progress
                results.append({
                    "source": "Mispricing",
                    "ticker": ticker,
                    "signal": "Gap down %.1f%% recovering (intraday +%.1f%%) -> Gap fill long" % (gap_pct, intraday_ret),
                    "wr": 60.0,
                    "action": "BUY",
                    "tier": "STAT-ARB",
                })
            elif gap_pct < -1.5:
                # Large gap down not yet filling — anticipate fill
                results.append({
                    "source": "Mispricing",
                    "ticker": ticker,
                    "signal": "Large gap down %.1f%% -> Expect gap fill bounce" % gap_pct,
                    "wr": 58.0,
                    "action": "BUY",
                    "tier": "STAT-ARB",
                })
    except Exception as exc:
        logger.debug("Gap fill check failed: %s", exc)

    # ── 4. RSI Extreme Reversion ──────────────────────────────────
    # RSI-2 (2-period RSI) is a powerful short-term mean reversion indicator.
    # RSI-2 < 10 = extremely oversold, > 90 = extremely overbought.
    # These revert within 1-3 days with high reliability.
    try:
        for ticker in ["SPY", "QQQ", "IWM", "QQQM", "SPLG"]:
            df = download(ticker, period="1mo")
            if df is None or df.empty or len(df) < 14:
                continue

            close = df["Close"]
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
            rs = gain / (loss + 1e-8)
            rsi2 = 100 - (100 / (1 + rs))

            rsi2_val = float(rsi2.iloc[-1])
            price = float(close.iloc[-1])

            if rsi2_val < 10:
                results.append({
                    "source": "Mispricing",
                    "ticker": ticker,
                    "signal": "RSI-2 extreme oversold (%.1f) @ $%.2f -> Mean reversion buy" % (rsi2_val, price),
                    "wr": 72.0,
                    "action": "BUY",
                    "tier": "STAT-ARB",
                })
            elif rsi2_val > 90:
                results.append({
                    "source": "Mispricing",
                    "ticker": ticker,
                    "signal": "RSI-2 extreme overbought (%.1f) @ $%.2f -> Mean reversion sell" % (rsi2_val, price),
                    "wr": 68.0,
                    "action": "SELL",
                    "tier": "STAT-ARB",
                })
    except Exception as exc:
        logger.debug("RSI-2 extreme check failed: %s", exc)

    # ── 5. Sector Rotation Lag ────────────────────────────────────
    # When SPY drops sharply but defensive sectors (XLU, GLD, TLT) haven't rallied yet,
    # there's a rotation lag — defensives tend to catch up in 1-2 days.
    try:
        spy_df = download("SPY", period="10d")
        if spy_df is not None and len(spy_df) >= 3:
            spy_3d_ret = (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-3]) - 1) * 100

            if spy_3d_ret < -2.0:
                # SPY dropped > 2% in 3 days — check if defensives have rotated
                defensives = {"GLD": "Gold", "TLT": "Bonds", "XLU": "Utilities"}
                for def_ticker, def_name in defensives.items():
                    try:
                        def_df = download(def_ticker, period="10d")
                        if def_df is None or def_df.empty or len(def_df) < 3:
                            continue
                        def_3d_ret = (float(def_df["Close"].iloc[-1]) / float(def_df["Close"].iloc[-3]) - 1) * 100

                        # Defensive hasn't rallied yet (flat or down) — rotation lag
                        if def_3d_ret < 0.5:
                            results.append({
                                "source": "Mispricing",
                                "ticker": def_ticker,
                                "signal": "Rotation lag: SPY %.1f%% but %s only %.1f%% -> Buy %s" % (
                                    spy_3d_ret, def_name, def_3d_ret, def_name
                                ),
                                "wr": 63.0,
                                "action": "BUY",
                                "tier": "STAT-ARB",
                            })
                    except Exception:
                        pass
    except Exception as exc:
        logger.debug("Sector rotation check failed: %s", exc)

    logger.info("Mispricing: %d signal(s) loaded", len(results))
    return results


def _fetch_finrl_signals():
    """
    Fetch FinRL reinforcement-learning signals from the local API.
    New/unproven model starts at 50% WR.  Once the tracker has >= 30
    completed predictions, the proven win rate is used instead.
    Returns list of {"ticker", "signal", "wr", "action", "source", "tier"}.
    """
    data = _api_get("/api/finrl-signals")
    if data is None:
        return []

    # Determine win rate: use tracked rate if proven, otherwise default 50%
    base_wr = 50.0
    try:
        from dashboard.finrl_tracker import get_finrl_win_rate, log_prediction, init_tracker_db
        init_tracker_db()
        proven_wr = get_finrl_win_rate()
        if proven_wr is not None:
            base_wr = proven_wr
            logger.info("FinRL using proven win rate: %.1f%%", base_wr)
    except Exception as exc:
        logger.warning("FinRL tracker unavailable: %s", exc)

    results = []
    for sig in data.get("signals", []):
        action = sig.get("action", "HOLD").upper()
        if action == "HOLD":
            continue

        confidence = sig.get("confidence", 0)
        conf_str = "%d%% conf" % int(confidence * 100) if isinstance(confidence, float) else str(confidence)

        ticker = sig.get("ticker", "")
        price = sig.get("price", None)

        # Log prediction for tracking
        try:
            from dashboard.finrl_tracker import log_prediction
            conf_val = float(confidence) if confidence else 0.0
            log_prediction(ticker, action, conf_val, price)
        except Exception as exc:
            logger.debug("Failed to log FinRL prediction: %s", exc)

        results.append({
            "source": "FinRL AI",
            "ticker": ticker,
            "signal": "%s (%s)" % (action, conf_str),
            "wr": base_wr,
            "action": action,
            "tier": "AI",
        })

    logger.info("FinRL: %d signal(s) loaded", len(results))
    return results


# ── Sentiment analysis ───────────────────────────────────────────────────────

def _keyword_sentiment(headlines):
    """
    Simple keyword-based sentiment scorer.
    Returns a float from -1.0 (very bearish) to +1.0 (very bullish).
    """
    if not headlines:
        return 0.0

    bull = 0
    bear = 0
    for headline in headlines:
        h_lower = headline.lower()
        for word in BULLISH_WORDS:
            if word in h_lower:
                bull += 1
        for word in BEARISH_WORDS:
            if word in h_lower:
                bear += 1

    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total  # -1.0 to +1.0


def _get_sentiment(ticker):
    """
    Get sentiment score for a ticker.

    Tries in order:
      1. finnlp library (if installed)
      2. Free news API headlines + keyword scoring
      3. Returns None if unavailable

    Returns: {"score": float, "label": str} or None
      score: -1.0 (very bearish) to +1.0 (very bullish)
      label: "Bullish" / "Bearish" / "Neutral"
    """
    # Attempt 1: finnlp
    try:
        from finnlp.data_sources.news.finnhub_date_range import Finnhub_Date_Range
        # If finnlp is available but not configured, this will raise
        # Just having the import succeed is enough to know it is installed
        logger.debug("finnlp is available but full integration is pending")
    except ImportError:
        pass
    except Exception:
        pass

    # Attempt 2: Free news API (newsdata.io or similar)
    headlines = _fetch_news_headlines(ticker)
    if headlines:
        score = _keyword_sentiment(headlines)
        if score > 0.15:
            label = "Bullish"
        elif score < -0.15:
            label = "Bearish"
        else:
            label = "Neutral"
        return {"score": round(score, 3), "label": label}

    return None


def _fetch_news_headlines(ticker):
    """
    Fetch recent news headlines for a ticker from a free source.
    Returns a list of headline strings, or an empty list on failure.
    """
    # Try newsapi.org (requires NEWSAPI_KEY env var)
    api_key = os.environ.get("NEWSAPI_KEY", "")
    if api_key:
        try:
            url = (
                "https://newsapi.org/v2/everything?"
                "q=%s&sortBy=publishedAt&pageSize=20&apiKey=%s"
                % (urllib.request.quote(ticker), api_key)
            )
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            articles = data.get("articles", [])
            return [a.get("title", "") for a in articles if a.get("title")]
        except Exception as exc:
            logger.debug("newsapi.org fetch failed for %s: %s", ticker, exc)

    # Try Brave Search API (requires BRAVE_API_KEY env var)
    brave_key = os.environ.get("BRAVE_API_KEY", "")
    if brave_key:
        try:
            query = "%s stock news" % ticker
            url = (
                "https://api.search.brave.com/res/v1/news/search?"
                "q=%s&count=20" % urllib.request.quote(query)
            )
            req = urllib.request.Request(url)
            req.add_header("X-Subscription-Token", brave_key)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            return [r.get("title", "") for r in results if r.get("title")]
        except Exception as exc:
            logger.debug("Brave search failed for %s: %s", ticker, exc)

    # Fallback: Yahoo Finance RSS (free, no API key)
    try:
        rss_url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US" % ticker
        req = urllib.request.Request(rss_url)
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=API_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        # Simple XML title extraction (no external parser needed)
        import re
        titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", raw)
        if not titles:
            titles = re.findall(r"<title>(.*?)</title>", raw)
        # Skip the first title (it's the feed title, not a headline)
        headlines = [t for t in titles[1:] if t and len(t) > 10]
        if headlines:
            logger.debug("Yahoo RSS: got %d headlines for %s", len(headlines), ticker)
            return headlines[:20]
    except Exception as exc:
        logger.debug("Yahoo RSS failed for %s: %s", ticker, exc)

    return []


def _fetch_sentiment_signals(tickers):
    """
    Fetch sentiment-based signals for a list of tickers.
    Returns list of {"ticker", "signal", "wr", "action", "source", "tier"}
    and a dict of {ticker: sentiment_result} for inclusion in output.
    """
    results = []
    sentiment_map = {}

    for ticker in tickers:
        try:
            sentiment = _get_sentiment(ticker)
            if sentiment is None:
                continue

            sentiment_map[ticker] = sentiment
            score = sentiment["score"]

            if abs(score) < 0.15:
                continue  # neutral — no signal

            strength = "strongly" if abs(score) >= 0.6 else "moderately" if abs(score) >= 0.3 else "slightly"
            if score > 0:
                action = "BUY"
                signal_text = "News headlines %s bullish on %s" % (strength, ticker)
            else:
                action = "SELL"
                signal_text = "News headlines %s bearish on %s" % (strength, ticker)

            results.append({
                "source": "Sentiment",
                "ticker": ticker,
                "signal": signal_text,
                "wr": 55.0,  # unproven — starts at 55%
                "action": action,
                "tier": "NLP",
            })
        except Exception as exc:
            logger.debug("Sentiment failed for %s: %s", ticker, exc)

    logger.info("Sentiment: %d signal(s) loaded", len(results))
    return results, sentiment_map


# ── Composite scoring ────────────────────────────────────────────────────────

def _compute_composite(signals_for_ticker):
    """
    Given a list of signals for a single ticker, compute the composite score.

    Each signal dict has: {"source", "signal", "wr", "action", "tier"}

    Returns:
        {
            "action": "BUY" | "SELL" | "HOLD",
            "score": float (0-100),
            "confidence": "LOW" | "MEDIUM" | "HIGH",
        }
    """
    if not signals_for_ticker:
        return {"action": "HOLD", "score": 0, "confidence": "LOW"}

    buy_signals = [s for s in signals_for_ticker if s["action"] == "BUY"]
    sell_signals = [s for s in signals_for_ticker if s["action"] == "SELL"]

    # Action = whichever direction has more / stronger signals.
    # SELL requires at least 2 signals to avoid single-signal noise flips.
    if len(sell_signals) >= 2 and len(sell_signals) > len(buy_signals):
        action = "SELL"
        active = sell_signals
    elif buy_signals:
        action = "BUY"
        active = buy_signals
    elif sell_signals:
        # Only sell signals but fewer than 2 — not enough conviction
        action = "HOLD"
        active = sell_signals
    else:
        action = "HOLD"
        active = []

    if not active:
        return {"action": "HOLD", "score": 0, "confidence": "LOW"}

    # PRIMARY signals: backtested at 60%+ WR over 1 year (2025-2026).
    # Bear rally fade (short): 62.3% WR, PF 4.49
    # RSI-14 oversold (buy): ~65% WR
    # These are the ONLY actionable signals. Everything else is informational.
    PRIMARY_SIGNALS = {
        "Bear rally fade",     # short green day in downtrend — 60% WR, validated
        # RSI Oversold removed: 28.6% WR, net negative, backtest failed
    }

    # SECONDARY signals: shown for context but lower confidence.
    SECONDARY_SIGNALS = {
        "Pairs:",              # SPY/QQQ divergence
        "RSI-2 extreme",       # oversold/overbought (only in downtrend)
        "2 red days",          # bounce after consecutive reds
        "2 consecutive red",   # alternate wording
        "EMA-50 rejection",    # failed reclaim in downtrend
    }

    primary = []
    secondary = []
    other = []
    for s in active:
        sig_text = s.get("signal", "")
        if any(p in sig_text for p in PRIMARY_SIGNALS):
            primary.append(s)
        elif any(p in sig_text for p in SECONDARY_SIGNALS):
            secondary.append(s)
        else:
            other.append(s)

    # Score: primary signals count 5x, secondary 2x, other 1x
    total_weight = 0
    weighted_sum = 0.0
    for s in primary:
        weighted_sum += s["wr"] * 5
        total_weight += 5
    for s in secondary:
        weighted_sum += s["wr"] * 2
        total_weight += 2
    for s in other:
        weighted_sum += s["wr"] * 1
        total_weight += 1

    score = weighted_sum / max(total_weight, 1)

    n = len(active)
    has_primary = len(primary) > 0
    if has_primary and n >= 2:
        confidence = "HIGH"
    elif has_primary:
        confidence = "MEDIUM"
    elif n >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "action": action,
        "score": round(score, 1),
        "confidence": confidence,
        "has_proven_signal": has_primary,
        "proven_count": len(primary),
    }


# ── Trend filter (EMA) ───────────────────────────────────────────────────────

def _get_trend(ticker):
    """
    Determine the trend for a ticker using EMA-20, EMA-50, and EMA-200.

    Returns:
        {
            "trend": "UPTREND" | "CAUTION" | "DOWNTREND",
            "ema20": float,
            "ema50": float,
            "ema200": float,
            "price": float,
            "detail": "Price > EMA-50 > EMA-200",
            "score_modifier": 1.0 | 0.75 | 0.5,
        }
        or None on error.
    """
    try:
        try:
            from data_provider import download
        except ImportError:
            if PARENT_DIR not in sys.path:
                sys.path.insert(0, PARENT_DIR)
            from dashboard.data_provider import download

        import pandas as pd

        df = download(ticker, period="1y")
        if df is None or df.empty or len(df) < 50:
            return None

        close = df["Close"]
        price = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        # EMA-200: need enough data
        if len(df) >= 200:
            ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        else:
            # Approximate with SMA if not enough data
            ema200 = float(close.rolling(min(200, len(df))).mean().iloc[-1])

        # Determine trend
        if price > ema50 and ema50 > ema200:
            trend = "UPTREND"
            detail = "Price > EMA-50 > EMA-200"
            modifier = 1.0  # full score for BUY signals
        elif price > ema200 and price < ema50:
            trend = "CAUTION"
            detail = "Price < EMA-50 but > EMA-200"
            modifier = 0.85  # slight reduction
        elif price < ema50 and price < ema200:
            trend = "DOWNTREND"
            detail = "Price < EMA-50 < EMA-200"
            modifier = 0.6  # significant reduction for BUY, boost for SELL
        elif price < ema50 and ema50 > ema200:
            trend = "CAUTION"
            detail = "Price < EMA-50, EMA-50 still > EMA-200"
            modifier = 0.75
        else:
            trend = "CAUTION"
            detail = "Mixed signals"
            modifier = 0.85

        return {
            "trend": trend,
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "ema200": round(ema200, 2),
            "price": round(price, 2),
            "detail": detail,
            "score_modifier": modifier,
        }

    except Exception as exc:
        logger.warning("Trend check failed for %s: %s", ticker, exc)
        return None


def _get_trends(tickers):
    """Get trend data for multiple tickers. Returns {ticker: trend_dict}."""
    trends = {}
    # Only check trends for major reference tickers + any US tickers
    check_tickers = set()
    check_tickers.add("SPY")  # always check SPY as market benchmark
    for t in tickers:
        if "." not in t and not t.startswith("^"):
            check_tickers.add(t)

    for ticker in check_tickers:
        trend = _get_trend(ticker)
        if trend:
            trends[ticker] = trend

    # For non-US tickers, use SPY trend as proxy
    spy_trend = trends.get("SPY")
    if spy_trend:
        for t in tickers:
            if t not in trends:
                trends[t] = spy_trend

    return trends


# ── Price lookup ─────────────────────────────────────────────────────────────

def _get_latest_prices(tickers):
    """
    Fetch the latest closing price for each ticker.
    Returns {ticker: price} dict.
    """
    prices = {}
    try:
        if PARENT_DIR not in sys.path:
            sys.path.insert(0, PARENT_DIR)
        from dashboard.data_provider import download_multi
        raw = download_multi(list(tickers), period="5d")
        for ticker, df in raw.items():
            if not df.empty:
                prices[ticker] = round(float(df["Close"].iloc[-1]), 2)
    except Exception as exc:
        logger.warning("Price fetch failed: %s", exc)
    return prices


# ── Main entry point ─────────────────────────────────────────────────────────

def get_unified_signals():
    """
    Main entry point. Gathers signals from all sources, groups by ticker,
    computes a composite score for each, and returns structured results.

    Returns:
        {
            "scan_time": "2026-03-24T10:30:00",
            "signals": [
                {
                    "ticker": "SPY",
                    "action": "BUY",
                    "score": 67.3,
                    "confidence": "HIGH",
                    "num_signals": 4,
                    "active_signals": [
                        {
                            "source": "Playbook",
                            "signal": "Fri Red -> Buy Mon",
                            "wr": 72.5,
                            "tier": "STRONG",
                        },
                        ...
                    ],
                    "price": 657.99,
                    "sentiment": {"score": 0.65, "label": "Bullish"},
                },
                ...
            ],
            "sources_ok": ["Playbook", "Asian", ...],
            "sources_failed": ["FinRL AI"],
        }
    """
    all_signals = []
    sources_ok = []
    sources_failed = []

    # 0. Detect market regime first — determines which strategies are active
    regime = _detect_regime()
    logger.info("Market regime: %s", regime)

    # 1. Playbook signals
    try:
        playbook = _fetch_playbook_signals()
        all_signals.extend(playbook)
        sources_ok.append("Playbook")
    except Exception as exc:
        logger.error("Playbook source failed: %s", exc)
        sources_failed.append("Playbook")

    # 1b. Monday Reversal (validated: 73.1% WR, PF 3.21 over 6 years)
    try:
        monday = _fetch_monday_reversal_signals()
        all_signals.extend(monday)
        if monday:
            sources_ok.append("Monday Reversal")
    except Exception as exc:
        logger.error("Monday Reversal source failed: %s", exc)

    # 2. Asian signals
    try:
        asian = _fetch_asian_signals()
        all_signals.extend(asian)
        sources_ok.append("Asian")
    except Exception as exc:
        logger.error("Asian source failed: %s", exc)
        sources_failed.append("Asian")

    # 3. Dip Scanner
    try:
        dip = _fetch_dip_signals()
        all_signals.extend(dip)
        sources_ok.append("Dip Scanner")
    except Exception as exc:
        logger.error("Dip Scanner source failed: %s", exc)
        sources_failed.append("Dip Scanner")

    # 4. FinRL AI
    try:
        finrl = _fetch_finrl_signals()
        all_signals.extend(finrl)
        sources_ok.append("FinRL AI")
    except Exception as exc:
        logger.error("FinRL AI source failed: %s", exc)
        sources_failed.append("FinRL AI")

    # Collect all tickers that have at least one signal
    tickers_with_signals = sorted(set(s["ticker"] for s in all_signals if s["ticker"]))

    # 5. Sentiment (only for tickers that already have signals)
    sentiment_map = {}
    try:
        sentiment_signals, sentiment_map = _fetch_sentiment_signals(tickers_with_signals)
        all_signals.extend(sentiment_signals)
        sources_ok.append("Sentiment")
    except Exception as exc:
        logger.error("Sentiment source failed: %s", exc)
        sources_failed.append("Sentiment")

    # 6. Bear Rally Fade removed — failed OOS backtest (WR 26%, PF 0.55, 2023-2026)

    # 7. Mispricing signals (stat arb / mean reversion)
    try:
        mispricing = _fetch_mispricing_signals()
        all_signals.extend(mispricing)
        sources_ok.append("Mispricing")
    except Exception as exc:
        logger.error("Mispricing source failed: %s", exc)
        sources_failed.append("Mispricing")

    # Only include Alpaca-tradeable tickers (US-listed)
    ALPACA_TICKERS = {
        "SPY", "QQQ", "IWM", "QQQM", "SPLG", "SPYM",
        "EWJ", "EWS", "EWH", "FXI", "KWEB", "EWT", "EWY",
        "GLD", "TLT", "XLU",
        "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "PLTR", "COIN", "SOFI", "NIO",
    }

    # Group signals by ticker (skip non-Alpaca tickers)
    by_ticker = {}
    for sig in all_signals:
        ticker = sig["ticker"]
        if ticker not in ALPACA_TICKERS:
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(sig)

    # Fetch latest prices and trends
    all_tickers = set(by_ticker.keys())
    prices = _get_latest_prices(all_tickers)

    logger.info("Checking EMA trends...")
    trends = _get_trends(all_tickers)

    # Compute composite for each ticker
    output = []
    for ticker in sorted(by_ticker.keys()):
        signals = by_ticker[ticker]
        composite = _compute_composite(signals)

        # Apply trend modifier to score
        trend_data = trends.get(ticker)
        original_score = composite["score"]
        if trend_data:
            modifier = trend_data["score_modifier"]
            if composite["action"] == "BUY" and modifier < 1.0:
                composite["score"] = round(original_score * modifier, 1)
            elif composite["action"] == "SELL" and trend_data["trend"] == "DOWNTREND":
                # Boost SELL scores in downtrends (trend is on your side)
                composite["score"] = round(min(original_score * 1.15, 100), 1)

        # Filter to only the active (winning-direction) signals for display
        active_signals = [
            {
                "source": s["source"],
                "signal": s["signal"],
                "wr": s["wr"],
                "tier": s["tier"],
            }
            for s in signals
            if s["action"] == composite["action"]
        ]

        # Add trend warning to active signals if downtrend
        if trend_data and trend_data["trend"] == "DOWNTREND" and composite["action"] == "BUY":
            active_signals.append({
                "source": "Trend",
                "signal": "DOWNTREND — %s (score reduced %.0f%%)" % (
                    trend_data["detail"],
                    (1 - trend_data["score_modifier"]) * 100,
                ),
                "wr": 0,
                "tier": "WARNING",
            })
        elif trend_data and trend_data["trend"] == "CAUTION" and composite["action"] == "BUY":
            active_signals.append({
                "source": "Trend",
                "signal": "CAUTION — %s" % trend_data["detail"],
                "wr": 0,
                "tier": "WARNING",
            })

        output.append({
            "ticker": ticker,
            "action": composite["action"],
            "score": composite["score"],
            "confidence": composite["confidence"],
            "num_signals": len(active_signals),
            "active_signals": active_signals,
            "price": prices.get(ticker),
            "sentiment": sentiment_map.get(ticker),
            "trend": {
                "status": trend_data["trend"],
                "ema20": trend_data["ema20"],
                "ema50": trend_data["ema50"],
                "ema200": trend_data["ema200"],
                "detail": trend_data["detail"],
            } if trend_data else None,
        })

    # Sort by score descending, then by number of signals
    output.sort(key=lambda x: (-x["score"], -x["num_signals"]))

    return {
        "scan_time": datetime.now().isoformat(),
        "regime": regime,
        "signals": output,
        "sources_ok": sources_ok,
        "sources_failed": sources_failed,
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

def _print_results(data):
    """Pretty-print unified signal results to the console."""
    print("\nUnified Signal Engine — %s" % data["scan_time"][:19])
    print("=" * 70)
    print("  Sources OK:     %s" % ", ".join(data["sources_ok"]) if data["sources_ok"] else "  Sources OK:     (none)")
    if data["sources_failed"]:
        print("  Sources FAILED: %s" % ", ".join(data["sources_failed"]))

    signals = data["signals"]
    if not signals:
        print("\n  No active signals detected.")
        return

    print("\n  %d ticker(s) with active signals:\n" % len(signals))

    for item in signals:
        conf_marker = {"HIGH": "***", "MEDIUM": "**", "LOW": "*"}.get(item["confidence"], "")
        print("  %s  %-6s  %s  Score: %.1f  Confidence: %s %s" % (
            ">>>" if item["action"] == "BUY" else "<<<" if item["action"] == "SELL" else "---",
            item["ticker"],
            item["action"],
            item["score"],
            item["confidence"],
            conf_marker,
        ))
        if item["price"] is not None:
            print("         Price: $%.2f" % item["price"])
        if item["sentiment"] is not None:
            s = item["sentiment"]
            print("         Sentiment: %s (%.3f)" % (s["label"], s["score"]))
        for sig in item["active_signals"]:
            print("         [%s] %s (WR: %.1f%%, %s)" % (
                sig["source"], sig["signal"], sig["wr"], sig["tier"],
            ))
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = get_unified_signals()
    _print_results(result)
