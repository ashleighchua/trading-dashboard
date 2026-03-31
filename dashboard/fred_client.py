"""
FRED Macro Regime Filter
========================
Fetches macro indicators from FRED (Federal Reserve Economic Data) and
classifies the current market regime as RISK_OFF, NEUTRAL, or RISK_ON.

Used as a pre-filter in premarket_scanner.py:
  - Bear rally fades: fire in RISK_OFF or NEUTRAL only
  - Monday reversal: fires in any regime (SPY-specific, already high WR)

Yield Curve Analysis (layered):
  T10Y2Y  — 10Y-2Y spread (classic inversion signal, widely watched)
  T10Y3M  — 10Y-3M spread (Fed/Estrella-Mishkin preferred, earlier signal)
  Slope momentum — 30-day change in T10Y2Y (steepening = risk-on rally forming)
  Inversion duration — consecutive days T10Y2Y < 0 (short inversion = noise)

Other Indicators:
  VIXCLS  — VIX daily close (fear gauge)
  FEDFUNDS — Effective Fed funds rate (monetary stance)

Regime rules:
  RISK_OFF : VIX > 25
           OR T10Y3M < -0.50 (deep inversion, Fed's preferred indicator)
           OR (T10Y2Y < -0.30 AND inversion_days >= 14)   (sustained, not noise)
           OR slope_30d < -0.30 (curve deteriorating fast)

  RISK_ON  : VIX < 15
           AND T10Y3M > 0.20 (curve clearly positive)
           AND slope_30d > -0.10 (not actively inverting)
           AND FEDFUNDS trending down or flat

  NEUTRAL  : everything else
             NOTE: rapid curve steepening (slope_30d > +0.50) suppresses fades
             even in NEUTRAL — steepener = risk-on rally forming, shorts get squeezed

Free FRED API: https://fred.stlouisfed.org/docs/api/fred/
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Series IDs
YIELD_CURVE_10Y2Y = "T10Y2Y"   # 10Y-2Y spread — classic inversion
YIELD_CURVE_10Y3M = "T10Y3M"   # 10Y-3M spread — Fed's preferred recession indicator
FED_FUNDS         = "FEDFUNDS"  # Effective Fed funds rate (monthly)
VIX_CLOSE         = "VIXCLS"   # VIX daily close


def _fetch_series(series_id: str, lookback_days: int = 90) -> list:
    """
    Fetch FRED series observations for the last `lookback_days` days.
    Returns list of {"date": "YYYY-MM-DD", "value": float} dicts,
    sorted ascending. Missing/invalid values are filtered out.
    """
    token = os.environ.get("FRED_API_KEY", "")
    if not token:
        raise RuntimeError("FRED_API_KEY not set in environment")

    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    url = (
        "{base}"
        "?series_id={sid}"
        "&observation_start={s}"
        "&observation_end={e}"
        "&api_key={key}"
        "&file_type=json"
    ).format(
        base=FRED_BASE,
        sid=series_id,
        s=start.strftime("%Y-%m-%d"),
        e=end.strftime("%Y-%m-%d"),
        key=token,
    )

    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())

    observations = []
    for obs in data.get("observations", []):
        try:
            val = float(obs["value"])
            observations.append({"date": obs["date"], "value": val})
        except (ValueError, KeyError):
            pass  # skip "." (missing) values

    return sorted(observations, key=lambda x: x["date"])


def _latest(series_id: str, lookback_days: int = 90) -> Optional[float]:
    """Return the most recent valid value for a FRED series."""
    try:
        obs = _fetch_series(series_id, lookback_days)
        return obs[-1]["value"] if obs else None
    except Exception as e:
        logger.warning("FRED fetch failed for %s: %s", series_id, e)
        return None


def _fed_funds_trend() -> str:
    """
    Return 'rising', 'falling', or 'flat' based on last 3 months of FEDFUNDS.
    Uses monthly data — compare earliest vs latest in the window.
    """
    try:
        obs = _fetch_series(FED_FUNDS, lookback_days=120)
        if len(obs) < 2:
            return "flat"
        delta = obs[-1]["value"] - obs[0]["value"]
        if delta > 0.10:
            return "rising"
        elif delta < -0.10:
            return "falling"
        return "flat"
    except Exception as e:
        logger.warning("FRED FEDFUNDS trend failed: %s", e)
        return "flat"


def _yield_curve_analysis() -> dict:
    """
    Deep yield curve analysis using both T10Y2Y and T10Y3M.

    Returns:
        t10y2y:           float | None — latest 10Y-2Y spread
        t10y3m:           float | None — latest 10Y-3M spread (Fed's preferred)
        slope_30d:        float | None — 30-day change in T10Y2Y (+ = steepening)
        inversion_days:   int — consecutive calendar days T10Y2Y has been < 0
        curve_shape:      "deeply_inverted" | "inverted" | "flat" | "normal" | "steep"
        steepening_rally: bool — curve normalising rapidly (dangerous for shorts)

    Interpretation guide:
        slope_30d > +0.50  → rapid steepener → risk-on rally forming → suppress shorts
        slope_30d < -0.30  → rapid inversion → regime deteriorating → favour shorts
        inversion_days < 14 → brief dip, probably noise
        T10Y3M < -0.50     → deep inversion, Fed's preferred recession signal
    """
    result = {
        "t10y2y": None,
        "t10y3m": None,
        "slope_30d": None,
        "inversion_days": 0,
        "curve_shape": "unknown",
        "steepening_rally": False,
    }

    # ── T10Y2Y: level + 30-day momentum + inversion duration ─────────────────
    try:
        obs_10y2y = _fetch_series(YIELD_CURVE_10Y2Y, lookback_days=90)
        if obs_10y2y:
            latest_val = obs_10y2y[-1]["value"]
            result["t10y2y"] = latest_val

            # 30-day slope: compare latest vs value ~30 days ago
            cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            older = [o for o in obs_10y2y if o["date"] <= cutoff]
            if older:
                result["slope_30d"] = round(latest_val - older[-1]["value"], 3)

            # Inversion duration: count consecutive days (from end) where value < 0
            days = 0
            for obs in reversed(obs_10y2y):
                if obs["value"] < 0:
                    days += 1
                else:
                    break
            result["inversion_days"] = days
    except Exception as e:
        logger.warning("FRED T10Y2Y analysis failed: %s", e)

    # ── T10Y3M: level only ────────────────────────────────────────────────────
    try:
        obs_10y3m = _fetch_series(YIELD_CURVE_10Y3M, lookback_days=30)
        if obs_10y3m:
            result["t10y3m"] = obs_10y3m[-1]["value"]
    except Exception as e:
        logger.warning("FRED T10Y3M fetch failed: %s", e)

    # ── Curve shape classification ────────────────────────────────────────────
    t2y = result["t10y2y"]
    t3m = result["t10y3m"]
    # Use T10Y3M if available (more sensitive), else fall back to T10Y2Y
    primary = t3m if t3m is not None else t2y
    if primary is not None:
        if primary < -0.50:
            result["curve_shape"] = "deeply_inverted"
        elif primary < 0:
            result["curve_shape"] = "inverted"
        elif primary < 0.30:
            result["curve_shape"] = "flat"
        elif primary < 1.0:
            result["curve_shape"] = "normal"
        else:
            result["curve_shape"] = "steep"

    # ── Steepening rally detection ────────────────────────────────────────────
    # Rapid steepening after inversion = risk-on recovery forming.
    # Bears get squeezed when the curve normalises fast.
    slope = result["slope_30d"]
    if slope is not None and slope > 0.50:
        result["steepening_rally"] = True

    return result


def get_macro_regime() -> dict:
    """
    Classify current macro regime using layered yield curve analysis.

    Returns:
        {
            "regime":           "RISK_OFF" | "NEUTRAL" | "RISK_ON",
            "vix":              float | None,
            "yield_curve":      float | None,   # T10Y2Y (backward compat)
            "t10y3m":           float | None,   # NEW: 10Y-3M spread
            "slope_30d":        float | None,   # NEW: 30-day momentum
            "inversion_days":   int,            # NEW: duration of inversion
            "curve_shape":      str,            # NEW: descriptive shape
            "steepening_rally": bool,           # NEW: rapid normalisation flag
            "fed_funds":        float | None,
            "fed_trend":        "rising" | "falling" | "flat",
            "reason":           str,
        }
    """
    vix = _latest(VIX_CLOSE)
    curve = _yield_curve_analysis()
    fed_funds = _latest(FED_FUNDS)
    fed_trend = _fed_funds_trend()

    t10y2y       = curve["t10y2y"]
    t10y3m       = curve["t10y3m"]
    slope_30d    = curve["slope_30d"]
    inv_days     = curve["inversion_days"]
    steep_rally  = curve["steepening_rally"]

    reasons = []
    risk_off = False
    risk_on = False

    # ── Risk-off signals ─────────────────────────────────────────────────────

    if vix is not None and vix > 25:
        risk_off = True
        reasons.append("VIX={:.1f} (elevated fear)".format(vix))

    if t10y3m is not None and t10y3m < -0.50:
        risk_off = True
        reasons.append("T10Y3M={:.2f}% (deep inversion — Fed recession signal)".format(t10y3m))

    if t10y2y is not None and t10y2y < -0.30 and inv_days >= 14:
        risk_off = True
        reasons.append(
            "T10Y2Y={:.2f}% inverted for {:d} days (sustained, not noise)".format(t10y2y, inv_days)
        )

    if slope_30d is not None and slope_30d < -0.30:
        risk_off = True
        reasons.append("Curve deteriorating fast ({:+.2f}% in 30d)".format(slope_30d))

    # ── Risk-on signals ──────────────────────────────────────────────────────
    # Require: calm VIX, positive T10Y3M (not inverted), curve not actively inverting, Fed easing
    if (
        vix is not None and vix < 15
        and (t10y3m is None or t10y3m > 0.20)
        and (slope_30d is None or slope_30d > -0.10)
        and fed_trend in ("falling", "flat")
    ):
        risk_on = True
        reasons.append(
            "VIX={:.1f} (calm), curve={:.2f}% (positive), Fed {}".format(
                vix,
                t10y3m if t10y3m is not None else (t10y2y or 0),
                fed_trend,
            )
        )

    # ── Steepening rally note (NEUTRAL regime, but suppress shorts) ──────────
    # Not risk_off or risk_on, but rapid steepening kills fade edge.
    if steep_rally and not risk_off and not risk_on:
        reasons.append(
            "Curve steepening fast ({:+.2f}% in 30d) — risk-on rally forming, fade edge suppressed".format(
                slope_30d
            )
        )

    # ── Classify ─────────────────────────────────────────────────────────────
    if risk_off:
        regime = "RISK_OFF"
    elif risk_on:
        regime = "RISK_ON"
    else:
        regime = "NEUTRAL"
        if not reasons:
            parts = []
            if vix is not None:
                parts.append("VIX={:.1f}".format(vix))
            if t10y2y is not None:
                parts.append("T10Y2Y={:.2f}%".format(t10y2y))
            if t10y3m is not None:
                parts.append("T10Y3M={:.2f}%".format(t10y3m))
            if fed_funds is not None:
                parts.append("FFR={:.2f}%".format(fed_funds))
            reasons.append(", ".join(parts) if parts else "no FRED data")

    return {
        "regime":           regime,
        "vix":              vix,
        "yield_curve":      t10y2y,     # backward compat key
        "t10y3m":           t10y3m,
        "slope_30d":        slope_30d,
        "inversion_days":   inv_days,
        "curve_shape":      curve["curve_shape"],
        "steepening_rally": steep_rally,
        "fed_funds":        fed_funds,
        "fed_trend":        fed_trend,
        "reason":           "; ".join(reasons),
    }


def regime_allows_short(regime: str, regime_data: Optional[dict] = None) -> bool:
    """
    Bear rally fades (shorts) are allowed in RISK_OFF and NEUTRAL regimes,
    UNLESS the curve is steepening rapidly (risk-on rally forming).

    In RISK_ON (everything is going up), fading bounces has negative edge.
    During a steepening rally (curve normalising fast), shorts also get squeezed.
    """
    if regime == "RISK_ON":
        return False
    # Suppress shorts during rapid steepening even in NEUTRAL
    if regime_data and regime_data.get("steepening_rally"):
        return False
    return True


def regime_allows_long(regime: str, regime_data: Optional[dict] = None) -> bool:
    """
    Return True if the macro regime allows long trades.

    RISK_OFF blocks longs — pullbacks extend in bear markets, not bounce.
    NEUTRAL and RISK_ON allow longs. Steepening rally also allows longs.
    """
    if regime == "RISK_OFF":
        return False
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = get_macro_regime()
    print("\nMacro Regime: {}".format(result["regime"]))
    print("  VIX:              {}".format(result["vix"]))
    print("  T10Y2Y:           {}%".format(result["yield_curve"]))
    print("  T10Y3M:           {}%  (Fed preferred)".format(result["t10y3m"]))
    print("  Slope 30d:        {:+}%  (+ = steepening)".format(result["slope_30d"]))
    print("  Inversion days:   {}".format(result["inversion_days"]))
    print("  Curve shape:      {}".format(result["curve_shape"]))
    print("  Steepening rally: {}".format(result["steepening_rally"]))
    print("  Fed Funds:        {}%".format(result["fed_funds"]))
    print("  Fed Trend:        {}".format(result["fed_trend"]))
    print("  Reason:           {}".format(result["reason"]))
    print("\nShorts allowed: {}".format(regime_allows_short(result["regime"], result)))
