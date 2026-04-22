# Scanner Fade Watchlist + EMA-200 Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the premarket scanner's bear-rally-fade strategy by narrowing the fade watchlist to PLTR+FXI and adding an EMA-200 long-term downtrend confirmation filter.

**Architecture:** Single-file change to `dashboard/premarket_scanner.py`. Three edits: (1) shrink `FADE_WATCHLIST`, (2) extend `bar_limit` to 260 for fade tickers, (3) compute EMA-200 inside `score_ticker()` and gate the fade signal on `price < EMA-200`.

**Tech Stack:** Python 3.9, Alpaca SDK, existing `fetch_bars()` helper, no new dependencies.

---

## File Map

| File | What changes |
|---|---|
| `dashboard/premarket_scanner.py` | All three edits — watchlist, bar_limit, EMA-200 gate |

No other files change.

---

### Task 1: Shrink FADE_WATCHLIST to PLTR + FXI

**Files:**
- Modify: `dashboard/premarket_scanner.py:64`

Context: `FADE_WATCHLIST` is defined at module level around line 64. It currently includes five tickers. KWEB, TSLA, and AMD have been backtested over 6 years and all fail the minimum quality bar (WR < 52%, PF < 1.5). They are removed.

- [ ] **Step 1: Open the file and locate the constant**

Open `dashboard/premarket_scanner.py`. Find this block (around line 62–64):

```python
# Tickers where bear rally fade has positive PF over 6-year backtest.
# SPY, QQQ, IWM, AAPL, NVDA, GLD, TLT, EWJ excluded — negative contribution.
FADE_WATCHLIST = ["KWEB", "FXI", "PLTR", "TSLA", "AMD"]
```

- [ ] **Step 2: Replace the constant and its comment**

Replace those three lines with:

```python
# Tickers where bear rally fade has positive PF over 6-year backtest.
# PLTR+FXI: 54.1% WR, PF 1.88 (74 trades, 2020-2026, with EMA-200 filter)
# Removed: KWEB (WR 29.8%, PF 0.61), TSLA (WR 40.8%, PF 1.14), AMD (WR 39.0%, PF 1.33)
FADE_WATCHLIST = ["PLTR", "FXI"]
```

- [ ] **Step 3: Verify**

Run:
```bash
grep -n "FADE_WATCHLIST" /Users/ashleighchua/trading\ analyses/dashboard/premarket_scanner.py
```

Expected (two lines):
```
64:FADE_WATCHLIST = ["PLTR", "FXI"]
198:    if (symbol in FADE_WATCHLIST
```

- [ ] **Step 4: Commit**

```bash
cd /Users/ashleighchua/trading\ analyses
git add dashboard/premarket_scanner.py
git commit -m "feat: narrow fade watchlist to PLTR+FXI (drop KWEB/TSLA/AMD)"
```

---

### Task 2: Extend bar_limit to 260 for FADE_WATCHLIST tickers

**Files:**
- Modify: `dashboard/premarket_scanner.py:172–175`

Context: `score_ticker()` currently fetches only 25 bars for PLTR and FXI. EMA-200 needs 201+ bars to be fully warmed up. The 260-bar pattern already exists for `LONG_WATCHLIST`.

- [ ] **Step 1: Find the bar_limit line and its comment block**

Inside `score_ticker()` (around lines 172–175):
```python
    # Long watchlist needs extra history for EMA warmup:
    #   GLD  → EMA(20,50):  needs 51+ bars
    #   NVDA → EMA(50,200): needs 201+ bars
    bar_limit = 260 if symbol in LONG_WATCHLIST else 25
```

- [ ] **Step 2: Replace with updated comment and extended condition**

```python
    # Extra history needed for EMA warmup:
    #   GLD       → EMA(20,50):   needs 51+ bars
    #   NVDA      → EMA(50,200):  needs 201+ bars
    #   PLTR, FXI → EMA-200 gate: needs 201+ bars
    bar_limit = 260 if symbol in LONG_WATCHLIST or symbol in FADE_WATCHLIST else 25
```

- [ ] **Step 3: Verify**

```bash
grep -n "bar_limit" /Users/ashleighchua/trading\ analyses/dashboard/premarket_scanner.py
```

Expected:
```
175:    bar_limit = 260 if symbol in LONG_WATCHLIST or symbol in FADE_WATCHLIST else 25
```

- [ ] **Step 4: Commit**

```bash
cd /Users/ashleighchua/trading\ analyses
git add dashboard/premarket_scanner.py
git commit -m "feat: fetch 260 bars for PLTR+FXI to support EMA-200 calculation"
```

---

### Task 3: Add EMA-200 gate to the bear-rally-fade signal

**Files:**
- Modify: `dashboard/premarket_scanner.py:163–215`

Context: `bars` is already in scope inside `score_ticker()` — a list of dicts with key `"c"` for close price. We compute EMA-200 from it using a standard EMA loop (k = 2/201), add a helper function, and gate the fade condition on `price < EMA-200`.

- [ ] **Step 1: Add compute_ema200 helper just before score_ticker**

Find the comment line `# ---------------------------------------------------------------------------` that precedes `def score_ticker(` (around line 163). Insert this function immediately before it:

```python
def compute_ema200(bars):
    """
    Compute EMA-200 from a list of bar dicts (key 'c' = close).
    Returns float or None if fewer than 201 bars (not enough warmup).
    """
    closes = [b["c"] for b in bars]
    if len(closes) < 201:
        return None
    k = 2.0 / 201.0
    ema = closes[0]
    for c in closes[1:]:
        ema = c * k + ema * (1.0 - k)
    return ema


```

- [ ] **Step 2: Call compute_ema200 inside score_ticker after full_analysis**

Find this block (around lines 186–191):
```python
    trend = analysis.get("trend", {})
    rsi = analysis.get("rsi_14")
    conviction = analysis.get("conviction", 0)
    current_price = analysis.get("current_price")

    setup = None
```

Add `ema200 = compute_ema200(bars)` after `current_price`:
```python
    trend = analysis.get("trend", {})
    rsi = analysis.get("rsi_14")
    conviction = analysis.get("conviction", 0)
    current_price = analysis.get("current_price")
    ema200 = compute_ema200(bars)

    setup = None
```

- [ ] **Step 3: Add price_below_ema200 to the fade condition**

Find the bear-rally-fade `if` block (around lines 198–201):
```python
    if (symbol in FADE_WATCHLIST
            and trend.get("trend") == "downtrend"
            and trend.get("bear_rally")
            and regime_allows_short(_regime, _regime_data)):
```

Replace with:
```python
    price_below_ema200 = (ema200 is not None and current_price is not None
                          and current_price < ema200)
    if (symbol in FADE_WATCHLIST
            and trend.get("trend") == "downtrend"
            and trend.get("bear_rally")
            and price_below_ema200
            and regime_allows_short(_regime, _regime_data)):
```

- [ ] **Step 4: Update the thesis string to include EMA-200 value**

Find the `thesis` key inside that same `if` block (around line 210):
```python
            "thesis": "Bear rally fade: {} in downtrend, bounced {:.1f}% off recent low, RSI {:.0f}".format(
                symbol, trend.get("recent_bounce_pct", 0), rsi
            ),
```

Replace with:
```python
            "thesis": "Bear rally fade: {} below EMA-200 (${:.2f}), bounced {:.1f}% off recent low, RSI {:.0f}".format(
                symbol, ema200, trend.get("recent_bounce_pct", 0), rsi
            ),
```

- [ ] **Step 5: Verify all ema200 references are consistent**

```bash
grep -n "ema200\|price_below_ema200\|compute_ema200" /Users/ashleighchua/trading\ analyses/dashboard/premarket_scanner.py
```

Expected (5 lines):
```
163:def compute_ema200(bars):
191:    ema200 = compute_ema200(bars)
198:    price_below_ema200 = (ema200 is not None and current_price is not None
201:            and price_below_ema200
211:            "thesis": "Bear rally fade: {} below EMA-200 (${:.2f}), ...
```

- [ ] **Step 6: Commit**

```bash
cd /Users/ashleighchua/trading\ analyses
git add dashboard/premarket_scanner.py
git commit -m "feat: add EMA-200 gate to bear-rally-fade — 54.1% WR, PF 1.88 (74 trades, 6yr)"
```

---

### Task 4: Smoke-test end-to-end

**Files:**
- Read: `dashboard/premarket_scanner.py`

- [ ] **Step 1: Confirm KWEB/TSLA/AMD never produce a setup**

```bash
cd /Users/ashleighchua/trading\ analyses && python3 - <<'EOF' 2>&1 | grep -v OpenSSL | grep -v warnings | grep -v urllib3
import sys, os
sys.path.insert(0, "dashboard")
os.chdir("dashboard")
from dotenv import load_dotenv
load_dotenv("../.env")
from premarket_scanner import score_ticker, FADE_WATCHLIST

print("FADE_WATCHLIST:", FADE_WATCHLIST)
for sym in ["KWEB", "TSLA", "AMD"]:
    result = score_ticker(sym)
    print(f"{sym}: {'SETUP (BUG!)' if result and result.get('side') == 'short' else 'no fade setup — correct'}")
EOF
```

Expected output:
```
FADE_WATCHLIST: ['PLTR', 'FXI']
KWEB: no fade setup — correct
TSLA: no fade setup — correct
AMD: no fade setup — correct
```

- [ ] **Step 2: Confirm PLTR/FXI signal fires only when below EMA-200**

```bash
cd /Users/ashleighchua/trading\ analyses && python3 - <<'EOF' 2>&1 | grep -v OpenSSL | grep -v warnings | grep -v urllib3
import sys, os
sys.path.insert(0, "dashboard")
os.chdir("dashboard")
from dotenv import load_dotenv
load_dotenv("../.env")
from premarket_scanner import score_ticker, compute_ema200, fetch_bars

for sym in ["PLTR", "FXI"]:
    bars = fetch_bars(sym, limit=260)
    ema200 = compute_ema200(bars)
    price = bars[-1]["c"] if bars else None
    below = price is not None and ema200 is not None and price < ema200
    result = score_ticker(sym)
    fade_fired = result is not None and result.get("side") == "short"
    status = "CONSISTENT" if (fade_fired == below) or (not below and not fade_fired) else "BUG"
    print(f"{sym}: price=${price:.2f}, EMA-200=${ema200:.2f if ema200 else 'N/A'}, below={below}, fade_fired={fade_fired} — {status}")
EOF
```

Expected: both lines end with `CONSISTENT`.

- [ ] **Step 3: Confirm NVDA/GLD/SPY strategies are unaffected**

```bash
cd /Users/ashleighchua/trading\ analyses && python3 - <<'EOF' 2>&1 | grep -v OpenSSL | grep -v warnings | grep -v urllib3
import sys, os
sys.path.insert(0, "dashboard")
os.chdir("dashboard")
from dotenv import load_dotenv
load_dotenv("../.env")
from premarket_scanner import score_ticker, LONG_WATCHLIST

print("LONG_WATCHLIST:", LONG_WATCHLIST)
for sym in ["NVDA", "GLD"]:
    result = score_ticker(sym)
    print(f"{sym}: {'setup=' + result['side'] if result else 'no setup'} (NVDA/GLD unaffected — OK either way)")
EOF
```

Expected: script runs without error. Result (setup or no setup) does not matter — just confirms no crash.

- [ ] **Step 4: Final commit if any last fixes were needed, otherwise confirm clean**

```bash
cd /Users/ashleighchua/trading\ analyses && git status
```

Expected: `nothing to commit, working tree clean`

If any files were modified during smoke-test fixes:
```bash
git add dashboard/premarket_scanner.py
git commit -m "fix: smoke-test corrections to scanner EMA-200 implementation"
```
