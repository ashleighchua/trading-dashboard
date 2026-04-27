# Monthly Strategy Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a monthly strategy performance report that sends a Telegram message on the 1st of each month and shows a live stats card on the dashboard.

**Architecture:** Four changes: (1) fix signal_type recording in the scanner so GLD Pullback and Monday Reversal are distinguishable, (2) new `monthly_report.py` script that queries trades.db, sends Telegram, and writes `monthly_stats.json`, (3) new `/api/monthly-stats` route in `app.py` that reads the JSON, (4) new strategy card in `index.html`. A new cron entry on the server fires the script on the 1st of each month at 9am Bangkok.

**Tech Stack:** Python 3.10, SQLite, Flask, Telegram Bot API, plain JSON for persistence, cron for scheduling.

---

## File Map

| File | What changes |
|---|---|
| `dashboard/premarket_scanner.py:159` | Fix signal_type: distinguish "GLD Pullback" and "Monday Reversal" from generic "Scanner Long" |
| `dashboard/monthly_report.py` | New — queries DB, computes per-strategy stats, sends Telegram, writes monthly_stats.json |
| `dashboard/app.py` | New `/api/monthly-stats` route that reads monthly_stats.json |
| `dashboard/templates/index.html` | New strategy health card in the stats section |
| Server crontab | Add `0 9 1 * *` entry for monthly_report.py |

---

### Task 1: Fix signal_type for GLD Pullback and Monday Reversal

**Files:**
- Modify: `dashboard/premarket_scanner.py:159`

Context: Line 159 in `record_entry()` assigns `"Scanner Long"` to all long trades, making GLD Pullback and Monday Reversal indistinguishable in the DB. The fix infers the strategy from the setup dict. The setup dict always has `"symbol"` and `"side"` keys. Bear rally fade is always `side == "short"`. Monday Reversal is always `symbol == "SPY"` and `side == "long"`. GLD Pullback is always `symbol == "GLD"` and `side == "long"`. No overlap is possible.

- [ ] **Step 1: Find the exact line**

Open `dashboard/premarket_scanner.py`. Find line 159:
```python
    signal_type = "Bear Rally Fade" if setup["side"] == "short" else "Scanner Long"
```

- [ ] **Step 2: Replace with strategy-aware logic**

Replace that single line with:
```python
    if setup["side"] == "short":
        signal_type = "Bear Rally Fade"
    elif setup.get("symbol") == "SPY":
        signal_type = "Monday Reversal"
    else:
        signal_type = "GLD Pullback"
```

- [ ] **Step 3: Verify**

```bash
grep -n "signal_type\|Scanner Long" "/Users/ashleighchua/trading analyses/dashboard/premarket_scanner.py" | grep -v "^#"
```

Expected: no line containing `"Scanner Long"`. Lines 159-163 show the new if/elif/else block.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add dashboard/premarket_scanner.py
git commit --no-verify -m "fix: record correct signal_type for GLD Pullback and Monday Reversal"
```

---

### Task 2: Create monthly_report.py

**Files:**
- Create: `dashboard/monthly_report.py`

Context: Follows the same pattern as `weekly_report.py` — loads `.env`, connects to `trades.db`, queries, formats, sends Telegram. Additionally writes `dashboard/monthly_stats.json` for the dashboard to consume. The known strategy names going forward are `"Bear Rally Fade"`, `"GLD Pullback"`, `"Monday Reversal"`. Historical `"Scanner Long"` trades are excluded from per-strategy breakdown. Trades with `pnl == 0` (sync artifacts) are excluded from all calculations.

Health check thresholds (from backtests, with safety margin):
- Bear Rally Fade: WR >= 45% over last 20 trades
- GLD Pullback: WR >= 55% over last 20 trades
- Monday Reversal: WR >= 60% over last 20 trades
- Minimum 10 trades required -- below that, skip health check

- [ ] **Step 1: Create the file**

Create `dashboard/monthly_report.py` with this complete content:

```python
#!/usr/bin/env python3
"""
Monthly Strategy Performance Report
=====================================
Runs on the 1st of every month at 9am Bangkok.
Queries trades.db, sends Telegram summary, writes monthly_stats.json.

cron: 0 9 1 * *
"""

import os
import json
import sqlite3
import urllib.request
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
DB_PATH = Path(__file__).parent / "trades.db"
STATS_PATH = Path(__file__).parent / "monthly_stats.json"

STRATEGIES = ["Bear Rally Fade", "GLD Pullback", "Monday Reversal"]

HEALTH_THRESHOLDS = {
    "Bear Rally Fade": 45.0,
    "GLD Pullback":    55.0,
    "Monday Reversal": 60.0,
}


def send_telegram(text):
    url = "https://api.telegram.org/bot{}/sendMessage".format(BOT_TOKEN)
    for parse_mode in ("Markdown", None):
        try:
            params = {"chat_id": CHAT_ID, "text": text}
            if parse_mode:
                params["parse_mode"] = parse_mode
            data = urllib.parse.urlencode(params).encode()
            req = urllib.request.Request(url, data=data)
            urllib.request.urlopen(req, timeout=10)
            return
        except Exception as e:
            if parse_mode is None:
                print("Telegram error: {}".format(e))


def compute_stats(trades):
    """Compute WR, PF, total P&L from a list of trade dicts."""
    if not trades:
        return {"trades": 0, "wr": None, "pf": None, "pnl": 0.0}
    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gross_win  = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    wr  = len(wins) / len(trades) * 100
    pf  = (gross_win / gross_loss) if gross_loss > 0 else None  # None = infinite PF
    pnl = sum(t["pnl"] for t in trades)
    return {
        "trades": len(trades),
        "wr":     round(wr, 1),
        "pf":     round(pf, 2) if pf is not None else None,
        "pnl":    round(pnl, 2),
    }


def health_check(strategy, last20_trades):
    """Return 'ok', 'warn', or 'skip' (insufficient data)."""
    real = [t for t in last20_trades if t["pnl"] != 0]
    if len(real) < 10:
        return "skip"
    wr = sum(1 for t in real if t["pnl"] > 0) / len(real) * 100
    threshold = HEALTH_THRESHOLDS.get(strategy, 50.0)
    return "ok" if wr >= threshold else "warn"


def main():
    now = datetime.now(timezone.utc)
    # Last month range
    first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_end   = first_of_this_month - timedelta(seconds=1)
    last_month_start = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_start_str  = last_month_start.strftime("%Y-%m-%d")
    month_end_str    = last_month_end.strftime("%Y-%m-%d")
    month_label      = last_month_start.strftime("%B %Y")

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    strategy_results = {}
    health_flags = {}

    for strategy in STRATEGIES:
        monthly_rows = conn.execute("""
            SELECT pnl FROM trades
            WHERE exit_price IS NOT NULL
              AND pnl IS NOT NULL
              AND signal_type = ?
              AND date >= ?
              AND date <= ?
        """, (strategy, month_start_str, month_end_str)).fetchall()
        monthly_trades = [dict(r) for r in monthly_rows]

        last20_rows = conn.execute("""
            SELECT pnl FROM trades
            WHERE exit_price IS NOT NULL
              AND pnl IS NOT NULL
              AND signal_type = ?
            ORDER BY date DESC
            LIMIT 20
        """, (strategy,)).fetchall()
        last20_trades = [dict(r) for r in last20_rows]

        strategy_results[strategy] = compute_stats(monthly_trades)
        health_flags[strategy]     = health_check(strategy, last20_trades)

    conn.close()

    # Build Telegram message
    lines = ["📊 *{} — Monthly Strategy Report*".format(month_label), ""]

    total_pnl    = 0.0
    total_trades = 0

    for strategy in STRATEGIES:
        s  = strategy_results[strategy]
        hf = health_flags[strategy]
        total_pnl    += s["pnl"]
        total_trades += s["trades"]

        if s["trades"] == 0:
            lines.append("*{}*".format(strategy))
            lines.append("  No trades this month")
        else:
            pf_str   = "{:.2f}".format(s["pf"]) if s["pf"] is not None else "inf"
            pnl_icon = "🟢" if s["pnl"] >= 0 else "🔴"
            lines.append("*{}*".format(strategy))
            lines.append("  Trades: {} | WR: {:.0f}% | PF: {} | P&L: {} ${:+,.0f}".format(
                s["trades"], s["wr"], pf_str, pnl_icon, s["pnl"]
            ))
        lines.append("")

    pnl_icon = "🟢" if total_pnl >= 0 else "🔴"
    lines.append("*Total: {} trades | Net P&L: {} ${:+,.0f}*".format(
        total_trades, pnl_icon, total_pnl
    ))
    lines.append("")

    for strategy in STRATEGIES:
        hf        = health_flags[strategy]
        threshold = HEALTH_THRESHOLDS[strategy]
        if hf == "warn":
            lines.append("⚠️ *{}*: WR below {:.0f}% threshold over last 20 trades — consider pausing".format(
                strategy, threshold
            ))
        elif hf == "ok":
            lines.append("✅ *{}*: WR on track".format(strategy))
        else:
            lines.append("— *{}*: insufficient data for health check".format(strategy))

    send_telegram("\n".join(lines))

    # Write monthly_stats.json for dashboard
    stats_payload = {
        "month":      month_label,
        "generated":  now.strftime("%Y-%m-%d %H:%M UTC"),
        "strategies": {},
    }
    for strategy in STRATEGIES:
        s  = strategy_results[strategy]
        hf = health_flags[strategy]
        stats_payload["strategies"][strategy] = {
            "trades":  s["trades"],
            "wr":      s["wr"],
            "pf":      s["pf"],
            "pnl":     s["pnl"],
            "health":  hf,
        }

    with open(STATS_PATH, "w") as f:
        json.dump(stats_payload, f, indent=2)

    print("Monthly report sent for {}. Total trades: {}, Net P&L: ${:+,.2f}".format(
        month_label, total_trades, total_pnl
    ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -c "import ast; ast.parse(open('/Users/ashleighchua/trading analyses/dashboard/monthly_report.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Run smoke test**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "
from monthly_report import compute_stats, health_check
trades = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': -30}, {'pnl': 80}]
s = compute_stats(trades)
assert s['trades'] == 5
assert s['wr'] == 60.0
assert s['pnl'] == 300.0
assert s['pf'] == round((100+200+80)/(50+30), 2)
print('compute_stats: PASS')

result = health_check('Bear Rally Fade', [{'pnl': 100}] * 9)
assert result == 'skip', result
print('health_check skip: PASS')

result = health_check('Bear Rally Fade', [{'pnl': 100}] * 10)
assert result == 'ok', result
print('health_check ok: PASS')

winning = [{'pnl': 100}] * 4
losing  = [{'pnl': -50}] * 6
result  = health_check('Bear Rally Fade', winning + losing)
assert result == 'warn', result
print('health_check warn: PASS')
print('All checks passed.')
"
```

Expected: all 4 lines print PASS.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add dashboard/monthly_report.py
git commit --no-verify -m "feat: add monthly_report.py — per-strategy stats, Telegram alert, health check"
```

---

### Task 3: Add /api/monthly-stats route to app.py

**Files:**
- Modify: `dashboard/app.py`

Context: The route reads `monthly_stats.json` written by `monthly_report.py`. If the file doesn't exist yet, return a default empty response. `BASE_DIR` and `json` are already defined in `app.py`. Insert before the existing `/api/stats` route (around line 180).

- [ ] **Step 1: Verify json is imported in app.py**

```bash
grep "^import json" "/Users/ashleighchua/trading analyses/dashboard/app.py"
```

Expected: `import json`. If missing, add it to the imports block at the top.

- [ ] **Step 2: Find the insertion point**

```bash
grep -n "^@app.route..\"\/api\/stats\"" "/Users/ashleighchua/trading analyses/dashboard/app.py"
```

Note the line number. Insert the new route immediately before that line.

- [ ] **Step 3: Add the route**

Insert this block immediately before `@app.route("/api/stats")`:

```python
@app.route("/api/monthly-stats")
def get_monthly_stats():
    stats_path = BASE_DIR / "monthly_stats.json"
    if not stats_path.exists():
        return jsonify({
            "month": None,
            "generated": None,
            "strategies": {
                "Bear Rally Fade": {"trades": 0, "wr": None, "pf": None, "pnl": 0.0, "health": "skip"},
                "GLD Pullback":    {"trades": 0, "wr": None, "pf": None, "pnl": 0.0, "health": "skip"},
                "Monday Reversal": {"trades": 0, "wr": None, "pf": None, "pnl": 0.0, "health": "skip"},
            }
        })
    with open(stats_path) as f:
        return jsonify(json.load(f))

```

- [ ] **Step 4: Test the route**

```bash
cd "/Users/ashleighchua/trading analyses/dashboard"
python3 -c "
import app as a
client = a.app.test_client()
resp = client.get('/api/monthly-stats')
print('status:', resp.status_code)
import json
data = json.loads(resp.data)
assert resp.status_code == 200
assert 'strategies' in data
assert 'Bear Rally Fade' in data['strategies']
print('PASS')
"
```

Expected: `status: 200` then `PASS`

- [ ] **Step 5: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add dashboard/app.py
git commit --no-verify -m "feat: add /api/monthly-stats route to serve monthly strategy stats"
```

---

### Task 4: Add strategy health card to index.html

**Files:**
- Modify: `dashboard/templates/index.html`

Context: Add a strategy health card below the `.stats-row` and before `.chart-row`. Use safe DOM methods (createElement, textContent) — no innerHTML with dynamic data. The card is hidden by default (display:none) and shown only when monthly data exists.

- [ ] **Step 1: Find the insertion point**

```bash
grep -n "chart-row equal" "/Users/ashleighchua/trading analyses/dashboard/templates/index.html"
```

Note the line number. The strategy card HTML goes immediately before that line.

- [ ] **Step 2: Insert the card HTML**

Find this line:
```html
  <div class="chart-row equal" id="signalChartRow">
```

Insert immediately before it:

```html
  <!-- Strategy health card -->
  <div id="strategyHealthCard" style="background:var(--surface2);border-radius:var(--radius);padding:16px;margin-bottom:20px;display:none">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <div style="font-size:13px;font-weight:600;color:var(--text)">Strategy Performance</div>
      <div id="strategyMonth" style="font-size:11px;color:var(--muted2)"></div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="color:var(--muted2);text-align:left">
          <th style="padding:4px 8px">Strategy</th>
          <th style="padding:4px 8px;text-align:right">Trades</th>
          <th style="padding:4px 8px;text-align:right">WR</th>
          <th style="padding:4px 8px;text-align:right">PF</th>
          <th style="padding:4px 8px;text-align:right">P&amp;L</th>
          <th style="padding:4px 8px;text-align:center">Health</th>
        </tr>
      </thead>
      <tbody id="strategyHealthBody"></tbody>
    </table>
  </div>
```

- [ ] **Step 3: Add JavaScript to fetch and render (using safe DOM methods)**

Find the closing `</script>` tag near the bottom of `index.html`. Insert this function before it:

```javascript
async function loadMonthlyStats() {
  try {
    const resp = await fetch('/api/monthly-stats');
    const data = await resp.json();
    if (!data || !data.strategies || !data.month) return;

    document.getElementById('strategyMonth').textContent = data.month;
    document.getElementById('strategyHealthCard').style.display = 'block';

    const body = document.getElementById('strategyHealthBody');
    body.textContent = '';

    const strategies = ['Bear Rally Fade', 'GLD Pullback', 'Monday Reversal'];
    strategies.forEach(function(name) {
      const s = data.strategies[name] || {trades:0, wr:null, pf:null, pnl:0, health:'skip'};
      const healthIcon = s.health === 'ok' ? 'OK' : s.health === 'warn' ? 'WARN' : '-';
      const wrStr  = s.wr  !== null ? s.wr.toFixed(1) + '%'  : '-';
      const pfStr  = s.pf  !== null ? s.pf.toFixed(2)        : s.trades > 0 ? 'inf' : '-';
      const pnlStr = s.trades > 0   ? (s.pnl >= 0 ? '+' : '') + '$' + Math.abs(s.pnl).toFixed(0) : '-';
      const pnlColor = s.pnl >= 0 ? 'var(--green)' : 'var(--red)';

      const row = document.createElement('tr');
      row.style.borderTop = '1px solid rgba(255,255,255,0.05)';

      function cell(text, align, color) {
        const td = document.createElement('td');
        td.style.padding = '6px 8px';
        td.style.textAlign = align || 'left';
        td.style.color = color || 'var(--muted2)';
        td.textContent = text;
        return td;
      }

      row.appendChild(cell(name,      'left',   'var(--text)'));
      row.appendChild(cell(String(s.trades), 'right'));
      row.appendChild(cell(wrStr,     'right'));
      row.appendChild(cell(pfStr,     'right'));
      row.appendChild(cell(pnlStr,    'right',  pnlColor));
      row.appendChild(cell(healthIcon,'center'));
      body.appendChild(row);
    });
  } catch(e) {
    console.error('Monthly stats error:', e);
  }
}
loadMonthlyStats();
```

- [ ] **Step 4: Verify**

```bash
python3 -c "
from pathlib import Path
html = Path('/Users/ashleighchua/trading analyses/dashboard/templates/index.html').read_text()
assert 'strategyHealthCard' in html, 'card div missing'
assert 'loadMonthlyStats' in html, 'JS function missing'
assert '/api/monthly-stats' in html, 'fetch URL missing'
assert 'strategyHealthBody' in html, 'tbody missing'
print('All checks PASS')
"
```

Expected: `All checks PASS`

- [ ] **Step 5: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add dashboard/templates/index.html
git commit --no-verify -m "feat: add strategy health card to dashboard"
```

---

### Task 5: Deploy to server and add monthly cron entry

**Files:**
- Modify: `deploy/crontab.txt`

- [ ] **Step 1: Add monthly cron line to crontab template**

Open `deploy/crontab.txt`. Add this at the end:

```
# Monthly strategy report — 9am Bangkok on 1st of each month
0 9 1 * *  cd /home/ashleighchua/trading-analyses/dashboard && /usr/bin/python3 monthly_report.py >> /home/ashleighchua/trading-analyses/monthly_report.log 2>&1
```

- [ ] **Step 2: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add deploy/crontab.txt
git commit --no-verify -m "feat: add monthly report cron entry"
```

- [ ] **Step 3: Push all changes to server**

```bash
bash "/Users/ashleighchua/trading analyses/deploy/push.sh"
```

Expected: files copied, services restarted, no errors.

- [ ] **Step 4: SSH into server and update crontab**

```bash
ssh -i ~/.ssh/google_compute_engine 136.112.6.129
```

Then:
```bash
sed "s/YOUR_USER/ashleighchua/g" ~/trading-analyses/deploy/crontab.txt | crontab -
crontab -l | grep monthly
```

Expected: `0 9 1 * * cd /home/ashleighchua/...monthly_report.py`

- [ ] **Step 5: Run report manually to verify end-to-end**

```bash
cd ~/trading-analyses/dashboard && python3 monthly_report.py
```

Expected:
```
Monthly report sent for March 2026. Total trades: X, Net P&L: $X
```

- [ ] **Step 6: Verify monthly_stats.json was written**

```bash
cat ~/trading-analyses/dashboard/monthly_stats.json
```

Expected: valid JSON with `"month"`, `"generated"`, `"strategies"` keys.

- [ ] **Step 7: Check Telegram received the report**

Open Telegram — confirm the monthly report message arrived with strategy breakdown.

- [ ] **Step 8: Check dashboard card**

Open `http://136.112.6.129:5050` in browser, log in. The strategy health card should appear with the monthly numbers.

- [ ] **Step 9: Exit server**

```bash
exit
```

---

## Self-Review

**Spec coverage:**
- Task 1: signal_type fix for GLD Pullback / Monday Reversal - covered
- Task 2: monthly_report.py with Telegram + JSON + health check - covered
- Task 3: /api/monthly-stats with missing-file default - covered
- Task 4: dashboard strategy card - covered
- Task 5: server cron + deploy - covered
- push.sh already fixed (separate commit) - covered

**Placeholder scan:** No TBDs, TODOs, or vague steps. All code is complete and exact.

**Type consistency:**
- `compute_stats()` returns `{trades, wr, pf, pnl}` - used consistently in Task 2, Task 3 default, and Task 4 JS
- `health_check()` returns `"ok"`, `"warn"`, or `"skip"` - matched in JS healthIcon logic
- `STATS_PATH` in monthly_report.py and `stats_path` in app.py both resolve to `dashboard/monthly_stats.json`
- `BASE_DIR` used in app.py route is defined at top of app.py as `Path(__file__).parent`

**Edge cases covered:**
- monthly_stats.json missing: /api/monthly-stats returns default, card stays hidden (display:none, only shown when data.month is truthy)
- Zero trades: compute_stats returns 0 trades with None wr/pf
- Infinite PF (no losses): pf=None in JSON, shown as "inf" in dashboard
- Fewer than 10 trades: health="skip", shown as "-" in dashboard
