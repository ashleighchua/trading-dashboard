# Monthly Strategy Performance Report

**Date:** 2026-04-27
**Status:** Approved for implementation

---

## Problem

The trading system has no way to automatically track whether live strategy performance matches backtest expectations. Three risks go undetected:

1. **Strategy decay** — edge erodes over time, WR drops below quality bar
2. **Backtest overestimation** — live WR consistently below backtest WR
3. **Signal type confusion** — GLD Pullback and Monday Reversal both recorded as `"Scanner Long"` in `trades.db`, making per-strategy analysis impossible

---

## What Gets Built

### Prerequisite fix: Correct signal_type recording

`dashboard/premarket_scanner.py` line 159 records all long trades as `"Scanner Long"`. Fix to record the actual strategy name:

```python
# Before
signal_type = "Bear Rally Fade" if setup["side"] == "short" else "Scanner Long"

# After — infer from thesis or symbol
if setup["side"] == "short":
    signal_type = "Bear Rally Fade"
elif setup.get("symbol") == "SPY":
    signal_type = "Monday Reversal"
else:
    signal_type = "GLD Pullback"
```

This fix applies to new trades only — historical `"Scanner Long"` records stay as-is.

### New script: dashboard/monthly_report.py

Runs on the 1st of every month at 9am Bangkok. Does three things:

1. **Queries trades.db** — pulls all closed trades grouped by `signal_type`, computes per-strategy stats for last month AND all-time last 20 trades (for health check)
2. **Sends Telegram message** — formatted monthly summary with health flags
3. **Writes JSON summary** — saves stats to `dashboard/monthly_stats.json` for the dashboard to read

### Telegram message format

```
📊 April 2026 — Monthly Strategy Report

Bear Rally Fade (Short)
  Trades: 5 | WR: 54% | PF: 1.88 | P&L: +$312

GLD Pullback (Long)
  Trades: 3 | WR: 67% | PF: 2.10 | P&L: +$180

Monday Reversal (Long)
  Trades: 4 | WR: 75% | PF: 3.20 | P&L: +$420

Total: 12 trades | Net P&L: +$912

⚠️ Bear Rally Fade: WR 43% over last 20 trades — below 45% threshold. Consider pausing.
✅ GLD Pullback and Monday Reversal within expected range.
```

Health check thresholds (from backtests):
- Bear Rally Fade: WR ≥ 45% over last 20 trades
- GLD Pullback: WR ≥ 55% over last 20 trades  
- Monday Reversal: WR ≥ 60% over last 20 trades

If fewer than 10 trades exist for a strategy, skip health check (insufficient data).

### Dashboard card: new `/api/monthly-stats` route + UI card

New API route in `app.py` reads `monthly_stats.json` and returns it. New card on `index.html` shows a table — one row per strategy, columns: Strategy, Trades (month), WR (month), PF (month), P&L (month), Health.

Health column shows: ✅ On track / ⚠️ Below threshold / — (insufficient data)

### New cron entry (server)

```
0 9 1 * *  cd /home/ashleighchua/trading-analyses/dashboard && /usr/bin/python3 monthly_report.py >> /home/ashleighchua/trading-analyses/monthly_report.log 2>&1
```

---

## Files Changed

| File | Change |
|---|---|
| `dashboard/premarket_scanner.py` | Fix signal_type for GLD Pullback and Monday Reversal |
| `dashboard/monthly_report.py` | New — monthly stats script |
| `dashboard/app.py` | New `/api/monthly-stats` route |
| `dashboard/templates/index.html` | New strategy stats card |
| Server crontab | Add monthly cron entry |

---

## Critical Fix: push.sh Must Not Overwrite trades.db

`deploy/push.sh` previously used `gcloud compute scp` to copy the entire `dashboard/` folder, which would overwrite `trades.db` on the server with the stale Mac version — wiping live trade history on every deploy. Fixed to use `rsync --exclude='trades.db' --exclude='monthly_stats.json'` so the live database and generated files are never touched.

## What Does NOT Change

- Weekly report (`weekly_report.py`) — unchanged
- Trade sync logic — unchanged
- Existing historical trades recorded as `"Scanner Long"` — left as-is, excluded from per-strategy breakdown (shown as "Other" if present)

## Edge Cases

- **monthly_stats.json missing** (before first report runs): `/api/monthly-stats` returns empty default data, dashboard card shows "No data yet"
- **Fewer than 10 trades** for a strategy: health check skipped, shows "— (insufficient data)"
- **Profit Factor formula**: gross wins / gross losses. If no losses, PF = ∞ (shown as "∞"). If no wins, PF = 0.0
- **Zero trades in month**: strategy row still shown with "0 trades — no activity this month"

---

## Success Criteria

- On the 1st of each month, Telegram receives a formatted strategy report
- Dashboard shows a strategy stats card with last month's numbers
- Health check flags strategies below WR threshold
- GLD Pullback and Monday Reversal are distinguishable in the DB going forward
