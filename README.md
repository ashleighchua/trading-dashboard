# Trading Journal Dashboard

**Live demo: [web-production-53f70.up.railway.app](https://web-production-53f70.up.railway.app)**

A Flask dashboard for logging trades, running pre-market scans, tracking signals, and generating weekly/monthly reports. Connects to Alpaca for live data and Telegram for alerts.

## Features

- Trade journal with SQLite storage — log entries, exits, notes, screenshots
- Pre-market scanner — flags setups before open
- Signal engine — daily signal checks with configurable strategies
- Price alerts — notifies via Telegram when levels are hit
- Weekly and monthly performance reports
- Post-open stop management
- Crypto weekly summary
- FinRL integration for ML-based backtesting

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your Alpaca API keys and Telegram bot credentials.

```bash
python3 app.py
```

Opens at [http://localhost:5050](http://localhost:5050).

## Stack

- Flask — web server
- SQLite — trade storage
- Alpaca API — market data and order management
- Telegram Bot API — alerts and reports
- FinRL — reinforcement learning backtests

## Notes

- `.env`, `trades.db`, and `price_alerts.json` are gitignored; bring your own
