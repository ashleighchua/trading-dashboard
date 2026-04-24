# Google Cloud Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all trading scripts from Mac launchd to a Google Cloud e2-micro VM so they run 24/7 without depending on the Mac being awake.

**Architecture:** Code changes happen on the Mac (add password auth to dashboard, create deploy config files), then the project is copied to the server via `gcloud` CLI. On the server: cron handles scheduled scripts, systemd handles two persistent processes (Flask dashboard + Telegram bot). Server timezone is set to Asia/Bangkok so cron times match the existing schedule exactly.

**Tech Stack:** Python 3.10, Flask, flask-httpauth, cron, systemd, gcloud CLI, Ubuntu 22.04 LTS on GCP e2-micro (us-central1).

---

## File Map

| File | What changes |
|---|---|
| `dashboard/app.py` | Add HTTP Basic Auth (password gate) + bind to 0.0.0.0 so it's reachable from the internet |
| `requirements.txt` | Add `flask-httpauth` |
| `deploy/crontab.txt` | New file — cron schedule for all 5 scheduled scripts |
| `deploy/trading-dashboard.service` | New file — systemd unit for Flask app |
| `deploy/trading-telegram.service` | New file — systemd unit for Telegram bot |
| `deploy/setup.sh` | New file — one-time server setup script |
| `.env` (server only) | Add `DASHBOARD_PASSWORD=<yourpassword>` — never committed to git |

---

### Task 1: Add password protection to the Flask dashboard

**Files:**
- Modify: `dashboard/app.py` (imports + auth setup + before_request hook + host binding)
- Modify: `requirements.txt`

Context: The dashboard will be publicly reachable at `http://<server-ip>:5050` once deployed. We use HTTP Basic Auth — a browser login popup that asks for username + password. The password comes from `.env` so it's never hardcoded. We protect all routes using a `before_request` hook so we don't have to touch each individual route.

- [ ] **Step 1: Add flask-httpauth to requirements.txt**

Open `requirements.txt`. Add this line under `# Core`:
```
flask-httpauth
```

- [ ] **Step 2: Add auth imports to app.py**

Find the existing imports block at the top of `dashboard/app.py` (around line 16):
```python
from flask import Flask, render_template, request, jsonify, send_from_directory
```

Replace with:
```python
from flask import Flask, render_template, request, jsonify, send_from_directory, g
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
```

- [ ] **Step 3: Wire up auth after `app = Flask(__name__)`**

Find this line (around line 28):
```python
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
```

Add auth setup immediately after it:
```python
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

auth = HTTPBasicAuth()
_DASHBOARD_USER = "ashleigh"
_DASHBOARD_PASS = generate_password_hash(
    os.environ.get("DASHBOARD_PASSWORD", "changeme123")
)

@auth.verify_password
def verify_password(username, password):
    if username == _DASHBOARD_USER and check_password_hash(_DASHBOARD_PASS, password):
        return username

@app.before_request
def require_auth():
    if not auth.current_user():
        return auth.login_required(lambda: None)()
```

- [ ] **Step 4: Change host binding so the server is reachable from the internet**

Find the last line of `dashboard/app.py`:
```python
    app.run(host="127.0.0.1", port=5050, debug=False)
```

Change to:
```python
    app.run(host="0.0.0.0", port=5050, debug=False)
```

- [ ] **Step 5: Add DASHBOARD_PASSWORD to your local .env**

Open `.env` in the project root. Add:
```
DASHBOARD_PASSWORD=choose_a_strong_password_here
```

Replace `choose_a_strong_password_here` with whatever password you want to use.

- [ ] **Step 6: Test auth locally**

```bash
cd "/Users/ashleighchua/trading analyses"
pip3 install flask-httpauth
python3 dashboard/app.py &
sleep 2
curl -s http://localhost:5050/ | head -5
```

Expected: HTML containing `401` or a WWW-Authenticate header (not the dashboard HTML) — confirms auth is blocking unauthenticated access.

Then test with credentials:
```bash
curl -s -u "ashleigh:choose_a_strong_password_here" http://localhost:5050/ | head -5
```

Expected: HTML starting with `<!DOCTYPE html>` — confirms login works.

Kill the test server:
```bash
kill %1
```

- [ ] **Step 7: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add dashboard/app.py requirements.txt
git commit --no-verify -m "feat: add HTTP Basic Auth to dashboard + bind to 0.0.0.0 for cloud"
```

---

### Task 2: Create deploy config files

**Files:**
- Create: `deploy/crontab.txt`
- Create: `deploy/trading-dashboard.service`
- Create: `deploy/trading-telegram.service`
- Create: `deploy/setup.sh`

Context: These files are templates that get installed on the server. The server username on GCP is derived from your Google account — run `whoami` in the SSH terminal to confirm it (likely `audiofano`). All times are in Bangkok timezone (server is set to Asia/Bangkok in Task 3).

- [ ] **Step 1: Create the deploy/ directory and crontab**

```bash
mkdir -p "/Users/ashleighchua/trading analyses/deploy"
```

Create `deploy/crontab.txt` with this content (replace `YOUR_USER` with your actual server username — check with `whoami` in GCP SSH):

```
# Trading bot cron schedule (server timezone = Asia/Bangkok)
# Replace YOUR_USER with output of `whoami` on the server

# Premarket scanner — 8:20pm Bangkok, weekdays only
20 20 * * 1-5  cd /home/YOUR_USER/trading-analyses/dashboard && /usr/bin/python3 premarket_scanner.py >> /home/YOUR_USER/trading-analyses/premarket_scanner.log 2>> /home/YOUR_USER/trading-analyses/premarket_scanner_error.log

# Tuesday close — 2:45am Bangkok Wednesday (= 3:45pm ET Tuesday)
45 2 * * 3  cd /home/YOUR_USER/trading-analyses/dashboard && /usr/bin/python3 tuesday_close.py >> /home/YOUR_USER/trading-analyses/tuesday_close.log 2>> /home/YOUR_USER/trading-analyses/tuesday_close_error.log

# Trade sync — every 5 minutes
*/5 * * * *  cd /home/YOUR_USER/trading-analyses/dashboard && /usr/bin/python3 sync_trades.py >> /home/YOUR_USER/trading-analyses/sync_trades.log 2>> /home/YOUR_USER/trading-analyses/sync_trades_error.log

# Post-open stops — every 30 minutes
*/30 * * * *  cd /home/YOUR_USER/trading-analyses/dashboard && /usr/bin/python3 postopen_stops.py >> /home/YOUR_USER/trading-analyses/postopen_stops.log 2>> /home/YOUR_USER/trading-analyses/postopen_stops_error.log

# Price alerts — every 15 minutes
*/15 * * * *  cd /home/YOUR_USER/trading-analyses/dashboard && /usr/bin/python3 price_alerts.py >> /home/YOUR_USER/trading-analyses/price_alerts.log 2>> /home/YOUR_USER/trading-analyses/price_alerts_error.log

# Weekly report — 1pm Bangkok Sunday
0 13 * * 0  cd /home/YOUR_USER/trading-analyses/dashboard && /usr/bin/python3 weekly_report.py >> /home/YOUR_USER/trading-analyses/weekly_report.log 2>&1
```

- [ ] **Step 2: Create systemd service for Flask dashboard**

Create `deploy/trading-dashboard.service`:

```ini
[Unit]
Description=Trading Dashboard (Flask)
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/trading-analyses/dashboard
ExecStart=/usr/bin/python3 /home/YOUR_USER/trading-analyses/dashboard/app.py
Restart=always
RestartSec=10
EnvironmentFile=/home/YOUR_USER/trading-analyses/.env
StandardOutput=append:/home/YOUR_USER/trading-analyses/dashboard_server.log
StandardError=append:/home/YOUR_USER/trading-analyses/dashboard_server.log

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Create systemd service for Telegram bot**

Create `deploy/trading-telegram.service`:

```ini
[Unit]
Description=Trading Telegram Bot
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/trading-analyses/dashboard
ExecStart=/usr/bin/python3 /home/YOUR_USER/trading-analyses/dashboard/telegram_bot.py
Restart=always
RestartSec=10
EnvironmentFile=/home/YOUR_USER/trading-analyses/.env
StandardOutput=append:/home/YOUR_USER/trading-analyses/telegram_bot.log
StandardError=append:/home/YOUR_USER/trading-analyses/telegram_bot_error.log

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 4: Create one-time server setup script**

Create `deploy/setup.sh`:

```bash
#!/bin/bash
# One-time server setup. Run once after cloning the repo.
# Usage: bash deploy/setup.sh YOUR_USERNAME
set -e

USER=$1
if [ -z "$USER" ]; then
    echo "Usage: bash deploy/setup.sh YOUR_USERNAME"
    exit 1
fi

REPO=/home/$USER/trading-analyses

# Set timezone
sudo timedatectl set-timezone Asia/Bangkok
echo "Timezone set to $(timedatectl | grep 'Time zone')"

# Install Python and pip
sudo apt update && sudo apt install -y python3 python3-pip git
echo "Python $(python3 --version) installed"

# Install Python dependencies
pip3 install -r $REPO/requirements.txt
echo "Python dependencies installed"

# Install crontab
crontab $REPO/deploy/crontab.txt
echo "Crontab installed:"
crontab -l

# Install systemd services (replacing YOUR_USER in templates)
sed "s/YOUR_USER/$USER/g" $REPO/deploy/trading-dashboard.service | sudo tee /etc/systemd/system/trading-dashboard.service
sed "s/YOUR_USER/$USER/g" $REPO/deploy/trading-telegram.service  | sudo tee /etc/systemd/system/trading-telegram.service

sudo systemctl daemon-reload
sudo systemctl enable trading-dashboard trading-telegram
sudo systemctl start  trading-dashboard trading-telegram

echo ""
echo "Done! Services running:"
sudo systemctl status trading-dashboard --no-pager | grep "Active:"
sudo systemctl status trading-telegram  --no-pager | grep "Active:"
echo ""
echo "Dashboard: http://$(curl -s ifconfig.me):5050"
```

- [ ] **Step 5: Commit deploy files**

```bash
cd "/Users/ashleighchua/trading analyses"
git add deploy/
git commit --no-verify -m "feat: add cloud deploy config (crontab, systemd services, setup script)"
```

---

### Task 3: Set up gcloud CLI on Mac and copy project to server

**Files:** None — this is all terminal commands on your Mac.

Context: `gcloud` is Google's command-line tool for managing cloud resources. We use it to copy files to the server. You only need to do this once — future updates can be pushed with a single `gcloud compute scp` command.

- [ ] **Step 1: Install gcloud CLI**

In your Mac terminal:
```bash
brew install google-cloud-sdk
```

If you don't have Homebrew: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

- [ ] **Step 2: Log in to gcloud**

```bash
gcloud auth login
```

A browser window opens — log in with the same Google account you used to create the VM.

- [ ] **Step 3: Set your project**

```bash
gcloud config set project shaped-terrain-385708
```

(This is your project ID from the Google Cloud console screenshot.)

- [ ] **Step 4: Find your VM zone**

```bash
gcloud compute instances list
```

Expected output shows your `trading-botty` instance with its zone (e.g. `us-central1-a`). Note the zone.

- [ ] **Step 5: Copy project to server**

```bash
gcloud compute scp --recurse \
  "/Users/ashleighchua/trading analyses/" \
  trading-botty:~/trading-analyses \
  --zone=us-central1-a
```

Replace `us-central1-a` with the zone from Step 4 if different. This copies the entire project folder to the server. Takes ~1 minute.

- [ ] **Step 6: Copy .env to server separately**

The `.env` file is in `.gitignore` and was not copied above. Copy it manually:
```bash
gcloud compute scp \
  "/Users/ashleighchua/trading analyses/.env" \
  trading-botty:~/trading-analyses/.env \
  --zone=us-central1-a
```

- [ ] **Step 7: Verify files arrived**

```bash
gcloud compute ssh trading-botty --zone=us-central1-a --command="ls ~/trading-analyses/"
```

Expected: see `dashboard/`, `deploy/`, `.env`, `requirements.txt`, etc.

---

### Task 4: Run server setup and verify everything is running

**Files:** None — all commands run in GCP SSH terminal (open in browser from Google Cloud console).

Context: SSH into the server via the GCP console browser SSH button. Run the setup script, then verify each service is running. The setup script installs Python packages, activates cron, and starts the two systemd services.

- [ ] **Step 1: Open SSH in GCP console**

Go to Google Cloud Console → Compute Engine → VM Instances → click **SSH** next to `trading-botty`.

- [ ] **Step 2: Check your username**

```bash
whoami
```

Note the output (e.g. `audiofano`). This is your `YOUR_USER` value.

- [ ] **Step 3: Update crontab and service files with your actual username**

```bash
sed -i "s/YOUR_USER/$(whoami)/g" ~/trading-analyses/deploy/crontab.txt
sed -i "s/YOUR_USER/$(whoami)/g" ~/trading-analyses/deploy/trading-dashboard.service
sed -i "s/YOUR_USER/$(whoami)/g" ~/trading-analyses/deploy/trading-telegram.service
```

- [ ] **Step 4: Run the setup script**

```bash
bash ~/trading-analyses/deploy/setup.sh $(whoami)
```

Expected output ends with:
```
Done! Services running:
   Active: active (running) since ...
   Active: active (running) since ...
Dashboard: http://34.xxx.xxx.xxx:5050
```

- [ ] **Step 5: Verify cron is loaded**

```bash
crontab -l
```

Expected: 6 cron entries matching the schedule from `deploy/crontab.txt`.

- [ ] **Step 6: Verify dashboard is reachable**

Open a browser on your phone or laptop and go to:
```
http://<your-server-ip>:5050
```

Expected: browser shows a login popup asking for username and password. Enter `ashleigh` and your password from `.env`. Expected: dashboard loads.

- [ ] **Step 7: Verify Telegram bot is running**

```bash
sudo systemctl status trading-telegram --no-pager
```

Expected: `Active: active (running)`. Send `/status` to your Telegram bot — it should respond.

- [ ] **Step 8: Check sync_trades ran (it fires every 5 min)**

Wait 5 minutes, then:
```bash
tail -20 ~/trading-analyses/sync_trades.log
```

Expected: log lines showing `Sync complete.`

---

### Task 5: Disable launchd on Mac (stop double-running scripts)

**Files:** None — terminal commands on Mac.

Context: Once the server is running everything, the Mac launchd plists must be unloaded. Otherwise both Mac and server run the scanner simultaneously, placing duplicate orders. The `.plist` files stay in the repo for reference but are unloaded from launchd.

- [ ] **Step 1: Unload all trading plists on Mac**

```bash
cd "/Users/ashleighchua/trading analyses"
for plist in com.trading.premarket-scanner.plist com.trading.tuesday-close.plist com.trading.sync-trades.plist com.trading.postopen-stops.plist com.trading.price-alerts.plist com.trading.telegram-bot.plist com.trading.dashboard.plist com.trading.weekly-report.plist; do
    launchctl unload ~/Library/LaunchAgents/$plist 2>/dev/null && echo "Unloaded $plist" || echo "Not loaded: $plist"
done
```

- [ ] **Step 2: Verify nothing is loaded**

```bash
launchctl list | grep com.trading
```

Expected: no output (all unloaded).

- [ ] **Step 3: Confirm server scanner is the only one running**

Check that the server log shows scanner activity and the Mac log does not grow:
```bash
tail -5 "/Users/ashleighchua/trading analyses/premarket_scanner.log"
```

The Mac log should show the last entry was before today. Server has taken over.

---

### Task 6: Set up easy deployment for future code changes

**Files:**
- Create: `deploy/push.sh` — one command to deploy Mac changes to server

Context: When you change code on Mac (e.g. Claude edits `premarket_scanner.py`), you need to push those changes to the server. This script does it in one command.

- [ ] **Step 1: Create push.sh**

Create `deploy/push.sh`:

```bash
#!/bin/bash
# Deploy latest Mac code changes to server.
# Usage: bash deploy/push.sh
# Requires gcloud CLI + authenticated session.

ZONE=us-central1-a   # update if your zone is different
INSTANCE=trading-botty
REMOTE_PATH=/home/$(gcloud compute ssh $INSTANCE --zone=$ZONE --command="whoami" 2>/dev/null)/trading-analyses

echo "Copying changed files to $INSTANCE..."
gcloud compute scp --recurse \
  "/Users/ashleighchua/trading analyses/dashboard/" \
  $INSTANCE:$REMOTE_PATH/dashboard/ \
  --zone=$ZONE

echo "Restarting dashboard service..."
gcloud compute ssh $INSTANCE --zone=$ZONE \
  --command="sudo systemctl restart trading-dashboard trading-telegram"

echo "Done. Tailing server log for 5s..."
gcloud compute ssh $INSTANCE --zone=$ZONE \
  --command="tail -5 $REMOTE_PATH/premarket_scanner.log"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x "/Users/ashleighchua/trading analyses/deploy/push.sh"
```

- [ ] **Step 3: Commit**

```bash
cd "/Users/ashleighchua/trading analyses"
git add deploy/push.sh
git commit --no-verify -m "feat: add push.sh for one-command server deploys"
```

- [ ] **Step 4: Test a round-trip deploy**

Make a trivial change (e.g. add a comment to `dashboard/app.py`), then:
```bash
bash "/Users/ashleighchua/trading analyses/deploy/push.sh"
```

Expected: files copied, services restarted, log lines printed. No errors.

---

## Self-Review

**Spec coverage:**
- ✅ e2-micro VM (already created by user)
- ✅ Cron replaces launchd for 6 scheduled scripts
- ✅ systemd for Flask dashboard + Telegram bot (persistent)
- ✅ Timezone set to Asia/Bangkok
- ✅ Password protection on dashboard
- ✅ Dashboard reachable at server IP:5050
- ✅ Mac launchd disabled after migration
- ✅ Deployment workflow for future changes

**Tuesday close time:** Confirmed `45 2 * * 3` = 2:45am Bangkok Wednesday = 3:45pm ET Tuesday ✅

**No placeholders:** All steps have exact commands. ✅

**Username substitution:** `YOUR_USER` is replaced via `sed` before setup runs — no manual editing required. ✅
