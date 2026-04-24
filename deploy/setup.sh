#!/bin/bash
# One-time server setup. Run once after copying the repo to the server.
# Usage: bash deploy/setup.sh YOUR_USERNAME
set -e

USER=$1
if [ -z "$USER" ]; then
    echo "Usage: bash deploy/setup.sh YOUR_USERNAME"
    exit 1
fi

REPO=/home/$USER/trading-analyses

# Install Python and pip
sudo apt update && sudo apt install -y python3 python3-pip git
echo "Python $(python3 --version) installed"

# Install Python dependencies
pip3 install -r $REPO/requirements.txt
echo "Python dependencies installed"

# Install crontab (replace YOUR_USER placeholder)
sed "s/YOUR_USER/$USER/g" $REPO/deploy/crontab.txt | crontab -
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
