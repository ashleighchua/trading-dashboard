#!/bin/bash
# Deploy latest Mac code changes to server.
# Usage: bash deploy/push.sh
# Requires gcloud CLI + authenticated session.

ZONE=us-central1-a   # update if your zone is different
INSTANCE=trading-botty
REMOTE_USER=$(gcloud compute ssh $INSTANCE --zone=$ZONE --command="whoami" 2>/dev/null)
REMOTE_PATH=/home/$REMOTE_USER/trading-analyses

echo "Copying dashboard/ to $INSTANCE (excluding trades.db and generated files)..."
rsync -avz --exclude='trades.db' --exclude='monthly_stats.json' --exclude='__pycache__' --exclude='*.pyc' \
  -e "ssh -i ~/.ssh/google_compute_engine" \
  "/Users/ashleighchua/trading analyses/dashboard/" \
  136.112.6.129:$REMOTE_PATH/dashboard/

echo "Restarting services..."
gcloud compute ssh $INSTANCE --zone=$ZONE \
  --command="sudo systemctl restart trading-dashboard trading-telegram"

echo "Done. Last 5 lines of scanner log:"
gcloud compute ssh $INSTANCE --zone=$ZONE \
  --command="tail -5 $REMOTE_PATH/premarket_scanner.log 2>/dev/null || echo '(no log yet)'"
