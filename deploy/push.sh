#!/bin/bash
# Deploy latest Mac code changes to server.
# Usage: bash deploy/push.sh
# Requires gcloud CLI + authenticated session.

ZONE=us-central1-a   # update if your zone is different
INSTANCE=trading-botty
REMOTE_USER=$(gcloud compute ssh $INSTANCE --zone=$ZONE --command="whoami" 2>/dev/null)
REMOTE_PATH=/home/$REMOTE_USER/trading-analyses

echo "Copying dashboard/ to $INSTANCE..."
gcloud compute scp --recurse \
  "/Users/ashleighchua/trading analyses/dashboard/" \
  $INSTANCE:$REMOTE_PATH/dashboard/ \
  --zone=$ZONE

echo "Restarting services..."
gcloud compute ssh $INSTANCE --zone=$ZONE \
  --command="sudo systemctl restart trading-dashboard trading-telegram"

echo "Done. Last 5 lines of scanner log:"
gcloud compute ssh $INSTANCE --zone=$ZONE \
  --command="tail -5 $REMOTE_PATH/premarket_scanner.log 2>/dev/null || echo '(no log yet)'"
