#!/bin/bash
# Daily Signal Checker — runs at 9:00 PM Thailand time (Mon-Fri)
# Logs output and sends macOS notifications

SCRIPT_DIR="/Users/ashleighchua/trading analyses"
LOG_FILE="$SCRIPT_DIR/signal_checker.log"
ERR_FILE="$SCRIPT_DIR/signal_checker_error.log"
PYTHON="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3"

echo "==============================" >> "$LOG_FILE"
echo "Run: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "==============================" >> "$LOG_FILE"

$PYTHON "$SCRIPT_DIR/daily_signal_checker.py" >> "$LOG_FILE" 2>> "$ERR_FILE"

echo "" >> "$LOG_FILE"
