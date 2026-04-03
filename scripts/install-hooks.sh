#!/bin/bash
# Install git hooks for trading system safety

set -e

HOOKS_DIR="$(git rev-parse --git-dir)/hooks"

echo "Installing pre-commit hook for stop-loss enforcement..."

cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook: Review gate for order-placement code
# Blocks commits to critical trading files without explicit approval

CRITICAL_FILES=(
    "dashboard/premarket_scanner.py"
    "dashboard/telegram_bot.py"
    "dashboard/postopen_stops.py"
)

# Check if any critical files are staged
STAGED_CRITICAL=()
for file in "${CRITICAL_FILES[@]}"; do
    if git diff --cached --name-only | grep -q "^$file$"; then
        STAGED_CRITICAL+=("$file")
    fi
done

if [ ${#STAGED_CRITICAL[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  CRITICAL FILES CHANGED — REVIEW REQUIRED"
    echo ""
    echo "You are about to commit changes to:"
    for file in "${STAGED_CRITICAL[@]}"; do
        echo "  • $file"
    done
    echo ""
    echo "These files control order placement and must have manual review."
    echo ""
    echo "Rules to check:"
    echo "  ✓ Every entry order has a trailing stop attached (default 1.5%)"
    echo "  ✓ No naked orders (entry without stop)"
    echo "  ✓ Stop-loss code is reached before order returns"
    echo "  ✓ Error handling wraps all Alpaca API calls"
    echo ""
    echo "Run: trading-code-review skill before committing"
    echo ""
    read -p "Do you understand and approve these changes? (type 'yes' to continue): " approval

    if [ "$approval" != "yes" ]; then
        echo "Commit cancelled."
        exit 1
    fi
    echo "✅ Approved. Proceeding with commit."
    echo ""
fi

exit 0
EOF

chmod +x "$HOOKS_DIR/pre-commit"

echo "✅ Pre-commit hook installed at $HOOKS_DIR/pre-commit"
echo ""
echo "How it works:"
echo "  • When you commit changes to critical order-placement files,"
echo "    git will ask for explicit approval"
echo "  • Type 'yes' to proceed with the commit"
echo "  • Always run 'trading-code-review' skill before committing"
