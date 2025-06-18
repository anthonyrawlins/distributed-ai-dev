#!/bin/bash
#
# Setup Monitoring Tools
# Adds monitoring commands to your PATH for easy access
#
# Usage: source setup-monitoring.sh
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"

# Check if bin directory exists
if [ ! -d "$BIN_DIR" ]; then
    echo "❌ Error: $BIN_DIR directory not found"
    return 1
fi

# Add to PATH if not already there
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    export PATH="$BIN_DIR:$PATH"
    echo "✅ Added monitoring tools to PATH"
else
    echo "ℹ️  Monitoring tools already in PATH"
fi

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
    echo "✅ Loaded environment configuration"
fi

echo ""
echo "🚀 Available monitoring commands:"
echo "   simple-monitor   - Simple terminal-based agent monitor"
echo "   advanced-monitor - Advanced curses-based agent monitor"
echo "   agent-monitor    - Agent 113 dedicated monitor"
echo ""
echo "📖 Usage examples:"
echo "   simple-monitor --refresh 3"
echo "   advanced-monitor --single-agent"
echo "   agent-monitor"
echo ""
echo "❓ Help:"
echo "   simple-monitor --help"
echo "   advanced-monitor --help"