#!/bin/bash
# IMCFlow CI Manager Script
# Manages the CI runner as a background process

set -e

# Get the directory where this script is located
CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_SCRIPT="$CI_DIR/ci_runner.py"
PID_FILE="$CI_DIR/logs/ci_runner.pid"
LOG_DIR="$CI_DIR/logs"
REPO_DIR="/root/project/tvm"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure log directory exists
mkdir -p "$LOG_DIR"

get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    fi
}

is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

start() {
    if is_running; then
        echo -e "${YELLOW}CI runner is already running (PID: $(get_pid))${NC}"
        return 1
    fi

    echo -e "${GREEN}Starting IMCFlow CI runner...${NC}"

    # Check if direnv is installed
    if ! command -v direnv &> /dev/null; then
        echo -e "${RED}❌ Error: direnv is not installed${NC}"
        echo "The CI runner requires direnv to load the correct environment."
        echo "Please install direnv: https://direnv.net/docs/installation.html"
        return 1
    fi

    # Check if .envrc files exist
    if [ ! -f "$REPO_DIR/.envrc" ] || [ ! -f "$REPO_DIR/tvm_practice/.envrc" ]; then
        echo -e "${YELLOW}⚠️  Warning: .envrc files not found${NC}"
        echo "Expected files:"
        echo "  - $REPO_DIR/.envrc"
        echo "  - $REPO_DIR/tvm_practice/.envrc"
    fi

    # Check if GitHub token is set
    if [ -z "$GITHUB_TOKEN" ]; then
        echo -e "${YELLOW}Warning: GITHUB_TOKEN not set. GitHub status updates will be disabled.${NC}"
        echo "To enable GitHub reporting, set the GITHUB_TOKEN environment variable."
        echo "Export it in your shell profile or create a .env file."
    fi

    # Start the CI runner in background
    # Use -u flag for unbuffered output so logs appear immediately
    cd "$REPO_DIR"
    nohup python3 -u "$CI_SCRIPT" > "$LOG_DIR/ci_runner_main.log" 2>&1 &
    local pid=$!

    # Save PID
    echo $pid > "$PID_FILE"

    # Wait a moment to see if it starts successfully
    sleep 2

    if is_running; then
        echo -e "${GREEN}✅ CI runner started successfully (PID: $pid)${NC}"
        echo "   Logs: $LOG_DIR"
        echo "   Main log: $LOG_DIR/ci_runner_main.log"
        echo "   Using direnv for environment management"
        echo ""
        echo "To view live logs: tail -f $LOG_DIR/ci_runner_main.log"
        return 0
    else
        echo -e "${RED}❌ Failed to start CI runner${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop() {
    if ! is_running; then
        echo -e "${YELLOW}CI runner is not running${NC}"
        rm -f "$PID_FILE"
        return 1
    fi

    local pid=$(get_pid)
    echo -e "${YELLOW}Stopping CI runner (PID: $pid)...${NC}"

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    # Wait for process to stop (max 30 seconds)
    local count=0
    while kill -0 "$pid" 2>/dev/null && [ $count -lt 30 ]; do
        sleep 1
        count=$((count + 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${RED}Process didn't stop gracefully, force killing...${NC}"
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ CI runner stopped${NC}"
}

restart() {
    echo "Restarting CI runner..."
    stop
    sleep 2
    start
}

status() {
    if is_running; then
        local pid=$(get_pid)
        echo -e "${GREEN}✅ CI runner is running (PID: $pid)${NC}"

        # Show some stats if state file exists
        if [ -f "$LOG_DIR/ci_state.json" ]; then
            echo ""
            echo "Last test run:"
            cat "$LOG_DIR/ci_state.json" | python3 -m json.tool 2>/dev/null || cat "$LOG_DIR/ci_state.json"
        fi

        # Show recent logs
        echo ""
        echo "Recent activity (last 10 lines):"
        tail -10 "$LOG_DIR/ci_runner_main.log" 2>/dev/null || echo "No logs yet"

        return 0
    else
        echo -e "${RED}❌ CI runner is not running${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

logs() {
    local lines=${1:-50}
    if [ -f "$LOG_DIR/ci_runner_main.log" ]; then
        tail -n "$lines" -f "$LOG_DIR/ci_runner_main.log"
    else
        echo "No logs found at $LOG_DIR/ci_runner_main.log"
        return 1
    fi
}

list_test_logs() {
    echo "Test run logs in $LOG_DIR:"
    ls -lht "$LOG_DIR"/test_*.log 2>/dev/null || echo "No test logs found"
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    list)
        list_test_logs
        ;;
    *)
        echo "IMCFlow CI Manager"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs [lines]|list}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the CI runner in background"
        echo "  stop     - Stop the CI runner"
        echo "  restart  - Restart the CI runner"
        echo "  status   - Show CI runner status"
        echo "  logs     - Show and follow CI runner logs (default: 50 lines)"
        echo "  list     - List all test run logs"
        echo ""
        echo "Examples:"
        echo "  $0 start"
        echo "  $0 logs 100    # Show last 100 lines"
        echo "  $0 status"
        exit 1
        ;;
esac
