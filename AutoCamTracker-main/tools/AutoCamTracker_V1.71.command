#!/bin/zsh
set -e

PROJECT_DIR="/Users/linen/.codex/worktrees/3267/AutoCamTracker"
LOG_FILE="/tmp/autocamtracker-v171-launch.log"

cd "$PROJECT_DIR"

echo "Starting AutoCamTracker V1.71..." | tee "$LOG_FILE"
echo "Project: $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"

if [[ -x ".venv/bin/autocamtracker-app" ]]; then
  exec .venv/bin/autocamtracker-app 2>&1 | tee -a "$LOG_FILE"
fi

if [[ -x ".venv/bin/python" ]]; then
  exec env PYTHONPATH=src .venv/bin/python -m autocamtracker.main 2>&1 | tee -a "$LOG_FILE"
fi

echo "Could not find .venv/bin/autocamtracker-app or .venv/bin/python." | tee -a "$LOG_FILE"
echo "Please install dependencies first:"
echo "  python -m venv .venv"
echo "  .venv/bin/python -m pip install -r requirements.txt"
echo "  .venv/bin/python -m pip install -e ."
read "?Press Return to close..."
