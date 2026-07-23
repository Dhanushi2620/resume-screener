#!/bin/bash
# Start Resume Screener on port 8010
set -e
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Start Ollama if available and not running
if [ -x "$HOME/ollama" ] && ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama..."
  nohup "$HOME/ollama" serve > /tmp/ollama.log 2>&1 &
  sleep 3
fi

# Stop previous instance of this app only
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

echo "Starting Resume Screener at http://127.0.0.1:8010 ..."
echo "Press Ctrl+C to stop."
exec uvicorn main:app --host 127.0.0.1 --port 8010
