#!/usr/bin/env bash
# Start the Maharat RAG Ops UI — backend (FastAPI) + frontend (Vite)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual env if present
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

# Install Python deps if fastapi is missing
python3 -c "import fastapi" 2>/dev/null || {
  echo "Installing Python dependencies…"
  pip3 install -r requirements.txt -q
}

# Install frontend deps if needed
if [ ! -d "ui/node_modules" ]; then
  echo "Installing frontend dependencies…"
  (cd ui && npm install)
fi

echo ""
echo "Starting Maharat RAG Ops UI"
echo "  Backend  → http://localhost:8000"
echo "  Frontend → http://localhost:5173"
echo ""

# Start backend
python3 -m uvicorn api.main:app --reload --port 8000 &
API_PID=$!

# Start frontend
(cd ui && npm run dev) &
UI_PID=$!

trap "echo ''; echo 'Stopping…'; kill $API_PID $UI_PID 2>/dev/null; wait" EXIT INT TERM
wait
