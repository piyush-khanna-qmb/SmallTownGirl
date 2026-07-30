#!/bin/bash
# SmallTownGirl.app launcher: bootstraps a Python 3.12 venv + model on first
# run (MediaPipe/OpenCV are too large to bundle), then starts the menu-bar app.
set -uo pipefail

RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
DATA="$HOME/Library/Application Support/SmallTownGirl"
VENV="$DATA/venv"
MODEL="$DATA/hand_landmarker.task"
LOG="$DATA/launch.log"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "$DATA"
exec >>"$LOG" 2>&1
echo "=== launch $(date) ==="

notify() { /usr/bin/osascript -e "display notification \"$2\" with title \"$1\"" >/dev/null 2>&1 || true; }
fail()   { /usr/bin/osascript -e "display alert \"SmallTownGirl\" message \"$1\"" >/dev/null 2>&1 || true; exit 1; }

# Locate Homebrew's python@3.12 (GUI launches have a minimal PATH).
PY=""
for p in /opt/homebrew/opt/python@3.12/bin/python3.12 /usr/local/opt/python@3.12/bin/python3.12; do
  [ -x "$p" ] && PY="$p" && break
done
[ -z "$PY" ] && PY="$(command -v python3.12 || true)"
[ -z "$PY" ] && fail "Python 3.12 was not found. Install it with:  brew install python@3.12"

if [ ! -x "$VENV/bin/python" ]; then
  notify "SmallTownGirl" "First-time setup (~1 min, one time)…"
  "$PY" -m venv "$VENV" || fail "Could not create the Python environment. See launch.log."
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null 2>&1
  if ! "$VENV/bin/python" -m pip install -r "$RES/requirements.txt"; then
    fail "Dependency install failed — check your network. Details in $LOG"
  fi
  notify "SmallTownGirl" "Ready — look for the icon in your menu bar."
fi

if [ ! -f "$MODEL" ]; then
  /usr/bin/curl -fsSL -o "$MODEL" "$MODEL_URL" || fail "Could not download the hand model — check your network."
fi

export GESTURE_SCROLL_MODEL="$MODEL"
exec "$VENV/bin/python" "$RES/menubar.py"
