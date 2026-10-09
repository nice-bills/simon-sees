#!/usr/bin/env bash
# One-command playtest: install (first time only), start the server, give you the phone link, and when you
# press Ctrl+C at the end, upload the recorded session (or zip it if there is no HF_TOKEN).
#
#   ./run_playtest.sh            # real referee: SAM 3 -> DINOv3 -> d1-3B on this laptop (GPU/Apple chip if any)
#   ./run_playtest.sh fast       # lighter models (RT-DETR + DINOv2-small + d1-omni-600M), for CPU laptops
#   ./run_playtest.sh mock       # no AI at all: checks camera, phone link, recording and upload (random calls)
#   ./run_playtest.sh modal      # vision + d1 on your Modal GPU (needs SIMON_VISION_URL, see docs/modal.md)
#
# Guide: docs/playtest.md
set -euo pipefail
cd "$(dirname "$0")"
PROFILE="${1:-real}"
PORT="${PORT:-8000}"
PY="${PYTHON:-python3}"

case "$PROFILE" in
  real)  FLAGS=();                                                                 REQ=requirements-model.txt ;;
  fast)  FLAGS=(--segmenter fallback --embedder dinov2 --model LiquidAI/d1-omni-600M); REQ=requirements-model.txt ;;
  mock)  FLAGS=(--segmenter mock --embedder mock --decider mock);                  REQ=requirements.txt ;;
  modal) FLAGS=(--backend modal);                                                  REQ=requirements.txt
         [ -n "${SIMON_VISION_URL:-}" ] || { echo "modal needs SIMON_VISION_URL (docs/modal.md)"; exit 1; } ;;
  *) echo "usage: $0 [real|fast|mock|modal]"; exit 2 ;;
esac

# 1. a private Python environment, installed once per profile
if [ ! -d .venv ]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [ ! -f ".venv/.installed-$REQ" ]; then
  echo "Installing $REQ (first run only, can take a few minutes)..."
  pip install -q --upgrade pip
  pip install -q -r "$REQ"
  touch ".venv/.installed-$REQ"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Note: HF_TOKEN is not set. The real models need it to download SAM 3, and upload needs it at the end."
  echo "      export HF_TOKEN=hf_...   (docs/playtest.md, step 3)"
fi

# 2. the server (recording is on by default; sessions/ fills up as you play)
python -m simon_sees serve --port "$PORT" ${FLAGS[@]+"${FLAGS[@]}"} &
SERVER=$!
TUNNEL=""

finish() {
  trap - INT TERM EXIT
  echo; echo "Stopping..."
  [ -n "$TUNNEL" ] && kill "$TUNNEL" 2>/dev/null || true
  kill "$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  if [ -n "${HF_TOKEN:-}" ]; then
    python -m simon_sees upload || python -m simon_sees bundle
  else
    python -m simon_sees bundle
  fi
}
trap finish INT TERM EXIT

# 3. an https link for the phone (browsers only allow the camera over https)
if command -v cloudflared >/dev/null 2>&1; then
  LOG="$(mktemp)"
  cloudflared tunnel --url "http://localhost:$PORT" >"$LOG" 2>&1 &
  TUNNEL=$!
  for _ in $(seq 1 30); do
    URL="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
    [ -n "$URL" ] && break
    sleep 1
  done
  echo; echo "  Open this on your phone:  ${URL:-(see $LOG)}"; echo
else
  echo; echo "  cloudflared is not installed, so there is no https link for the phone yet."
  echo "  Install it (docs/playtest.md, step 4) or follow docs/phone-setup.md (option C, same Wi-Fi)."; echo
fi
echo "  When you are done playing, press Ctrl+C here: the session is uploaded (or zipped) for you."
wait "$SERVER"
