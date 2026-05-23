#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/TinyLlama-1.1B-Chat-v1.0}"
HF_HOME="${HF_HOME:-$ROOT_DIR/hf_cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
PYTHON="$ROOT_DIR/reference/.venv/bin/python"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

if [ -f "$ROOT_DIR/.env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env.local"
  set +a
fi

export HF_HOME
export UV_CACHE_DIR

if ! command -v uv >/dev/null 2>&1; then
  echo "missing dependency: uv" >&2
  exit 1
fi

echo "syncing reference environment"
uv sync --project reference --locked

echo "running local reference tests"
"$PYTHON" -m unittest discover -s tests

echo "running TinyLlama checkpoint smoke test"
"$PYTHON" reference/tiny_llama.py \
  --model-dir "$MODEL_DIR" \
  --model-id "$MODEL_ID" \
  --hf-home "$HF_HOME" \
  --max-new-tokens 1
