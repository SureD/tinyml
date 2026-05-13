#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/TinyLlama-1.1B-Chat-v1.0}"
HF_HOME="${HF_HOME:-$ROOT_DIR/hf_cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
PYTHON="$ROOT_DIR/ref/.venv/bin/python"

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

echo "syncing ref environment"
uv sync --project ref --locked

REQUIRED_MODEL_FILES=(
  "config.json"
  "generation_config.json"
  "model.safetensors"
  "special_tokens_map.json"
  "tokenizer.json"
  "tokenizer.model"
  "tokenizer_config.json"
)

MODEL_READY=1
for file in "${REQUIRED_MODEL_FILES[@]}"; do
  if [ ! -f "$MODEL_DIR/$file" ]; then
    MODEL_READY=0
    break
  fi
done

if [ "$MODEL_READY" -eq 1 ]; then
  echo "model already present: $MODEL_DIR"
else
  echo "downloading model: $MODEL_ID"
  "$PYTHON" tools/download_tinyllama.py \
    --model-id "$MODEL_ID" \
    --model-dir "$MODEL_DIR" \
    --hf-home "$HF_HOME"
fi

echo "running TinyLlama checkpoint smoke test"
"$PYTHON" ref/tiny_llama.py --model-dir "$MODEL_DIR" --max-new-tokens 1

echo "running local harness tests"
"$PYTHON" -m unittest discover -s tests
