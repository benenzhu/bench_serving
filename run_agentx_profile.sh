#!/usr/bin/env bash
# Run agentx_profile_window.py from a vLLM client image against a profiler-enabled server on BASE_URL.
#   MODEL=<path as served> [TOKENIZER=$MODEL] [BASE_URL=http://127.0.0.1:8888] [CLI_IMG=...] [MODELS_DIR=...]
#   [RESULT_DIR=$PWD/results] [SEED=0] run_agentx_profile.sh <conc> <mode> <tag> [extra agentx_profile_window.py args]
# Writes $RESULT_DIR/agentx_profile_<tag>.log. The traces are written by the server into its own profiler dir.
set -uo pipefail
CONC=$1; MODE=$2; TAG=$3; shift 3
: "${MODEL:?set MODEL to the served model name / path}"
TOKENIZER="${TOKENIZER:-$MODEL}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8888}"
CLI_IMG="${CLI_IMG:-vllm/vllm-openai-rocm:nightly-6d4562c59b97b4e35d459ff9389e71b6fe4995de}"
MODELS_DIR="${MODELS_DIR:-/mnt/disk/huggingface/hub}"
RESULT_DIR="${RESULT_DIR:-$PWD/results}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RESULT_DIR"
for i in $(seq 1 120); do curl -sf "$BASE_URL/health" >/dev/null 2>&1 && break; sleep 10; done
curl -sf "$BASE_URL/health" >/dev/null || { echo "server not healthy at $BASE_URL"; exit 1; }
docker run --rm --name "agentx_prof_cli_$$" --network=host \
  -v "$HERE:/bs:ro" -v "$MODELS_DIR:/models:ro" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint python3 "$CLI_IMG" \
  /bs/agentx_profile_window.py --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
  --conc "$CONC" --mode "$MODE" --seed "${SEED:-0}" --profile /bs/profile_tp4.json "$@" \
  2>&1 | tee "$RESULT_DIR/agentx_profile_${TAG}.log"
