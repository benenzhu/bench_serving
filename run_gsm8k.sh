#!/usr/bin/env bash
# gsm8k against a running OpenAI-compatible server, lm-eval inside a docker container (never the host python).
# InferenceX CI convention: local-chat-completions + --apply_chat_template + eval/ci/gsm8k_ci.yaml,
# max_tokens 12288 / max_length 16384, the reasoning_content fallback of eval/ci/sitecustomize.py.
#
#   LIMIT=100 CONCURRENT=20 PORT=8888 OUT=c20 MODEL_NAME=amd/MiniMax-M3-MXFP4 ./run_gsm8k.sh
#     LIMIT       questions (default 100; empty = the full 1319)
#     CONCURRENT  parallel requests (default 32; set it to the concurrency under test)
#     OUT         name of the result directory under $RESULT_DIR/eval/ (default gsm8k_<timestamp>)
#     RESULT_DIR  default $PWD/results (mounted as /workspace in the container)
#     LMEVAL_IMAGE / LMEVAL_CTR  the client image and container name (reused across calls; docker rm -f to reset)
#
# The server's speculative decoding must be off or use real rejection sampling: with synthetic acceptance
# the draft tokens are accepted at random and the answers are wrong by construction.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${LMEVAL_IMAGE:-vllm/vllm-openai-rocm:nightly-6d4562c59b97b4e35d459ff9389e71b6fe4995de}"
CTR="${LMEVAL_CTR:-bench_lmeval}"
LM_EVAL_REF=b315ef3b05176acc9732bb7fdec116abe1ecc476   # the ref InferenceX CI pins
PORT="${PORT:-8888}"
LIMIT="${LIMIT-100}"
CONCURRENT="${CONCURRENT:-32}"
MODEL_NAME="${MODEL_NAME:-amd/MiniMax-M3-MXFP4}"
RESULT_DIR="${RESULT_DIR:-$PWD/results}"
OUT="${OUT:-gsm8k_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR/eval/hf_cache"
cp -r "$HERE/eval/ci" "$RESULT_DIR/eval/" 2>/dev/null || true

if ! docker ps --format '{{.Names}}' | grep -qx "$CTR"; then
    docker rm -f "$CTR" 2>/dev/null || true
    docker run -d --name "$CTR" --network=host -v "$RESULT_DIR":/workspace -e HF_HOME=/workspace/eval/hf_cache \
        --entrypoint bash "$IMAGE" -c 'sleep infinity' >/dev/null
    echo "[$CTR] installing lm-eval (${LM_EVAL_REF:0:8})..."
    docker exec "$CTR" bash -c "pip install -q --no-cache-dir 'lm-eval[api]' && pip install -q --no-cache-dir --no-deps \
        'https://github.com/EleutherAI/lm-evaluation-harness/archive/${LM_EVAL_REF}.tar.gz' && python3 -c 'import lm_eval; print(\"lm-eval\", lm_eval.__version__)'"
fi
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "no server on :$PORT"; exit 1; }

LIMIT_ARG=(); [ -n "$LIMIT" ] && LIMIT_ARG=(--limit "$LIMIT")
docker exec -e PYTHONPATH=/workspace/eval/ci "$CTR" python3 -m lm_eval \
    --model local-chat-completions --apply_chat_template \
    --tasks /workspace/eval/ci/gsm8k_ci.yaml "${LIMIT_ARG[@]}" \
    --output_path "/workspace/eval/$OUT" --log_samples \
    --model_args "model=${MODEL_NAME},base_url=http://127.0.0.1:${PORT}/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=${CONCURRENT},timeout=1800,tokenized_requests=False,max_length=16384" \
    --gen_kwargs "max_tokens=12288,temperature=0,top_p=1" \
    2>&1 | tee "$RESULT_DIR/eval/$OUT.log" | grep -E "^\|\s*gsm8k|exact_match|Traceback|Error" | tail -n 6

python3 - "$RESULT_DIR/eval/$OUT" <<'EOF'
import glob, json, sys
fs = sorted(glob.glob(sys.argv[1] + "/**/results_*.json", recursive=True))
if not fs:
    sys.exit("no results json under " + sys.argv[1])
r = json.load(open(fs[-1]))["results"]["gsm8k"]
n = json.load(open(fs[-1])).get("n-samples", {}).get("gsm8k", {}).get("effective", "?")
print(f"GSM8K n={n} flexible-extract={r['exact_match,flexible-extract']:.4f} strict-match={r['exact_match,strict-match']:.4f}")
EOF
