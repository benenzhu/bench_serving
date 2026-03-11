#!/bin/bash
# Usage: bash bench.sh <CONC_START> <CONC_END>
# Example: bash bench.sh 256 32  => tests 256 128 64 32

export MODEL="/A/MiniMax-M2.5"
export ISL=8192
export OSL=1024
export RANDOM_RANGE_RATIO=0.8
export RESULT_FILENAME="dsr1_fp8_mi300x_docker.json"
export port=30000
export TP=8

CONC_START=${1:?Usage: bash bench.sh <CONC_START> <CONC_END>}
CONC_END=${2:?Usage: bash bench.sh <CONC_START> <CONC_END>}

set -x
cd /A

until curl --output /dev/null --silent --fail http://0.0.0.0:$port/health; do
    sleep 3
done

if [ ! -d /A/bench_serving ]; then
    git clone https://github.com/benenzhu/bench_serving.git bench_serving
fi

CONC=$CONC_START
while [ "$CONC" -ge "$CONC_END" ]; do
    echo "========== Testing CONC=$CONC =========="
    python3 /A/bench_serving/benchmark_serving.py \
        --model "$MODEL" \
        --backend "vllm" \
        --base-url "http://0.0.0.0:30000" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts $(( CONC * 4 )) \
        --max-concurrency "$CONC" \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --percentile-metrics 'ttft,tpot,itl,e2el' \
        --result-dir "/app/" \
        --result-filename "$RESULT_FILENAME.json" \
        2>&1 | tee "result_${CONC}_$(date +%Y%m%d_%H%M%S).log"
    CONC=$((CONC / 2))
done
