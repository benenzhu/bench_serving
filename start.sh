set -x 
export MODEL="/A/MiniMax-M2.5"
export PORT=30000
export TP=8
ps aux | grep vllm |awk '{print $2}' | xargs kill -9
sleep 1
export VLLM_ROCM_USE_AITER=1
export SGLANG_TORCH_PROFILER_DIR=/A
export SGLANG_USE_AITER=1
export SERVER_LOG=a.log
export MAX_MODEL_LEN=10240
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.8 \
--max-num-seqs 512 \
--max-num-batched-tokens 196608 \
--max-model-len $MAX_MODEL_LEN \
--block-size=32 \
--disable-log-requests \
--profiler-config.profiler torch \
--profiler-config.torch_profiler_dir /app/profiler_traces \
--profiler-config.torch_profiler_record_shapes true \
--profiler-config.torch_profiler_with_stack true \
--profiler-config.delay_iterations 5 \
--profiler-config.max_iterations 3 \
--profiler-config.ignore_frontend true \
--enable-expert-parallel \
--trust-remote-code 2>&1 |tee $SERVER_LOG






--data-parallel-size=8 \
