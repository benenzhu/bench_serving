export VLLM_ROCM_USE_AITER=1
vllm serve MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 8 \
    --enable_expert-parallel \
    --max-num-batched-tokens 196608 \
    --max-model-len=10240 \
    --max-num-seqs 512 \
    --block-size=32 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --port=30000
