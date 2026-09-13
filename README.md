# bench_serving

Serving benchmarks for MI355X / vLLM work. `benchmark_serving.py` + `backend_request_func.py` are the InferenceX
random-prompt client (`1_bench.py` / `1_bench.sh` drive it over a concurrency sweep).

## Fast agentX proxy (`agentx_fast.py`)

The InferenceX agentic point (trace replay, one hour per concurrency) in minutes: N concurrent sessions, each a
long cached context plus K turns of (append U fresh prompt tokens, decode D tokens), the shape measured from the
InferenceX run (MiniMax-M3 MXFP4, MI355X TP4): see `PLAN_AGENTX_FAST.md` in benenzhu/m3-compare.

- `profile_tp4.json`: per-concurrency session contexts (quantiles of the recorded per-request ISL) and the
  append / decode ladders (8 stratum means of the recorded distributions); built by `build_profile.py` from
  the run's artifacts.
- prompts are token-id lists on `/v1/completions` (exact prefix-cache hits, no tokenizer); one `--seed`
  fixes every length and id, so before/after runs send identical work.
- `--uniform` replaces the ladders by U[0.8x, x] draws around the recorded means (the `--random-range-ratio 0.8`
  style of `benchmark_serving.py`).

```
python3 agentx_fast.py --base-url http://127.0.0.1:8888 --model <served name> --tokenizer <path> \
    --conc 20 --turns 8 --seed 0 --result-dir out --result-filename agentx_fast_c20.json --verbose
```

`run_agentx_fast.sh <conc> <turns> <tag>` runs it from the vLLM client image (env `MODEL`, `TOKENIZER`, `BASE_URL`,
`CLI_IMG`, `MODELS_DIR`, `RESULT_DIR`, `SEED`); `agentx_fast_compare.py <result.json ...>` prints the runs next to the
InferenceX numbers at the same concurrency.

Output: benchmark_serving-style keys (`median_ttft_ms`, `p99_tpot_ms`, `output_throughput`, ...) plus every
turn's record. With the server's `--stream-interval 20`, ITL is per 20-token chunk; use TPOT for per-token
decode latency.

Server recipe for MiniMax-M3 (the InferenceX one): TP4, `--kv-cache-dtype fp8`, `--enable-prefix-caching`,
`--max-model-len 1048576`, `--max-num-batched-tokens 32768`, `--max-num-seqs 2*conc`, MTP eagle3 draft.

## InferenceX artifact analysis (`infx_agentic_summary.py`)

Per-job table (cache hit rates, nominal and uncached ISL, OSL, TTFT) from the raw aiperf artifacts of an
InferenceX agentic run, unpacked as `<root>/tp4_conc<N>/`. The uncached prompt tokens per request come from
vLLM's per-second prefix-cache counters (exact in aggregate; the per-request split is approximate above
conc ~8).

## gsm8k check on demand (`run_gsm8k.sh`)

lm-eval in a long-lived CPU-only docker container (never the host python), the InferenceX CI convention
(`local-chat-completions`, chat template, `eval/ci/gsm8k_ci.yaml`, max_tokens 12288, reasoning_content fallback via
`eval/ci/sitecustomize.py`), on a subset: `LIMIT=100 CONCURRENT=20 PORT=8888 OUT=c20 ./run_gsm8k.sh` prints
flexible-extract / strict-match exact_match (M3 MXFP4: 0.98 / 0.97 on the 100-question subset at conc 4 / 20,
~0.96 on the full set). Not part of the perf A/B — run it when the accuracy is in doubt. The server must run
speculative decoding **off or with real rejection sampling** — synthetic acceptance (the InferenceX perf recipe)
accepts draft tokens at random and the score is meaningless; the method is fixed at engine start, so that
means a server restart.
