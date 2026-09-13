#!/usr/bin/env python3
"""Compare agentx_fast runs with the InferenceX agentic numbers at the same concurrency (MiniMax-M3 MXFP4, MI355X TP4,
run 33889796302, profiling requests: TTFT median / p90, ITL median / p90 = per-token decode latency there, TPOT here).

    python3 agentx_fast_compare.py work/agentx_fast_val_c4.json work/agentx_fast_val_c20.json
"""
import json, sys

INFX = {  # conc: (ttft_median_ms, ttft_p90_ms, itl_median_ms, itl_p90_ms, uncached_isl_mean, osl_mean)
    1: (929, None, 3.31, None, 5709, 1944),
    4: (478, 1537, 3.31, 4.32, 4351, 1276),
    20: (579, 1259, 8.27, 15.57, 4731, 1038),
}

print("| run | conc | turns | TTFT median / p90 ms (InferenceX) | TPOT median / p90 ms (InferenceX ITL) | new tokens per turn (InfX) | decode per turn (InfX) | cached | cold prefill tok/s | out tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    r = json.load(open(path))
    c = r["conc"]
    t = INFX.get(c, (None,) * 6)
    f = lambda v: "-" if v is None else f"{v:,.0f}" if v >= 100 else f"{v:.2f}"
    n = max(1, r["completed"])
    print(f"| {path.split('/')[-1]} | {c} | {r['turns']} | {r['median_ttft_ms']:.0f} / {r['p90_ttft_ms']:.0f} ({f(t[0])} / {f(t[1])}) | "
          f"{r['median_tpot_ms']:.2f} / {r['p90_tpot_ms']:.2f} ({f(t[2])} / {f(t[3])}) | "
          f"{r['total_new_prompt_tokens']/n:,.0f} ({f(t[4])}) | {r['total_output_tokens']/n:,.0f} ({f(t[5])}) | "
          f"{r['cached_fraction']:.1%} | {r['cold_prefill_throughput']:,.0f} | {r['output_throughput']:,.0f} |")
