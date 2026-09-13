#!/usr/bin/env python3
"""Compare agentx_fast runs with the InferenceX agentic numbers at the same concurrency (MiniMax-M3 MXFP4, MI355X TP4,
run 33889796302, profiling requests: TTFT median / p90, ITL median / p90 = per-token decode latency there, TPOT here;
interactivity = aiperf's output_token_throughput_per_user = 1000 / TPOT per request, p50 / p90 across requests).

    python3 agentx_fast_compare.py work/agentx_fast_val_c4.json work/agentx_fast_val_c20.json
"""
import json, sys
import numpy as np

INFX = {  # conc: (ttft_median_ms, ttft_p90_ms, itl_median_ms, itl_p90_ms, uncached_isl_mean, osl_mean, intvty_p50, intvty_p90)
    1: (929, None, 3.31, None, 5709, 1944, None, None),
    4: (478, 1537, 3.31, 4.32, 4351, 1276, 302.1, 372.4),
    20: (579, 1259, 8.27, 15.57, 4731, 1038, 120.9, 230.1),
}


def intvty(r, p):
    """pNN interactivity (tok/s/user); computed from the per-turn records for files older than the key."""
    if f"p{p}_interactivity" in r:
        return r[f"p{p}_interactivity"]
    v = [1000.0 / t["tpot_ms"] for t in r["turns_detail"] if t["ok"] and t["tpot_ms"] > 0]
    return float(np.percentile(v, p)) if v else float("nan")


print("| run | conc | turns | TTFT median / p90 ms (InferenceX) | TPOT median / p90 ms (InferenceX ITL) | "
      "interactivity p50 / p90 tok/s/user (InfX) | new tokens per turn (InfX) | decode per turn (InfX) | cached | cold prefill tok/s | out tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for path in sys.argv[1:]:
    r = json.load(open(path))
    c = r["conc"]
    t = INFX.get(c, (None,) * 8)
    f = lambda v: "-" if v is None else f"{v:,.0f}" if v >= 100 else f"{v:.2f}"
    n = max(1, r["completed"])
    print(f"| {path.split('/')[-1]} | {c} | {r['turns']} | {r['median_ttft_ms']:.0f} / {r['p90_ttft_ms']:.0f} ({f(t[0])} / {f(t[1])}) | "
          f"{r['median_tpot_ms']:.2f} / {r['p90_tpot_ms']:.2f} ({f(t[2])} / {f(t[3])}) | "
          f"{intvty(r, 50):.0f} / {intvty(r, 90):.0f} ({f(t[6])} / {f(t[7])}) | "
          f"{r['total_new_prompt_tokens']/n:,.0f} ({f(t[4])}) | {r['total_output_tokens']/n:,.0f} ({f(t[5])}) | "
          f"{r['cached_fraction']:.1%} | {r['cold_prefill_throughput']:,.0f} | {r['output_throughput']:,.0f} |")
