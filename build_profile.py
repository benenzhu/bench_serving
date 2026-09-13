#!/usr/bin/env python3
"""Build profile_tp4.json for agentx_fast.py from InferenceX agentic artifacts.

Input: a directory of unpacked `agentic_<model>_tp4_conc<N>_...` artifacts, one per concurrency, named
`tp4_conc<N>/` (each holding `aiperf_artifacts/profile_export.jsonl` and `server_metrics_export.json`);
download them with `gh api repos/SemiAnalysisAI/InferenceX/actions/artifacts/<id>/zip`.

Per request the uncached prompt tokens are attributed from vLLM's per-second prefix-cache counters
(queries = prompt length, hits = cached tokens, both counted once per request when it is scheduled;
the requests are matched to the counter slices cumulatively in start order). Output:
- contexts[N]: N evenly spaced quantiles of the per-request ISL of the conc-N job (the session contexts)
- append_ladder / decode_ladder: 8 stratum means of the uncached-prompt-tokens and decode-length
  distributions pooled over the eviction-free jobs (--ladder-concs, default 8 10 12 15 20)

    python3 build_profile.py --root art --out profile_tp4.json
"""
import argparse
import glob
import json
import os
import re
import statistics as st

K = 8


def per_request(job_dir):
    A = f"{job_dir}/aiperf_artifacts"
    recs = [json.loads(l) for l in open(f"{A}/profile_export.jsonl")]
    d = json.load(open(f"{A}/server_metrics_export.json"))
    S = lambda sec, k: d[sec][k]["series"][0]  # noqa: E731
    sl_h = S("metrics", "vllm:prefix_cache_hits")["timeslices"]
    sl_q = S("metrics", "vllm:prefix_cache_queries")["timeslices"]
    slices = [(a["start_ns"], a["total"], b["total"]) for a, b in zip(sl_q, sl_h) if a["total"] > 0]
    si, q_left, out = 0, slices[0][1], []
    for r in sorted(recs, key=lambda r: r["metadata"]["request_start_ns"]):
        m, md = r["metrics"], r["metadata"]
        if "input_sequence_length" not in m or "output_sequence_length" not in m:
            continue
        isl = m["input_sequence_length"]["value"]
        need, hits = isl, 0.0
        while need > 0 and si < len(slices):
            take = min(need, q_left)
            hits += take / slices[si][1] * slices[si][2]
            need -= take
            q_left -= take
            if q_left <= 0:
                si += 1
                q_left = slices[si][1] if si < len(slices) else 0
        if md["benchmark_phase"] == "profiling":
            out.append((int(isl), max(isl - hits, 0.0), int(m["output_sequence_length"]["value"])))
    pool = None
    info = d["metrics"].get("vllm:cache_config_info", {}).get("series", [{}])[0].get("labels", {})
    if info.get("kv_cache_size_tokens"):
        pool = int(float(info["kv_cache_size_tokens"]))
    return out, pool


def ladder(values):
    s = sorted(values)
    n = len(s)
    return [round(st.mean(s[int(k / K * n): int((k + 1) / K * n)])) for k in range(K)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="art")
    ap.add_argument("--out", default="profile_tp4.json")
    ap.add_argument("--ladder-concs", type=int, nargs="+", default=[8, 10, 12, 15, 20])
    ap.add_argument("--source", default="InferenceX run 33889796302, MiniMax-M3 MXFP4 MI355X TP4")
    args = ap.parse_args()
    jobs = {}
    for jd in glob.glob(f"{args.root}/tp4_conc*"):
        m = re.search(r"conc(\d+)$", jd)
        if m and os.path.isdir(f"{jd}/aiperf_artifacts"):
            jobs[int(m.group(1))] = jd
    rows, pools = {}, set()
    for conc, jd in sorted(jobs.items()):
        rows[conc], pool = per_request(jd)
        if pool:
            pools.add(pool)
    pooled = [x for c in args.ladder_concs if c in rows for x in rows[c]]
    prof = dict(
        source=args.source + ", profiling requests",
        turns_per_session=K,
        append_ladder=ladder([x[1] for x in pooled]),
        decode_ladder=ladder([x[2] for x in pooled]),
        append_mean=round(st.mean(x[1] for x in pooled)),
        decode_mean=round(st.mean(x[2] for x in pooled)),
        ladder_source=f"conc {args.ladder_concs} pooled ({len(pooled)} requests)",
        kv_pool_tokens=max(pools) if pools else None,
        contexts={},
    )
    for conc in sorted(rows):
        isl = sorted(x[0] for x in rows[conc])
        prof["contexts"][str(conc)] = [isl[int((i + 0.5) / conc * len(isl))] for i in range(conc)]
    json.dump(prof, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}: contexts for conc {sorted(rows)}, append ladder {prof['append_ladder']}, "
          f"decode ladder {prof['decode_ladder']}")


if __name__ == "__main__":
    main()
