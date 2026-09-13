#!/usr/bin/env python3
"""Fast agentX proxy benchmark: N concurrent sessions, each a long cached context plus K turns of
(append U new prompt tokens, decode D tokens), the shape of the InferenceX agentic trace replay
(MiniMax-M3, MI355X TP4) in minutes instead of an hour.

Per session i: a context of C_i tokens (from the profile's per-concurrency table), sent once in the
warm phase (not scored); then K scored turns, turn k re-sends the whole context + every earlier append
(all prefix-cache hits) + U_k fresh tokens and decodes exactly D_k tokens (ignore_eos). The context
grows by U_k per turn, as in the trace (the model's answer is discarded; the recorded U already stands
for the next turn's answer + tool result). U_k / D_k come from the profile's ladders (8 stratum means of
the recorded distributions, each session plays every rung once in a seeded order) or, with --uniform,
from U[0.8 x, x] draws around the recorded means. Prompts are token-id lists on /v1/completions, so
prefix hits are exact and no tokenizer round trip is involved. One seed fixes every length and id.

    python3 agentx_fast.py --base-url http://127.0.0.1:8888 --model <path> --conc 4 --turns 8 --seed 0 \
        --profile profile_tp4.json --result-dir out --result-filename agentx_fast_c4.json

The result JSON uses benchmark_serving.py's key names (median_ttft_ms, p99_tpot_ms, output_throughput,
...) plus the per-turn records. ITL is per streamed chunk (the server's --stream-interval); TPOT is the
per-token decode latency, (latency - ttft) / (tokens - 1).
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend_request_func import RequestFuncInput, async_request_openai_completions  # noqa: E402

PERCENTILES = (50, 90, 95, 99)


def pct(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def vocab_range(tokenizer_path, lo, hi):
    """[lo, hi) of plain token ids: below the smallest special id when a tokenizer is given."""
    if not tokenizer_path:
        return lo, hi
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        specials = [i for i in tok.all_special_ids if i is not None]
        top = min([len(tok)] + [s for s in specials if s >= 1000])
        return lo, min(hi, top - 1)
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer not loaded ({e}); using ids [{lo}, {hi})", file=sys.stderr)
        return lo, hi


def make_sessions(args, profile):
    """Deterministic per-session plan: context length, K (append, decode) pairs, token ids."""
    K = args.turns
    ctx_table = profile["contexts"].get(str(args.conc))
    if ctx_table is None:
        # nearest recorded concurrency, resampled to N evenly spaced quantiles
        keys = sorted(int(k) for k in profile["contexts"])
        near = min(keys, key=lambda k: abs(k - args.conc))
        src = sorted(profile["contexts"][str(near)])
        ctx_table = [src[int((i + 0.5) / args.conc * len(src))] for i in range(args.conc)]
        print(f"no context table for conc {args.conc}; resampled the conc {near} one", file=sys.stderr)
    u_ladder, d_ladder = profile["append_ladder"], profile["decode_ladder"]
    lo, hi = vocab_range(args.tokenizer, args.vocab_lo, args.vocab_hi)
    sessions = []
    for i in range(args.conc):
        rng = random.Random(args.seed * 100_003 + i)
        if args.uniform:
            c_target = int(statistics.mean(ctx_table))
            C = rng.randint(int(0.8 * c_target), c_target)
            U = [rng.randint(int(0.8 * profile["append_mean"]), profile["append_mean"]) for _ in range(K)]
            D = [rng.randint(int(0.8 * profile["decode_mean"]), profile["decode_mean"]) for _ in range(K)]
        else:
            C = int(ctx_table[i])
            order_u = [u_ladder[j % len(u_ladder)] for j in range(K)]
            order_d = [d_ladder[j % len(d_ladder)] for j in range(K)]
            rng.shuffle(order_u)
            rng.shuffle(order_d)
            U, D = order_u, order_d
        gen = np.random.default_rng(args.seed * 100_003 + i)
        ids = gen.integers(lo, hi, size=C + sum(U), dtype=np.int64).tolist()
        sessions.append(dict(i=i, C=C, U=U, D=D, ids=ids))
    return sessions


async def request(api_url, model, prompt_ids, max_tokens):
    inp = RequestFuncInput(
        prompt=prompt_ids,  # a token-id list: vLLM's completions endpoint takes it as is
        api_url=api_url,
        prompt_len=len(prompt_ids),
        output_len=max_tokens,
        model=model,
        ignore_eos=True,
    )
    return await async_request_openai_completions(inp)


async def warm(sess, api_url, model):
    t0 = time.perf_counter()
    out = await request(api_url, model, sess["ids"][: sess["C"]], 1)
    return dict(session=sess["i"], context=sess["C"], ok=out.success, ttft_s=out.ttft,
                latency_s=time.perf_counter() - t0, error=out.error)


async def turns(sess, api_url, model, records, verbose):
    end = sess["C"]
    for k, (u, d) in enumerate(zip(sess["U"], sess["D"])):
        cached = end
        end += u
        prompt = sess["ids"][:end]
        t0 = time.perf_counter()
        out = await request(api_url, model, prompt, d)
        rec = dict(session=sess["i"], turn=k, prompt_len=len(prompt), cached_len=cached, new_tokens=u,
                   max_tokens=d, output_tokens=out.output_tokens or 0, ok=out.success, ttft_ms=out.ttft * 1e3,
                   latency_ms=out.latency * 1e3, itl_ms=[x * 1e3 for x in out.itl], t_start=t0, error=out.error)
        n = rec["output_tokens"]
        rec["tpot_ms"] = (out.latency - out.ttft) * 1e3 / (n - 1) if n > 1 else float("nan")
        records.append(rec)
        if verbose:
            print(f"  s{sess['i']:02d} t{k}: prompt {len(prompt):>7,} (new {u:>6,}) decode {n:>5,} "
                  f"ttft {rec['ttft_ms']:7.1f} ms tpot {rec['tpot_ms']:6.2f} ms lat {rec['latency_ms']/1e3:6.2f} s"
                  + ("" if out.success else f"  ERROR {out.error[:80]}"), flush=True)


def summarize(args, sessions, warm_recs, records, t_warm, t_turns):
    ok = [r for r in records if r["ok"]]
    ttft = [r["ttft_ms"] for r in ok]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] == r["tpot_ms"]]
    itl = [x for r in ok for x in r["itl_ms"]]
    lat = [r["latency_ms"] for r in ok]
    out_tokens = sum(r["output_tokens"] for r in ok)
    new_tokens = sum(r["new_tokens"] for r in ok)
    cached = sum(r["cached_len"] for r in ok)
    warm_tokens = sum(w["context"] for w in warm_recs if w["ok"])
    res = dict(
        benchmark="agentx_fast", conc=args.conc, turns=args.turns, seed=args.seed, uniform=args.uniform,
        profile=os.path.basename(args.profile), model=args.model,
        completed=len(ok), failed=len(records) - len(ok), sessions=len(sessions),
        duration=t_turns, warm_duration=t_warm,
        total_input_tokens=sum(r["prompt_len"] for r in ok), total_new_prompt_tokens=new_tokens,
        total_output_tokens=out_tokens, cached_fraction=cached / max(1, cached + new_tokens),
        cold_prefill_tokens=warm_tokens, cold_prefill_throughput=warm_tokens / t_warm if t_warm else 0.0,
        request_throughput=len(ok) / t_turns, output_throughput=out_tokens / t_turns,
        new_prompt_token_throughput=new_tokens / t_turns,
        contexts=[s["C"] for s in sessions],
    )
    for name, vals in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2el", lat)):
        res[f"mean_{name}_ms"] = statistics.mean(vals) if vals else float("nan")
        res[f"median_{name}_ms"] = statistics.median(vals) if vals else float("nan")
        res[f"std_{name}_ms"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        for p in PERCENTILES:
            res[f"p{p}_{name}_ms"] = pct(vals, p)
    res["warm"] = warm_recs
    res["turns_detail"] = records
    return res


def print_summary(res):
    print("=" * 72)
    print(f"agentx_fast  conc {res['conc']}  turns {res['turns']}  seed {res['seed']}  "
          f"{'uniform' if res['uniform'] else 'ladder'}  contexts {[f'{c//1000}K' for c in res['contexts']]}")
    print(f"warm phase: {res['cold_prefill_tokens']:,} context tokens in {res['warm_duration']:.1f} s "
          f"({res['cold_prefill_throughput']:,.0f} tok/s cold prefill)")
    print(f"scored: {res['completed']} turns ({res['failed']} failed) in {res['duration']:.1f} s; "
          f"cached fraction of prompts {res['cached_fraction']:.1%}; new prompt tokens {res['total_new_prompt_tokens']:,} "
          f"({res['new_prompt_token_throughput']:,.0f}/s); output tokens {res['total_output_tokens']:,} "
          f"({res['output_throughput']:,.0f}/s); {res['request_throughput']:.2f} turns/s")
    print(f"{'metric':<10}{'mean':>10}{'median':>10}{'p90':>10}{'p95':>10}{'p99':>10}")
    for name in ("ttft", "tpot", "itl", "e2el"):
        print(f"{name+' ms':<10}{res[f'mean_{name}_ms']:>10.2f}{res[f'median_{name}_ms']:>10.2f}"
              f"{res[f'p90_{name}_ms']:>10.2f}{res[f'p95_{name}_ms']:>10.2f}{res[f'p99_{name}_ms']:>10.2f}")
    print("=" * 72)


async def main_async(args):
    profile = json.load(open(args.profile))
    sessions = make_sessions(args, profile)
    pool = profile.get("kv_pool_tokens")
    need = sum(s["C"] + sum(s["U"]) for s in sessions)
    if pool:
        print(f"resident context at the end: {need:,} tokens = {need / pool:.0%} of the {pool:,}-token KV pool"
              + ("  (WARNING: eviction likely)" if need > 0.8 * pool else ""))
    api_url = f"{args.base_url.rstrip('/')}/v1/completions"
    print(f"warm phase: {len(sessions)} sessions, {sum(s['C'] for s in sessions):,} context tokens", flush=True)
    t0 = time.perf_counter()
    warm_recs = await asyncio.gather(*(warm(s, api_url, args.model) for s in sessions))
    t_warm = time.perf_counter() - t0
    bad = [w for w in warm_recs if not w["ok"]]
    if bad:
        print(f"{len(bad)} warm requests failed: {bad[0]['error'][:200]}", file=sys.stderr)
        sys.exit(1)
    print(f"warm phase done in {t_warm:.1f} s; scored phase: {args.turns} turns per session", flush=True)
    records = []
    t1 = time.perf_counter()
    await asyncio.gather(*(turns(s, api_url, args.model, records, args.verbose) for s in sessions))
    t_turns = time.perf_counter() - t1
    res = summarize(args, sessions, warm_recs, records, t_warm, t_turns)
    print_summary(res)
    if args.result_filename:
        os.makedirs(args.result_dir, exist_ok=True)
        path = os.path.join(args.result_dir, args.result_filename)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True, help="model name as served (the path when --served-model-name is not set)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path, only to bound the random ids below the special ids")
    ap.add_argument("--conc", type=int, required=True, help="number of concurrent sessions")
    ap.add_argument("--turns", type=int, default=8, help="scored turns per session (8 = one pass over the ladders)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_tp4.json"))
    ap.add_argument("--uniform", action="store_true", help="U[0.8x, x] lengths around the recorded means instead of the ladders")
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=199_000)
    ap.add_argument("--result-dir", default=".")
    ap.add_argument("--result-filename", default="")
    ap.add_argument("--verbose", action="store_true", help="print every turn")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
