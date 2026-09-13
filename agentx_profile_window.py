#!/usr/bin/env python3
"""Open a torch-profiler window (/start_profile, /stop_profile) on a vLLM server in the agentic regime of
agentx_fast.py: N sessions with cached long contexts (the profile's context table for that concurrency), each
appending U tokens from the ladder and decoding. One invocation = one or more windows on the same server.

  --mode decode    warm the N contexts; start every session's first turn together (append U_i, long decode);
                   once all N have produced their first token the batch is pure decode: settle, /start_profile,
                   WINDOW s (or until the server's own max_iterations stops it), /stop_profile, drain, cancel.
  --mode prefill   warm the N contexts; with --background keep the other N-1 sessions decoding; /start_profile;
                   one session's first turn (append --append tokens, --prefill-decode decode tokens);
                   /stop_profile; drain.  Several modes: --mode decode,prefill (one warm phase, windows in order).

Server side (m3-compare): PROFILE=1 GPU_ONLY_TRACE=1 PROF_MAX_ITERS=100 -> --profiler-config.profiler=torch with a
GPU-only trace (CPU flow events make libkineto segfault at conc > 1) and the worker stops itself after 100 steps.
Traces land in the server's torch_profiler_dir, one file per rank per window (timestamped).

    python3 agentx_profile_window.py --base-url http://127.0.0.1:8888 --model <path> --conc 20 --mode decode,prefill \
        --profile profile_tp4.json --window 12 --append 4410 --background
"""
import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_fast import AIOHTTP_TIMEOUT, make_sessions, request, warm  # noqa: E402


async def stream(http, api_url, model, prompt_ids, max_tokens, first, st):
    """One streaming request that is meant to be cancelled: sets `first` at the first completion chunk and keeps
    st['chunks'] / st['tokens'] up to date so the caller can see how much was generated inside the window."""
    payload = {"model": model, "prompt": prompt_ids, "temperature": 0.0, "max_tokens": max_tokens,
               "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    try:
        async with http.post(api_url, json=payload) as resp:
            if resp.status != 200:
                st["error"] = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                first.set()
                return
            buf = b""
            async for raw in resp.content.iter_any():
                buf += raw
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    body = line[5:].strip()
                    if body == b"[DONE]":
                        continue
                    data = json.loads(body)
                    if data.get("choices"):
                        if not first.is_set():
                            st["ttft"] = time.perf_counter() - t0
                            first.set()
                        st["chunks"] += 1
                    elif data.get("usage"):
                        st["tokens"] = data["usage"].get("completion_tokens", 0)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        st["error"] = repr(e)
        first.set()
    finally:
        st["done"] = True


async def ctl(http, base, path, timeout):
    try:
        async with http.post(base + path, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            return f"{path} -> HTTP {r.status} {(await r.text())[:120]!r}"
    except Exception as e:  # noqa: BLE001
        return f"{path} -> {e!r}"


async def start_background(http, api_url, model, sessions, max_tokens, label):
    """Start every session's first turn (append U[0], long decode) and wait until all have their first token."""
    firsts = [asyncio.Event() for _ in sessions]
    stats = [dict(chunks=0, tokens=0, ttft=None, error="", done=False) for _ in sessions]
    tasks = [asyncio.create_task(stream(http, api_url, model, s["ids"][: s["C"] + s["U"][0]], max_tokens, f, st))
             for s, f, st in zip(sessions, firsts, stats)]
    t0 = time.perf_counter()
    await asyncio.wait_for(asyncio.gather(*(f.wait() for f in firsts)), timeout=1800)
    bad = [st["error"] for st in stats if st["error"]]
    if bad:
        raise RuntimeError(f"{len(bad)} {label} streams failed: {bad[0]}")
    ttfts = [st["ttft"] for st in stats]
    print(f"[{label}] {len(sessions)} streams past prefill in {time.perf_counter() - t0:.1f} s "
          f"(ttft min {min(ttfts):.2f} / max {max(ttfts):.2f} s); appends "
          f"{[s['U'][0] for s in sessions]}", flush=True)
    return tasks, stats


async def cancel_all(tasks):
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def window_decode(args, http, base, api_url, sessions):
    tasks, stats = await start_background(http, api_url, args.model, sessions, args.decode_tokens, "decode")
    await asyncio.sleep(args.settle)
    c0 = [st["chunks"] for st in stats]
    print(f"[decode] START_PROFILE ({time.strftime('%H:%M:%S')}), window {args.window} s", flush=True)
    print("  " + await ctl(http, base, "/start_profile", 120), flush=True)
    await asyncio.sleep(args.window)
    print("  " + await ctl(http, base, "/stop_profile", 300), flush=True)
    c1 = [st["chunks"] for st in stats]
    d = [b - a for a, b in zip(c0, c1)]
    print(f"[decode] STOP_PROFILE ({time.strftime('%H:%M:%S')}); stream chunks in the window: {sum(d)} total, "
          f"per session {d}; still running {sum(not st['done'] for st in stats)}/{len(stats)}", flush=True)
    print(f"[decode] draining {args.drain} s for the trace to reach disk", flush=True)
    await asyncio.sleep(args.drain)
    await cancel_all(tasks)


async def window_prefill(args, http, base, api_url, sessions):
    k = args.session if args.session >= 0 else len(sessions) // 2
    sess = sessions[k]
    others = [s for i, s in enumerate(sessions) if i != k]
    tasks = []
    if args.background and others:
        tasks, _ = await start_background(http, api_url, args.model, others, args.decode_tokens, "background")
        await asyncio.sleep(args.settle)
    U = args.append if args.append > 0 else sess["U"][0]
    prompt = sess["ids"][: sess["C"] + U]
    print(f"[prefill] session {k}: context {sess['C']:,} cached + {U:,} new tokens, {args.prefill_decode} decode tokens, "
          f"{len(tasks)} sessions decoding in the background", flush=True)
    print(f"[prefill] START_PROFILE ({time.strftime('%H:%M:%S')})", flush=True)
    print("  " + await ctl(http, base, "/start_profile", 120), flush=True)
    out = await request(http, api_url, args.model, prompt, args.prefill_decode)
    print("  " + await ctl(http, base, "/stop_profile", 300), flush=True)
    print(f"[prefill] STOP_PROFILE ({time.strftime('%H:%M:%S')}); ttft {out.ttft * 1e3:.0f} ms, latency {out.latency:.2f} s, "
          f"{out.output_tokens} tokens" + ("" if out.success else f"  ERROR {out.error[:120]}"), flush=True)
    print(f"[prefill] draining {args.drain} s for the trace to reach disk", flush=True)
    await asyncio.sleep(args.drain)
    await cancel_all(tasks)


async def main_async(args):
    profile = json.load(open(args.profile))
    ns = argparse.Namespace(conc=args.conc, turns=1, seed=args.seed, uniform=False, tokenizer=args.tokenizer,
                            vocab_lo=args.vocab_lo, vocab_hi=args.vocab_hi)
    sessions = make_sessions(ns, profile)
    # the prefill window may need more new tokens than the ladder value the session was given
    extra = max(0, args.append - min(s["U"][0] for s in sessions))
    if extra:
        import numpy as np

        for s in sessions:
            gen = np.random.default_rng(args.seed * 7919 + s["i"] + 1)
            s["ids"] = s["ids"] + gen.integers(args.vocab_lo, args.vocab_hi, size=extra, dtype=np.int64).tolist()
    base = args.base_url.rstrip("/")
    api_url = base + "/v1/completions"
    http = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, connector=aiohttp.TCPConnector(limit=0))
    t0 = time.perf_counter()
    ctxs = [f"{s['C'] // 1000}K" for s in sessions]
    print(f"warm phase: {len(sessions)} sessions, contexts {ctxs}", flush=True)
    recs = await asyncio.gather(*(warm(s, http, api_url, args.model) for s in sessions))
    bad = [w for w in recs if not w["ok"]]
    if bad:
        print(f"{len(bad)} warm requests failed: {bad[0]['error'][:200]}", file=sys.stderr)
        sys.exit(1)
    print(f"warm phase done in {time.perf_counter() - t0:.1f} s", flush=True)
    for mode in args.mode.split(","):
        if mode == "decode":
            await window_decode(args, http, base, api_url, sessions)
        elif mode == "prefill":
            await window_prefill(args, http, base, api_url, sessions)
        else:
            sys.exit(f"unknown mode {mode}")
        await asyncio.sleep(3)
    await http.close()
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8888")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--conc", type=int, required=True, help="number of sessions (the context table of the profile)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_tp4.json"))
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=199_000)
    ap.add_argument("--mode", default="decode", help="decode | prefill | decode,prefill")
    ap.add_argument("--window", type=float, default=12.0, help="seconds the decode window stays open (upper bound)")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds after the batch is full before /start_profile")
    ap.add_argument("--drain", type=float, default=60.0, help="seconds to keep the streams alive after /stop_profile")
    ap.add_argument("--decode-tokens", type=int, default=4096, help="max_tokens of the streams that fill the batch")
    ap.add_argument("--append", type=int, default=0, help="prefill window: new tokens of the profiled turn (0 = the ladder value)")
    ap.add_argument("--prefill-decode", type=int, default=8, help="prefill window: decode tokens after the append")
    ap.add_argument("--session", type=int, default=-1, help="prefill window: which session (-1 = the middle context)")
    ap.add_argument("--background", action="store_true", help="prefill window: keep the other sessions decoding")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
