"""Per-job summary of an InferenceX agentic run from the raw aiperf artifacts (`gh api repos/SemiAnalysisAI/InferenceX/actions/artifacts/<id>/zip`, unpacked as <root>/tp4_conc<N>/): cache hit rates and the
nominal / uncached ISL and OSL of the profiling requests.

Uncached ISL per request: vLLM counts prefix-cache queries (= prompt length) and hits once per request at
scheduling; the 1-s counter slices are matched cumulatively to the requests in start order (exact for
queries; the hits of a slice shared by several requests are split in proportion to their prompt lengths,
so the per-request median/p90 are approximate at high concurrency -- the mean is exact).
"""
import json, statistics as st, sys, glob, os, re

pct = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]


def theoretical(path):
    d = json.load(open(path))
    top = d.get('theoretical_prefix_cache_hit')  # the profiling-scoped value; warmup_metrics holds its own copy
    if isinstance(top, dict) and top.get('avg') is not None:
        return top['avg']
    found = []

    def walk(o):
        if isinstance(o, dict):
            if o.get('tag') == 'theoretical_prefix_cache_hit' and 'avg' in o:
                found.append(o['avg'])
            for k, v in o.items():
                if k == 'theoretical_prefix_cache_hit' and isinstance(v, dict) and 'avg' in v:
                    found.append(v['avg'])
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return found[0] if found else None


def analyze(job_dir):
    A = f'{job_dir}/aiperf_artifacts'
    recs = [json.loads(l) for l in open(f'{A}/profile_export.jsonl')]
    d = json.load(open(f'{A}/server_metrics_export.json'))
    S = lambda sec, k: d[sec][k]['series'][0]
    wh, wq = S('warmup_metrics', 'vllm:prefix_cache_hits')['stats']['total'], S('warmup_metrics', 'vllm:prefix_cache_queries')['stats']['total']
    ah, aq = S('metrics', 'vllm:prefix_cache_hits')['stats']['total'], S('metrics', 'vllm:prefix_cache_queries')['stats']['total']
    sl_h, sl_q = S('metrics', 'vllm:prefix_cache_hits')['timeslices'], S('metrics', 'vllm:prefix_cache_queries')['timeslices']
    slices = [(a['start_ns'], a['total'], b['total']) for a, b in zip(sl_q, sl_h) if a['total'] > 0]
    reqs = sorted(recs, key=lambda r: r['metadata']['request_start_ns'])
    rows = []
    si, q_left = 0, slices[0][1]
    shared = 0
    skipped = 0
    for r in reqs:
        m, md = r['metrics'], r['metadata']
        if 'input_sequence_length' not in m or 'output_sequence_length' not in m:
            skipped += 1  # errored / cancelled records carry no token counts
            continue
        isl = m['input_sequence_length']['value']
        need, hits, nslices = isl, 0.0, 0
        while need > 0 and si < len(slices):
            take = min(need, q_left)
            hits += take / slices[si][1] * slices[si][2]
            need -= take; q_left -= take; nslices += 1
            if q_left <= 0:
                si += 1
                q_left = slices[si][1] if si < len(slices) else 0
        if nslices == 1 and take < slices[si - 1][1] if si else False:
            shared += 1
        rows.append(dict(phase=md['benchmark_phase'], isl=isl, osl=m['output_sequence_length']['value'],
                         uniq=max(isl - hits, 0), ttft=m['time_to_first_token']['value']))
    prof = [x for x in rows if x['phase'] == 'profiling']
    isl, u, osl, ttft = ([x[k] for x in prof] for k in ('isl', 'uniq', 'osl', 'ttft'))
    ph, pq = ah - wh, aq - wq
    dur_s = (max(r['metadata']['request_end_ns'] for r in recs) - min(r['metadata']['request_start_ns'] for r in recs)) / 1e9
    return dict(n=len(prof), skipped=skipped, chip=ah / aq, chip_prof=ph / pq, theo=theoretical(f'{A}/profile_export_aiperf.json'),
                isl_mean=st.mean(isl), isl_med=st.median(isl), isl_p90=pct(isl, .9),
                u_mean=pq and (pq - ph) / len(prof), u_med=st.median(u), u_p90=pct(u, .9), u_p95=pct(u, .95),
                osl_mean=st.mean(osl), osl_med=st.median(osl), osl_p90=pct(osl, .9),
                ttft_med=st.median(ttft), out_tps=sum(osl) / dur_s, uncached_in_tps=(pq - ph) / dur_s)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'art'
    jobs = []
    for jd in sorted(glob.glob(f'{root}/tp*_conc*')):
        mm = re.match(r'.*/(tp\d+)_conc(\d+)$', jd)
        if mm and os.path.isdir(jd) and os.path.isdir(f'{jd}/aiperf_artifacts'):
            jobs.append((mm.group(1), int(mm.group(2)), jd))
    jobs.sort()
    print('| TP | conc | requests (dropped) | chip hit rate (whole run / profiling) | theoretical | nominal ISL mean / median / p90 | uncached ISL mean / median / p90 / p95 | OSL mean / median / p90 | TTFT median ms | out tok/s |')
    print('|' + '---|' * 10)
    for tp, c, jd in jobs:
        r = analyze(jd)
        th = f"{r['theo']:.1f}%" if r['theo'] is not None else 'n/a'
        print(f"| {tp} | {c} | {r['n']} ({r['skipped']}) | {r['chip']:.1%} / {r['chip_prof']:.1%} | {th} | "
              f"{r['isl_mean']:,.0f} / {r['isl_med']:,.0f} / {r['isl_p90']:,.0f} | "
              f"**{r['u_mean']:,.0f}** / {r['u_med']:,.0f} / {r['u_p90']:,.0f} / {r['u_p95']:,.0f} | "
              f"{r['osl_mean']:,.0f} / {r['osl_med']:,.0f} / {r['osl_p90']:,.0f} | {r['ttft_med']:,.0f} | {r['out_tps']:,.0f} |")


if __name__ == '__main__':
    main()
