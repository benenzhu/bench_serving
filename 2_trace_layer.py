#!/usr/bin/env python3
"""Extract per-layer kernel breakdown from ATOM trace files.

Auto-detects kernels-per-layer and number of model layers from the trace.
Slices PA-to-PA layers and computes average kernel durations per position.

Usage:
    python 2_trace_layer.py <trace.json.gz>
    python 2_trace_layer.py <trace.json.gz> --csv out.csv
"""

import gzip
import json
import statistics
from collections import Counter
from pathlib import Path

import typer

app = typer.Typer()

# Short display names for common kernels
KERNEL_ALIASES = {
    "pa_bf16": "PA (paged_attn)",
    "kernel_gemm_xdl_cshuffle_v3": "GEMM_FP8 (CK dense)",
    "kernel_moe_gemm": "MoE_GEMM (CK)",
    "dynamic_per_group_scaled_quant": "FP8_quant",
    "add_rmsnorm_quant": "RMSNorm+Quant",
    "grouped_topk": "MoE_gate (topk)",
    "MoeSortingMultiPhaseKernel_P0": "MoE_sort_P0",
    "MoeSortingMultiPhaseKernel_P23": "MoE_sort_P23",
    "MoeSortingKernel": "MoE_sort",
    "Cijk_": "GEMM_BF16 (rocBLAS)",
    "triton_poi_fused__to_copy": "Triton_copy",
    "triton_poi_fused_embedding": "Embedding",
    "triton_red_fused": "RMSNorm_triton",
    "kn_entry_2c_sbhd": "RoPE",
    "reshape_and_cache": "KV_cache_quant",
    "fmoe_bf16_blockscaleFp8": "MoE_GEMM_fused (ASM)",
    "MoeFlatmm": "MoE_GEMM_flatmm",
    "mix_sample": "Sampling",
    "fmha_fwd": "FlashAttn",
    "wv_splitk": "PA_reduce",
}


def short_name(name: str) -> str:
    for pattern, alias in KERNEL_ALIASES.items():
        if pattern in name:
            return alias
    return name[:50]


def load_trace(trace_path: str):
    path = Path(trace_path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    else:
        with open(path) as f:
            return json.load(f)


def extract_layers(trace_data: dict, pa_pattern: str = "pa_bf16"):
    """Extract PA-to-PA layers from GPU kernel timeline."""
    kernels = sorted(
        [
            (ev["ts"], ev.get("dur", 0), ev["name"].replace("void ", ""))
            for ev in trace_data["traceEvents"]
            if ev.get("cat") == "kernel"
        ],
        key=lambda x: x[0],
    )

    if not kernels:
        typer.echo("ERROR: No GPU kernel events found in trace.", err=True)
        raise typer.Exit(1)

    # Find PA kernel indices as layer boundaries
    pa_indices = [i for i, (_, _, name) in enumerate(kernels) if pa_pattern in name]

    if len(pa_indices) < 2:
        typer.echo(
            f"ERROR: Found only {len(pa_indices)} PA kernels, need at least 2.",
            err=True,
        )
        raise typer.Exit(1)

    # Determine dominant gap (kernels per layer)
    gaps = [pa_indices[i + 1] - pa_indices[i] for i in range(len(pa_indices) - 1)]
    dominant_gap = Counter(gaps).most_common(1)[0][0]

    # Extract layers matching the dominant gap
    layers = []
    for i in range(len(pa_indices) - 1):
        s, e = pa_indices[i], pa_indices[i + 1]
        if e - s == dominant_gap:
            layers.append([(dur, name) for _, dur, name in kernels[s:e]])

    return layers, dominant_gap, len(kernels)


@app.command()
def main(
    trace_path: str = typer.Argument(..., help="Path to trace .json.gz file"),
    csv_out: str = typer.Option("", "--csv", help="Output CSV path (optional)"),
    pa_pattern: str = typer.Option(
        "pa_bf16", "--pa-pattern", help="PA kernel name pattern for layer boundary"
    ),
):
    """Extract per-layer kernel breakdown from an ATOM trace file."""
    typer.echo(f"Loading trace: {trace_path}")
    trace_data = load_trace(trace_path)

    # Detect dominant decode batch size from annotations
    import re
    bs_counts: dict[int, int] = Counter()
    for ev in trace_data["traceEvents"]:
        m = re.match(r"decode\[bs=(\d+)", ev.get("name", ""))
        if m:
            bs_counts[int(m.group(1))] += 1
    dominant_bs = 0
    if bs_counts:
        dominant_bs = bs_counts.most_common(1)[0][0]
        typer.echo(f"Decode batch sizes: {dict(bs_counts.most_common())}")
        typer.echo(f"Dominant decode bs: {dominant_bs}")
    else:
        typer.echo("No decode annotations found in trace")

    # Auto-generate CSV filename with conc{bs} prefix
    if csv_out and dominant_bs:
        csv_path = Path(csv_out)
        csv_out = str(csv_path.parent / f"conc{dominant_bs}_{csv_path.name}")
    elif not csv_out and dominant_bs:
        csv_out = f"conc{dominant_bs}_layer.csv"

    layers, kernels_per_layer, total_kernels = extract_layers(trace_data, pa_pattern)

    typer.echo(f"Total GPU kernels: {total_kernels}")
    typer.echo(f"Kernels per layer: {kernels_per_layer}")
    typer.echo(f"Sampled layers: {len(layers)}\n")

    # Compute per-position stats
    header = f"{'Pos':>3}  {'Avg(us)':>8}  {'Std':>6}  {'Min':>6}  {'Max':>8}  {'%':>5}  {'N':>5}  {'Short':20s}  Kernel"
    typer.echo(header)
    typer.echo("-" * len(header))

    total_avg = 0
    rows = []
    for pos in range(kernels_per_layer):
        durs = [layer[pos][0] for layer in layers]
        name = layers[0][pos][1]
        sname = short_name(name)
        avg = statistics.mean(durs)
        std = statistics.stdev(durs) if len(durs) > 1 else 0
        total_avg += avg
        rows.append((pos, sname, name, avg, std, min(durs), max(durs), len(durs)))

    for pos, sname, name, avg, std, mn, mx, n in rows:
        pct = avg / total_avg * 100 if total_avg else 0
        typer.echo(
            f"{pos:>3}  {avg:>8.1f}  {std:>6.1f}  {mn:>6.1f}  {mx:>8.1f}  {pct:>4.1f}%  {n:>5}  {sname:20s}  {name}"
        )

    typer.echo("-" * len(header))
    typer.echo(f"{'':>3}  {total_avg:>8.1f} us  TOTAL per layer")

    # CSV output
    if csv_out:
        import csv

        with open(csv_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["pos", "short_name", "kernel", "avg_us", "std_us", "min_us", "max_us", "pct", "count"]
            )
            for pos, sname, name, avg, std, mn, mx, n in rows:
                pct = avg / total_avg * 100 if total_avg else 0
                w.writerow(
                    [pos, sname, name, f"{avg:.1f}", f"{std:.1f}", f"{mn:.1f}", f"{mx:.1f}", f"{pct:.1f}", n]
                )
        typer.echo(f"\nCSV saved to: {csv_out}")


if __name__ == "__main__":
    app()
