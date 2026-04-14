#!/usr/bin/env python3
import os
import shlex
import subprocess
import time
from datetime import datetime

import typer

app = typer.Typer()


def wait_for_server(port: int):
    url = f"http://0.0.0.0:{port}/health"
    typer.echo(f"Waiting for server at {url} ...")
    while True:
        try:
            result = subprocess.run(
                ["curl", "--output", "/dev/null", "--silent", "--fail", url],
                capture_output=True,
            )
            if result.returncode == 0:
                typer.echo("Server is ready.")
                return
        except Exception:
            pass
        time.sleep(3)


@app.command()
def bench(
    conc_end: int = typer.Option(..., "--conc-end", "-c", help="Max concurrency (e.g. 256)"),
    conc_start: int = typer.Option(4, "--conc-start", "-s", help="Starting concurrency"),
    isl: int = typer.Option(8192, "--isl", help="Random input sequence length"),
    osl: int = typer.Option(1024, "--osl", help="Random output sequence length"),
    port: int = typer.Option(30000, "--port", "-p", help="Server port"),
    model: str = typer.Option("MiniMaxAI/MiniMax-M2.5", "--model", "-m", help="Model name"),
    random_range_ratio: float = typer.Option(0.8, "--random-range-ratio", help="Random range ratio"),
    result_filename: str = typer.Option("dsr1_fp8_mi300x_docker", "--result-filename", "-r", help="Result filename prefix"),
    num_prompts_mul: int = typer.Option(4, "--num-prompts-mul", help="num_prompts = CONC * this value"),
    result_dir: str = typer.Option("/app/", "--result-dir", help="Result directory"),
    profile: bool = typer.Option(False, "--profile", help="Enable torch profiler (start/stop via server API)"),
):
    # Build concurrency list
    conc_list = []
    c = conc_start
    while c <= conc_end:
        conc_list.append(c)
        c *= 2

    typer.echo("\n" + "=" * 50)
    typer.echo("  Benchmark Configuration")
    typer.echo("=" * 50)
    typer.echo(f"  Model:              {model}")
    typer.echo(f"  Port:               {port}")
    typer.echo(f"  ISL:                {isl}")
    typer.echo(f"  OSL:                {osl}")
    typer.echo(f"  Random Range Ratio: {random_range_ratio}")
    typer.echo(f"  Concurrency:        {conc_list}")
    typer.echo(f"  num_prompts:        CONC * {num_prompts_mul}")
    typer.echo(f"  Result filename:    {result_filename}.json")
    typer.echo(f"  Result dir:         {result_dir}")
    typer.echo(f"  Profile:            {profile}")
    typer.echo("=" * 50 + "\n")

    if not typer.confirm("Proceed?"):
        typer.echo("Aborted.")
        raise typer.Abort()

    wait_for_server(port)

    conc = conc_start
    while conc <= conc_end:
        num_prompts = conc * num_prompts_mul
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"result_{conc}_{timestamp}.log"

        typer.echo(f"========== Testing CONC={conc} ==========")

        cmd = (
            f"python3 benchmark_serving.py"
            f" --model {shlex.quote(model)}"
            f" --backend vllm"
            f" --base-url http://0.0.0.0:{port}"
            f" --dataset-name random"
            f" --random-input-len {isl}"
            f" --random-output-len {osl}"
            f" --random-range-ratio {random_range_ratio}"
            f" --num-prompts {num_prompts}"
            f" --max-concurrency {conc}"
            f" --request-rate inf"
            f" --ignore-eos"
            f" --save-result"
            f" --percentile-metrics ttft,tpot,itl,e2el"
            f" --result-dir {shlex.quote(result_dir)}"
            f" --result-filename {shlex.quote(result_filename + '.json')}"
        )

        if profile:
            cmd += " --profile"

        cmd += f" 2>&1 | tee {shlex.quote(log_file)}"

        typer.echo(cmd)
        os.system(cmd)

        conc *= 2


if __name__ == "__main__":
    app()
