#!/usr/bin/env python3
import os
import signal
import subprocess
import typer

app = typer.Typer()


def kill_existing_atom():
    result = subprocess.run(
        ["pgrep", "-f", "atom.entrypoints"], capture_output=True, text=True
    )
    for pid in result.stdout.strip().split("\n"):
        if pid:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


@app.command()
def serve(
    model: str = typer.Option("MiniMaxAI/MiniMax-M2.5", "--model", "-m", help="Model name or path"),
    tp: int = typer.Option(..., "--tp", help="Tensor parallel size"),
    ep: bool = typer.Option(..., "--ep / --no-ep", help="Enable expert parallel"),
    port: int = typer.Option(30000, "--port", "-p", help="Server port"),
    max_model_len: int = typer.Option(10240, "--max-model-len", help="Max model length"),
    kv_cache_dtype: str = typer.Option("fp8", "--kv-cache-dtype", help="KV cache dtype: bf16 or fp8"),
    gpu_mem_util: float = typer.Option(0.9, "--gpu-mem-util", help="GPU memory utilization"),
    gpus: str = typer.Option(None, "--gpus", help="Comma-separated GPU IDs, e.g. '6,7'. Uses all if not set."),
    log_file: str = typer.Option("server.log", "--log-file", "-l", help="Server log file"),
):
    kill_existing_atom()

    os.environ.update({
        "VLLM_ROCM_USE_AITER": "1",
        "OMP_NUM_THREADS": "1",
    })

    if gpus:
        # CUDA_VISIBLE_DEVICES works on ROCm; ROCR_VISIBLE_DEVICES breaks aiter triton JIT
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    cmd = [
        "python3", "-m", "atom.entrypoints.openai_server",
        "--model", model,
        "--server-port", str(port),
        "-tp", str(tp),
        "--kv_cache_dtype", kv_cache_dtype,
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
    ]

    if ep:
        cmd.append("--enable-expert-parallel")

    typer.echo(f"Running: {' '.join(cmd)}")
    typer.echo(f"GPUs: {gpus or 'all'}")
    typer.echo(f"Logging to: {log_file}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/app/ATOM",
        )
        try:
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()


if __name__ == "__main__":
    app()
