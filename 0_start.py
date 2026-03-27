#!/usr/bin/env python3
import os
import signal
import subprocess
import typer

app = typer.Typer()


def kill_existing_vllm():
    result = subprocess.run(
        ["pgrep", "-f", "vllm"], capture_output=True, text=True
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
    max_num_seqs: int = typer.Option(512, "--max-num-seqs", help="Max number of sequences"),
    # max_num_batched_tokens: int = typer.Option(196608, "--max-num-batched-tokens", help="Max batched tokens"),
    gpu_mem_util: float = typer.Option(0.95, "--gpu-mem-util", help="GPU memory utilization"),
    profile: bool = typer.Option(True, "--profile", help="Enable torch profiler"),
    log_file: str = typer.Option("server.log", "--log-file", "-l", help="Server log file"),
):
    kill_existing_vllm()

    os.environ.update({
        "VLLM_ROCM_USE_AITER": "1",
        "SGLANG_TORCH_PROFILER_DIR": "/app",
        "SGLANG_USE_AITER": "1",
    })

    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-num-seqs", str(max_num_seqs),
        # "--max-num-batched-tokens", str(max_num_batched_tokens),
        "--max-model-len", str(max_model_len),
        "--block-size", "32",
        "--trust-remote-code",
    ]

    if ep:
        cmd.append("--enable-expert-parallel")

    cmd.extend([
        "--profiler-config.profiler", "torch",
        "--profiler-config.torch_profiler_dir", "/mnt/hf_hub_cache",
        "--profiler-config.torch_profiler_record_shapes", "true",
        "--profiler-config.torch_profiler_with_stack", "true",
        "--profiler-config.delay_iterations", "5",
        "--profiler-config.max_iterations", "3",
        "--profiler-config.ignore_frontend", "true",
    ])

    typer.echo(f"Running: {' '.join(cmd)}")
    typer.echo(f"Logging to: {log_file}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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
