"""Integration tests for inference with perplexity scoring.

Tests are parametrized by YAML spec files that include an 'inference' section.
Each test trains a model, starts the inference server, generates text, and
scores the output with GPT-2 perplexity.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from .assertions import assert_exit_code
from .perplexity import compute_perplexity
from .runner import run_forgather_train

# Resolve inference server script path
_SERVER_SCRIPT = (
    Path(__file__).resolve().parents[2] / "tools" / "inference_server" / "server.py"
)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_model_dir(output_dir: Path) -> Path:
    """Find the model directory containing config.json within output_dir.

    With --output-dir, the output_dir itself is the model directory.
    """
    if (output_dir / "config.json").exists():
        return output_dir

    # Search one level down in case the layout nests a model subdirectory
    for candidate in sorted(output_dir.iterdir()):
        if candidate.is_dir() and (candidate / "config.json").exists():
            return candidate

    raise FileNotFoundError(f"No directory with config.json found in {output_dir}")


def _wait_for_health(port: int, timeout: int, proc: subprocess.Popen) -> None:
    """Poll the server health endpoint until it responds or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # Check if server process died
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"Server exited with code {proc.returncode}.\n"
                f"stdout: {stdout[-2000:] if stdout else ''}\n"
                f"stderr: {stderr[-2000:] if stderr else ''}"
            )
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200 and r.json().get("model_loaded"):
                return
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)

    proc.kill()
    raise TimeoutError(f"Server did not become healthy within {timeout}s")


@pytest.mark.integration
@pytest.mark.slow
def test_inference_with_perplexity(spec, output_dir):
    """Train a model, serve it, generate text, and score with GPT-2 perplexity."""
    # 1. Train
    result = run_forgather_train(spec, output_dir)
    assert_exit_code(result)

    # 2. Find model directory
    model_dir = _find_model_dir(output_dir)

    # 3. Start inference server (load from checkpoint since training
    #    saves checkpoints, not standalone model weights)
    port = _find_free_port()
    server_proc = subprocess.Popen(
        [
            sys.executable,
            str(_SERVER_SCRIPT),
            "--model",
            str(model_dir),
            "--from-checkpoint",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "WARNING",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_health(port, spec.inference.server_timeout, server_proc)

        # 4. Send completion request
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/completions",
            json={
                "model": "test",
                "prompt": spec.inference.prompt,
                "max_tokens": spec.inference.max_tokens,
                "temperature": spec.inference.temperature,
            },
            timeout=30,
        )
        assert (
            response.status_code == 200
        ), f"Completion request failed: {response.status_code} {response.text}"

        generated_text = response.json()["choices"][0]["text"]
        assert len(generated_text.strip()) > 0, "Server returned empty completion"

        # 5. Score with GPT-2 perplexity
        full_text = spec.inference.prompt + generated_text
        ppl = compute_perplexity(full_text)

        assert ppl < spec.inference.perplexity_max, (
            f"GPT-2 perplexity {ppl:.1f} exceeds threshold "
            f"{spec.inference.perplexity_max}.\n"
            f"Generated text: {full_text!r}"
        )

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
