"""One JSON file per run: the config, the environment, the summary, one record per request
and the server's step log, so a run is measured once and plotted any number of times.
Load sweeps go in results/serving/, the request scenarios of docs/SCENARIOS.md in
results/scenarios/.

Per request: the server's record (engine.request_record, from /internal/log) merged with the
load client's, timestamps in seconds since the run started. Queued is
arrival <= t < prefill_start, active is prefill_start <= t < finished_at; rejected requests
have no prefill_start.
"""

import dataclasses
import json
import math
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# The load client's absolute times (time.perf_counter()); request_rows() makes them relative.
TIMES = ("scheduled", "sent_at", "finished_at", "server_arrival", "prefill_start")
TIME_LISTS = ("token_times", "server_token_times")


def run_name(config: dict, rate: float, seed: int) -> str:
    """continuous_fcfs_slots-8_chunk-none_lambda-2_seed-0, from the server's /internal/config."""
    chunk = "none" if config["chunk_size"] is None else config["chunk_size"]
    return (
        f"{config['batching']}_{config['policy']}_slots-{config['slots']}_chunk-{chunk}"
        f"_lambda-{rate:g}_seed-{seed}"
    )


def environment() -> dict:
    """What a number depends on besides the config: the code version, software and GPU."""
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": shlex.join(["python3", *sys.argv]),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def request_rows(records: list, t0: float) -> list[dict]:
    """The load client's records (merged with the server's) as dicts, times since t0."""
    rows = []
    for r in records:
        row = dataclasses.asdict(r)
        for k in TIMES:
            if row[k] is not None:
                row[k] -= t0
        for k in TIME_LISTS:
            if row[k] is not None:
                row[k] = [t - t0 for t in row[k]]
        rows.append(row)
    return rows


def step_rows(steps: list[dict], t0: float) -> list[dict]:
    """The server's step log, times since t0."""
    return [s | {"t_start": s["t_start"] - t0, "t_end": s["t_end"] - t0} for s in steps]


def save_run(
    name: str,
    config: dict,
    requests: list[dict],
    summary: dict,
    steps: list[dict],
    subdir: str = "serving",
    scenario: dict | None = None,
) -> Path:
    """Writes results/<subdir>/<name>.json and returns its path. Overwrites a run of the
    same name, so rerunning a config replaces its old numbers.
    config: {"server": /internal/config, "client": the load client's settings}.
    scenario: {"id", "name", "seed", "specs"} for a run of docs/SCENARIOS.md."""
    out_dir = RESULTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    run = {} if scenario is None else {"scenario": scenario}
    run |= {
        "config": config,
        "environment": environment(),
        "summary": summary,
        "requests": requests,
        "steps": steps,
    }
    path.write_text(json.dumps(_finite(run), indent=1, allow_nan=False))
    return path


def _finite(x):
    """NaN and inf become null: JSON has no NaN, and the page's JSON.parse would fail on it."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()
