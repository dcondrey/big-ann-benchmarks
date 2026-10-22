"""Two-image streaming experiment; all search/policy sweeps are runtime parameters."""
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import modal

HMS_SHA = os.environ["HMS_COMMIT"]
if re.fullmatch(r"[0-9a-f]{40}", HMS_SHA) is None:
    raise ValueError("HMS_COMMIT must be a full commit SHA")
BAB_SHA = "89a3aba"
ADAPTER = Path(__file__).resolve().parent
app = modal.App("hms-bigann")
volume = modal.Volume.from_name("hms-bigann", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "build-essential", "git", "time")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.99.0")
    .run_commands(
        f"git clone -q https://github.com/harsha-simhadri/big-ann-benchmarks.git /bab && cd /bab && git checkout -q {BAB_SHA}",
        "pip install -q -r /bab/requirements_py3.10.txt maturin",
    )
    .run_commands(
        "GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/writerslogic/holographic-memory.git /hms"
        f" && cd /hms && git checkout -q {HMS_SHA}"
        " && PATH=/root/.cargo/bin:$PATH maturin build -q --release --features python -i python3 -o /wheels"
        " && pip install -q /wheels/*.whl"
    )
    .add_local_dir(ADAPTER, "/bab/neurips23/streaming/hms")
)


def execute(args, log, timeout, env):
    with log.open("w") as output:
        process = subprocess.Popen(args, cwd="/bab", stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        raise RuntimeError(f"{args[0]} failed with {code}; see {log}")


@app.function(image=image, volumes={"/vol": volume}, cpu=(8.0, 8.0), memory=(8192, 8192),
              timeout=5400, max_containers=1)
def run(mode: str, build: dict, search: dict):
    if mode not in ("diagnostic", "delete", "final"):
        raise ValueError(mode)
    started = time.time()
    tag = f"{mode}-{HMS_SHA[:12]}-{int(started)}"
    root = Path("/vol/streaming-recall") / tag
    root.mkdir(parents=True)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 8:
        raise RuntimeError(f"only {len(affinity)} schedulable CPUs; need eight")
    os.sched_setaffinity(0, affinity[:8])
    env = dict(os.environ, RAYON_NUM_THREADS="8", OMP_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1")
    receipt = {"mode": mode, "hms_sha": HMS_SHA, "harness_sha": BAB_SHA,
               "build": build, "search": search, "cpus": affinity[:8],
               "memory_limit_bytes": 8 * 1024**3, "started": started}
    if mode == "diagnostic":
        from textwrap import dedent
        dataset = "msturing-1M-stream-diagnostic"
        book = root / "diagnostic.yaml"
        book.write_text(dedent(f"""\
            {dataset}:
              max_pts: 1000000
              1: {{operation: insert, start: 0, end: 1000000}}
              2: {{operation: search}}
              3: {{operation: delete, start: 0, end: 100000}}
              4: {{operation: search}}
              5: {{operation: search}}
            """))
        env.update(HMS_STREAM_PROBE="1", HMS_STREAM_DIAGNOSTICS="1",
                   HMS_PROBE_OUTPUT=str(root / "diagnostic.json"),
                   PYTHONPATH="/bab/neurips23/streaming/hms:/bab")
        module, constructor = "neurips23.streaming.hms.streaming_probe", "HMSDiagnostic"
    elif mode in ("delete", "final"):
        dataset = "msturing-10M-clustered" if mode == "delete" else "msturing-30M-clustered"
        book = Path("/bab/neurips23/runbooks") / f"{mode}_runbook.yaml"
        module, constructor = "neurips23.streaming.hms.hms", "HMS"
    else:
        raise ValueError(mode)
    config = {dataset: {"hms": {"docker-tag": "neurips23-streaming-hms", "module": module,
              "constructor": constructor, "base-args": ["@metric"],
              "run-groups": {"base": {"args": [build], "query-args": [search]}}}}}
    import yaml
    Path("/bab/neurips23/streaming/hms/config.yaml").write_text(yaml.safe_dump(config))
    receipt["dataset"] = dataset
    receipt["runbook"] = str(book)
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
    try:
        if mode != "diagnostic":
            cached = Path("/vol/streaming-recall/data")
            cached.mkdir(parents=True, exist_ok=True)
            data = Path("/bab/data")
            if data.is_dir() and not data.is_symlink():
                data.rmdir()
            if not data.exists():
                data.symlink_to(cached, target_is_directory=True)
            execute(["python", "create_dataset.py", "--dataset", dataset], root / "prepare.log", 900, env)
            execute(["python", "-m", "benchmark.streaming.download_gt", "--runbook_file", str(book),
                     "--dataset", dataset], root / "ground-truth.log", 900, env)
            volume.commit()
        execute(["/usr/bin/time", "-v", "-o", str(root / "time.txt"), "python", "run.py", "--nodocker",
                 "--neurips23track", "streaming", "--algorithm", "hms", "--dataset", dataset,
                 "--runbook_path", str(book)], root / "run.log", 1200 if mode == "diagnostic" else 3600, env)
        if mode != "diagnostic":
            execute(["python", "data_export.py", "--out", str(root / "results.csv")], root / "export.log", 180, env)
            entries = yaml.safe_load(book.read_text())[dataset]
            totals = {"search": [0, 0.0], "insert": [0, 0.0], "delete": [0, 0.0], "replace": [0, 0.0]}
            for step, seconds in re.findall(r"Step (\d+) took ([\d.]+)s", (root / "run.log").read_text()):
                entry = entries[int(step)]
                operation = entry["operation"]
                count = 10_000 if operation == "search" else (
                    entry["tags_end"] - entry["tags_start"] if operation == "replace" else entry["end"] - entry["start"])
                totals[operation][0] += count
                totals[operation][1] += float(seconds)
            receipt["throughput"] = {op: {"count": count, "seconds": seconds,
                                      "per_second": count / seconds if seconds else None}
                                     for op, (count, seconds) in totals.items()}
        shutil.copytree("/bab/results", root / "raw-results", dirs_exist_ok=True)
        receipt["status"] = "complete"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.time() - started
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
        volume.commit()
        print(json.dumps(receipt), flush=True)
    return str(root)


@app.local_entrypoint()
def main(mode: str = "diagnostic", build: str = '{"R":32,"L":64,"T":8}', search: str = '{"ef":64}'):
    print(run.remote(mode, json.loads(build), json.loads(search)))
