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

HMS_SHA = os.environ.get("HMS_COMMIT") or subprocess.check_output(
    ["git", "-C", "/hms", "rev-parse", "HEAD"], text=True
).strip()
if re.fullmatch(r"[0-9a-f]{40}", HMS_SHA) is None:
    raise ValueError("HMS_COMMIT must be a full commit SHA")
BAB_SHA = "89a3aba"
ADAPTER = Path(__file__).resolve().parent
app = modal.App("hms-bigann")
volume = modal.Volume.from_name("hms-bigann", create_if_missing=False)
IMAGE_ID = os.environ.get("HMS_IMAGE_ID")
image = modal.Image.from_id(IMAGE_ID) if IMAGE_ID else (
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
)
image = image.add_local_dir(ADAPTER, "/bab/neurips23/streaming/hms")


def execute(args, log, timeout, env):
    import psutil
    deadline = time.monotonic() + timeout
    last_commit = time.monotonic()
    peak = 0
    with log.open("w") as output:
        process = subprocess.Popen(args, cwd="/bab", stdout=output, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            with log.with_suffix(".rss.jsonl").open("w") as samples:
                while True:
                    try:
                        code = process.wait(timeout=min(0.25, max(0.001, deadline - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            raise
                        try:
                            child = psutil.Process(process.pid)
                            processes = [psutil.Process(), child, *child.children(recursive=True)]
                            rss_by_pid = {p.pid: p.memory_info().rss for p in processes}
                            rss = sum(rss_by_pid.values())
                        except psutil.NoSuchProcess:
                            continue
                        peak = max(peak, rss)
                        sample = {"elapsed_seconds": time.monotonic() - (deadline - timeout),
                                  "summed_process_rss_bytes": rss, "peak_sampled_rss_bytes": peak,
                                  "rss_by_pid": rss_by_pid}
                        samples.write(json.dumps(sample) + "\n")
                        samples.flush()
                        if rss > 8 * 1024**3:
                            try:
                                descendants = child.children(recursive=True)
                            except psutil.NoSuchProcess:
                                descendants = []
                            leaves = []
                            for descendant in descendants:
                                try:
                                    if not descendant.children():
                                        leaves.append(descendant)
                                except psutil.NoSuchProcess:
                                    continue
                            for leaf in leaves or [child]:
                                try:
                                    leaf.kill()
                                except psutil.NoSuchProcess:
                                    pass
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            raise MemoryError(f"summed benchmark and parent RSS {rss} exceeds 8 GiB")
                        if time.monotonic() - last_commit >= 60:
                            volume.commit()
                            print("STREAMING_PROGRESS " + json.dumps(sample), flush=True)
                            last_commit = time.monotonic()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        raise RuntimeError(f"{args[0]} failed with {code}; see {log}")


@app.function(image=image, volumes={"/vol": volume}, cpu=(8.0, 8.0), memory=(8192, 8192),
              timeout=3 * 3600, max_containers=1, nonpreemptible=True)
def run(mode: str, build: dict, search: dict, run_id: str):
    if mode not in ("diagnostic", "delete", "final"):
        raise ValueError(mode)
    started = time.time()
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("run_id must be a 32-character hex identifier")
    native_sha = subprocess.check_output(["git", "-C", "/hms", "rev-parse", "HEAD"], text=True).strip()
    if native_sha != HMS_SHA:
        raise ValueError("cached native image does not match HMS_COMMIT")
    tag = f"{mode}-{HMS_SHA[:12]}-{run_id}"
    root = Path("/vol/streaming-recall") / tag
    if root.exists():
        previous = root / "receipt.json"
        if previous.exists() and json.loads(previous.read_text()).get("status") == "complete":
            return str(root)
        raise RuntimeError(f"previous attempt interrupted; refusing automatic replay of {tag}")
    root.mkdir(parents=True)
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 8:
        raise RuntimeError(f"only {len(affinity)} schedulable CPUs; need eight")
    os.sched_setaffinity(0, affinity[:8])
    env = dict(os.environ, RAYON_NUM_THREADS="8", OMP_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1")
    receipt = {"mode": mode, "hms_sha": HMS_SHA, "harness_sha": BAB_SHA,
               "build": build, "search": search, "cpus": affinity[:8],
               "memory_limit_bytes": 8 * 1024**3, "started": started, "status": "running"}
    from hashlib import sha256
    adapter_root = Path("/bab/neurips23/streaming/hms")
    receipt["adapter_source_sha256"] = {
        name: sha256((adapter_root / name).read_bytes()).hexdigest()
        for name in ("hms.py", "streaming_io.py", "prepare_gt.py", "streaming_results.py")}
    receipt["launcher_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["dataset_io"] = "bounded-fbin-range"
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
    volume.commit()
    try:
        if mode != "diagnostic":
            folder = "MSTuring-10M-clustered" if mode == "delete" else "MSTuring-30M-clustered"
            cached = Path("/vol/streaming-recall/data") / folder
            cached.mkdir(parents=True, exist_ok=True)
            data = Path("/bab/data")
            data.mkdir(exist_ok=True)
            dataset_dir = data / folder
            if not dataset_dir.exists():
                dataset_dir.symlink_to(cached, target_is_directory=True)
            execute(["python", "create_dataset.py", "--dataset", dataset], root / "prepare.log", 1800, env)
            execute(["python", "-m", "neurips23.streaming.hms.prepare_gt", "--runbook", str(book),
                     "--dataset", dataset], root / "ground-truth.log", 1800, env)
            volume.commit()
        execute(["/usr/bin/time", "-v", "-o", str(root / "time.txt"), "python", "run.py", "--nodocker",
                 "--neurips23track", "streaming", "--algorithm", "hms", "--dataset", dataset,
                 "--runbook_path", str(book)], root / "run.log", 1200 if mode == "diagnostic" else 3600, env)
        if mode != "diagnostic":
            entries = yaml.safe_load(book.read_text())[dataset]
            import sys
            sys.path.insert(0, "/bab/neurips23/streaming/hms")
            from streaming_results import validate_results
            receipt["eligibility_validation"] = validate_results(
                "/bab/results", entries, 10_000_000 if mode == "delete" else 29_998_994)
            execute(["python", "data_export.py", "--out", str(root / "results.csv")], root / "export.log", 180, env)
            totals = {"search": [0, 0.0], "insert": [0, 0.0], "delete": [0, 0.0], "replace": [0, 0.0]}
            timings = re.findall(r"Step (\d+) took ([0-9.eE+-]+)s", (root / "run.log").read_text())
            expected_steps = {k for k in entries if isinstance(k, int)}
            if len(timings) != len(expected_steps) or {int(step) for step, _ in timings} != expected_steps:
                raise ValueError("incomplete or duplicate runbook timing records")
            for step, seconds in timings:
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
    except BaseException as error:
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
def main(mode: str = "diagnostic", build: str = '{"R":32,"L":64,"T":8}', search: str = '{"ef":64}', run_id: str = ""):
    import uuid
    print(run.remote(mode, json.loads(build), json.loads(search), run_id or uuid.uuid4().hex))
