"""Run a small discriminating probe using the already-built native image."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import modal

ADAPTER = Path(__file__).resolve().parent
IMAGE_ID = "im-b2vpMMYmHN4fYGbms2isdz"
app = modal.App("hms-bigann")
volume = modal.Volume.from_name("hms-bigann", create_if_missing=False)
image = modal.Image.from_id(IMAGE_ID).add_local_dir(ADAPTER, "/bab/neurips23/streaming/hms")


@app.function(image=image, volumes={"/vol": volume}, cpu=(8.0, 8.0), memory=(8192, 8192),
              timeout=600, max_containers=1, nonpreemptible=True)
def probe(run_id: str):
    import psutil

    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("invalid run ID")
    root = Path("/vol/streaming-recall") / f"runtime-probe-{run_id}"
    if root.exists():
        raise RuntimeError("refusing automatic probe replay")
    root.mkdir()
    cpus = sorted(os.sched_getaffinity(0))[:8]
    if len(cpus) != 8:
        raise RuntimeError("eight CPUs required")
    os.sched_setaffinity(0, cpus)
    receipt = {"image_id": IMAGE_ID, "cpus": cpus, "rss_guard_bytes": 1024**3,
               "status": "running", "runs": []}
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
    volume.commit()
    env = dict(os.environ, RAYON_NUM_THREADS="8", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
    for key in ("MALLOC_ARENA_MAX", "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "GLIBC_TUNABLES"):
        env.pop(key, None)
    cases = [("full-map", "full-map", {}), ("range-read", "range-read", {}),
             ("allocator-default", "allocator", {}),
             ("allocator-tuned", "allocator", {"MALLOC_ARENA_MAX": "2",
              "MALLOC_MMAP_THRESHOLD_": "131072", "MALLOC_TRIM_THRESHOLD_": "131072"})]
    started = time.monotonic()
    try:
        for name, mode, extra in cases:
            peak = 0
            with (root / f"{name}.jsonl").open("w") as output:
                process = subprocess.Popen(["python", "/bab/neurips23/streaming/hms/runtime_probe.py", mode],
                                           cwd="/bab", env=env | extra, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    deadline = time.monotonic() + 150
                    with (root / f"{name}.rss.jsonl").open("w") as samples:
                        while process.poll() is None:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(name)
                            try:
                                child = psutil.Process(process.pid)
                                processes = [psutil.Process(), child, *child.children(recursive=True)]
                                rss = sum(p.memory_info().rss for p in processes)
                            except psutil.NoSuchProcess:
                                continue
                            peak = max(peak, rss)
                            samples.write(json.dumps({"elapsed_seconds": time.monotonic() - started,
                                                      "summed_rss_bytes": rss}) + "\n")
                            samples.flush()
                            if rss > 1024**3:
                                raise MemoryError(f"probe exceeds 1 GiB: {rss}")
                            try:
                                process.wait(timeout=0.1)
                            except subprocess.TimeoutExpired:
                                pass
                    if process.returncode != 0:
                        raise RuntimeError(f"{name} exited {process.returncode}")
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            receipt["runs"].append({"name": name, "peak_summed_rss_bytes": peak, "environment": extra})
            volume.commit()
        hashes = [json.loads((root / f"{name}.jsonl").read_text().splitlines()[-1])["sha256"]
                  for name in ("full-map", "range-read")]
        if hashes[0] != hashes[1]:
            raise AssertionError("dataset readers differ")
        receipt["dataset_readers_equal"] = True
        receipt["status"] = "complete"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
        volume.commit()
        print(json.dumps(receipt), flush=True)
    return str(root)


@app.local_entrypoint()
def main(run_id: str):
    print(probe.remote(run_id))
