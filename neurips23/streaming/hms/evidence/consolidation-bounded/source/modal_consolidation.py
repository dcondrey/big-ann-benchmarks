# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run one bounded critical-shape probe with the streaming native image."""
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import modal

sys.path.insert(0, "/bab")
from neurips23.streaming.hms.modal_stream import HMS_SHA, execute, image, volume

app = modal.App("hms-bigann")


@app.function(image=image, volumes={"/vol": volume}, cpu=(8.0, 8.0), memory=(8192, 8192),
              timeout=3700, max_containers=1, nonpreemptible=True)
def probe(run_id: str):
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError("invalid run ID")
    actual_sha = subprocess.check_output(["git", "-C", "/hms", "rev-parse", "HEAD"], text=True).strip()
    if actual_sha != HMS_SHA:
        raise ValueError("native image and declared commit differ")
    root = Path("/vol/streaming-recall") / f"consolidation-{HMS_SHA[:12]}-{run_id}"
    if root.exists():
        raise RuntimeError("refusing automatic probe replay")
    root.mkdir()
    cpus = sorted(os.sched_getaffinity(0))[:8]
    if len(cpus) != 8:
        raise RuntimeError("eight CPUs required")
    os.sched_setaffinity(0, cpus)
    folder = Path("/bab/data/MSTuring-30M-clustered")
    if not folder.exists():
        folder.parent.mkdir(exist_ok=True)
        folder.symlink_to("/vol/streaming-recall/data/MSTuring-30M-clustered", target_is_directory=True)
    receipt = dict(mode="critical-shape-probe-not-official", hms_sha=HMS_SHA, cpus=cpus,
                   memory_limit_bytes=8 * 1024**3, status="running", started=time.time())
    adapter = Path("/bab/neurips23/streaming/hms")
    receipt["source_sha256"] = {
        name: hashlib.sha256((adapter / name).read_bytes()).hexdigest()
        for name in ("consolidation_probe.py", "consolidation_counts.py", "modal_consolidation.py", "streaming_io.py")
    }
    (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
    volume.commit()
    env = dict(os.environ, RAYON_NUM_THREADS="8", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="8", PYTHONUNBUFFERED="1")
    for key in ("MALLOC_ARENA_MAX", "MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_", "GLIBC_TUNABLES", "HMS_STREAM_DIAGNOSTICS"):
        env.pop(key, None)
    try:
        execute(["/usr/bin/time", "-v", "-o", str(root / "time.txt"), "python", "-m",
                 "neurips23.streaming.hms.consolidation_probe", "--output", str(root / "probe.json")],
                root / "run.log", 3600, env)
        receipt["status"] = "complete"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.time() - receipt["started"]
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2))
        volume.commit()
        print(json.dumps(receipt), flush=True)
    return str(root)


@app.local_entrypoint()
def main(run_id: str):
    print(probe.remote(run_id))
