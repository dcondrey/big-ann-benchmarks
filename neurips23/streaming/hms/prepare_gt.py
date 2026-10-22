"""Fetch the official checkpoint GT with bounded concurrency and an input manifest."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from urllib.request import urlopen


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_gt(url, target, nq, expected_hash=None):
    target = Path(target)
    size = 8 + nq * 100 * 8
    header = struct.pack("<II", nq, 100)
    if target.exists() and target.stat().st_size == size:
        with target.open("rb") as source:
            cached_header = source.read(8)
        if cached_header == header:
            digest = file_sha256(target)
            if expected_hash is None or digest == expected_hash:
                return {"url": url, "file": target.name, "bytes": size, "sha256": digest}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with urlopen(url, timeout=120) as source, tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".gt-", delete=False) as output:
            temp = Path(output.name)
            actual_header = source.read(8)
            if actual_header != header:
                raise ValueError(f"unexpected GT header for {target.name}")
            digest = hashlib.sha256(actual_header)
            output.write(actual_header)
            remaining = size - 8
            while remaining:
                block = source.read(min(1 << 20, remaining))
                if not block:
                    raise ValueError(f"truncated GT for {target.name}")
                output.write(block)
                digest.update(block)
                remaining -= len(block)
            if source.read(1):
                raise ValueError(f"oversized GT for {target.name}")
            checksum = digest.hexdigest()
            if expected_hash is not None and checksum != expected_hash:
                raise ValueError(f"GT source changed for {target.name}")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, target)
        return {"url": url, "file": target.name, "bytes": size, "sha256": checksum}
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    import argparse
    from benchmark.datasets import DATASETS
    from benchmark.streaming.compute_gt import gt_dir
    from benchmark.streaming.load_runbook import get_gt_url, load_runbook

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]()
    _, entries = load_runbook(args.dataset, dataset.nb, args.runbook)
    url = get_gt_url(args.dataset, args.runbook)
    if not url.startswith("https://"):
        raise ValueError("official ground truth must use HTTPS")
    root = Path(gt_dir(dataset, args.runbook))
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "gt-input-manifest.json"
    runbook_hash = file_sha256(args.runbook)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None and previous["runbook_sha256"] != runbook_hash:
        raise ValueError("cached GT belongs to another runbook revision")
    hashes = {r["file"]: r["sha256"] for r in previous["files"]} if previous else {}
    files = [f"step{step}.gt100" for step, entry in enumerate(entries, 1) if entry["operation"] == "search"]
    def fetch(name):
        return fetch_gt(url + "/" + name, root / name, dataset.nq, hashes.get(name))
    with ThreadPoolExecutor(max_workers=4) as workers:
        records = list(workers.map(fetch, files))
    result = {"dataset": args.dataset, "runbook_sha256": runbook_hash, "files": records}
    temporary = manifest_path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(manifest_path)
    print(json.dumps({"checkpoint_files": len(records), "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
