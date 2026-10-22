# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stress the final policy's peak consolidation shape; this is not an official replay."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import yaml

from neurips23.streaming.hms.consolidation_counts import count_triggers


def critical_plan(entries, source_rows):
    triggers = count_triggers(entries)
    target = max(triggers, key=lambda row: 916 * row["live"] + 548 * row["delta_allocated"])
    previous = max((row["step"] for row in triggers if row["step"] < target["step"]), default=0)
    alive = np.zeros(source_rows, dtype=bool)
    base_ids = np.empty(0, dtype=np.int64)
    changes = []
    for step in sorted(key for key in entries if isinstance(key, int) and key <= target["step"]):
        entry = entries[step]
        operation = entry["operation"]
        if operation == "search":
            continue
        if operation not in ("insert", "delete"):
            raise ValueError("shape probe requires insert/delete-only inputs")
        start, end = entry["start"], entry["end"]
        if not 0 <= start <= end <= source_rows:
            raise ValueError("source range out of bounds")
        if operation == "insert":
            if np.any(alive[start:end]):
                raise ValueError("insertion of live tags")
            alive[start:end] = True
        else:
            if not np.all(alive[start:end]):
                raise ValueError("deletion of absent tags")
            alive[start:end] = False
        if step == previous:
            base_ids = np.flatnonzero(alive)
        if previous < step:
            changes.append((step, entry))
    if len(base_ids) != target["old_base_slots"] or int(alive.sum()) != target["live"]:
        raise ValueError("shape does not match policy counts")
    return dict(base_step=previous, target=target, base_ids=base_ids, alive=alive, changes=changes)


def source_batches(ids, limit=65536):
    if limit <= 0:
        raise ValueError("positive batch limit required")
    at = 0
    while at < len(ids):
        stop = min(len(ids), at + limit)
        gaps = np.flatnonzero(np.diff(ids[at:stop]) != 1)
        if len(gaps):
            stop = at + int(gaps[0]) + 1
        yield int(ids[at]), int(ids[stop - 1]) + 1, ids[at:stop]
        at = stop


def main():
    import psutil
    from benchmark.dataset_io import knn_result_read
    from benchmark.streaming.compute_gt import gt_dir
    from holographic_memory import UpdatableIndex
    from neurips23.streaming.hms.prepare_gt import file_sha256
    from neurips23.streaming.hms.streaming_io import MSTuring30MRangeReader

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    book = Path("/bab/neurips23/runbooks/final_runbook.yaml")
    dataset = MSTuring30MRangeReader()
    entries = yaml.safe_load(book.read_text())["msturing-30M-clustered"]
    plan = critical_plan(entries, dataset.nb)
    gt_root = Path(gt_dir(dataset, book))
    manifest = json.loads((gt_root / "gt-input-manifest.json").read_text())
    if manifest["runbook_sha256"] != file_sha256(book):
        raise ValueError("GT manifest and runbook differ")
    gt_hashes = {row["file"]: row["sha256"] for row in manifest["files"]}
    result = dict(mode="critical-shape-probe-not-official", build=dict(R=32, L=64, T=8),
                  search=dict(ef=1024, rerank=1024), target=plan["target"], base_step=plan["base_step"],
                  runbook_sha256=manifest["runbook_sha256"], source_rows=dataset.nb,
                  base_ids_sha256=hashlib.sha256(memoryview(plan["base_ids"])).hexdigest(),
                  retained_result_bytes=512_000_000, phases=[], checkpoints=[])
    args.output.write_text(json.dumps(result, indent=2))
    retained = np.ones(result["retained_result_bytes"], dtype=np.uint8)
    index_dir = tempfile.TemporaryDirectory(prefix="hms-consolidation-")
    index = UpdatableIndex.create(str(Path(index_dir.name) / "index"), dataset.d,
                                  degree=32, build_ef=64, threads=8)
    index.set_query_arguments(1024, 1024)

    def phase(name, **extra):
        row = dict(phase=name, live=len(index), rss_bytes=psutil.Process().memory_info().rss,
                   hwm_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, **extra)
        result["phases"].append(row)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)

    def insert(ids):
        for start, end, tags in source_batches(ids):
            vectors = dataset.get_data_in_range(start, end)
            index.insert(vectors, tags)

    def checkpoint(step, live):
        filename = f"step{step}.gt100"
        gt_path = gt_root / filename
        if file_sha256(gt_path) != gt_hashes[filename]:
            raise ValueError("cached checkpoint GT hash mismatch")
        truth, _ = knn_result_read(gt_path)
        queries = dataset.get_queries()
        started = time.monotonic()
        ids = index.search(queries, 10)
        seconds = time.monotonic() - started
        if ids.shape != (10_000, 10) or np.any(ids < 0) or np.any(ids >= dataset.nb) or not np.all(live[ids]):
            raise ValueError("dead, missing, or invalid answers")
        if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
            raise ValueError("duplicate answers")
        recall = float(np.any(ids[:, :, None] == truth[:, None, :10], axis=2).sum() / ids.size)
        row = dict(step=step, seconds=seconds, qps=len(ids) / seconds, recall10=recall,
                   gt_sha256=gt_hashes[filename],
                   answers_sha256=hashlib.sha256(memoryview(np.ascontiguousarray(ids))).hexdigest(),
                   all_answers_live_and_unique=True)
        result["checkpoints"].append(row)
        phase("query", **row)

    done, total = 0, len(plan["base_ids"])
    target = min(100_000, total)
    while done < total:
        started = time.monotonic()
        insert(plan["base_ids"][done:target])
        phase("base_insert", count=target - done, seconds=time.monotonic() - started)
        started = time.monotonic()
        phase("base_compact_start")
        index.consolidate()
        phase("base_compact_finish", seconds=time.monotonic() - started)
        done, target = target, min(total, target * 2)
    live = np.zeros(dataset.nb, dtype=bool)
    live[plan["base_ids"]] = True
    checkpoint(plan["base_step"] + 1, live)
    del live
    for step, entry in plan["changes"]:
        started = time.monotonic()
        tags = np.arange(entry["start"], entry["end"], dtype=np.int64)
        if entry["operation"] == "insert":
            insert(tags)
        else:
            index.delete(tags)
        phase("update", step=step, operation=entry["operation"], seconds=time.monotonic() - started)
        if step != plan["target"]["step"] and index.needs_consolidation(1.0, 0.3, 100_000):
            raise ValueError("unexpected earlier consolidation")
        if step == plan["changes"][-2][0]:
            before = plan["alive"].copy()
            last = plan["changes"][-1][1]
            before[last["start"]:last["end"]] = last["operation"] == "delete"
            checkpoint(step + 1, before)
            del before
    if len(index) != plan["target"]["live"] or not index.needs_consolidation(1.0, 0.3, 100_000):
        raise ValueError("critical policy shape mismatch")
    phase("critical_compact_start")
    started = time.monotonic()
    index.consolidate()
    phase("critical_compact_finish", seconds=time.monotonic() - started)
    checkpoint(plan["target"]["step"] + 1, plan["alive"])
    if int(retained[0]) != 1 or int(retained[-1]) != 1:
        raise ValueError("retained-result allocation changed")
    result["status"] = "complete"
    args.output.write_text(json.dumps(result, indent=2))
    del index
    index_dir.cleanup()


if __name__ == "__main__":
    main()
