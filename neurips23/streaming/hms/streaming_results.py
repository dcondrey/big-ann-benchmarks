"""Validate recorded streaming answers against the runbook's live tag set."""
from pathlib import Path

import h5py
import numpy as np


def validate_results(directory, entries, source_rows, expected_queries=10_000):
    files = list(Path(directory).rglob("*.hdf5"))
    if len(files) != 1:
        raise ValueError(f"expected one completed replay, found {len(files)} result files")
    alive = np.zeros(source_rows, dtype=bool)
    searches = queries = returned = 0
    with h5py.File(files[0], "r") as data:
        for step in sorted(k for k in entries if isinstance(k, int)):
            entry = entries[step]
            operation = entry["operation"]
            if operation == "search":
                ids = data[f"neighbors_step{step}"][:]
                if ids.shape != (expected_queries, 10):
                    raise ValueError("unexpected answer shape")
                if np.any(ids < 0) or np.any(ids >= source_rows) or not np.all(alive[ids]):
                    raise ValueError(f"dead or invalid answer at step {step}")
                if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
                    raise ValueError(f"duplicate answer at step {step}")
                searches += 1
                queries += len(ids)
                returned += ids.size
            elif operation in ("insert", "delete", "replace"):
                start = entry["tags_start"] if operation == "replace" else entry["start"]
                end = entry["tags_end"] if operation == "replace" else entry["end"]
                if not 0 <= start <= end <= source_rows:
                    raise ValueError("runbook tag range outside source bounds")
                if operation == "insert":
                    if np.any(alive[start:end]):
                        raise ValueError("insert reuses a live tag")
                    alive[start:end] = True
                else:
                    if not np.all(alive[start:end]):
                        raise ValueError("delete/replace uses an absent tag")
                    if operation == "delete":
                        alive[start:end] = False
            else:
                raise ValueError(f"unknown operation {operation}")
    return {"search_checkpoints": searches, "queries": queries, "returned_ids": returned,
            "all_returned_ids_live_and_unique": True}
