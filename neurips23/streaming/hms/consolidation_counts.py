"""Count the adapter's consolidation triggers without constructing an ANN index."""
import argparse
import json
import math
import operator

import numpy as np
import yaml


def count_triggers(entries, *, delta_ratio=1.0, deleted_ratio=0.3, min_delta=100_000):
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (delta_ratio, deleted_ratio)):
        raise ValueError("consolidation ratios must be in 0..=1")
    min_delta = operator.index(min_delta)
    if min_delta < 0:
        raise ValueError("minimum delta must be nonnegative")
    updates = [v for k, v in entries.items() if isinstance(k, int) and v["operation"] != "search"]
    extent = max(v.get("tags_end", v.get("end", 0)) for v in updates)
    alive = np.zeros(extent, dtype=bool)
    base_live = np.zeros(extent, dtype=bool)
    live = base = deleted = delta = 0
    result = []
    for step in sorted(k for k in entries if isinstance(k, int)):
        item = entries[step]
        operation = item["operation"]
        if operation == "search":
            continue
        start = item["tags_start"] if operation == "replace" else item["start"]
        end = item["tags_end"] if operation == "replace" else item["end"]
        count = end - start
        if operation == "insert":
            if np.any(alive[start:end]):
                raise ValueError("insertion of live tags")
            alive[start:end] = True
            live += count
            delta += count
        elif operation in ("delete", "replace"):
            if not np.all(alive[start:end]):
                raise ValueError("update of absent tags")
            deleted += int(np.count_nonzero(base_live[start:end]))
            base_live[start:end] = False
            if operation == "delete":
                alive[start:end] = False
                live -= count
            else:
                delta += count
        else:
            raise ValueError(operation)
        if (delta >= min_delta and delta / max(base, 1) >= delta_ratio) or (deleted and deleted / max(base, 1) >= deleted_ratio):
            result.append(dict(step=step, operation=operation, live=live, old_base_slots=base,
                               old_base_dead=deleted, delta_allocated=delta,
                               delta_live=live - (base - deleted)))
            base_live[:] = alive
            base, deleted, delta = live, 0, 0
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runbook")
    parser.add_argument("dataset")
    parser.add_argument("--delta-ratio", type=float, default=1.0)
    parser.add_argument("--deleted-ratio", type=float, default=0.3)
    parser.add_argument("--min-delta", type=int, default=100_000)
    args = parser.parse_args()
    with open(args.runbook) as source:
        entries = yaml.safe_load(source)[args.dataset]
    print(json.dumps(count_triggers(entries, delta_ratio=args.delta_ratio,
                                   deleted_ratio=args.deleted_ratio, min_delta=args.min_delta), indent=2))
