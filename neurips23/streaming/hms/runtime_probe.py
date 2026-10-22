"""Bounded allocation and dataset-I/O probes, separate from official scores."""
import argparse
import ctypes
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import tempfile
import time

import numpy as np
import psutil


def sample(phase, **extra):
    record = {"phase": phase, "rss_bytes": psutil.Process().memory_info().rss,
              "hwm_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **extra}
    print(json.dumps(record), flush=True)


def allocator():
    from holographic_memory import UpdatableIndex

    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    vectors = np.random.default_rng(817).random((100_000, 100), dtype=np.float32)
    ids = np.arange(len(vectors), dtype=np.int64)
    queries = vectors[:64].copy()
    sample("start", libc=platform.libc_ver(), rows=len(vectors), dim=vectors.shape[1])
    with tempfile.TemporaryDirectory(prefix="hms-allocation-") as root:
        for cycle in range(3):
            index = UpdatableIndex.create(str(Path(root) / str(cycle)), 100,
                                          degree=32, build_ef=64, threads=8)
            index.set_query_arguments(64, 64)
            index.insert(vectors, ids)
            sample("insert", cycle=cycle)
            before = index.search(queries, 10)
            sample("before_insert_trim", cycle=cycle)
            released = libc.malloc_trim(0)
            np.testing.assert_array_equal(before, index.search(queries, 10))
            sample("after_insert_trim", cycle=cycle, trim_return=released)
            begin = time.perf_counter()
            index.consolidate()
            sample("consolidate", cycle=cycle, seconds=time.perf_counter() - begin)
            before = index.search(queries, 10)
            sample("before_compact_trim", cycle=cycle)
            released = libc.malloc_trim(0)
            np.testing.assert_array_equal(before, index.search(queries, 10))
            sample("after_compact_trim", cycle=cycle, trim_return=released)
            del index
            gc.collect()
            sample("drop", cycle=cycle)
            released = libc.malloc_trim(0)
            sample("after_drop_trim", cycle=cycle, trim_return=released)


def dataset_io(mode):
    source = Path("/vol/streaming-recall/diagnostic-data/base-1M.fbin")
    header = np.fromfile(source, dtype="<u4", count=2)
    if tuple(header) != (1_000_000, 100) or source.stat().st_size != 400_000_008:
        raise ValueError("unexpected diagnostic input")
    sample("before_io", mode=mode)
    if mode == "full-map":
        vectors = np.memmap(source, dtype="<f4", mode="r", offset=8,
                            shape=(1_000_000, 100))[:38_806]
    else:
        from neurips23.streaming.hms.streaming_io import read_fbin_range

        vectors = read_fbin_range(source, 0, 38_806, 1_000_000, 100)
    sample("after_io", mode=mode, rows=len(vectors))
    digest = hashlib.sha256(vectors.tobytes()).hexdigest()
    sample("after_touch", mode=mode, sha256=digest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("allocator", "full-map", "range-read"))
    args = parser.parse_args()
    if args.mode == "allocator":
        allocator()
    else:
        dataset_io(args.mode)
