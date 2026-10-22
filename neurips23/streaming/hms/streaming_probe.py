"""A cropped diagnostic workload; its results are never official runbook scores."""
import hashlib
import json
import os
from pathlib import Path
import resource
import struct
import time
import urllib.request

import numpy as np

from benchmark.datasets import DATASETS, MSTuringClustered10M
from neurips23.streaming.hms.hms import HMS
from neurips23.streaming.hms.streaming_truth import exact_topk

NAME = "msturing-1M-stream-diagnostic"
ROWS = 1_000_000
NQ = 256


def download_prefix(url, path, rows, dim):
    size = 8 + rows * dim * 4
    if path.exists():
        if path.stat().st_size != size:
            raise ValueError("invalid cached diagnostic file length")
        return
    temp = path.with_suffix(".partial")
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as out:
        header = response.read(8)
        source_rows, source_dim = struct.unpack("<II", header)
        if source_rows < rows or source_dim != dim:
            raise ValueError("unexpected source dimensions")
        out.write(struct.pack("<II", rows, dim))
        remaining = size - 8
        while remaining:
            chunk = response.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError("truncated diagnostic download")
            out.write(chunk)
            remaining -= len(chunk)
    temp.replace(path)


class DiagnosticDataset(MSTuringClustered10M):
    def __init__(self):
        super().__init__()
        self.nb = ROWS
        self.nq = NQ
        self.basedir = "/vol/streaming-recall/diagnostic-data"
        self.ds_fn = "base-1M.fbin"
        self.qs_fn = "queries-256.fbin"
        self.gt_fn = None

    def prepare(self, skip_data=False, original_size=None):
        root = Path(self.basedir)
        root.mkdir(parents=True, exist_ok=True)
        download_prefix(self.base_url + "msturing-10M-clustered.fbin", root / self.ds_fn, ROWS, self.d)
        download_prefix(self.base_url + "testQuery10K.fbin", root / self.qs_fn, NQ, self.d)


def register_dataset():
    DATASETS[NAME] = DiagnosticDataset




class HMSDiagnostic(HMS):
    def setup(self, dtype, max_pts, ndim):
        super().setup(dtype, max_pts, ndim)
        self.checkpoint = 0
        self.records = []

    def _maybe_consolidate(self):
        self._used = True

    def query(self, X, k):
        if k != 10 or len(X) != NQ:
            raise ValueError("diagnostic requires fixed 256 queries and k=10")
        compact_seconds = None
        if self.checkpoint == 2:
            before = time.perf_counter()
            self.index.consolidate()
            compact_seconds = time.perf_counter() - before
        dataset = DiagnosticDataset()
        vectors = dataset.get_data_in_range(0, ROWS)
        truth = exact_topk(vectors, X, 0 if self.checkpoint == 0 else 100_000)
        record = {
            "checkpoint": ("insert", "delete", "compact")[self.checkpoint],
            "R": self.R, "L": self.L, "T": self.threads,
            "queries": NQ, "k": k, "ground_truth_live_start": 0 if self.checkpoint == 0 else 100_000,
            "consolidate_seconds": compact_seconds,
            "topology": self.index.streaming_diagnostics(), "sweep": [],
            "query_sha256": hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
        }
        for ef in (64, 128, 256, 512):
            self.index.set_query_arguments(ef, ef)
            begin = time.perf_counter()
            found = self.index.search(X, k)
            seconds = time.perf_counter() - begin
            if np.any(found < (0 if self.checkpoint == 0 else 100_000)) or np.any(found >= ROWS):
                raise AssertionError("returned a dead or invalid generation")
            hits = sum(len(set(a).intersection(b)) for a, b in zip(found, truth))
            record["sweep"].append({"ef": ef, "recall10": hits / (len(X) * k), "qps": len(X) / seconds})
            if ef == 64:
                self.res = found
        record["counters_after"] = self.index.streaming_diagnostics()
        record["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.records.append(record)
        Path(os.environ["HMS_PROBE_OUTPUT"]).write_text(json.dumps(self.records, indent=2))
        print("STREAMING_DIAGNOSTIC " + json.dumps(record), flush=True)
        self.checkpoint += 1
