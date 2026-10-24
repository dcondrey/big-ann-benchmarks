"""OOD-track wrapper for HMS (holographic-memory): a quantized proximity graph routed by the
dataset's training queries.

The graph is built over the base vectors (inner product reduced to Euclidean distance by one
appended coordinate), then gains projected edges from the nearest base rows of the first
`learn` vectors of the provided training-query file (`query.learn.50M.fbin`, never the
evaluation queries). Search reads 8-bit codes of the original coordinates, scores by inner
product directly and re-ranks the best `rerank` candidates with an 8-bit residual code.
"""
import os

# Back the index arrays with transparent huge pages (a no-op where the kernel refuses them).
os.environ.setdefault("HMS_QGRAPH_HUGEPAGES", "1")

import numpy as np
from holographic_memory import VectorIndex

from benchmark.dataset_io import download, xbin_mmap
from benchmark.datasets import DATASETS, download_accelerated
from neurips23.ood.base import BaseOODANN

LEARN_URL = "https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin"
METRICS = {"ip": "ip", "euclidean": "euclidean", "angular": "angular"}


class HMS(BaseOODANN):
    def __init__(self, metric, index_params):
        if metric not in METRICS:
            raise NotImplementedError(f"hms does not support the {metric} distance")
        self.metric = metric
        self.degree = int(index_params.get("degree", 32))
        self.build_ef = int(index_params.get("build_ef", 200))
        self.code_bits = int(index_params.get("code_bits", 8))
        self.direct_ip = bool(index_params.get("direct_ip", metric == "ip"))
        # Training queries, in millions of rows of the learn file (0 builds the plain graph).
        self.learn_m = float(index_params.get("learn_M", 2))
        self.training_ef = int(index_params.get("training_ef", 200))
        self.training_neighbors = int(index_params.get("training_neighbors", 100))
        self.projected_degree = int(index_params.get("projected_degree", 16))
        self.threads = int(index_params.get("threads", 8))
        self.url = index_params.get("url")
        self.ef = 64
        self.rerank = 64
        self.index = None
        self.res = None

    def _new_index(self):
        return VectorIndex(METRICS[self.metric], degree=self.degree, build_ef=self.build_ef,
                           threads=self.threads, code_bits=self.code_bits,
                           direct_ip=self.direct_ip)

    def index_name(self):
        return (f"hms_R{self.degree}_L{self.build_ef}_b{self.code_bits}_learn{self.learn_m:g}M"
                f"_tef{self.training_ef}_k{self.training_neighbors}_p{self.projected_degree}"
                f"{'_dip' if self.direct_ip else ''}")

    def create_index_dir(self, ds):
        index_dir = os.path.join(os.getcwd(), "data", "indices", "ood", "hms", ds.short_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        return index_dir

    def _learn_queries(self, ds, index_dir):
        rows = int(self.learn_m * 1_000_000)
        if rows == 0 or ds.d != 200 or ds.distance() != "ip":
            # Only the text-to-image set ships a training-query file.
            return None
        path = os.path.join(index_dir, f"learn.{rows}.fbin")
        if not os.path.exists(path):
            download(LEARN_URL, path, max_size=8 + rows * ds.d * 4)
        learn = xbin_mmap(path, dtype="float32", maxn=rows)
        assert learn.shape == (rows, ds.d), learn.shape
        return learn

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        index_dir = self.create_index_dir(ds)
        base = xbin_mmap(ds.get_dataset_fn(), dtype=ds.dtype, maxn=ds.nb)
        learn = self._learn_queries(ds, index_dir)
        self.index = self._new_index()
        if learn is None:
            self.index.fit(base)
        else:
            self.index.fit(base, training_queries=learn, training_limit=len(learn),
                           training_neighbors=self.training_neighbors,
                           projected_degree=self.projected_degree,
                           training_ef=self.training_ef)
        self.index.save(os.path.join(index_dir, self.index_name()))

    def index_files_to_store(self, dataset):
        return [self.create_index_dir(DATASETS[dataset]()), self.index_name(), [""]]

    def load_index(self, dataset):
        path = os.path.join(self.create_index_dir(DATASETS[dataset]()), self.index_name())
        if not os.path.exists(path):
            if not self.url:
                return False
            download_accelerated(f"{self.url}/{self.index_name()}", path, quiet=True)
        self.index = self._new_index()
        self.index.load(path)
        return True

    def set_query_arguments(self, query_args):
        self.ef = int(query_args["ef"])
        self.rerank = int(query_args.get("rerank", self.ef))
        self.index.set_query_arguments(self.ef, self.rerank)

    def query(self, X, k):
        self.res = self.index.batch_query_array(np.ascontiguousarray(X, dtype=np.float32), k)

    def __str__(self):
        return f"hms({self.index_name()}, ef={self.ef}, rerank={self.rerank})"
