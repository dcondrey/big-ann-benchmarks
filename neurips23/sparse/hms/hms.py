import numpy as np

from neurips23.sparse.base import BaseSparseANN
from benchmark.datasets import DATASETS

import holographic_memory


class HmsSparse(BaseSparseANN):
    """Blocked learned-sparse inner-product index of Holographic Memory (HMS).

    Build: every coordinate keeps its `postings` largest postings, split into blocks
    (`block_fraction` of the list length) by shallow clustering; a block summary keeps the
    coordinates holding `energy` of its mass. A forward index scores candidates exactly.
    Search: walk the lists of the `cut` heaviest query coordinates and score a block's
    documents when its summary score times `heap_factor` reaches the current 10th best.
    """

    def __init__(self, metric, index_params):
        assert metric == "ip"
        self.name = "hms"
        self.postings = int(index_params.get("postings", 4000))
        self.block_fraction = float(index_params.get("block_fraction", 0.1))
        self.energy = float(index_params.get("energy", 0.4))
        self.index = holographic_memory.SparseIndex(
            postings_per_term=self.postings,
            block_fraction=self.block_fraction,
            summary_energy=self.energy,
        )
        self.cut = 10
        self.heap_factor = 1.0

    def fit(self, dataset):
        ds = DATASETS[dataset]()
        assert ds.data_type() == "sparse"
        self.index.build_from_spmat(ds.get_dataset_fn())

    def load_index(self, dataset):
        return False

    def set_query_arguments(self, query_args):
        self.cut = int(query_args["cut"])
        self.heap_factor = float(query_args["heap_factor"])
        self.index.set_search(self.cut, self.heap_factor)

    def query(self, X, k):
        self.res = self.index.search(
            np.ascontiguousarray(X.indptr, dtype=np.int64),
            np.ascontiguousarray(X.indices, dtype=np.int32),
            np.ascontiguousarray(X.data, dtype=np.float32),
            k,
        )

    def get_results(self):
        return self.res

    def __str__(self):
        return (
            f"hms(p{self.postings}_b{self.block_fraction}_e{self.energy}"
            f"_cut{self.cut}_hf{self.heap_factor})"
        )
