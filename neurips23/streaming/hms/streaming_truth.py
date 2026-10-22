"""Bounded exact ground truth for the diagnostic live state."""
import numpy as np
from scipy.spatial.distance import cdist


def exact_topk(vectors, queries, start):
    if not 0 <= start <= len(vectors) - 10:
        raise ValueError("diagnostic ground truth requires at least ten live rows")
    distances = np.full((len(queries), 10), np.inf)
    ids = np.full((len(queries), 10), np.iinfo(np.uint32).max, dtype=np.uint32)
    query64 = np.asarray(queries, dtype=np.float64)
    for offset in range(start, len(vectors), 8192):
        block = np.asarray(vectors[offset:offset + 8192], dtype=np.float64)
        scores = cdist(query64, block, metric="sqeuclidean")
        take = min(10, len(block))
        for q, row in enumerate(scores):
            part = np.argpartition(row, take - 1)[:take]
            boundary = row[part].max()
            below = np.flatnonzero(row < boundary)
            ties = np.flatnonzero(row == boundary)[:take - len(below)]
            chosen = np.concatenate((below, ties))
            ds = np.concatenate((distances[q], row[chosen]))
            tags = np.concatenate((ids[q], (chosen + offset).astype(np.uint32)))
            order = np.lexsort((tags, ds))[:10]
            distances[q], ids[q] = ds[order], tags[order]
    return ids

