import unittest

import numpy as np
from scipy.spatial.distance import cdist

from streaming_truth import exact_topk


class ExactTruthTests(unittest.TestCase):
    def test_chunk_boundaries_live_states_and_ties(self):
        rng = np.random.default_rng(20261022)
        vectors = rng.normal(size=(8193, 3)).astype(np.float32)
        vectors[:12] = vectors[8192]
        queries = np.vstack((vectors[8192], rng.normal(size=(3, 3))))
        for n in (8191, 8192, 8193):
            for start in (0, 10, n - 10):
                with self.subTest(n=n, start=start):
                    ids = np.arange(start, n, dtype=np.uint32)
                    scores = cdist(queries, vectors[start:n], metric="sqeuclidean")
                    expected = np.array([ids[np.lexsort((ids, row))[:10]] for row in scores])
                    np.testing.assert_array_equal(exact_topk(vectors[:n], queries, start), expected)

    def test_invalid_live_population(self):
        for start in (-1, 1, 10, 11):
            with self.subTest(start=start), self.assertRaises(ValueError):
                exact_topk(np.zeros((10, 3)), np.zeros((2, 3)), start)


if __name__ == "__main__":
    unittest.main()
