# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np

from consolidation_probe import critical_plan, source_batches
from consolidation_counts import count_triggers


class ConsolidationProbeTests(unittest.TestCase):
    def test_native_tombstone_threshold_at_adjacent_population_counts(self):
        for end in [29_999, 30_000, 30_001]:
            entries = {1: dict(operation="insert", start=0, end=100_000),
                       2: dict(operation="delete", start=0, end=end)}
            triggers = count_triggers(entries)
            self.assertEqual([row["step"] for row in triggers], [1] if end < 30_000 else [1, 2])
            self.assertEqual(triggers[-1]["live"], 100_000 if end < 30_000 else 100_000 - end)
        for ratio in [-0.1, float("nan"), 1.1]:
            with self.assertRaises(ValueError):
                count_triggers(entries, deleted_ratio=ratio)

    def test_peak_shape_keeps_the_exact_base_and_live_populations(self):
        entries = {
            1: dict(operation="insert", start=0, end=100_000),
            2: dict(operation="search"),
            3: dict(operation="insert", start=100_000, end=200_000),
            4: dict(operation="search"),
            5: dict(operation="delete", start=0, end=60_000),
            6: dict(operation="search"),
            7: dict(operation="insert", start=200_000, end=300_000),
            8: dict(operation="search"),
            9: dict(operation="delete", start=60_000, end=100_000),
            10: dict(operation="search"),
            11: dict(operation="insert", start=300_000, end=350_000),
        }
        plan = critical_plan(entries, 400_000)
        self.assertEqual((plan["base_step"], plan["target"]["step"]), (5, 11))
        np.testing.assert_array_equal(plan["base_ids"], np.arange(60_000, 200_000))
        np.testing.assert_array_equal(np.flatnonzero(plan["alive"]), np.arange(100_000, 350_000))
        self.assertEqual([step for step, _ in plan["changes"]], [7, 9, 11])

    def test_source_batches_keep_row_id_alignment_at_limits_and_gaps(self):
        for count in [2, 3, 4]:
            ids = np.r_[np.arange(count), [10, 11, 20]].astype(np.int64)
            batches = list(source_batches(ids, 3))
            np.testing.assert_array_equal(np.concatenate([tags for _, _, tags in batches]), ids)
            for start, end, tags in batches:
                np.testing.assert_array_equal(tags, np.arange(start, end))
                self.assertLessEqual(end - start, 3)
        with self.assertRaises(ValueError):
            list(source_batches(np.array([1]), 0))


if __name__ == "__main__":
    unittest.main()
