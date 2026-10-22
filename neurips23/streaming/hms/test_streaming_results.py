from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from streaming_results import validate_results


class EligibilityTests(unittest.TestCase):
    def test_reused_tags_and_deleted_answers(self):
        entries = {
            1: {"operation": "insert", "start": 0, "end": 12},
            2: {"operation": "search"},
            3: {"operation": "delete", "start": 0, "end": 2},
            4: {"operation": "search"},
            5: {"operation": "insert", "start": 0, "end": 2},
            6: {"operation": "search"},
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "answers.hdf5"
            with h5py.File(path, "w") as output:
                for step, start in ((2, 0), (4, 2), (6, 0)):
                    output[f"neighbors_step{step}"] = np.tile(np.arange(start, start + 10), (2, 1))
            valid = validate_results(root, entries, 12, expected_queries=2)
            self.assertEqual(valid["returned_ids"], 60)
            for invalid in (-1, 0, 2, 12):
                with self.subTest(invalid=invalid):
                    with h5py.File(path, "r+") as output:
                        output["neighbors_step4"][0, 1] = invalid
                    with self.assertRaises(ValueError):
                        validate_results(root, entries, 12, expected_queries=2)
                    with h5py.File(path, "r+") as output:
                        output["neighbors_step4"][0, 1] = 3


if __name__ == "__main__":
    unittest.main()
