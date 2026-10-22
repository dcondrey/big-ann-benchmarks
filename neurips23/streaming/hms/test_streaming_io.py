from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from streaming_io import read_fbin_range


class SourceRangeTests(unittest.TestCase):
    def test_exact_rows_and_range_boundaries(self):
        expected = np.arange(13 * 7, dtype="<f4").reshape(13, 7) / 3
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.fbin"
            path.write_bytes(struct.pack("<II", *expected.shape) + expected.tobytes())
            for start, end in ((0, 12), (0, 13), (12, 13), (13, 13), (3, 8)):
                with self.subTest(start=start, end=end):
                    actual = read_fbin_range(path, start, end, *expected.shape)
                    np.testing.assert_array_equal(actual, expected[start:end])
            for start, end in ((0, 14), (-1, 13), (14, 14), (4, 3)):
                with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                    read_fbin_range(path, start, end, *expected.shape)
            with self.assertRaises(TypeError):
                read_fbin_range(path, 0.5, 1, *expected.shape)

    def test_rejects_malformed_source_before_returning_rows(self):
        header = struct.pack("<II", 13, 7)
        valid = header + bytes(13 * 7 * 4)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.fbin"
            for body in (valid[:7], valid[:-1], valid + b"x",
                         struct.pack("<II", 12, 7) + valid[8:],
                         struct.pack("<II", 13, 8) + valid[8:]):
                with self.subTest(length=len(body)), self.assertRaises(ValueError):
                    path.write_bytes(body)
                    read_fbin_range(path, 0, 1, 13, 7)


if __name__ == "__main__":
    unittest.main()
