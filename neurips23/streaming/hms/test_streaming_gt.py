from pathlib import Path
import struct
import tempfile
import unittest

from prepare_gt import fetch_gt


class GroundTruthDownloadTests(unittest.TestCase):
    def test_atomic_length_header_and_hash_validation(self):
        payload = struct.pack("<II", 2, 100) + bytes(2 * 100 * 8)
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            target = Path(root) / "cache" / "step2.gt100"
            source.write_bytes(payload)
            original = fetch_gt(source.as_uri(), target, 2)
            for body in (payload[:-1], payload + b"x", struct.pack("<II", 3, 100) + payload[8:]):
                with self.subTest(length=len(body)):
                    source.write_bytes(body)
                    with self.assertRaises(ValueError):
                        fetch_gt(source.as_uri(), target, 2, expected_hash="0" * 64)
                    self.assertEqual(target.read_bytes(), payload)
            source.write_bytes(payload)
            target.write_bytes(payload[:-1] + b"x")
            restored = fetch_gt(source.as_uri(), target, 2, expected_hash=original["sha256"])
            self.assertEqual(restored["sha256"], original["sha256"])
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(list(target.parent.glob(".gt-*")))


if __name__ == "__main__":
    unittest.main()
