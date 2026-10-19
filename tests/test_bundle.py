import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.verify_bundle import verify


class BundleTests(unittest.TestCase):
    def test_detects_modified_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "train.jsonl").write_bytes(b"data")
            manifest = {"status": "data_ready_not_model_evaluated", "files_sha256": {
                "train.jsonl": hashlib.sha256(b"data").hexdigest()}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify(root), manifest)
            (root / "train.jsonl").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                verify(root)

    def test_rejects_path_outside_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"status": "data_ready_not_model_evaluated",
                                                            "files_sha256": {"../outside": "x"}}))
            with self.assertRaisesRegex(ValueError, "escapes"):
                verify(root)
