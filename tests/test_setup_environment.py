import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.setup_environment import download_pip, setup


class SetupTests(unittest.TestCase):
    def test_unverified_bootstrap_is_not_written_or_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.setup_environment.urllib.request.urlopen", return_value=io.BytesIO(b"modified wheel")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    download_pip(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_environment_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing-file"
            existing.write_text("preserve me")
            with self.assertRaisesRegex(ValueError, "not overwritten"):
                setup(directory)
            self.assertEqual(existing.read_text(), "preserve me")
