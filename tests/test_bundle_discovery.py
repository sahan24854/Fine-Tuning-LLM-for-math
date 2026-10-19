import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from scripts.find_bundle import find_bundle


def manifest(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({"status": "data_ready_not_model_evaluated"}))


class DiscoveryTests(unittest.TestCase):
    def test_linked_mounts_and_cycles_are_followed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "real-output" / "data-protocol-run"
            manifest(data)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "notebook-output").symlink_to(data.parent, target_is_directory=True)
            (inputs / "alias").symlink_to(data.parent, target_is_directory=True)
            (data.parent / "cycle").symlink_to(inputs, target_is_directory=True)
            self.assertEqual(find_bundle(inputs), data.resolve())
            script = Path(__file__).resolve().parents[1] / "scripts/find_bundle.py"
            output = subprocess.check_output([sys.executable, str(script), str(inputs)], text=True).strip()
            self.assertEqual(Path(output), data.resolve())

    def test_renamed_zip_in_linked_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "real-output").mkdir()
            archive = root / "real-output" / "uploaded-results.zip"
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("nested/manifest.json", json.dumps({"status": "data_ready_not_model_evaluated"}))
            (root / "input").mkdir()
            (root / "input/attached").symlink_to(archive.parent, target_is_directory=True)
            discovered = find_bundle(root / "input")
            self.assertTrue((discovered / "manifest.json").is_file())

    def test_failure_reports_visible_paths_and_rejected_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "only-notebook.ipynb").write_text("{}")
            (root / "manifest.json").write_text('{"status": "unfinished"}')
            with self.assertRaises(ValueError) as error:
                find_bundle(root)
            self.assertIn("only-notebook.ipynb", str(error.exception))
            self.assertIn("Rejected manifests", str(error.exception))

    def test_broken_links_are_reported_instead_of_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "missing-output").symlink_to(root / "not-there", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Broken input link"):
                find_bundle(root)
