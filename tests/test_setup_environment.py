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


    def test_training_profile_installs_pinned_torch_before_peft_and_checks_dependencies(self):
        from scripts.setup_environment import ROOT
        with tempfile.TemporaryDirectory() as directory:
            with patch("scripts.setup_environment.subprocess.run") as run, patch(
                    "scripts.setup_environment.download_pip", return_value=Path(directory)/"pip.whl"):
                setup(Path(directory)/"env", profile="training", torch_variant="cpu")
            commands = [call.args[0] for call in run.call_args_list]
            torch_index = next(i for i, command in enumerate(commands) if "torch==2.7.1+cpu" in command)
            peft_index = next(i for i, command in enumerate(commands) if str(ROOT/"requirements-training.txt") in command)
            self.assertLess(torch_index, peft_index)
            self.assertIn("https://download.pytorch.org/whl/cpu", commands[torch_index])
            self.assertEqual(commands[-1][-1], "check")
            self.assertIn("--without-pip", commands[0])
