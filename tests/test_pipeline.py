"""Offline orchestration test. The tokenizer stub cannot validate SmolLM token lengths."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from math_project.data import sha256
from scripts.prepare_data import main
from scripts.verify_bundle import verify


class FixtureTokenizer:
    eos_token_id = 0
    model_max_length = 2048

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def save_pretrained(self, path):
        path.mkdir()
        (path / "test_fixture_tokenizer.json").write_text('{"fixture_only": true}')


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_bundle_repeatability_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            train = [dict(question=f"What is {i} plus one?", answer=f"Add 1 to {i}.\n#### {i + 1}") for i in range(30)]
            test = [dict(question="What is 1000 plus one?", answer="Add one.\n#### 1001")]
            svamp = [dict(ID="fixture", Body="There are 7 eggs.", Question="How many eggs?", Answer=7, Type="Addition")]
            sources = {}
            for name, data in [("gsm8k_train", "\n".join(map(json.dumps, train)).encode()),
                               ("gsm8k_test", "\n".join(map(json.dumps, test)).encode()),
                               ("svamp_test", json.dumps(svamp).encode())]:
                digest = sha256(data)
                (cache / digest).write_bytes(data)
                sources[name] = {"url": "https://example.invalid/fixture", "sha256": digest}
            config = {"model": {"id": "fixture/model", "revision": "main"}, "seed": 42,
                      "sources": sources, "validation_fraction": .1, "arithmetic_per_category": 1,
                      "max_training_tokens": 256, "few_shot_count": 3, "demonstration_max_tokens": 80,
                      "context_window_tokens": 2048, "generation": {"max_new_tokens": 512}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            fake_hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha="a"*40)))
            fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: FixtureTokenizer()))
            manifests = []
            with patch.dict("sys.modules", {"huggingface_hub": fake_hub, "transformers": fake_transformers}):
                for name in ["first", "second"]:
                    argv = ["prepare_data.py", "--config", str(config_path), "--cache", str(cache), "--output", str(root / name)]
                    with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                        main()
                    manifests.append(verify(root / name))
                with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 2)
            self.assertEqual(manifests[0]["model"]["revision"], "a"*40)
            self.assertEqual(manifests[0]["files_sha256"], manifests[1]["files_sha256"])
            self.assertEqual(manifests[0]["counts"]["train"], 27)
            self.assertEqual(manifests[0]["counts"]["validation"], 3)


if __name__ == "__main__":
    unittest.main()
