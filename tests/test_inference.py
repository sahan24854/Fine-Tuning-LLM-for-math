"""Real CPU generation with a tiny randomly initialized Llama; no network/downloads."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_evaluation import fixture_bundle
from scripts.run_baseline import evaluate


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "requires baseline environment")
class InferenceTests(unittest.TestCase):
    def make_runner(self, bundle):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from math_project.inference import ModelRunner
        torch.set_num_threads(2)
        torch.manual_seed(42)
        words = ["[PAD]", "[UNK]", "[EOS]", "What", "is", "1", "2", "4", "+", "?", "Final", "answer", ":", "Solution", "Question", "."]
        backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(words), hidden_size=32, intermediate_size=64,
                                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                max_position_embeddings=128, pad_token_id=0, eos_token_id=2))
        return ModelRunner(model, tokenizer, bundle["config"], "cpu")

    def test_real_generation_and_complete_prediction_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(Path(tmp) / "data")
            runner = self.make_runner(bundle)
            result = runner.generate(["What is 1 + 1?", "What is 4?"])
            self.assertEqual(len(result), 2)
            self.assertTrue(all(0 < r["generated_tokens"] <= 8 for r in result))
            self.assertTrue(all(r["prompt_tokens"] > 0 for r in result))
            result = evaluate(bundle, runner, {"fixture": "tiny_random_llama_cpu"}, Path(tmp) / "run", "full", 2)
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_predictions"], 12)

    def test_decode_only_new_tokens_and_stop_at_first_eos_per_row(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.make_runner(fixture_bundle(Path(tmp) / "data"))
            def generate(input_ids, attention_mask, generation_config):
                suffix = torch.tensor([[7, 2, 0], [6, 7, 2]])
                return torch.cat([input_ids, suffix], dim=1)
            runner.model.generate = generate
            outputs = runner.generate(["Question: 1", "Question: 1 + 1"])
            self.assertEqual([r["completion"] for r in outputs], ["4", "2 4"])
            self.assertEqual([r["generated_tokens"] for r in outputs], [2, 3])
            self.assertEqual([r["finish_reason"] for r in outputs], ["eos", "eos"])

    def test_real_model_smoke_and_guard_against_unhealthy_full_run(self):
        from math_project.data import sha256, write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = fixture_bundle(root / "data")
            runner = self.make_runner(bundle)
            model_path = root / "tiny-model"
            runner.model.save_pretrained(model_path)
            runner.tokenizer.save_pretrained(bundle["path"] / "tokenizer")
            config = bundle["config"]
            config["model"]["id"] = str(model_path)
            write_json(bundle["path"] / "config.resolved.json", config)
            manifest = bundle["manifest"]
            manifest["model"] = config["model"]
            manifest["files_sha256"] = {str(p.relative_to(bundle["path"])): sha256(p.read_bytes())
                for p in bundle["path"].rglob("*") if p.is_file() and p.name != "manifest.json"}
            write_json(bundle["path"] / "manifest.json", manifest)
            script = Path(__file__).resolve().parents[1] / "scripts/run_baseline.py"
            for stage in ["smoke", "full"]:
                command = [sys.executable, str(script), "--bundle", str(bundle["path"]), "--output", str(root / stage),
                           "--stage", stage, "--device", "cpu", "--batch-size", "2", "--smoke-count", "1"]
                if stage == "full":
                    command += ["--smoke-output", str(root / "smoke")]
                result = subprocess.run(command, text=True, capture_output=True)
                if stage == "smoke":
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("Full benchmark blocked by validation smoke health checks", result.stderr)
            summary = json.loads((root / "smoke/summary.json").read_text())
            self.assertTrue(summary["complete"])
            self.assertFalse((root / "full").exists())
