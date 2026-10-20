"""Loss boundaries and real tiny-model adapter training, without remote weights."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from math_project.data import completion, prompt, sha256, write_json, write_jsonl
from math_project.training import ResponseCollator, encode_rows, response_loss

ROOT = Path(__file__).resolve().parents[1]
HAS_TRAINING = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "accelerate"))


@unittest.skipUnless(HAS_TRAINING, "requires training environment")
class TrainingTests(unittest.TestCase):
    def test_segmented_encoding_eos_and_padding_never_supervise_prompt(self):
        class Tokenizer:
            eos_token_id = 99
            def encode(self, text, add_special_tokens=False):
                # A context-sensitive tokenizer would differ if concatenated text
                # were encoded once. Respect separate audited segments instead.
                return [len(text), 2] if text.startswith("Solve") else [3, 4, 5]
        row = {"id": "train/0", "question": "What is 1 + 1?", "solution": "1 + 1 = 2.", "answer": "2"}
        encoded = encode_rows([row], Tokenizer(), 20)
        self.assertEqual(encoded[0]["labels"], [-100, -100, 3, 4, 5, 99])
        short = {"input_ids": [8, 7, 6], "attention_mask": [1, 1, 1], "labels": [-100, 7, 6]}
        batch = ResponseCollator(99)(encoded+[short])
        self.assertEqual(batch["labels"][1].tolist(), [-100, 7, 6, -100, -100, -100])
        self.assertEqual(batch["attention_mask"][1].tolist(), [1, 1, 1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            encode_rows([row], Tokenizer(), 5)
        with self.assertRaisesRegex(ValueError, "Stored training text changed"):
            encode_rows([row], Tokenizer(), 20, check_stored=True)

    def test_loss_masks_prompt_and_padding_and_accumulates_by_response_tokens(self):
        import torch
        labels = torch.tensor([[-100, -100, 2, 3, -100], [-100, 1, 2, -100, -100]])
        logits = torch.randn(2, 5, 4, requires_grad=True)
        loss = response_loss(SimpleNamespace(logits=logits), labels)
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1), ignore_index=-100)
        self.assertTrue(torch.allclose(loss, expected))
        split = sum(response_loss(SimpleNamespace(logits=logits[i:i+1]), labels[i:i+1], 4) for i in range(2))
        self.assertTrue(torch.allclose(split, loss))
        loss.backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 3:].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, 1:3].abs().sum().item(), 0)

    def make_training_fixture(self, root):
        from math_project.evaluation import load_bundle
        from test_evaluation import fixture_bundle
        from test_inference import InferenceTests
        bundle = fixture_bundle(root/"data")
        base = InferenceTests().make_runner(bundle)
        config = bundle["config"]
        config["max_training_tokens"] = 128
        train = []
        for i in range(8):
            row = {"id": f"train/{i}", "question": "What is 1 + 1?", "solution": "1 + 1 = 2.", "answer": "2", "answer_canonical": "2"}
            row.update(prompt=prompt(row["question"]), completion=completion(row))
            row["prompt_token_count"] = len(base.tokenizer.encode(row["prompt"], add_special_tokens=False))
            row["completion_token_count"] = len(base.tokenizer.encode(row["completion"], add_special_tokens=False))+1
            train.append(row)
        write_jsonl(bundle["path"]/"train.jsonl", train)
        write_json(bundle["path"]/"demonstrations.json", train[:1])
        validation = [{"id": f"validation/{i}", "question": "What is 1 + 1?", "solution": "1 + 1 = 2.", "answer": "2", "answer_canonical": "2"} for i in range(2)]
        write_jsonl(bundle["path"]/"validation.jsonl", validation)
        base.model.save_pretrained(root/"tiny-model")
        base.tokenizer.save_pretrained(bundle["path"]/"tokenizer")
        config["model"]["id"] = str(root/"tiny-model")
        write_json(bundle["path"]/"config.resolved.json", config)
        manifest = bundle["manifest"]
        manifest["model"] = config["model"]
        manifest["counts"]["train"] = 8
        manifest["files_sha256"] = {str(p.relative_to(bundle["path"])): sha256(p.read_bytes())
            for p in bundle["path"].rglob("*") if p.is_file() and p.name != "manifest.json"}
        write_json(bundle["path"]/"manifest.json", manifest)
        bundle = load_bundle(bundle["path"])
        training = json.loads((ROOT/"configs/lora_training.json").read_text())
        training.update(epochs=1, micro_batch_size=2, gradient_accumulation_steps=2,
                        eval_batch_size=2, max_sequence_tokens=128, checkpoint_every_updates=1)
        training["smoke"] = {"train_rows": 8, "validation_rows": 2, "max_updates": 2}
        training["lora"].update(r=2, lora_alpha=4, lora_dropout=0.)
        return bundle, base, training

    def test_real_training_freezes_base_selects_validation_and_reloads_adapter(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from math_project.training import datasets_for_training
        from scripts.train_lora import run_training, verify_files, verify_smoke
        from math_project.inference import load_runner
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, base, config = self.make_training_fixture(root)
            frozen = {name: value.detach().clone() for name, value in base.model.named_parameters()}
            smoke = run_training(bundle, base, {"fixture": "cpu"}, config, root/"smoke", "smoke")
            self.assertTrue(smoke["health_passed"])
            self.assertTrue(smoke["adapter_parameters_changed"])
            self.assertEqual(smoke["optimizer_updates"], 2)
            # PEFT prefixes the original names but leaves all base weights frozen.
            current = base.model.state_dict()
            for name, value in frozen.items():
                actual = name.replace(".weight", ".base_layer.weight") if any(f".{part}." in name for part in config["lora"]["target_modules"]) else name
                self.assertTrue(torch.equal(current[actual], value), name)
            verify_smoke(root/"smoke", bundle, config)
            fresh, _ = load_runner(bundle, "cpu")
            full = run_training(bundle, fresh, {"fixture": "cpu"}, config, root/"full", "full", root/"smoke")
            self.assertEqual(full["optimizer_updates"], 2)
            self.assertEqual(full["train_rows"], 8)
            self.assertAlmostEqual(full["selected_validation_response_token_nll"], full["best_checkpoint_validation_loss"], places=5)
            model = AutoModelForCausalLM.from_pretrained(root/"tiny-model")
            adapter = PeftModel.from_pretrained(model, root/"full/adapter")
            data, _, _ = datasets_for_training(bundle, base.tokenizer, config, "smoke")
            batch = ResponseCollator(base.tokenizer.pad_token_id)(data[:1])
            with torch.inference_mode():
                self.assertTrue(torch.isfinite(adapter(**batch).loss))
            manifest_before = (root/"full/result_manifest.json").read_bytes()
            again, _ = load_runner(bundle, "cpu")
            self.assertEqual(run_training(bundle, again, {"fixture": "cpu"}, config, root/"full", "full", root/"smoke"), full)
            self.assertEqual((root/"full/result_manifest.json").read_bytes(), manifest_before)
            verify_files(root/"full", "result_manifest.json")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                run_training(bundle, again, {"fixture": "changed"}, config, root/"smoke", "smoke")
            with self.assertRaisesRegex(ValueError, "successful smoke"):
                verify_smoke(root/"smoke", bundle, {**config, "seed": 43})
            (root/"smoke/adapter/adapter_config.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_smoke(root/"smoke", bundle, config)

    def test_real_checkpoint_resume_restores_training_and_matches_uninterrupted_adapter(self):
        import torch
        from safetensors.torch import load_file
        from transformers import TrainerCallback
        from unittest.mock import patch
        from math_project.inference import load_runner
        from math_project.training import trainer_class
        from scripts.train_lora import run_training
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, _, config = self.make_training_fixture(root)
            original_class = trainer_class()

            class Interrupt(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    raise RuntimeError("simulated checkpoint interruption")

            class InterruptingTrainer(original_class):
                def __init__(self, **kwargs):
                    kwargs["callbacks"].append(Interrupt())
                    super().__init__(**kwargs)

            first, _ = load_runner(bundle, "cpu")
            with patch("scripts.train_lora.trainer_class", return_value=InterruptingTrainer):
                with self.assertRaisesRegex(RuntimeError, "checkpoint interruption"):
                    run_training(bundle, first, {"fixture": "cpu"}, config, root/"resumed", "smoke")
            self.assertTrue((root/"resumed/checkpoints/checkpoint-1/checkpoint_manifest.json").exists())
            self.assertFalse((root/"resumed/result_manifest.json").exists())
            fresh, _ = load_runner(bundle, "cpu")
            resumed = run_training(bundle, fresh, {"fixture": "cpu"}, config, root/"resumed", "smoke")
            other, _ = load_runner(bundle, "cpu")
            complete = run_training(bundle, other, {"fixture": "cpu"}, config, root/"complete", "smoke")
            self.assertEqual(resumed["optimizer_updates"], complete["optimizer_updates"])
            self.assertEqual(resumed["selected_validation_response_token_nll"], complete["selected_validation_response_token_nll"])
            lhs = load_file(root/"resumed/adapter/adapter_model.safetensors")
            rhs = load_file(root/"complete/adapter/adapter_model.safetensors")
            self.assertEqual(set(lhs), set(rhs))
            for key in lhs:
                self.assertTrue(torch.equal(lhs[key], rhs[key]), key)

    def test_cli_smoke_and_audit_use_only_training_and_validation(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, _, config = self.make_training_fixture(root)
            write_json(root/"training.json", config)
            command = [sys.executable, str(ROOT/"scripts/train_lora.py"), "--bundle", str(bundle["path"]),
                       "--config", str(root/"training.json"), "--stage", "smoke", "--device", "cpu"]
            audit = subprocess.run(command+["--output", str(root/"audit"), "--audit-only"], capture_output=True, text=True)
            self.assertEqual(audit.returncode, 0, audit.stdout+audit.stderr)
            data = json.loads((root/"audit/data_audit.json").read_text())
            self.assertEqual(data["train_rows"], 8)
            self.assertEqual(data["validation_rows"], 2)
            self.assertTrue(all(key.startswith("train/") for key in data["train_ids"]))
            self.assertTrue(all(key.startswith("validation/") for key in data["validation_ids"]))
            trained = subprocess.run(command+["--output", str(root/"smoke")], capture_output=True, text=True)
            self.assertEqual(trained.returncode, 0, trained.stdout+trained.stderr)
            summary = json.loads((root/"smoke/summary.json").read_text())
            self.assertTrue(summary["health_passed"])
            self.assertEqual(summary["optimizer_updates"], 2)

    def test_validation_loss_is_token_weighted_and_independent_of_eval_batch_size(self):
        import torch
        from transformers import TrainingArguments
        from math_project.training import trainer_class
        from test_evaluation import fixture_bundle
        from test_inference import InferenceTests
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = InferenceTests().make_runner(fixture_bundle(root/"data"))
            rows = [
                {"input_ids": [3, 5, 7, 2], "attention_mask": [1]*4, "labels": [-100, -100, 7, 2]},
                {"input_ids": [3, 5, 6, 7, 4, 2], "attention_mask": [1]*6, "labels": [-100, -100, 6, 7, 4, 2]}]
            model = base.model.eval()
            collator = ResponseCollator(base.tokenizer.pad_token_id)
            totals = []
            with torch.inference_mode():
                for row in rows:
                    batch = collator([row])
                    labels = batch.pop("labels")
                    totals.append(float(response_loss(model(**batch), labels, 1)))
            expected = sum(totals)/6
            for size in (1, 2):
                args = TrainingArguments(output_dir=str(root/f"batch{size}"), use_cpu=True,
                    per_device_eval_batch_size=size, report_to="none", remove_unused_columns=False,
                    disable_tqdm=True)
                trainer = trainer_class()(model=model, args=args, eval_dataset=rows,
                    data_collator=collator, compute_loss_func=response_loss)
                metrics = trainer.evaluate()
                self.assertEqual(metrics["eval_response_tokens"], 6)
                self.assertAlmostEqual(metrics["eval_loss"], expected, places=6)
