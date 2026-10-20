"""Paired statistics, artifact verification, and actual local adapter inference."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from math_project.comparison import arithmetic_evidence, paired_stats, verify_inputs, verify_runtime
from math_project.data import arithmetic_set, sha256, write_json, write_jsonl
from scripts.find_results import find_result

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/"configs/comparison_v1.json").read_text())
HAS_TRAINING = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "accelerate"))


class ComparisonContractTests(unittest.TestCase):
    def test_equation_audit_accepts_only_complete_matching_single_equation(self):
        row = {"split": "arithmetic_test", "correct": False, "question": "What is 18 + 9?",
               "reference_answer": "27", "completion": " 18 + 9 = 27\n\n", "finish_reason": "next_question"}
        audit = arithmetic_evidence(row)
        self.assertTrue(audit["single_equation_correct"])
        self.assertTrue(audit["answer_evidence_correct"])
        self.assertFalse(audit["official_correct"])
        for changes in ({"completion": "18 + 9 = 28"}, {"completion": "18 + 8 = 27"},
                        {"completion": "Explanation 18 + 9 = 27"}, {"completion": "Final answer: wrong\n18 + 9 = 27"},
                        {"completion": "18 + 9 = 27\nFinal answer: 28"}, {"finish_reason": "length"},
                        {"question": "A story problem with answer 27"}):
            self.assertFalse(arithmetic_evidence({**row, **changes})["answer_evidence_correct"])
        self.assertTrue(arithmetic_evidence({**row, "correct": True})["official_correct"])

    def test_paired_counts_confidence_intervals_and_stratification(self):
        config = {**CONFIG, "paired_bootstrap_resamples": 1000}
        pairs = [({"correct": False, "category": "a"}, {"correct": True}),
                 ({"correct": True, "category": "b"}, {"correct": False})]
        result = paired_stats(pairs, config)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["regressed"], 1)
        self.assertEqual(result["accuracy_change"], 0)
        self.assertEqual(result["paired_bootstrap_95"], [-1, 1])
        self.assertEqual(paired_stats(pairs, config, stratified=True)["paired_bootstrap_95"], [0, 0])
        gain = paired_stats([pairs[0]]*10, config)
        self.assertEqual(gain["accuracy_change_percentage_points"], 100)
        self.assertEqual(gain["paired_bootstrap_95"], [1, 1])
        self.assertEqual(gain, paired_stats([pairs[0]]*10, config))
        self.assertEqual(paired_stats([], config)["paired_bootstrap_95"], [None, None])
        with self.assertRaisesRegex(ValueError, "runtime differs"):
            verify_runtime({"dtype": "torch.float16"}, {"dtype": "torch.float32"})

    def test_discovery_supports_zip_symlinks_and_refuses_smoke_or_ambiguous_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root/"notebook/full"
            training.mkdir(parents=True)
            write_json(training/"result_manifest.json", {"status": "full_training_complete", "files_sha256": {}})
            (root/"mount").symlink_to(root/"notebook", target_is_directory=True)
            self.assertEqual(find_result(root, "training"), training.resolve())
            archive = shutil.make_archive(str(root/"training"), "zip", root_dir=root/"notebook")
            self.assertEqual(json.loads((find_result(archive, "training")/"result_manifest.json").read_text())["status"], "full_training_complete")
            write_json(training/"result_manifest.json", {"status": "smoke_training_complete", "files_sha256": {}})
            with self.assertRaisesRegex(ValueError, "No completed"):
                find_result(root/"notebook", "training")
            write_json(training/"result_manifest.json", {"status": "full_training_complete", "files_sha256": {}})
            other = root/"notebook/other"
            other.mkdir()
            write_json(other/"result_manifest.json", {"status": "full_training_complete", "different": True})
            with self.assertRaisesRegex(ValueError, "Multiple"):
                find_result(root/"notebook", "training")


@unittest.skipUnless(HAS_TRAINING, "requires training environment")
class AdapterComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_training import TrainingTests
        from math_project.evaluation import load_bundle
        from math_project.inference import load_runner
        from math_project.pilot import load_protocol
        from scripts.train_lora import run_training
        from scripts.run_pilot import run_pilot
        from test_pilot import Runner
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        bundle, model, config = TrainingTests().make_training_fixture(cls.root)
        model.model.config.max_position_embeddings = 2048
        model.model.generation_config.eos_token_id = list(range(model.model.config.vocab_size))
        model.model.save_pretrained(cls.root/"tiny-model")
        bundle["config"].update(context_window_tokens=2048, few_shot_count=3)
        train = [json.loads(line) for line in (bundle["path"]/"train.jsonl").read_text().splitlines()]
        write_json(bundle["path"]/"demonstrations.json", train[:3])
        write_json(bundle["path"]/"config.resolved.json", bundle["config"])
        for split in ("gsm8k_test", "svamp_test"):
            rows = [{"id": f"{split}/{i}", "question": f"What is {i} + 4?", "answer_canonical": str(i+4)} for i in range(120)]
            write_jsonl(bundle["path"]/f"{split}.jsonl", rows)
            bundle["manifest"]["counts"][split] = len(rows)
        arithmetic = arithmetic_set(42, 12)
        write_jsonl(bundle["path"]/"arithmetic_test.jsonl", arithmetic)
        bundle["manifest"]["counts"]["arithmetic_test"] = len(arithmetic)
        bundle["manifest"]["files_sha256"] = {str(p.relative_to(bundle["path"])): sha256(p.read_bytes())
            for p in bundle["path"].rglob("*") if p.is_file() and p.name != "manifest.json"}
        write_json(bundle["path"]/"manifest.json", bundle["manifest"])
        cls.bundle = load_bundle(bundle["path"])
        base, runtime = load_runner(cls.bundle, "cpu")
        run_training(cls.bundle, base, runtime, config, cls.root/"smoke", "smoke")
        base, runtime = load_runner(cls.bundle, "cpu")
        run_training(cls.bundle, base, runtime, config, cls.root/"training", "full", cls.root/"smoke")
        cls.runtime = {**runtime, "model_eos_ids": base.eos_ids, "pad_token_id": base.tokenizer.pad_token_id,
                       "tokenizer_eos_id": base.tokenizer.eos_token_id, "stopping": "protocol_v2_per_sequence"}
        pilot, scoring = load_protocol(ROOT/"configs/pilot_v2.json")
        run_pilot(cls.bundle, Runner(), cls.runtime, cls.root/"baseline", pilot, scoring)
        cls.inputs = verify_inputs(cls.bundle, cls.root/"baseline", cls.root/"training")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_input_checks_detect_tampering_and_wrong_bundle(self):
        self.assertEqual(len(self.inputs["jobs"]), 300)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(self.root/"baseline", root/"baseline")
            path = root/"baseline/predictions.jsonl"
            before = path.read_bytes()
            path.write_bytes(before[:-5])
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_inputs(self.bundle, root/"baseline", self.root/"training")
            self.assertEqual(path.read_bytes(), before[:-5])  # Read-only inputs must not be repaired.
            changed = {**self.bundle, "fingerprint": "different"}
            with self.assertRaisesRegex(ValueError, "selection differs"):
                verify_inputs(changed, self.root/"baseline", self.root/"training")
            shutil.copytree(self.root/"training", root/"training")
            (root/"training/adapter/extra.bin").write_bytes(b"unverified")
            with self.assertRaisesRegex(ValueError, "Unverified extra"):
                verify_inputs(self.bundle, self.root/"baseline", root/"training")
            with patch("math_project.comparison.FROZEN_EVALUATOR_FILES", ["README.md"]):
                with self.assertRaisesRegex(ValueError, "evaluator code changed"):
                    verify_inputs(self.bundle, self.root/"baseline", self.root/"training")

    def test_partial_resume_reports_matched_questions_and_preserves_complete_results(self):
        from scripts.compare_pilot import run_comparison
        from test_pilot import Runner
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"run"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_comparison(self.inputs, Runner(fail_after=1), self.runtime, output, CONFIG)
            summary = json.loads((output/"summary.json").read_text())
            self.assertFalse(summary["execution_complete"])
            group = summary["groups"]["gsm8k_test"]
            self.assertEqual(group["baseline_on_paired_questions"]["count"], 4)
            self.assertEqual(group["tuned_on_paired_questions"]["count"], 4)
            self.assertFalse((output/"result_manifest.json").exists())
            path = output/"predictions.jsonl"
            first = path.read_bytes().split(b"\n")[0]+b"\n"
            path.write_bytes(first+b'{"partial":')
            runner = Runner()
            summary = run_comparison(self.inputs, runner, self.runtime, output, CONFIG)
            self.assertTrue(summary["execution_complete"])
            self.assertEqual(summary["predictions"], 300)
            self.assertEqual(summary["groups"]["gsm8k_test"]["paired"]["accuracy_change"], 0)
            audit = summary["supplementary_arithmetic"]
            self.assertEqual(audit["answer_evidence_improved"], 0)
            self.assertEqual(audit["answer_evidence_regressed"], 0)
            self.assertEqual(sum(audit[name] for name in ("answer_evidence_improved", "answer_evidence_regressed",
                "both_answer_evidence_correct", "both_answer_evidence_incorrect")), 100)
            self.assertEqual(len({json.loads(x)["key"] for x in path.read_bytes().splitlines()}), 300)
            self.assertEqual(path.read_bytes().split(b"\n")[0]+b"\n", first)
            before = (output/"result_manifest.json").read_bytes()
            calls = runner.calls
            run_comparison(self.inputs, runner, self.runtime, output, CONFIG)
            self.assertEqual(runner.calls, calls)
            self.assertEqual((output/"result_manifest.json").read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                run_comparison(self.inputs, runner, self.runtime, output, {**CONFIG, "bootstrap_seed": 43})
            (output/"report.html").write_text('changed')
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_comparison(self.inputs, runner, self.runtime, output, CONFIG)

    def test_real_adapter_cli_and_verification_only_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, str(ROOT/"scripts/compare_pilot.py"),
                "--bundle", str(self.bundle["path"]), "--baseline", str(self.root/"baseline"),
                "--training", str(self.root/"training"), "--output", str(root/"run"), "--device", "cpu"]
            check = subprocess.run(command+["--verify-only"], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stdout+check.stderr)
            self.assertTrue(json.loads(check.stdout)["inputs_verified"])
            self.assertFalse((root/"run").exists())
            complete = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(complete.returncode, 0, complete.stdout+complete.stderr)
            summary = json.loads((root/"run/summary.json").read_text())
            self.assertTrue(summary["execution_complete"])
            self.assertEqual(summary["predictions"], 300)
            identity = json.loads((root/"run/identity.json").read_text())
            self.assertEqual(identity["model_role"], "validation_selected_lora_adapter")
            self.assertFalse(identity["runtime"]["adapter_runtime"]["merged"])
            self.assertEqual(sum(g["tuned_on_paired_questions"]["generated_tokens"] for g in summary["groups"].values()), 300)
