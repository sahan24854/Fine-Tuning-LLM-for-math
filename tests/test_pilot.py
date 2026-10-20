"""Pilot selection, immutable resume, and reporting contracts."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from math_project.data import arithmetic_set, sha256, write_json, write_jsonl
from math_project.evaluation import load_bundle
from math_project.pilot import ARITHMETIC_STRATA, load_protocol, make_plan, save_plan
from scripts.run_pilot import run_pilot
from test_evaluation import fixture_bundle

ROOT = Path(__file__).resolve().parents[1]


def pilot_bundle(path):
    bundle = fixture_bundle(path)
    config = bundle["config"]
    config["few_shot_count"] = 3
    config["context_window_tokens"] = 2048
    demonstrations = [{**bundle["demonstrations"][0], "id": f"train/{i}"} for i in range(3)]
    write_json(path/"config.resolved.json", config)
    write_json(path/"demonstrations.json", demonstrations)
    write_jsonl(path/"train.jsonl", demonstrations)
    counts = {"train": 3}
    for split in ("validation", "gsm8k_test", "svamp_test"):
        rows = [{"id": f"{split}/{i}", "question": f"What is {i} + 4?", "answer_canonical": str(i+4)} for i in range(120)]
        write_jsonl(path/f"{split}.jsonl", rows)
        counts[split] = len(rows)
    rows = arithmetic_set(42, 12)
    write_jsonl(path/"arithmetic_test.jsonl", rows)
    counts["arithmetic_test"] = len(rows)
    manifest = bundle["manifest"]
    manifest["counts"] = counts
    manifest["files_sha256"] = {str(p.relative_to(path)): sha256(p.read_bytes())
        for p in path.rglob("*") if p.is_file() and p.name != "manifest.json"}
    write_json(path/"manifest.json", manifest)
    return load_bundle(path)


class Runner:
    device = "cpu"
    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after
    def generate(self, prompts):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("interrupted")
        return [{"completion": "Final answer: $4\n", "raw_completion": "Final answer: $4\n",
                 "finish_reason": "final_answer_line", "generated_tokens": 6, "prompt_tokens": 5,
                 "amortized_generation_seconds": .1} for _ in prompts]


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.pilot, self.scoring = load_protocol(ROOT/"configs/pilot_v2.json")

    def test_selection_is_300_balanced_unique_and_independent_of_source_order_or_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = pilot_bundle(Path(tmp)/"data")
            plan = make_plan(bundle, self.pilot, self.scoring)
            self.assertEqual(plan["prediction_count"], 300)
            jobs = plan["jobs"]
            self.assertEqual(Counter(j["split"] for j in jobs), {"gsm8k_test": 100, "svamp_test": 100, "arithmetic_test": 100})
            self.assertEqual(len({j["key"] for j in jobs}), 300)
            self.assertNotIn("validation", {j["split"] for j in jobs})
            self.assertEqual(Counter(j["category"] for j in jobs if j["split"] == "arithmetic_test"),
                             {f"{a}/{b}": 10 for a, b in ARITHMETIC_STRATA})
            for rows in bundle["splits"].values():
                rows.reverse()
            self.assertEqual(make_plan(bundle, self.pilot, self.scoring), plan)
            for rows in bundle["splits"].values():
                for row in rows:
                    row["answer_canonical"] = "-12345"
            other = make_plan(bundle, self.pilot, self.scoring)
            self.assertEqual([j["id"] for j in other["jobs"]], [j["id"] for j in jobs])
            self.assertNotEqual(other["comparison_contract_sha256"], plan["comparison_contract_sha256"])

    def test_scope_hash_and_count_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"protocol.json"
            for changes in ({"questions_per_dataset": 101}, {"mode": "zero_shot"},
                            {"scoring_protocol_file_sha256": "0"*64}, {"scope": "full"}):
                write_json(path, {**self.pilot, **changes})
                with self.assertRaises(ValueError):
                    load_protocol(path)
            bundle = pilot_bundle(Path(tmp)/"data")
            with self.assertRaisesRegex(ValueError, "ten strata"):
                make_plan(bundle, {**self.pilot, "questions_per_dataset": 99}, self.scoring)
            del bundle["splits"]["gsm8k_test"][30:]
            with self.assertRaisesRegex(ValueError, "Not enough"):
                make_plan(bundle, self.pilot, self.scoring)

    def test_interruption_partial_line_resume_and_complete_archive_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = pilot_bundle(root/"data")
            pilot = {**self.pilot, "questions_per_dataset": 10}
            output = root/"run"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_pilot(bundle, Runner(fail_after=1), {}, output, pilot, self.scoring)
            summary = json.loads((output/"summary.json").read_text())
            self.assertFalse(summary["execution_complete"])
            self.assertEqual(summary["predictions"], 4)
            self.assertFalse((output/"result_manifest.json").exists())
            # Keep one flushed record from a batch and an incomplete next write.
            path = output/"predictions.jsonl"
            first_line = path.read_bytes().split(b"\n")[0]+b"\n"
            path.write_bytes(first_line+b'{"completion": "partial')
            runner = Runner()
            summary = run_pilot(bundle, runner, {}, output, pilot, self.scoring)
            self.assertTrue(summary["execution_complete"])
            self.assertEqual(summary["predictions"], 30)
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            self.assertEqual(len({r["key"] for r in rows}), 30)
            self.assertEqual(path.read_bytes().split(b"\n")[0]+b"\n", first_line)
            self.assertEqual(summary["groups"]["gsm8k_test"]["parse_rate"], 1)
            self.assertEqual(summary["groups"]["gsm8k_test"]["strict_numeric_format_rate"], 0)
            self.assertIn("accuracy_wilson_95", summary["groups"]["gsm8k_test"])
            before = (output/"result_manifest.json").read_bytes()
            calls = runner.calls
            run_pilot(bundle, runner, {}, output, pilot, self.scoring)
            self.assertEqual(runner.calls, calls)
            self.assertEqual((output/"result_manifest.json").read_bytes(), before)
            (output/"report.html").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_pilot(bundle, runner, {}, output, pilot, self.scoring)

    def test_resume_rejects_changed_plan_runtime_scores_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = pilot_bundle(root/"data")
            pilot = {**self.pilot, "questions_per_dataset": 10}
            output = root/"run"
            with self.assertRaises(RuntimeError):
                run_pilot(bundle, Runner(fail_after=1), {}, output, pilot, self.scoring)
            with self.assertRaisesRegex(ValueError, "selection changed"):
                run_pilot(bundle, Runner(), {}, output, {**pilot, "seed": 43}, self.scoring)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                run_pilot(bundle, Runner(), {"changed": True}, output, pilot, self.scoring)
            path = output/"predictions.jsonl"
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            for field, value in (("correct", True), ("split", "validation"), ("question", "changed")):
                changed = copy.deepcopy(rows)
                if changed[0][field] == value:
                    value = False
                changed[0][field] = value
                write_jsonl(path, changed)
                with self.assertRaisesRegex(ValueError, "score or metadata"):
                    run_pilot(bundle, Runner(), {}, output, pilot, self.scoring)

    def test_cli_plan_only_never_loads_missing_model_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = pilot_bundle(root/"data")
            result = subprocess.run([sys.executable, str(ROOT/"scripts/run_pilot.py"),
                "--bundle", str(bundle["path"]), "--output", str(root/"plan"), "--plan-only"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(json.loads(result.stdout)["predictions"], 300)
            self.assertTrue((root/"plan/selection.json").exists())
            self.assertFalse((root/"plan/predictions.jsonl").exists())
            saved = json.loads((root/"plan/selection.json").read_text())
            with self.assertRaisesRegex(ValueError, "selection changed"):
                save_plan(root/"plan", {**saved, "prediction_count": 1})


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "requires baseline environment")
class PilotInferenceTests(unittest.TestCase):
    def test_real_cli_loads_model_and_saves_complete_pilot(self):
        from test_inference import InferenceTests
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = pilot_bundle(root/"data")
            bundle["config"]["context_window_tokens"] = 128
            base = InferenceTests().make_runner(bundle)
            base.model.config.max_position_embeddings = 2048
            # Every generated token is EOS: exercise actual loading/generation
            # without spending the 512-token safety cap on random-model text.
            base.model.generation_config.eos_token_id = list(range(base.model.config.vocab_size))
            base.model.save_pretrained(root/"tiny-model")
            base.tokenizer.save_pretrained(bundle["path"]/"tokenizer")
            bundle["config"]["context_window_tokens"] = 2048
            bundle["config"]["model"]["id"] = str(root/"tiny-model")
            write_json(bundle["path"]/"config.resolved.json", bundle["config"])
            manifest = bundle["manifest"]
            manifest["model"] = bundle["config"]["model"]
            manifest["files_sha256"] = {str(p.relative_to(bundle["path"])): sha256(p.read_bytes())
                for p in bundle["path"].rglob("*") if p.is_file() and p.name != "manifest.json"}
            write_json(bundle["path"]/"manifest.json", manifest)
            pilot, _ = load_protocol(ROOT/"configs/pilot_v2.json")
            write_json(root/"pilot.json", {**pilot, "questions_per_dataset": 10})
            result = subprocess.run([sys.executable, str(ROOT/"scripts/run_pilot.py"),
                "--bundle", str(bundle["path"]), "--output", str(root/"run"),
                "--protocol", str(root/"pilot.json"), "--device", "cpu"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((root/"run/summary.json").read_text())
            self.assertTrue(summary["execution_complete"])
            self.assertEqual(summary["predictions"], 30)
            self.assertEqual(sum(g["generated_tokens"] for g in summary["groups"].values()), 30)
            self.assertEqual(summary["model_role"], "untouched_base")
