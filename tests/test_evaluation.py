import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from math_project.data import sha256, write_json, write_jsonl
from math_project.evaluation import load_bundle, make_jobs, read_predictions, summarize, wilson_interval
from scripts.find_bundle import find_bundle, unpack
from scripts.run_baseline import evaluate


def fixture_bundle(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = {"model": {"id": "fixture/model", "revision": "a" * 40}, "seed": 42,
              "few_shot_count": 1, "context_window_tokens": 128,
              "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 8}}
    demonstration = {"id": "train/0", "question": "What is 1 + 1?", "solution": "1 + 1 = 2.", "answer": "2", "answer_canonical": "2"}
    write_json(root / "config.resolved.json", config)
    write_json(root / "demonstrations.json", [demonstration])
    write_jsonl(root / "train.jsonl", [demonstration])
    counts = {"train": 1}
    for split in ["validation", "gsm8k_test", "svamp_test", "arithmetic_test"]:
        rows = [{"id": f"{split}/{i}", "question": f"What is {i} + 4?", "answer_canonical": str(i+4),
                 "number_range": "small", "operation": "add"} for i in range(2)]
        write_jsonl(root / f"{split}.jsonl", rows)
        counts[split] = len(rows)
    (root / "tokenizer").mkdir()
    write_json(root / "tokenizer/fixture.json", {"test": True})
    manifest = {"status": "data_ready_not_model_evaluated", "model": config["model"], "counts": counts,
                "files_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()) for p in root.rglob("*") if p.is_file()}}
    write_json(root / "manifest.json", manifest)
    return load_bundle(root)


class FixedRunner:
    device = "cpu"

    def __init__(self, fail_after=None):
        self.calls = 0
        self.fail_after = fail_after

    def generate(self, prompts, max_new_tokens=None):
        if max_new_tokens is None:
            self.calls += 1
            if self.fail_after is not None and self.calls > self.fail_after:
                raise RuntimeError("simulated interruption")
        return [{"completion": "Calculation.\nFinal answer: 4", "generated_tokens": 5,
                 "prompt_tokens": 10, "finish_reason": "eos", "amortized_generation_seconds": .1}
                for _ in prompts]


class EvaluationTests(unittest.TestCase):
    def test_smoke_only_uses_validation_and_full_only_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(Path(tmp) / "data")
            smoke = make_jobs(bundle, "smoke", 1)
            self.assertEqual(len(smoke), 2)
            self.assertEqual({j["split"] for j in smoke}, {"validation"})
            full = make_jobs(bundle, "full")
            self.assertEqual(len(full), 12)
            self.assertNotIn("validation", {j["split"] for j in full})
            for job in full:
                self.assertTrue(job["prompt"].endswith("\nSolution:"))

    def test_resume_after_interruption_without_duplicate_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(Path(tmp) / "data")
            output = Path(tmp) / "run"
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                evaluate(bundle, FixedRunner(fail_after=1), {"fixture": True}, output, "full", batch_size=2)
            partial = json.loads((output / "summary.json").read_text())
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["completed_predictions"], 2)
            self.assertFalse((output / "result_manifest.json").exists())
            result = evaluate(bundle, FixedRunner(), {"fixture": True}, output, "full", batch_size=2)
            self.assertTrue(result["complete"])
            self.assertEqual(result["completed_predictions"], 12)
            self.assertEqual(result["groups"]["gsm8k_test/zero_shot"]["accuracy"], .5)
            # Completed runs perform no inference on rerun.
            original_manifest = (output / "result_manifest.json").read_bytes()
            no_calls = FixedRunner(fail_after=0)
            evaluate(bundle, no_calls, {"fixture": True}, output, "full", batch_size=2)
            self.assertEqual(no_calls.calls, 0)
            self.assertEqual((output / "result_manifest.json").read_bytes(), original_manifest)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                evaluate(bundle, FixedRunner(), {"fixture": True}, output, "full", batch_size=1)

    def test_corrupt_final_fragment_recovered_but_complete_bad_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            path.write_bytes(b'{"ok":1}\n{"partial":')
            self.assertEqual(read_predictions(path), [{"ok": 1}])
            self.assertEqual(path.read_bytes(), b'{"ok":1}\n')
            path.write_bytes(b'not-json\n')
            with self.assertRaises(ValueError):
                read_predictions(path)

    def test_invalid_and_length_limited_answers_remain_in_denominator(self):
        base = {"amortized_generation_seconds": .2, "generated_tokens": 8}
        records = [{**base, "correct": True, "extracted_answer": "4", "finish_reason": "length"},
                   {**base, "correct": False, "extracted_answer": None, "finish_reason": "eos"}]
        summary = summarize(records)
        self.assertEqual(summary["accuracy"], .5)
        self.assertEqual(summary["invalid_answers"], 1)
        self.assertEqual(summary["length_limited_answers"], 1)
        self.assertEqual(summary["generated_tokens_per_second"], 40)
        self.assertEqual(wilson_interval(0, 0), [None, None])
        low, high = wilson_interval(5, 10)
        self.assertTrue(low < .5 < high)

    def test_bundle_selection_and_zip_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_bundle(root / "first")
            self.assertEqual(find_bundle(root), root / "first")
            fixture_bundle(root / "second")
            with self.assertRaisesRegex(ValueError, "Multiple"):
                find_bundle(root)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as file:
                file.writestr("../../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                unpack(archive)

    def test_modified_data_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_bundle(tmp)
            (Path(tmp) / "gsm8k_test.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                load_bundle(tmp)

    def test_unicode_line_separator_in_verified_bundle_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(tmp)
            path = Path(tmp) / "validation.jsonl"
            rows = bundle["splits"]["validation"]
            rows[0]["question"] = "A question\u2028with a Unicode separator\u2029and another\u0085character."
            write_jsonl(path, rows)
            manifest = bundle["manifest"]
            manifest["files_sha256"]["validation.jsonl"] = sha256(path.read_bytes())
            write_json(Path(tmp) / "manifest.json", manifest)
            loaded = load_bundle(tmp)
            self.assertEqual(loaded["splits"]["validation"], rows)

    def test_unicode_in_saved_predictions_does_not_break_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = [{"completion": "Solution\u2028continues\u2029here\u0085\nFinal answer: 4"}]
            path = Path(tmp) / "predictions.jsonl"
            write_jsonl(path, records)
            self.assertEqual(read_predictions(path), records)
