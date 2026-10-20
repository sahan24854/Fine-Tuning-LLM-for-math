import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from math_project.protocol_v2 import boundary, normalize_label, parse_target, score_target, validate_protocol
from scripts.run_protocol_v2 import jobs_for_validation, run_diagnostic
from test_evaluation import fixture_bundle

ROOT = Path(__file__).resolve().parents[1]


def protocol_fixture():
    return json.loads((ROOT / "configs/protocol_v2_diagnostic.json").read_text())


class ProtocolV2Tests(unittest.TestCase):
    def test_wait_for_line_ending_instead_of_partial_digits(self):
        self.assertIsNone(boundary("Final answer: 5"))
        self.assertIsNone(boundary("Final answer: 550"))
        self.assertEqual(boundary("Final answer: 550\n")["reason"], "final_answer_line")

    def test_completed_answer_precedes_later_problem_and_labels(self):
        text = "200 + 250 + 100 = 550\nFinal answer: 550\n\nQuestion: Another problem\nFinal answer: 700 dollars"
        score = score_target(text, "length", "550", protocol_fixture())
        self.assertEqual(score["answer"], "550")
        self.assertTrue(score["correct"])
        self.assertIsNone(score["v1_strict_answer_on_same_text"])

    def test_never_extract_answer_from_next_question(self):
        text = "No target answer. Question: New problem\nFinal answer: 550\n"
        self.assertEqual(boundary(text)["reason"], "next_question")
        self.assertIsNone(parse_target(text, "length", protocol_fixture())["answer"])

    def test_first_malformed_label_is_not_replaced_by_a_later_correct_number(self):
        text = "Final answer: maybe 4 or 5\nFinal answer: 39\n"
        self.assertIsNone(score_target(text, "final_answer_line", "39", protocol_fixture())["answer"])

    def test_eos_and_next_question_finalize_a_line_but_cap_does_not(self):
        protocol = protocol_fixture()
        self.assertEqual(parse_target("Final answer: 550", "length", protocol)["parse_status"], "incomplete_final_line")
        for reason in ["eos", "next_question"]:
            self.assertEqual(parse_target("Final answer: 550", reason, protocol)["answer"], "550")
        self.assertEqual(parse_target("Final answer: 550\n", "length", protocol)["answer"], "550")

    def test_declared_normalization_only(self):
        protocol = protocol_fixture()
        for text, answer in [("$39", "39"), ("39 dollars.", "39"), ("\\boxed{39}", "39"),
                             ("**39**", "39"), ("$39$", "39"), ("1 / 2", "1/2"), ("-0.5 kg", "-1/2")]:
            self.assertEqual(normalize_label(text, protocol), answer, text)
        for text in ["39 or 40", "39 dollars and 5 cents", "39+1", "1,00", "NaN", "The answer is 39"]:
            self.assertIsNone(normalize_label(text, protocol), text)
        score = score_target("Final answer: $9\n", "final_answer_line", "39", protocol)
        self.assertEqual(score["answer"], "9")
        self.assertFalse(score["correct"])
        self.assertFalse(score["strict_numeric_format"])

    def test_validation_only_hard_cap_and_unchanged_prompts(self):
        protocol = protocol_fixture()
        self.assertEqual(protocol["validation_questions"], 10)
        protocol["validation_questions"] = 21
        with self.assertRaisesRegex(ValueError, "capped"):
            validate_protocol(protocol)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(Path(tmp)/"data")
            protocol["validation_questions"] = 2
            jobs = jobs_for_validation(bundle, protocol)
            self.assertEqual(len(jobs), 4)
            self.assertTrue(all(j["key"].startswith("validation/") for j in jobs))
            self.assertTrue(all(j["prompt"].endswith("\nSolution:") for j in jobs))

    def test_pipeline_separates_format_parseability_correctness_and_preserves_resume(self):
        class Runner:
            calls = 0
            def generate(self, prompts):
                self.calls += 1
                return [{"completion": "Final answer: $4\n", "raw_completion": "Final answer: $4\n",
                         "finish_reason": "final_answer_line", "generated_tokens": 6, "prompt_tokens": 5,
                         "amortized_generation_seconds": .1} for _ in prompts]
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(Path(tmp)/"data")
            protocol = protocol_fixture()
            protocol["validation_questions"] = 2
            runner = Runner()
            output = Path(tmp)/"diagnostic"
            result = run_diagnostic(bundle, runner, {"fixture": True}, output, protocol)
            self.assertTrue(result["execution_complete"])
            self.assertTrue(result["review_required"])
            self.assertFalse(result["full_benchmark_authorized"])
            group = result["groups"]["few_shot"]
            self.assertEqual(group["strict_numeric_format_rate"], 0)
            self.assertEqual(group["normalized_parse_rate"], 1)
            self.assertEqual(group["numeric_accuracy_all_questions"], .5)
            before = (output/"result_manifest.json").read_bytes()
            calls = runner.calls
            run_diagnostic(bundle, runner, {"fixture": True}, output, protocol)
            self.assertEqual(runner.calls, calls)
            self.assertEqual((output/"result_manifest.json").read_bytes(), before)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "requires baseline environment")
class PerSequenceStoppingTests(unittest.TestCase):
    def test_completed_demonstrations_in_prompt_never_trigger_a_stop(self):
        import torch
        from math_project.inference_v2 import stopping_criteria
        class Characters:
            def decode(self, tokens, skip_special_tokens=True):
                return "".join(chr(value) for value in tokens)
        text = "Question: demo\nFinal answer: 999\nQuestion: target\nSolution:"
        criterion = stopping_criteria(Characters(), len(text), [9999])
        ids = torch.tensor([[ord(c) for c in text + "5"]])
        self.assertFalse(criterion(ids, None).item())
        self.assertEqual(criterion.finished, {})

    def test_huggingface_generate_stops_rows_independently_and_keeps_full_number(self):
        import torch
        from tokenizers import Tokenizer, decoders
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, LogitsProcessor, PreTrainedTokenizerFast
        from math_project.inference import ModelRunner
        from math_project.inference_v2 import BoundedRunner
        torch.set_num_threads(2)
        vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "Final answer: ": 3, "4": 4, "5": 5, "50": 6, "\n": 7}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        backend.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        model = LlamaForCausalLM(LlamaConfig(vocab_size=8, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=128, pad_token_id=1, eos_token_id=2))
        config = {"context_window_tokens": 128, "generation": {"max_new_tokens": 8}}
        base = ModelRunner(model, tokenizer, config, "cpu")
        original = model.generate
        class Scripted(LogitsProcessor):
            offset = None
            def __call__(self, input_ids, scores):
                if self.offset is None:
                    self.offset = input_ids.shape[1]
                step = input_ids.shape[1]-self.offset
                scores.fill_(-float("inf"))
                for index, sequence in enumerate([[3, 4, 7], [3, 5, 6, 7]]):
                    scores[index, sequence[min(step, len(sequence)-1)]] = 0
                return scores
        def generate(*args, **kwargs):
            return original(*args, logits_processor=[Scripted()], **kwargs)
        model.generate = generate
        protocol = protocol_fixture()
        protocol["generation"]["max_new_tokens"] = 8
        outputs = BoundedRunner(base, protocol).generate(["Question: x", "A longer prompt with Final answer: 999\nQuestion: y"])
        self.assertEqual([o["completion"] for o in outputs], ["Final answer: 4\n", "Final answer: 550\n"])
        self.assertEqual([o["generated_tokens"] for o in outputs], [3, 4])
        self.assertEqual([o["finish_reason"] for o in outputs], ["final_answer_line", "final_answer_line"])

    def test_real_model_validation_diagnostic_executes(self):
        from test_inference import InferenceTests
        from math_project.inference_v2 import BoundedRunner
        with tempfile.TemporaryDirectory() as tmp:
            bundle = fixture_bundle(Path(tmp)/"data")
            base = InferenceTests().make_runner(bundle)
            protocol = protocol_fixture()
            protocol["validation_questions"] = 2
            protocol["generation"]["max_new_tokens"] = 8
            result = run_diagnostic(bundle, BoundedRunner(base, protocol), {"fixture": "real_tiny_llama"},
                                    Path(tmp)/"diagnostic", protocol)
            self.assertTrue(result["execution_complete"])
            self.assertEqual(result["predictions"], 4)

    def test_diagnostic_cli_loads_saved_model_and_emits_review_only_results(self):
        from test_inference import InferenceTests
        from math_project.data import sha256, write_json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = fixture_bundle(root/"data")
            base = InferenceTests().make_runner(bundle)
            base.model.save_pretrained(root/"tiny-model")
            base.tokenizer.save_pretrained(bundle["path"]/"tokenizer")
            bundle["config"]["model"]["id"] = str(root/"tiny-model")
            write_json(bundle["path"]/"config.resolved.json", bundle["config"])
            manifest = bundle["manifest"]
            manifest["model"] = bundle["config"]["model"]
            manifest["files_sha256"] = {str(p.relative_to(bundle["path"])): sha256(p.read_bytes())
                for p in bundle["path"].rglob("*") if p.is_file() and p.name != "manifest.json"}
            write_json(bundle["path"]/"manifest.json", manifest)
            protocol = protocol_fixture()
            protocol["validation_questions"] = 2
            protocol["generation"]["max_new_tokens"] = 8
            write_json(root/"protocol.json", protocol)
            result = subprocess.run([sys.executable, str(ROOT/"scripts/run_protocol_v2.py"),
                "--bundle", str(bundle["path"]), "--protocol", str(root/"protocol.json"),
                "--output", str(root/"diagnostic"), "--device", "cpu"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            summary = json.loads((root/"diagnostic/summary.json").read_text())
            self.assertEqual(summary["predictions"], 4)
            self.assertFalse(summary["full_benchmark_authorized"])
            self.assertTrue(summary["review_required"])
