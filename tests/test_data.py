import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

from math_project.data import (
    arithmetic_set, canonical_number, download_source, extract_final_answer,
    parse_gsm8k, prompt, question_key, sha256, split_training, tokenize_and_audit,
)


def row(index, question=None):
    return {"id": str(index), "question": question or f"What is {index} + 1?",
            "solution": f"Adding one gives {index + 1}.", "answer": str(index + 1),
            "answer_canonical": str(index + 1)}


class WordTokenizer:
    """Deterministic test double, never used for production token length claims."""
    eos_token_id = 0
    model_max_length = 2048

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text.split())))


class DataTests(unittest.TestCase):
    def test_numeric_normalization_and_rejection(self):
        for text, expected in [("1,234.50", "2469/2"), ("-0.5", "-1/2"), ("2/4", "1/2"), ("−3", "-3")]:
            self.assertEqual(canonical_number(text), expected)
        for text in ["1,2", "NaN", "inf", "4 meters", "1/0", "1 or 2"]:
            with self.assertRaises(ValueError):
                canonical_number(text)

    def test_score_only_explicit_single_final_answer(self):
        self.assertEqual(extract_final_answer("Step 1: 2 + 2 = 4\nFinal answer: 4"), "4")
        self.assertIsNone(extract_final_answer("I tried 1, 2 and 4"))
        self.assertIsNone(extract_final_answer("Final answer: 2\nFinal answer: 4"))
        self.assertIsNone(extract_final_answer("Final answer: 4 pencils"))

    def test_split_is_deterministic_order_independent_and_disjoint(self):
        rows = [row(i) for i in range(40)]
        held_out = [row(5)]
        train, validation, report = split_training(rows + [row(3)], held_out, 42, .1)
        again, val_again, _ = split_training(list(reversed(rows)), held_out, 42, .1)
        self.assertEqual(train, again)
        self.assertEqual(validation, val_again)
        self.assertEqual(report["training_duplicates_removed"], 1)
        self.assertEqual(report["training_test_overlap_removed"], 1)
        keys = lambda examples: {question_key(r["question"]) for r in examples}
        self.assertFalse(keys(train) & keys(validation))
        self.assertFalse((keys(train) | keys(validation)) & keys(held_out))

    def test_conflicting_labels_fail(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            split_training([row(1), {**row(1), "answer_canonical": "5"}] + [row(i) for i in range(2, 20)], [], 42, .1)

    def test_gsm8k_removes_annotations_and_separates_label(self):
        data = (json.dumps({"question": "How many?", "answer": "2+2=<<2+2=4>>4.\n#### 4"}) + "\n").encode()
        parsed = parse_gsm8k(data, "train")[0]
        self.assertEqual(parsed["solution"], "2+2=4.")
        self.assertEqual(parsed["answer_canonical"], "4")

    def test_arithmetic_labels_and_unique_questions(self):
        rows = arithmetic_set(42, 50)
        self.assertEqual(len(rows), 500)
        self.assertEqual(len({r["question"] for r in rows}), 500)
        self.assertEqual(rows, arithmetic_set(42, 50))
        for r in rows:
            a, b, *rest = r["operands"]
            expected = {"add": a+b, "subtract": a-b, "multiply": a*b,
                        "divide": a, "two_step": a*b-(rest[0] if rest else 0)}[r["operation"]]
            self.assertEqual(Fraction(r["answer"]), expected)

    def test_corrupt_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected = sha256(b"expected")
            Path(tmp, expected).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                download_source({"url": "https://example.invalid/data", "sha256": expected}, tmp)

    def test_prompt_has_no_query_answer_and_demos_are_explicit(self):
        text = prompt("Held-out question?", [row(1)])
        self.assertTrue(text.endswith("Question: Held-out question?\nSolution:"))
        self.assertIn("Final answer: 2", text)

    def test_length_filter_does_not_drop_evaluation_and_demos_are_training_only(self):
        config = {"max_training_tokens": 100, "demonstration_max_tokens": 80, "few_shot_count": 2,
                  "context_window_tokens": 2048, "generation": {"max_new_tokens": 512}}
        long = {**row(99), "solution": "long " * 200}
        splits = {"train": [row(i) for i in range(5)] + [long], "validation": [row(10)],
                  "arithmetic_test": arithmetic_set(42, 1)}
        audit = {}
        result, demos, excluded = tokenize_and_audit(splits, config, WordTokenizer(), audit)
        self.assertEqual([r["id"] for r in excluded], ["99"])
        self.assertEqual(result["validation"], splits["validation"])
        self.assertTrue({r["id"] for r in demos} <= {r["id"] for r in result["train"]})
        self.assertIn("prompt_token_count", result["train"][0])
        config["context_window_tokens"] = 520
        with self.assertRaisesRegex(ValueError, "exceeds prompt budget"):
            tokenize_and_audit(copy.deepcopy(splits), config, WordTokenizer(), {})


if __name__ == "__main__":
    unittest.main()
