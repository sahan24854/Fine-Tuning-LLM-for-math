import json
from pathlib import Path
import tempfile
import unittest

from math_project.evaluation import smoke_health_issues
from scripts.audit_baseline import audit, inspect_completion
from scripts.run_baseline import evaluate
from test_evaluation import FixedRunner, fixture_bundle


class AuditTests(unittest.TestCase):
    def test_health_check_flags_the_observed_smoke_pattern(self):
        summary = {"stage": "smoke", "complete": True, "groups": {
            "validation/zero_shot": {"count": 20, "invalid_answers": 20, "length_limited_answers": 20},
            "validation/few_shot": {"count": 20, "invalid_answers": 19, "length_limited_answers": 20}}}
        self.assertEqual(len(smoke_health_issues(summary)), 4)

    def test_low_accuracy_alone_is_not_a_health_failure(self):
        summary = {"stage": "smoke", "complete": True, "groups": {
            "validation/zero_shot": {"count": 20, "accuracy": 0., "invalid_answers": 0, "length_limited_answers": 0}}}
        self.assertEqual(smoke_health_issues(summary), [])
        self.assertTrue(smoke_health_issues({"stage": "smoke", "complete": True, "groups": {}}))

    def test_continuation_after_a_correct_answer_can_be_strictly_invalid(self):
        text = "Final answer: 4\n\nQuestion: another problem\nSolution: 6\nFinal answer: 6"
        details = inspect_completion(text)
        self.assertIsNone(details["strict_answer"])
        self.assertEqual(details["first_problem_first_label_numeric"], "4")
        self.assertEqual(details["marker_count"], 2)
        self.assertTrue(details["continued_to_another_question"])

    def test_no_search_for_reference_in_unlabelled_text(self):
        details = inspect_completion("Maybe 4? Then 42. The answer is 9.")
        self.assertIsNone(details["first_problem_first_label_numeric"])

    def test_audit_verifies_saved_results_and_leaves_them_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = fixture_bundle(root / "data")
            evaluate(bundle, FixedRunner(), {"fixture": True}, root / "run", "smoke", 2)
            before = {p.name: p.read_bytes() for p in (root / "run").iterdir()}
            report = audit(root / "run", root / "audit")
            self.assertEqual(report["prediction_count"], 4)
            self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "run").iterdir()})
            with self.assertRaisesRegex(ValueError, "outside"):
                audit(root / "run", root / "run/audit")
