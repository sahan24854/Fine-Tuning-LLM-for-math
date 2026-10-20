"""CPU-only audit of existing predictions. Never rerun inference or change scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_project.data import canonical_number, extract_final_answer, read_jsonl, sha256, write_json


def inspect_completion(text):
    markers = re.findall(r"^[ \t]*Final answer:[ \t]*([^\n]*)$", text, re.MULTILINE | re.IGNORECASE)
    # Exploratory diagnostic only. Original strict scores remain untouched.
    boundary = re.search(r"\n[ \t]*(?:###[ \t]*)?Question:", text, re.IGNORECASE)
    first_problem = text[:boundary.start()] if boundary else text
    first_markers = re.findall(r"^[ \t]*Final answer:[ \t]*([^\n]*)$", first_problem, re.MULTILINE | re.IGNORECASE)
    first_numeric = None
    if first_markers:
        try:
            first_numeric = canonical_number(first_markers[0])
        except ValueError:
            pass
    return {"strict_answer": extract_final_answer(text), "marker_count": len(markers),
            "continued_to_another_question": boundary is not None,
            "first_problem_first_label_numeric": first_numeric}


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.is_relative_to(run):
        raise ValueError("Write the audit outside the original run directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Audit output directory must be new or empty")
    manifest = json.loads((run / "result_manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        target = (run / name).resolve()
        if not target.is_relative_to(run) or sha256(target.read_bytes()) != digest:
            raise ValueError(f"Original result verification failed: {name}")
    identity = json.loads((run / "identity.json").read_text())
    records = read_jsonl(run / "predictions.jsonl")
    if len(records) != manifest["prediction_count"]:
        raise ValueError("Original prediction count mismatch")
    groups, samples = {}, []
    sampled = Counter()
    for record in records:
        key = f"{record['split']}/{record['mode']}"
        group = groups.setdefault(key, Counter())
        group["count"] += 1
        details = inspect_completion(record["completion"])
        strict = details["strict_answer"]
        correct = strict == record["reference_answer"]
        if strict != record["extracted_answer"] or correct != record["correct"]:
            raise ValueError(f"Stored strict score differs from recomputation: {record['key']}")
        group["strict_correct"] += correct
        group["strict_invalid"] += strict is None
        group["length_limited"] += record["finish_reason"] == "length"
        group["no_final_label"] += details["marker_count"] == 0
        group["multiple_final_labels"] += details["marker_count"] > 1
        group["single_unparseable_label"] += details["marker_count"] == 1 and strict is None
        group["continued_to_another_question"] += details["continued_to_another_question"]
        group["exploratory_first_problem_label_correct"] += details["first_problem_first_label_numeric"] == record["reference_answer"]
        if sampled[key] < 5:
            # Stable first examples, never choose examples based on correctness.
            samples.append({"key": record["key"], "completion": record["completion"],
                            "reference_answer": record["reference_answer"], **details})
            sampled[key] += 1
    report = {"source_run": str(run), "source_code": identity["code"], "model": identity["model"],
              "generation": identity["generation"], "prediction_count": len(records), "groups": groups,
              "interpretation": [
                  "Original strict scores are verified and unchanged.",
                  "Length-limited means no configured EOS before the cap; it does not prove an unfinished target solution.",
                  "First-problem-label counts are post-hoc exploratory diagnostics, not replacement benchmark accuracy.",
                  "Do not infer answers from arbitrary last numbers or match the reference anywhere in the output.",
                  "Use validation (smoke) completions to design a revised protocol; retain test results as an audit trail."]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", report)
    write_json(output / "completion_samples.json", samples)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Existing smoke or full run directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the original run")
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.output), indent=2))
