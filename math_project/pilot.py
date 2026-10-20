"""Deterministic development pilot selection and reporting; no model dependencies."""
from collections import Counter
import csv
import html
import json
from pathlib import Path

from math_project.data import prompt, sha256
from math_project.evaluation import TEST_SPLITS, atomic_json, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
ARITHMETIC_STRATA = tuple((size, operation) for size in ("small", "large")
                          for operation in ("add", "subtract", "multiply", "divide", "two_step"))


def load_protocol(path):
    pilot = json.loads(Path(path).read_text())
    if (pilot["scope"] != "development_comparison_previously_inspected_test_sets" or
            pilot["mode"] != "few_shot" or pilot["batch_size"] != 4 or
            type(pilot["seed"]) is not int or type(pilot["questions_per_dataset"]) is not int or
            not 1 <= pilot["questions_per_dataset"] <= 100):
        raise ValueError("Pilot requires three-shot, batch 4, an integer seed, and at most 100 questions per dataset")
    scoring_path = (ROOT / pilot["scoring_protocol_file"]).resolve()
    if not scoring_path.is_relative_to(ROOT) or sha256(scoring_path.read_bytes()) != pilot["scoring_protocol_file_sha256"]:
        raise ValueError("Reviewed scoring protocol checksum changed")
    scoring = json.loads(scoring_path.read_text())
    from math_project.protocol_v2 import validate_protocol
    validate_protocol(scoring)
    if scoring["generation"]["max_new_tokens"] != 512:
        raise ValueError("Pilot retains the reviewed 512-token safety cap")
    return pilot, scoring


def make_plan(bundle, pilot, scoring):
    if len(bundle["demonstrations"]) != 3:
        raise ValueError("Pilot requires exactly three frozen training demonstrations")
    count = pilot["questions_per_dataset"]
    if count % len(ARITHMETIC_STRATA):
        raise ValueError("Arithmetic count must divide equally across its ten strata")
    selected = []
    for split in TEST_SPLITS:
        rows = bundle["splits"][split]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate IDs in {split}")
        def rank(row):
            payload = json.dumps([pilot["seed"], split, row["id"]], separators=(",", ":"))
            return sha256(payload.encode()), row["id"]
        if split == "arithmetic_test":
            strata = {(r["number_range"], r["operation"]) for r in rows}
            if strata != set(ARITHMETIC_STRATA):
                raise ValueError("Unexpected arithmetic strata")
            targets = []
            for size, operation in ARITHMETIC_STRATA:
                pool = sorted([r for r in rows if (r["number_range"], r["operation"]) == (size, operation)], key=rank)
                n = count // len(ARITHMETIC_STRATA)
                if len(pool) < n:
                    raise ValueError("Not enough rows in an arithmetic stratum")
                targets.extend(pool[:n])
        else:
            if len(rows) < count:
                raise ValueError(f"Not enough rows in {split}")
            targets = sorted(rows, key=rank)[:count]
        for row in targets:
            text = prompt(row["question"], bundle["demonstrations"])
            selected.append({"key": f"{split}/few_shot/{row['id']}", "split": split,
                "mode": "few_shot", "id": row["id"], "question_sha256": sha256(row["question"].encode()),
                "reference_sha256": sha256(row["answer_canonical"].encode()),
                "prompt_sha256": sha256(text.encode()),
                "category": f"{row['number_range']}/{row['operation']}" if split == "arithmetic_test" else row.get("category")})
    plan = {"schema_version": 1, "pilot": pilot, "scoring_protocol": scoring,
            "bundle_sha256": bundle["fingerprint"], "model": bundle["config"]["model"],
            "context_window_tokens": bundle["config"]["context_window_tokens"],
            "demonstrations_sha256": sha256(json.dumps(bundle["demonstrations"], sort_keys=True).encode()),
            "prediction_count": len(selected), "jobs": selected}
    # The comparison fingerprint pins the base revision and evaluation inputs,
    # independently of the later adapter artifact and execution environment.
    plan["comparison_contract_sha256"] = sha256(json.dumps(plan, sort_keys=True).encode())
    return plan


def save_plan(output, plan):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "selection.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Frozen pilot selection changed; use a new output directory")
    else:
        if any((output / name).exists() for name in ("identity.json", "predictions.jsonl")):
            raise ValueError("Cannot reconstruct a missing selection for existing results")
        atomic_json(path, plan)


def jobs_from_plan(bundle, plan):
    lookup = {(split, row["id"]): row for split in TEST_SPLITS for row in bundle["splits"][split]}
    return [{**item, "row": lookup[item["split"], item["id"]],
             "prompt": prompt(lookup[item["split"], item["id"]]["question"], bundle["demonstrations"])}
            for item in plan["jobs"]]


def group_metrics(rows, expected):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    valid = sum(r["answer"] is not None for r in rows)
    tokens = sum(r["generated_tokens"] for r in rows)
    return {"count": n, "expected": expected, "correct": correct,
            "accuracy": correct/n if n else None, "accuracy_wilson_95": wilson_interval(correct, n),
            "parseable": valid, "parse_rate": valid/n if n else None,
            "strict_numeric_format_rate": sum(r["strict_numeric_format"] for r in rows)/n if n else None,
            "invalid": n-valid, "cap_hits": sum(r["finish_reason"] == "length" for r in rows),
            "stop_reasons": dict(Counter(r["finish_reason"] for r in rows)),
            "generated_tokens": tokens, "mean_generated_tokens": tokens/n if n else None,
            "generation_seconds": sum(r["amortized_generation_seconds"] for r in rows)}


def write_report(output, records, plan):
    groups = {split: group_metrics([r for r in records if r["split"] == split],
                                  sum(j["split"] == split for j in plan["jobs"])) for split in TEST_SPLITS}
    categories = {f"{size}/{operation}": group_metrics(
        [r for r in records if r["split"] == "arithmetic_test" and r["category"] == f"{size}/{operation}"],
        sum(j["split"] == "arithmetic_test" and j["category"] == f"{size}/{operation}" for j in plan["jobs"]))
        for size, operation in ARITHMETIC_STRATA}
    summary = {"stage": "pilot_v2", "scope": plan["pilot"]["scope"], "model_role": "untouched_base",
               "execution_complete": len(records) == plan["prediction_count"], "predictions": len(records),
               "expected_predictions": plan["prediction_count"], "groups": groups, "arithmetic_categories": categories,
               "comparison_contract_sha256": plan["comparison_contract_sha256"],
               "interpretation": "Development comparison on previously inspected test sets; not an untouched final holdout. Invalid answers count as incorrect. Report datasets separately. Wilson intervals are descriptive and do not address pretraining contamination or development reuse.",
               "timing_note": "Amortized generation time includes stopping checks and first-call initialization; no warmup or end-to-end latency claim."}
    atomic_json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "count", "expected", "accuracy", "wilson_95_low", "wilson_95_high", "parse_rate", "invalid", "cap_hits", "generated_tokens"])
        for split, g in groups.items():
            writer.writerow([split, g["count"], g["expected"], g["accuracy"], *g["accuracy_wilson_95"], g["parse_rate"], g["invalid"], g["cap_hits"], g["generated_tokens"]])
    document = "<!doctype html><meta charset='utf-8'><title>SmolLM2 baseline pilot</title><style>body{font:16px system-ui;max-width:1100px;margin:2em auto}td,th{padding:.6em;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}details{margin:1em 0}</style>"
    document += f"<h1>SmolLM2 — three-shot baseline pilot</h1><p>{'Complete' if summary['execution_complete'] else 'PARTIAL'}: {len(records)}/{plan['prediction_count']} predictions.</p>"
    document += "<table><tr><th>Dataset</th><th>Count</th><th>Accuracy (95% Wilson interval)</th><th>Parse rate</th><th>Cap hits</th></tr>"
    for split, g in groups.items():
        percent = lambda value: f"{100*value:.1f}%" if value is not None else "—"
        low, high = g["accuracy_wilson_95"]
        document += f"<tr><td>{split}</td><td>{g['count']}/{g['expected']}</td><td>{percent(g['accuracy'])} ({percent(low)}–{percent(high)})</td><td>{percent(g['parse_rate'])}</td><td>{g['cap_hits']}</td></tr>"
    document += "</table><p>" + html.escape(summary["interpretation"]) + "</p><p>" + html.escape(summary["timing_note"]) + "</p>"
    for r in records:
        details = {name: r[name] for name in ("reference_answer", "answer", "correct", "parse_status", "finish_reason", "generated_tokens")}
        document += f"<details><summary>{html.escape(r['key'])}</summary><p>{html.escape(r['question'])}</p><pre>{html.escape(json.dumps(details, indent=2))}</pre><h3>Scored target</h3><pre>{html.escape(r['completion'])}</pre><h3>Raw generated text</h3><pre>{html.escape(r['raw_completion'])}</pre></details>"
    (output / "report.html").write_text(document, encoding="utf-8")
    return summary
