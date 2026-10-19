"""Baseline protocol, resumable predictions, and reports (no ML dependencies)."""
import csv
import html
import json
import math
from pathlib import Path

from math_project.data import extract_final_answer, prompt, read_jsonl, sha256
from scripts.verify_bundle import verify

TEST_SPLITS = ("gsm8k_test", "svamp_test", "arithmetic_test")
MODES = ("zero_shot", "few_shot")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_bundle(path):
    path = Path(path)
    manifest = verify(path)
    required = {"config.resolved.json", "demonstrations.json", "validation.jsonl", "train.jsonl"}
    required.update(f"{name}.jsonl" for name in TEST_SPLITS)
    if not required <= manifest["files_sha256"].keys():
        raise ValueError("Bundle manifest is missing required evaluation files")
    for file in (path / "tokenizer").rglob("*"):
        if file.is_file() and str(file.relative_to(path)) not in manifest["files_sha256"]:
            raise ValueError("Unverified extra tokenizer file")
    config = json.loads((path / "config.resolved.json").read_text())
    if config["model"] != manifest["model"]:
        raise ValueError("Model revision differs between config and manifest")
    revision = config["model"]["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Bundle must pin a full model commit")
    splits = {name: read_jsonl(path / f"{name}.jsonl")
              for name in ("train", "validation", *TEST_SPLITS)}
    for name, rows in splits.items():
        if len(rows) != manifest["counts"][name] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Count or ID mismatch in {name}")
    demonstrations = json.loads((path / "demonstrations.json").read_text())
    train = {r["id"]: r for r in splits.pop("train")}
    if len(demonstrations) != config["few_shot_count"]:
        raise ValueError("Wrong number of demonstrations")
    for row in demonstrations:
        if row != train.get(row["id"]):
            raise ValueError("Demonstration is not an unchanged training example")
    return {"path": path, "manifest": manifest, "config": config, "splits": splits,
            "demonstrations": demonstrations, "fingerprint": sha256((path / "manifest.json").read_bytes())}


def make_jobs(bundle, stage, smoke_count=20):
    if stage not in ("smoke", "full") or smoke_count <= 0:
        raise ValueError("Invalid evaluation stage or smoke count")
    jobs = []
    names = ("validation",) if stage == "smoke" else TEST_SPLITS
    for name in names:
        rows = bundle["splits"][name]
        if stage == "smoke":
            rows = rows[:smoke_count]
        for mode in MODES:
            shots = bundle["demonstrations"] if mode == "few_shot" else []
            for row in rows:
                text = prompt(row["question"], shots)
                jobs.append({"key": f"{name}/{mode}/{row['id']}", "split": name, "mode": mode,
                             "row": row, "prompt": text, "prompt_sha256": sha256(text.encode())})
    return jobs


def read_predictions(path):
    """Keep complete flushed records; discard only an interrupted final fragment."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n") + 1
    if end != len(data):
        with path.open("r+b") as file:
            file.truncate(end)
    return read_jsonl(path)


def validate_records(records, jobs):
    expected = {job["key"]: job for job in jobs}
    seen = set()
    for record in records:
        key = record["key"]
        if key not in expected or key in seen:
            raise ValueError("Duplicate or unexpected prediction; refusing resume")
        seen.add(key)
        job = expected[key]
        if record["prompt_sha256"] != job["prompt_sha256"]:
            raise ValueError("Prediction prompt differs from frozen protocol")
        extracted = extract_final_answer(record["completion"])
        if (record["extracted_answer"] != extracted or
                record["reference_answer"] != job["row"]["answer_canonical"] or
                record["correct"] != (extracted == job["row"]["answer_canonical"]) or
                record["split"] != job["split"] or record["mode"] != job["mode"]):
            raise ValueError("Prediction score or metadata is inconsistent")
    return seen


def wilson_interval(correct, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def summarize(records):
    correct = sum(r["correct"] for r in records)
    count = len(records)
    seconds = sum(r["amortized_generation_seconds"] for r in records)
    tokens = sum(r["generated_tokens"] for r in records)
    return {"count": count, "correct": correct, "accuracy": correct/count if count else None,
            "accuracy_wilson_95": wilson_interval(correct, count),
            "invalid_answers": sum(r["extracted_answer"] is None for r in records),
            "length_limited_answers": sum(r["finish_reason"] == "length" for r in records),
            "generation_seconds": seconds, "generated_tokens": tokens,
            "generated_tokens_per_second": tokens/seconds if seconds else None,
            "mean_generated_tokens": tokens/count if count else None,
            "mean_amortized_generation_seconds": seconds/count if count else None}


def write_reports(output, records, jobs, stage):
    output = Path(output)
    groups = {}
    for split, mode in sorted({(j["split"], j["mode"]) for j in jobs}):
        rows = [r for r in records if r["split"] == split and r["mode"] == mode]
        group = summarize(rows)
        group["expected_count"] = sum(j["split"] == split and j["mode"] == mode for j in jobs)
        groups[f"{split}/{mode}"] = group
    categories = {}
    for mode, category in sorted({(r["mode"], r["category"]) for r in records if r.get("category")}):
        categories[f"{mode}/{category}"] = summarize([r for r in records if r["mode"] == mode and r.get("category") == category])
    complete = len(records) == len(jobs)
    summary = {"schema_version": 1, "stage": stage, "complete": complete,
               "completed_predictions": len(records), "expected_predictions": len(jobs),
               "groups": groups, "arithmetic_categories": categories,
               "timing_note": "GPU-synchronized generation only, divided by batch size; not per-request latency. Model loading, warmup, tokenization and scoring excluded.",
               "uncertainty_note": "Wilson 95% intervals describe baseline proportions. Paired bootstrap of baseline versus tuned results belongs in the later comparison.",
               "scope_note": "Smoke results use validation only and are not final benchmark scores." if stage == "smoke" else "Untouched base model; test benchmarks reported separately."}
    atomic_json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "count", "expected_count", "accuracy", "wilson_95_low", "wilson_95_high", "invalid", "length_limited"])
        for name, group in groups.items():
            writer.writerow([name, group["count"], group["expected_count"], group["accuracy"], *group["accuracy_wilson_95"], group["invalid_answers"], group["length_limited_answers"]])
    rows = []
    for name, group in groups.items():
        percent = 100 * (group["accuracy"] or 0)
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{group['count']}/{group['expected_count']}</td>"
                    f"<td><progress max='100' value='{percent}'></progress> {percent:.2f}%</td>"
                    f"<td>{group['invalid_answers']}</td><td>{group['length_limited_answers']}</td></tr>")
    document = "<!doctype html><meta charset='utf-8'><title>SmolLM2 baseline</title>"
    document += "<style>body{font:16px system-ui;max-width:1100px;margin:3em auto}td,th{padding:.7em;text-align:left}progress{width:120px}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>"
    document += f"<h1>SmolLM2 baseline — {html.escape(stage)}</h1><p>Status: {'complete' if complete else 'PARTIAL'}. {len(records)}/{len(jobs)} predictions.</p>"
    document += "<table><tr><th>Benchmark / mode</th><th>Completed</th><th>Accuracy</th><th>Invalid</th><th>Token limit reached</th></tr>" + "".join(rows) + "</table>"
    document += "<p>Numeric final-answer accuracy includes invalid and length-limited answers in the denominator. See metrics.csv for Wilson intervals.</p>"
    document += f"<p>{html.escape(summary['timing_note'])}</p><p>{html.escape(summary['scope_note'])}</p>"
    (output / "report.html").write_text(document)
    return summary
