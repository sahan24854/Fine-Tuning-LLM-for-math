"""Run only a bounded, versioned validation diagnostic. No full-benchmark mode."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_project.data import prompt, sha256
from math_project.evaluation import atomic_json, load_bundle, read_predictions
from math_project.protocol_v2 import repetition_flags, score_target, validate_protocol


def jobs_for_validation(bundle, protocol):
    validate_protocol(protocol)
    rows = bundle["splits"]["validation"][:protocol["validation_questions"]]
    if len(rows) != protocol["validation_questions"]:
        raise ValueError("Not enough validation rows for the configured diagnostic")
    jobs = []
    for mode in protocol["modes"]:
        examples = bundle["demonstrations"] if mode == "few_shot" else []
        for row in rows:
            text = prompt(row["question"], examples)
            jobs.append({"key": f"validation/{mode}/{row['id']}", "mode": mode,
                         "row": row, "prompt": text, "prompt_sha256": sha256(text.encode())})
    return jobs


def write_report(output, records, jobs, protocol):
    groups, alarms = {}, []
    limits = protocol["reporting"]["quality_alarms"]
    for mode in protocol["modes"]:
        rows = [r for r in records if r["mode"] == mode]
        n = len(rows)
        valid = sum(r["answer"] is not None for r in rows)
        correct = sum(r["correct"] for r in rows)
        strict = sum(r["strict_numeric_format"] for r in rows)
        reasons = dict(Counter(r["finish_reason"] for r in rows))
        groups[mode] = {"count": n, "expected": sum(j["mode"] == mode for j in jobs),
                        "correct": correct, "strict_numeric_format_count": strict,
                        "normalized_parseable_count": valid,
                        "strict_numeric_format_rate": strict/n if n else None,
                        "normalized_parse_rate": valid/n if n else None,
                        "numeric_accuracy_all_questions": correct/n if n else None,
                        "accuracy_among_parseable_selection_biased": correct/valid if valid else None,
                        "stop_reasons": reasons,
                        "mean_generated_tokens": sum(r["generated_tokens"] for r in rows)/n if n else None,
                        "generation_seconds": sum(r["amortized_generation_seconds"] for r in rows),
                        "repetition_flagged": sum(any(r["repetition_flags"].values()) for r in rows)}
        if n and (n-valid)/n >= limits["unparseable_fraction_at_least"]:
            alarms.append(f"{mode}: {n-valid}/{n} unparseable target answers")
        if n and reasons.get("length", 0)/n >= limits["length_limited_fraction_at_least"]:
            alarms.append(f"{mode}: {reasons.get('length', 0)}/{n} generation-cap stops")
    complete = len(records) == len(jobs)
    summary = {"protocol_id": protocol["protocol_id"], "scope": protocol["scope"],
               "execution_complete": complete, "review_required": True,
               "full_benchmark_authorized": False, "predictions": len(records),
               "expected_predictions": len(jobs), "groups": groups, "quality_alarms": alarms,
               "timing_note": "Amortized batch generation time includes per-token stopping checks and first-call initialization; there is no warmup. This diagnostic is not a latency benchmark.",
               "interpretation": "Validation development diagnostic, not final test accuracy. Parseable-only accuracy is selection-biased. Low accuracy alone is not a runtime failure."}
    atomic_json(output / "summary.json", summary)
    with (output / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mode", "completed", "strict_numeric_format_rate", "normalized_parse_rate",
                         "numeric_accuracy_all_questions", "accuracy_among_parseable_selection_biased", "mean_generated_tokens"])
        for mode, values in groups.items():
            writer.writerow([mode, values["count"], *[values[name] for name in
                ["strict_numeric_format_rate", "normalized_parse_rate", "numeric_accuracy_all_questions",
                 "accuracy_among_parseable_selection_biased", "mean_generated_tokens"]]])
    rows_html = []
    for mode, group in groups.items():
        percent = lambda key: f"{100*group[key]:.1f}%" if group[key] is not None else "—"
        rows_html.append(f"<tr><td>{html.escape(mode)}</td><td>{group['count']}/{group['expected']}</td>"
                         f"<td>{percent('strict_numeric_format_rate')}</td><td>{percent('normalized_parse_rate')}</td>"
                         f"<td>{percent('numeric_accuracy_all_questions')}</td><td>{html.escape(str(group['stop_reasons']))}</td></tr>")
    document = "<!doctype html><meta charset='utf-8'><title>Protocol v2 validation review</title>"
    document += "<style>body{font:16px system-ui;max-width:1100px;margin:2em auto}td,th{padding:.6em;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}details{margin:1em 0}</style>"
    document += f"<h1>Protocol v2 — validation only</h1><p>Execution {'complete' if complete else 'PARTIAL'}. {len(records)}/{len(jobs)} predictions. Human review required. No full benchmark is launched.</p>"
    document += "<table><tr><th>Mode</th><th>Count</th><th>Strict numeric format</th><th>Normalized parse rate</th><th>Numeric accuracy, all questions</th><th>Stopping</th></tr>" + "".join(rows_html) + "</table>"
    document += "<p>Numeric correctness is exact rational equality after documented normalization. It is still performance under this protocol. Unit text may be removed, but units are not converted or validated. Conditional accuracy among parseable outputs is selection-biased.</p>"
    document += "<h2>Quality alarms</h2><ul>" + "".join(f"<li>{html.escape(a)}</li>" for a in alarms) + "</ul><p>Absence of alarms does not establish valid maths measurement. Review raw answers below.</p>"
    for record in records:
        details = {key: record[key] for key in ["finish_reason", "parse_status", "answer", "reference_answer", "correct", "generated_tokens"]}
        document += f"<details><summary>{html.escape(record['key'])} — {'correct' if record['correct'] else 'incorrect/unparseable'}</summary>"
        document += f"<p>{html.escape(record['question'])}</p><pre>{html.escape(json.dumps(details, indent=2))}</pre>"
        document += f"<h3>Scored target text</h3><pre>{html.escape(record['completion'])}</pre>"
        document += f"<h3>Raw generated text through stopping token</h3><pre>{html.escape(record['raw_completion'])}</pre></details>"
    (output / "report.html").write_text(document, encoding="utf-8")
    return summary


def run_diagnostic(bundle, runner, runtime, output, protocol):
    jobs = jobs_for_validation(bundle, protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_names = ["math_project/data.py", "math_project/inference.py", "math_project/protocol_v2.py",
                    "math_project/inference_v2.py", "math_project/evaluation.py", "scripts/run_protocol_v2.py", "scripts/verify_bundle.py"]
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    identity = {"protocol": protocol, "model": bundle["config"]["model"], "runtime": runtime,
                "bundle_sha256": bundle["fingerprint"], "python": platform.python_version(),
                "packages_sha256": sha256("\n".join(packages).encode()),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_files_sha256": {name: sha256((ROOT/name).read_bytes()) for name in source_names},
                "job_prompts_sha256": sha256(json.dumps([(j["key"], j["prompt_sha256"]) for j in jobs]).encode())}
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is running this diagnostic") from exc
        path = output / "identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Diagnostic resume identity changed; use a new output directory")
        if not path.exists():
            if (output / "predictions.jsonl").exists():
                raise ValueError("Cannot resume predictions without identity")
            atomic_json(path, identity)
        records = read_predictions(output / "predictions.jsonl")
        expected = {job["key"]: job for job in jobs}
        seen = set()
        for record in records:
            key = record["key"]
            if key in seen or key not in expected:
                raise ValueError("Unexpected or duplicate diagnostic record")
            job = expected[key]
            scores = score_target(record["completion"], record["finish_reason"], job["row"]["answer_canonical"], protocol)
            if (record["prompt_sha256"] != job["prompt_sha256"] or record["reference_answer"] != job["row"]["answer_canonical"] or
                    any(record[field] != value for field, value in scores.items())):
                raise ValueError("Diagnostic prediction changed; refusing resume")
            seen.add(key)
        if len(seen) == len(jobs) and (output / "result_manifest.json").exists():
            manifest = json.loads((output / "result_manifest.json").read_text())
            for name, digest in manifest["files_sha256"].items():
                target = (output/name).resolve()
                if not target.is_relative_to(output.resolve()) or sha256(target.read_bytes()) != digest:
                    raise ValueError("Diagnostic result checksum mismatch")
            return json.loads((output / "summary.json").read_text())
        (output / "result_manifest.json").unlink(missing_ok=True)
        atomic_json(output / "protocol.json", protocol)
        atomic_json(output / "source_data_manifest.json", bundle["manifest"])
        (output / "environment.txt").write_text("\n".join(packages)+"\n")
        print(f"Validation-only diagnostic: {len(seen)}/{len(jobs)} predictions saved; hard limit is 40", flush=True)
        try:
            with (output / "predictions.jsonl").open("a", encoding="utf-8") as sink:
                batch_size = protocol["batch_size"]
                for offset in range(0, len(jobs), batch_size):
                    batch = jobs[offset:offset+batch_size]
                    if all(job["key"] in seen for job in batch):
                        continue
                    answers = runner.generate([job["prompt"] for job in batch])
                    if len(answers) != len(batch):
                        raise ValueError("Wrong number of diagnostic completions")
                    for job, answer in zip(batch, answers):
                        if job["key"] in seen:
                            continue
                        reference = job["row"]["answer_canonical"]
                        scores = score_target(answer["completion"], answer["finish_reason"], reference, protocol)
                        record = {**answer, **scores, "key": job["key"], "split": "validation", "mode": job["mode"],
                                  "question": job["row"]["question"], "reference_answer": reference,
                                  "prompt_sha256": job["prompt_sha256"], "repetition_flags": repetition_flags(answer["completion"])}
                        sink.write(json.dumps(record, ensure_ascii=False)+"\n")
                        records.append(record)
                        seen.add(job["key"])
                    sink.flush()
                    os.fsync(sink.fileno())
                    summary = write_report(output, records, jobs, protocol)
                    print(f"Saved {len(seen)}/{len(jobs)} validation predictions", flush=True)
        finally:
            summary = write_report(output, records, jobs, protocol)
        files = ["identity.json", "protocol.json", "source_data_manifest.json", "environment.txt", "predictions.jsonl", "summary.json", "metrics.csv", "report.html"]
        atomic_json(output / "result_manifest.json", {"status": "diagnostic_complete_review_required",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "prediction_count": len(records),
            "peak_cuda_allocated_bytes_completing_process": runner.torch.cuda.max_memory_allocated()
                if str(getattr(runner, "device", "cpu")).startswith("cuda") else None,
            "files_sha256": {name: sha256((output/name).read_bytes()) for name in files}})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol_v2_diagnostic.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validate_protocol(protocol)
    bundle = load_bundle(args.bundle)
    jobs_for_validation(bundle, protocol)  # Fail before loading weights on invalid scope/counts.
    from math_project.inference import load_runner
    from math_project.inference_v2 import BoundedRunner
    base, runtime = load_runner(bundle, args.device)
    runtime = {**runtime, "model_eos_ids": base.eos_ids, "pad_token_id": base.tokenizer.pad_token_id,
               "tokenizer_eos_id": base.tokenizer.eos_token_id, "stopping": "protocol_v2_per_sequence"}
    result = run_diagnostic(bundle, BoundedRunner(base, protocol), runtime, args.output, protocol)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
