"""Bounded validation-only selection, review flags, and paired reports."""
import csv
from fractions import Fraction
import html
import json
from pathlib import Path
import re

from math_project.comparison import verify_result
from math_project.data import prompt, read_jsonl, sha256
from math_project.evaluation import atomic_json
from math_project.pilot import group_metrics
from math_project.protocol_v2 import validate_protocol

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    if (config["scope"] != "validation_only_development_not_final_test" or
            type(config["questions"]) is not int or not 1 <= config["questions"] <= min(config["max_questions"], 50) or
            config["modes"] != ["zero_shot", "few_shot"] or config["batch_size"] != 4 or type(config["seed"]) is not int):
        raise ValueError("Expected validation-only analysis, both modes, batch four, and at most 50 questions")


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate_config(config)
    path = (ROOT/config["scoring_protocol_file"]).resolve()
    if not path.is_relative_to(ROOT) or sha256(path.read_bytes()) != config["scoring_protocol_file_sha256"]:
        raise ValueError("Reviewed scoring protocol changed")
    scoring = json.loads(path.read_text())
    validate_protocol(scoring)
    return config, scoring


def make_jobs(bundle, config):
    validate_config(config)
    rows = bundle["splits"]["validation"]
    if len(rows) < config["questions"]:
        raise ValueError("Not enough validation rows")
    selected = sorted(rows, key=lambda r: (sha256(json.dumps([config["seed"], r["id"]], separators=(",", ":")).encode()), r["id"]))[:config["questions"]]
    jobs = []
    for mode in config["modes"]:
        for row in selected:
            text = prompt(row["question"], bundle["demonstrations"] if mode == "few_shot" else [])
            jobs.append({"key": f"validation/{mode}/{row['id']}", "split": "validation", "mode": mode,
                "category": None, "row": row, "prompt": text, "prompt_sha256": sha256(text.encode())})
    return jobs


def verify_training(bundle, path):
    path = Path(path)
    manifest, fingerprint = verify_result(path, "full_training_complete", (
        "identity.json", "summary.json", "data_audit.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"))
    identity = json.loads((path/"identity.json").read_text())
    summary = json.loads((path/"summary.json").read_text())
    audit = json.loads((path/"data_audit.json").read_text())
    adapter = json.loads((path/"adapter/adapter_config.json").read_text())
    if (identity["stage"] != "full" or identity["bundle_sha256"] != bundle["fingerprint"] or
            identity["model"] != bundle["config"]["model"] or summary["stage"] != "full" or
            not summary["health_passed"] or not summary["adapter_parameters_changed"] or
            summary["bundle_sha256"] != bundle["fingerprint"] or summary["base_model"] != bundle["config"]["model"] or
            summary["selection_rule"] != "lowest_validation_response_token_nll"):
        raise ValueError("Require the original full validation-selected training artifact")
    if (audit["train_ids"] != [r["id"] for r in read_jsonl(bundle["path"]/"train.jsonl")] or
            audit["validation_ids"] != [r["id"] for r in bundle["splits"]["validation"]]):
        raise ValueError("Training/validation scope differs from the original bundle")
    if (adapter["base_model_name_or_path"] != bundle["config"]["model"]["id"] or
            adapter["revision"] != bundle["config"]["model"]["revision"] or
            adapter["peft_type"] != "LORA" or adapter["task_type"] != "CAUSAL_LM"):
        raise ValueError("Adapter base revision/type differs")
    for file in (path/"adapter").rglob("*"):
        if file.is_file() and str(file.relative_to(path)) not in manifest["files_sha256"]:
            raise ValueError("Unverified adapter file")
    checkpoint = summary["best_checkpoint"]
    if Path(checkpoint).name != checkpoint or manifest["files_sha256"].get(
            f"checkpoints/{checkpoint}/adapter_model.safetensors") != manifest["files_sha256"]["adapter/adapter_model.safetensors"]:
        raise ValueError("Adapter is not the selected checkpoint")
    return {"path": path, "fingerprint": fingerprint, "manifest": manifest, "summary": summary}


NUMBER = r"[+-]?\d+(?:\.\d+)?"
EQUALITY = re.compile(rf"\s*({NUMBER})\s*([+*/×÷−\-])\s*({NUMBER})\s*=\s*({NUMBER})\s*")


def equality_checks(text):
    """Review flags for whole-line literal equalities; never evaluate arbitrary code."""
    checks = []
    for index, line in enumerate(text.split("\n"), 1):
        match = EQUALITY.fullmatch(line)
        if not match:
            continue
        a, operator, b, result = match.groups()
        a, b, result = Fraction(a), Fraction(b), Fraction(result)
        if operator in ("/", "÷") and b == 0:
            status = "undefined_division"
        else:
            actual = a+b if operator == "+" else a-b if operator in ("-", "−") else a*b if operator in ("*", "×") else a/b
            status = "exact_equality" if actual == result else "numeric_equality_mismatch"
        checks.append({"line_number": index, "line": line, "status": status})
    return checks


def paired_counts(base, tuned):
    base = {r["key"]: r for r in base}
    pairs = [(base[r["key"]], r) for r in tuned if r["key"] in base]
    return {"paired": len(pairs), "improved": sum(not b["correct"] and t["correct"] for b, t in pairs),
            "regressed": sum(b["correct"] and not t["correct"] for b, t in pairs),
            "both_correct": sum(b["correct"] and t["correct"] for b, t in pairs),
            "both_incorrect": sum(not b["correct"] and not t["correct"] for b, t in pairs)}


def write_review(output, records_by_model, jobs, config):
    groups = {}
    for model in ("base", "tuned"):
        for mode in config["modes"]:
            rows = [r for r in records_by_model.get(model, []) if r["mode"] == mode]
            metrics = group_metrics(rows, config["questions"])
            metrics["repetition_flagged"] = sum(any(r["repetition_flags"].values()) for r in rows)
            metrics["numeric_equality_review_flagged"] = sum(any(c["status"] != "exact_equality" for c in equality_checks(r["completion"])) for r in rows)
            groups[f"{model}/{mode}"] = metrics
    paired = {mode: paired_counts([r for r in records_by_model.get("base", []) if r["mode"] == mode],
                                  [r for r in records_by_model.get("tuned", []) if r["mode"] == mode]) for mode in config["modes"]}
    complete = all(len(records_by_model.get(model, [])) == len(jobs) for model in ("base", "tuned"))
    summary = {"stage": "validation_error_analysis", "analysis_id": config["analysis_id"],
        "execution_complete": complete, "questions": config["questions"],
        "predictions": sum(len(v) for v in records_by_model.values()), "expected_predictions": 2*len(jobs),
        "groups": groups, "base_vs_tuned_by_mode": paired, "review_required": True,
        "next_step": "Review validation evidence before choosing a controlled second experiment.", "interpretation": config["interpretation"],
        "flag_note": "Literal numeric equality checks are review flags, not reasoning grades. Mismatches may involve rounding or quoted calculations; unsupported expressions are unchecked. Official scores are unchanged."}
    atomic_json(output/"summary.json", summary)
    with (output/"metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "count", "accuracy", "parse_rate", "cap_hits", "repetition_flagged", "numeric_equality_review_flagged"])
        for key, g in groups.items():
            writer.writerow([key, *[g[k] for k in ("count", "accuracy", "parse_rate", "cap_hits", "repetition_flagged", "numeric_equality_review_flagged")]])
    indices = {model:{r["key"]:r for r in rows} for model,rows in records_by_model.items()}
    with (output/"paired_predictions.jsonl").open("w",encoding="utf-8") as file:
        for job in jobs:
            file.write(json.dumps({"key":job["key"],"question":job["row"]["question"],
                "reference_answer":job["row"]["answer_canonical"],"reference_solution":job["row"]["solution"],
                "base":indices.get("base",{}).get(job["key"]),"tuned":indices.get("tuned",{}).get(job["key"])},ensure_ascii=False)+"\n")
    # This is an immutable template. Reviewers annotate a separate copy.
    with (output/"review_template.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "key", "official_correct", "parse_status", "finish_reason", "numeric_equality_flags", "review_category", "notes"])
        for model, rows in records_by_model.items():
            for row in rows:
                flags = [c for c in equality_checks(row["completion"]) if c["status"] != "exact_equality"]
                writer.writerow([model, row["key"], row["correct"], row["parse_status"], row["finish_reason"], json.dumps(flags), "", ""])
    document = "<!doctype html><meta charset='utf-8'><title>Validation error review</title><style>body{font:16px system-ui;max-width:1100px;margin:2em auto}td,th{padding:.6em;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}</style>"
    document += f"<h1>Validation error analysis</h1><p>{summary['predictions']}/{summary['expected_predictions']} predictions. {'Complete' if complete else 'PARTIAL'}; human review required.</p><p>{html.escape(config['interpretation'])}</p>"
    document += "<table><tr><th>Model / mode</th><th>Count</th><th>Accuracy</th><th>Parse rate</th><th>Cap hits</th><th>Repetition</th><th>Equality review flags</th></tr>"
    pct = lambda x: f"{100*x:.1f}%" if x is not None else "—"
    for key,g in groups.items():
        document += f"<tr><td>{key}</td><td>{g['count']}/{g['expected']}</td><td>{pct(g['accuracy'])}</td><td>{pct(g['parse_rate'])}</td><td>{g['cap_hits']}</td><td>{g['repetition_flagged']}</td><td>{g['numeric_equality_review_flagged']}</td></tr>"
    document += "</table><p>"+html.escape(summary["flag_note"])+"</p><h2>Paired base/adapter outcomes</h2><pre>"+html.escape(json.dumps(paired,indent=2))+"</pre>"
    document += "<p>Review categories: wrong operation/quantity, arithmetic error, format failure, repetition/no progress, correct answer with reasoning concern, ambiguous, or other. Annotate a copy of review_template.csv; automatic flags do not determine the cause.</p>"
    indices = {model:{r["key"]:r for r in rows} for model,rows in records_by_model.items()}
    for job in jobs:
        document += f"<details><summary>{html.escape(job['key'])}</summary><p>{html.escape(job['row']['question'])}</p><h3>Reference solution — for review only</h3><pre>{html.escape(job['row']['solution'])}</pre><p>Reference answer: {html.escape(job['row']['answer_canonical'])}</p>"
        for model in ("base", "tuned"):
            row = indices.get(model, {}).get(job["key"])
            if row:
                metadata = {key: row[key] for key in ("answer", "correct", "parse_status", "finish_reason", "generated_tokens")}
                document += f"<h3>{model}</h3><pre>{html.escape(json.dumps(metadata,indent=2))}</pre><pre>{html.escape(row['completion'])}</pre><h4>Raw text</h4><pre>{html.escape(row['raw_completion'])}</pre><h4>Literal equality checks</h4><pre>{html.escape(json.dumps(equality_checks(row['completion']),indent=2))}</pre>"
        document += "</details>"
    (output/"report.html").write_text(document,encoding="utf-8")
    return summary
