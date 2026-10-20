"""Verified saved artifacts and paired maths comparisons; no ML dependencies."""
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import random
import re

from math_project.data import canonical_number, read_jsonl, sha256
from math_project.evaluation import TEST_SPLITS, atomic_json
from math_project.pilot import group_metrics, jobs_from_plan, load_protocol, make_plan
from scripts.run_pilot import RESULT_FILES, validate_records

ROOT = Path(__file__).resolve().parents[1]
FROZEN_EVALUATOR_FILES = ("math_project/data.py", "math_project/evaluation.py", "math_project/inference.py",
    "math_project/inference_v2.py", "math_project/protocol_v2.py", "math_project/pilot.py",
    "scripts/run_pilot.py", "scripts/verify_bundle.py")


def verify_result(path, status, required):
    path = Path(path)
    manifest_path = path/"result_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest["files_sha256"]
    if manifest["status"] != status or not set(required) <= hashes.keys():
        raise ValueError(f"Incomplete or wrong result artifact: expected {status}")
    for name, digest in hashes.items():
        target = (path/name).resolve()
        if not target.is_relative_to(path.resolve()) or sha256(target.read_bytes()) != digest:
            raise ValueError(f"Result checksum mismatch: {name}")
    return manifest, sha256(manifest_path.read_bytes())


def verify_inputs(bundle, baseline_path, training_path):
    baseline_path, training_path = Path(baseline_path), Path(training_path)
    baseline_manifest, baseline_sha = verify_result(baseline_path, "pilot_complete", RESULT_FILES)
    plan = json.loads((baseline_path/"selection.json").read_text())
    pilot, scoring = load_protocol(ROOT/"configs/pilot_v2.json")
    # Recompute only to verify the saved selection, never to replace it.
    expected_plan = make_plan(bundle, pilot, scoring)
    if plan != expected_plan:
        raise ValueError("Saved pilot selection differs from the original bundle/frozen protocol")
    baseline_identity = json.loads((baseline_path/"identity.json").read_text())
    if baseline_identity["plan"] != plan or baseline_identity["model_role"] != "untouched_base":
        raise ValueError("Baseline identity does not match the saved pilot")
    for name in FROZEN_EVALUATOR_FILES:
        if baseline_identity["code_files_sha256"].get(name) != sha256((ROOT/name).read_bytes()):
            raise ValueError(f"Frozen baseline evaluator code changed: {name}")
    jobs = jobs_from_plan(bundle, plan)
    records = read_jsonl(baseline_path/"predictions.jsonl")  # Never truncate read-only Kaggle inputs.
    seen = validate_records(records, jobs, scoring)
    if len(seen) != len(jobs) or baseline_manifest["prediction_count"] != len(jobs):
        raise ValueError("Baseline pilot is incomplete")
    summary = json.loads((baseline_path/"summary.json").read_text())
    if not summary["execution_complete"] or summary["comparison_contract_sha256"] != plan["comparison_contract_sha256"]:
        raise ValueError("Baseline summary is inconsistent")
    for split in TEST_SPLITS:
        actual = group_metrics([r for r in records if r["split"] == split], sum(j["split"] == split for j in jobs))
        if actual != summary["groups"][split]:
            raise ValueError(f"Baseline metrics differ from raw predictions: {split}")
    training_manifest, training_sha = verify_result(training_path, "full_training_complete",
        ("identity.json", "summary.json", "data_audit.json", "training_config.json",
         "source_data_manifest.json", "adapter/adapter_config.json", "adapter/adapter_model.safetensors"))
    training_identity = json.loads((training_path/"identity.json").read_text())
    trained = json.loads((training_path/"summary.json").read_text())
    config = json.loads((training_path/"training_config.json").read_text())
    if (training_identity["stage"] != "full" or training_identity["bundle_sha256"] != bundle["fingerprint"] or
            training_identity["model"] != bundle["config"]["model"] or training_identity["training_config"] != config or
            trained["stage"] != "full" or not trained["health_passed"] or not trained["adapter_parameters_changed"] or
            trained["bundle_sha256"] != bundle["fingerprint"] or trained["base_model"] != bundle["config"]["model"]):
        raise ValueError("Training artifact is not the full adapter from the original bundle/model")
    if config["checkpoint_selection"] != "lowest_validation_response_token_nll":
        raise ValueError("Training checkpoint was not selected using the declared validation loss")
    audit = json.loads((training_path/"data_audit.json").read_text())
    if (audit["train_ids"] != [r["id"] for r in read_jsonl(bundle["path"]/"train.jsonl")] or
            audit["validation_ids"] != [r["id"] for r in bundle["splits"]["validation"]] or
            trained["train_rows"] != len(audit["train_ids"]) or trained["validation_rows"] != len(audit["validation_ids"])):
        raise ValueError("Training/validation scope differs from the original bundle")
    adapter_config = json.loads((training_path/"adapter/adapter_config.json").read_text())
    if (adapter_config["base_model_name_or_path"] != bundle["config"]["model"]["id"] or
            adapter_config["revision"] != bundle["config"]["model"]["revision"] or
            adapter_config["peft_type"] != "LORA" or adapter_config["task_type"] != "CAUSAL_LM"):
        raise ValueError("Adapter base model, revision, or type differs")
    for name in ("r", "lora_alpha", "lora_dropout", "bias"):
        if adapter_config[name] != config["lora"][name]:
            raise ValueError(f"Exported adapter configuration differs: {name}")
    if set(adapter_config["target_modules"]) != set(config["lora"]["target_modules"]):
        raise ValueError("Exported adapter target modules differ")
    for file in (training_path/"adapter").rglob("*"):
        if file.is_file() and str(file.relative_to(training_path)) not in training_manifest["files_sha256"]:
            raise ValueError("Unverified extra adapter file")
    checkpoint = trained["best_checkpoint"]
    if Path(checkpoint).name != checkpoint or not checkpoint.startswith("checkpoint-"):
        raise ValueError("Invalid selected checkpoint name")
    best_name = f"checkpoints/{checkpoint}/adapter_model.safetensors"
    hashes = training_manifest["files_sha256"]
    if hashes.get(best_name) != hashes["adapter/adapter_model.safetensors"]:
        raise ValueError("Exported adapter weights do not match the selected checkpoint")
    if trained["selected_validation_response_token_nll"] != trained["best_checkpoint_validation_loss"]:
        raise ValueError("Selected adapter validation loss differs from the best checkpoint")
    return {"plan": plan, "jobs": jobs, "baseline_records": records,
        "baseline_identity": baseline_identity, "training_identity": training_identity,
        "training_summary": trained, "adapter_path": training_path/"adapter",
        "baseline_sha256": baseline_sha, "training_sha256": training_sha,
        "baseline_manifest": baseline_manifest, "training_manifest": training_manifest}


def validate_comparison(config):
    if (config["scope"] != "development_comparison_previously_inspected_test_sets" or
            config["supplementary_audit_id"] != "single-arithmetic-equation-evidence-1" or
            config["confidence_level"] != .95 or type(config["paired_bootstrap_resamples"]) is not int or
            not 1 <= config["paired_bootstrap_resamples"] <= 100000 or type(config["bootstrap_seed"]) is not int):
        raise ValueError("Unsupported paired comparison configuration")


def verify_runtime(saved, current):
    # PEFT adds packages; record the full new environment separately. The base
    # inference settings and core framework/hardware identity must still match.
    for name, value in saved.items():
        if current.get(name) != value:
            raise ValueError(f"Baseline/tuned inference runtime differs: {name}; use the original runtime settings")


def arithmetic_evidence(record):
    result = {"official_correct": record["correct"], "single_equation_eligible": False,
              "single_equation_correct": False, "answer_evidence_correct": record["correct"]}
    if record["split"] != "arithmetic_test" or record["finish_reason"] == "length":
        return result
    question = re.fullmatch(r"What is (\d+) ([+*/-]) (\d+)\?", record["question"])
    equation = re.fullmatch(r"\s*(\d+)\s*([+*/-])\s*(\d+)\s*=\s*([+-]?\d+)\s*", record["completion"])
    if question and equation and question.groups() == equation.groups()[:3]:
        result["single_equation_eligible"] = True
        result["single_equation_correct"] = canonical_number(equation[4]) == record["reference_answer"]
        result["answer_evidence_correct"] = record["correct"] or result["single_equation_correct"]
    return result


def paired_stats(pairs, config, stratified=False):
    counts = {"improved": 0, "regressed": 0, "both_correct": 0, "both_incorrect": 0}
    for base, tuned in pairs:
        key = ("both_correct" if base["correct"] else "improved") if tuned["correct"] else (
               "regressed" if base["correct"] else "both_incorrect")
        counts[key] += 1
    n = len(pairs)
    delta = (counts["improved"]-counts["regressed"])/n if n else None
    interval = [None, None]
    if n:
        pools = defaultdict(list)
        for base, tuned in pairs:
            pools[base["category"] if stratified else "all"].append(int(tuned["correct"])-int(base["correct"]))
        rng = random.Random(config["bootstrap_seed"])
        samples = sorted(sum(sum(rng.choices(pool, k=len(pool))) for pool in pools.values())/n
                         for _ in range(config["paired_bootstrap_resamples"]))
        def quantile(p):
            index = p*(len(samples)-1)
            low = math.floor(index)
            return samples[low]+(samples[math.ceil(index)]-samples[low])*(index-low)
        interval = [quantile(.025), quantile(.975)]
    return {**counts, "paired_count": n, "accuracy_change": delta,
            "accuracy_change_percentage_points": 100*delta if delta is not None else None,
            "paired_bootstrap_95": interval}


def write_report(output, records, inputs, config):
    plan = inputs["plan"]
    base_index = {r["key"]: r for r in inputs["baseline_records"]}
    pairs = [(base_index[r["key"]], r) for r in records]
    complete = len(records) == plan["prediction_count"]
    groups = {}
    for split in TEST_SPLITS:
        matched = [(b, t) for b, t in pairs if b["split"] == split]
        expected = sum(j["split"] == split for j in inputs["jobs"])
        # Until complete, compare only paired questions and defer bootstrap work.
        stats = paired_stats(matched, config, split == "arithmetic_test") if complete else {
            "paired_count": len(matched), "paired_bootstrap_95": [None, None], "status": "partial_defer_inference"}
        groups[split] = {"expected": expected,
            "baseline_on_paired_questions": group_metrics([b for b, _ in matched], expected),
            "tuned_on_paired_questions": group_metrics([t for _, t in matched], expected), "paired": stats}
    arithmetic = [(b, t) for b, t in pairs if b["split"] == "arithmetic_test"]
    audits = [(arithmetic_evidence(b), arithmetic_evidence(t)) for b, t in arithmetic]
    supplementary = {"audit_id": config["supplementary_audit_id"], "scope": "exploratory_arithmetic_only",
        "paired_questions": len(arithmetic),
        "baseline_additional_correct_unlabelled_equations": sum(a["single_equation_correct"] and not a["official_correct"] for a, _ in audits),
        "tuned_additional_correct_unlabelled_equations": sum(a["single_equation_correct"] and not a["official_correct"] for _, a in audits),
        "baseline_answer_evidence_correct": sum(a["answer_evidence_correct"] for a, _ in audits),
        "tuned_answer_evidence_correct": sum(a["answer_evidence_correct"] for _, a in audits),
        "answer_evidence_improved": sum(not b["answer_evidence_correct"] and t["answer_evidence_correct"] for b, t in audits),
        "answer_evidence_regressed": sum(b["answer_evidence_correct"] and not t["answer_evidence_correct"] for b, t in audits),
        "both_answer_evidence_correct": sum(b["answer_evidence_correct"] and t["answer_evidence_correct"] for b, t in audits),
        "both_answer_evidence_incorrect": sum(not b["answer_evidence_correct"] and not t["answer_evidence_correct"] for b, t in audits),
        "interpretation": "Exploratory lower-bound answer evidence using the same narrow rule for both models; not a replacement official score or pure reasoning metric. The rule was designed after inspecting the baseline. No arbitrary last-number or reference search."}
    summary = {"stage": "paired_pilot_comparison", "comparison_id": config["comparison_id"],
        "execution_complete": complete, "predictions": len(records), "expected_predictions": plan["prediction_count"],
        "scope": config["scope"], "comparison_contract_sha256": plan["comparison_contract_sha256"],
        "groups": groups, "supplementary_arithmetic": supplementary,
        "training_checkpoint": inputs["training_summary"]["best_checkpoint"],
        "interpretation": "Official numeric accuracy includes invalid answers; parseability measures format compliance. A score gain can include formatting gains. This is development evaluation on previously inspected tests, not an untouched final holdout. Bootstrap intervals are descriptive paired uncertainty, not corrections for development reuse, pretraining contamination, or multiple comparisons.",
        "timing_note": config["timing_policy"]}
    atomic_json(output/"summary.json", summary)
    with (output/"metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dataset", "paired_count", "expected", "baseline_accuracy", "tuned_accuracy", "delta_percentage_points", "paired_ci_low_percentage_points", "paired_ci_high_percentage_points", "improved", "regressed", "baseline_parse_rate", "tuned_parse_rate", "baseline_cap_hits", "tuned_cap_hits"])
        for split, group in groups.items():
            b, t, p = group["baseline_on_paired_questions"], group["tuned_on_paired_questions"], group["paired"]
            writer.writerow([split, p["paired_count"], group["expected"], b["accuracy"], t["accuracy"], p.get("accuracy_change_percentage_points"), *[100*v if v is not None else None for v in p["paired_bootstrap_95"]], p.get("improved"), p.get("regressed"), b["parse_rate"], t["parse_rate"], b["cap_hits"], t["cap_hits"]])
    with (output/"paired_predictions.jsonl").open("w", encoding="utf-8") as file:
        for b, t in pairs:
            file.write(json.dumps({"key": b["key"], "baseline": b, "tuned": t,
                "baseline_arithmetic_audit": arithmetic_evidence(b) if b["split"] == "arithmetic_test" else None,
                "tuned_arithmetic_audit": arithmetic_evidence(t) if t["split"] == "arithmetic_test" else None}, ensure_ascii=False)+"\n")
    document = "<!doctype html><meta charset='utf-8'><title>Paired maths comparison</title><style>body{font:16px system-ui;max-width:1200px;margin:2em auto}td,th{padding:.6em;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:1em}details{margin:1em 0}</style>"
    document += f"<h1>Original model vs LoRA adapter</h1><p>{'Complete' if complete else 'PARTIAL'}: {len(records)}/{plan['prediction_count']} paired predictions.</p>"
    document += "<table><tr><th>Dataset</th><th>Paired</th><th>Original accuracy</th><th>Tuned accuracy</th><th>Change (percentage points)</th><th>95% paired interval (pp)</th><th>Improved / regressed</th><th>Original / tuned parse rate</th></tr>"
    pct = lambda x: f"{100*x:.1f}%" if x is not None else "—"
    for split, g in groups.items():
        b, t, p = g["baseline_on_paired_questions"], g["tuned_on_paired_questions"], g["paired"]
        delta = f"{p['accuracy_change_percentage_points']:+.1f}" if complete else "pending"
        interval = " to ".join(f"{100*v:+.1f}" for v in p["paired_bootstrap_95"]) if complete else "pending"
        document += f"<tr><td>{split}</td><td>{p['paired_count']}/{g['expected']}</td><td>{pct(b['accuracy'])}</td><td>{pct(t['accuracy'])}</td><td>{delta}</td><td>{interval}</td><td>{p.get('improved', '—')} / {p.get('regressed', '—')}</td><td>{pct(b['parse_rate'])} / {pct(t['parse_rate'])}</td></tr>"
    document += "</table><p>"+html.escape(summary["interpretation"])+"</p><p>Paired bootstrap confidence intervals, tokens, cap hits, and times are saved in summary.json/metrics.csv. "+html.escape(config["timing_policy"])+"</p>"
    document += "<h2>Supplementary arithmetic audit — exploratory</h2><pre>"+html.escape(json.dumps(supplementary, indent=2))+"</pre>"
    for b, t in pairs:
        label = "improved" if not b["correct"] and t["correct"] else "regressed" if b["correct"] and not t["correct"] else "unchanged correctness"
        document += f"<details><summary>{html.escape(b['key'])} — {label}</summary><p>{html.escape(b['question'])}</p><p>Reference: {html.escape(b['reference_answer'])}</p>"
        for title, row in (("Original", b), ("LoRA", t)):
            metadata = {name: row[name] for name in ("answer", "correct", "parse_status", "finish_reason", "generated_tokens")}
            document += f"<h3>{title}</h3><pre>{html.escape(json.dumps(metadata, indent=2))}</pre><h4>Scored target</h4><pre>{html.escape(row['completion'])}</pre><h4>Raw text</h4><pre>{html.escape(row['raw_completion'])}</pre>"
        document += "</details>"
    (output/"report.html").write_text(document, encoding="utf-8")
    return summary
