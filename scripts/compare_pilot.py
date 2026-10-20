"""Evaluate the saved full LoRA adapter against the original saved pilot."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_project.comparison import (FROZEN_EVALUATOR_FILES, validate_comparison, verify_inputs,
                                     verify_result, verify_runtime, write_report)
from math_project.data import sha256
from math_project.evaluation import atomic_json, load_bundle, read_predictions
from math_project.protocol_v2 import repetition_flags, score_target
from scripts.run_pilot import validate_records

SOURCE_FILES = [*FROZEN_EVALUATOR_FILES, "math_project/comparison.py", "scripts/compare_pilot.py"]
RESULT_FILES = ("identity.json", "selection.json", "comparison_config.json", "environment.txt",
    "baseline_predictions.jsonl", "baseline_result_manifest.json", "training_result_manifest.json",
    "training_summary.json", "predictions.jsonl", "paired_predictions.jsonl", "summary.json", "metrics.csv", "report.html")


def load_adapter_runner(bundle, inputs, device):
    from math_project.inference import load_runner
    from math_project.inference_v2 import BoundedRunner
    from peft import PeftModel, __version__ as peft_version
    base, runtime = load_runner(bundle, device)
    runtime.update(model_eos_ids=base.eos_ids, pad_token_id=base.tokenizer.pad_token_id,
                   tokenizer_eos_id=base.tokenizer.eos_token_id, stopping="protocol_v2_per_sequence")
    verify_runtime(inputs["baseline_identity"]["runtime"], runtime)
    generation_before = base.model.generation_config.to_dict()
    # Use the original bundle tokenizer and base revision. The adapter is loaded
    # frozen and remains unmerged, with FP32 adapter weights as used in training.
    model = PeftModel.from_pretrained(base.model, inputs["adapter_path"], is_trainable=False,
                                      local_files_only=True, autocast_adapter_dtype=True)
    model.eval()
    model.config.use_cache = True
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Evaluation adapter/base must be frozen")
    if model.generation_config.to_dict() != generation_before:
        raise ValueError("Adapter loading changed base generation settings")
    runtime["adapter_runtime"] = {"peft": peft_version, "merged": False,
        "parameter_dtypes": sorted({str(value.dtype) for name, value in model.named_parameters() if "lora_" in name})}
    base.model = model
    return BoundedRunner(base, inputs["plan"]["scoring_protocol"]), runtime


def run_comparison(inputs, runner, runtime, output, config):
    validate_comparison(config)
    verify_runtime(inputs["baseline_identity"]["runtime"], runtime)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan, jobs = inputs["plan"], inputs["jobs"]
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    identity = {"comparison_config": config, "plan": plan, "runtime": runtime,
        "baseline_artifact_sha256": inputs["baseline_sha256"], "training_artifact_sha256": inputs["training_sha256"],
        "model_role": "validation_selected_lora_adapter", "python": platform.python_version(),
        "packages_sha256": sha256("\n".join(packages).encode()),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_files_sha256": {name: sha256((ROOT/name).read_bytes()) for name in SOURCE_FILES}}
    with (output/".comparison.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is running this comparison") from exc
        path = output/"identity.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("Comparison resume identity changed; use a new output directory")
        else:
            if (output/"predictions.jsonl").exists():
                raise ValueError("Cannot resume comparison predictions without identity")
            atomic_json(path, identity)
        records = read_predictions(output/"predictions.jsonl")
        seen = validate_records(records, jobs, plan["scoring_protocol"])
        if (output/"result_manifest.json").exists():
            manifest, _ = verify_result(output, "comparison_complete", RESULT_FILES)
            if len(seen) != len(jobs) or manifest["prediction_count"] != len(jobs):
                raise ValueError("Completed comparison has missing predictions")
            return json.loads((output/"summary.json").read_text())
        atomic_json(output/"selection.json", plan)
        atomic_json(output/"comparison_config.json", config)
        atomic_json(output/"baseline_result_manifest.json", inputs["baseline_manifest"])
        atomic_json(output/"training_result_manifest.json", inputs["training_manifest"])
        atomic_json(output/"training_summary.json", inputs["training_summary"])
        (output/"environment.txt").write_text("\n".join(packages)+"\n")
        with (output/"baseline_predictions.jsonl").open("w", encoding="utf-8") as sink:
            for record in inputs["baseline_records"]:
                sink.write(json.dumps(record, ensure_ascii=False)+"\n")
        print(f"Saved-baseline paired comparison: {len(seen)}/{len(jobs)} adapter predictions saved", flush=True)
        try:
            with (output/"predictions.jsonl").open("a", encoding="utf-8") as sink:
                batch_size = plan["pilot"]["batch_size"]
                for offset in range(0, len(jobs), batch_size):
                    batch = jobs[offset:offset+batch_size]
                    if all(j["key"] in seen for j in batch):
                        continue
                    answers = runner.generate([j["prompt"] for j in batch])
                    if len(answers) != len(batch):
                        raise ValueError("Wrong number of adapter completions")
                    new_records = []
                    for job, answer in zip(batch, answers):
                        if job["key"] in seen:
                            continue
                        reference = job["row"]["answer_canonical"]
                        scores = score_target(answer["completion"], answer["finish_reason"], reference, plan["scoring_protocol"])
                        new_records.append({**answer, **scores, "key": job["key"], "split": job["split"],
                            "mode": job["mode"], "category": job["category"], "question": job["row"]["question"],
                            "reference_answer": reference, "prompt_sha256": job["prompt_sha256"],
                            "repetition_flags": repetition_flags(answer["completion"])})
                    validate_records(new_records, jobs, plan["scoring_protocol"])
                    for record in new_records:
                        sink.write(json.dumps(record, ensure_ascii=False)+"\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                    records.extend(new_records)
                    seen.update(record["key"] for record in new_records)
                    if len(seen) < len(jobs):
                        write_report(output, records, inputs, config)
                    print(f"Saved {len(seen)}/{len(jobs)} adapter predictions ({batch[-1]['split']})", flush=True)
        finally:
            summary = write_report(output, records, inputs, config)
        atomic_json(output/"result_manifest.json", {"status": "comparison_complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "prediction_count": len(records),
            "peak_cuda_allocated_bytes_completing_process": runner.torch.cuda.max_memory_allocated()
                if str(getattr(runner, "device", "cpu")).startswith("cuda") else None,
            "files_sha256": {name: sha256((output/name).read_bytes()) for name in RESULT_FILES}})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True, help="Notebook 03 full/ directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/comparison_v1.json")
    parser.add_argument("--verify-only", action="store_true", help="Validate saved inputs without model weights or ML imports")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_comparison(config)
    bundle = load_bundle(args.bundle)
    inputs = verify_inputs(bundle, args.baseline, args.training)
    if args.verify_only:
        print(json.dumps({"inputs_verified": True, "predictions": inputs["plan"]["prediction_count"],
            "comparison_contract_sha256": inputs["plan"]["comparison_contract_sha256"],
            "baseline_sha256": inputs["baseline_sha256"], "training_sha256": inputs["training_sha256"],
            "selected_checkpoint": inputs["training_summary"]["best_checkpoint"],
            "base_model": bundle["config"]["model"]}, indent=2))
        return
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    runner, runtime = load_adapter_runner(bundle, inputs, args.device)
    print(json.dumps(run_comparison(inputs, runner, runtime, args.output, config), indent=2))


if __name__ == "__main__":
    main()
