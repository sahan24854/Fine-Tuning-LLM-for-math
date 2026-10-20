"""Run the bounded three-shot development pilot with the reviewed v2 evaluator."""
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
from math_project.data import sha256
from math_project.evaluation import atomic_json, load_bundle, read_predictions
from math_project.pilot import jobs_from_plan, load_protocol, make_plan, save_plan, write_report
from math_project.protocol_v2 import boundary, repetition_flags, score_target


SOURCE_FILES = ["math_project/data.py", "math_project/evaluation.py", "math_project/inference.py",
                "math_project/inference_v2.py", "math_project/protocol_v2.py", "math_project/pilot.py",
                "scripts/run_pilot.py", "scripts/verify_bundle.py"]
RESULT_FILES = ["selection.json", "identity.json", "source_data_manifest.json", "environment.txt",
                "predictions.jsonl", "summary.json", "metrics.csv", "report.html"]


def validate_records(records, jobs, scoring):
    expected = {job["key"]: job for job in jobs}
    seen = set()
    for record in records:
        key = record["key"]
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected pilot record")
        job = expected[key]
        event = boundary(record["raw_completion"])
        target = record["raw_completion"][:event["cut"]] if event else record["raw_completion"]
        if (record["completion"] != target or
                (event and record["finish_reason"] != event["reason"]) or
                record["finish_reason"] not in ("length", "eos", "final_answer_line", "next_question")):
            raise ValueError("Pilot target boundary changed")
        scores = score_target(target, record["finish_reason"], job["row"]["answer_canonical"], scoring)
        metadata = {"split": job["split"], "mode": job["mode"], "category": job["category"],
                    "question": job["row"]["question"], "prompt_sha256": job["prompt_sha256"],
                    "reference_answer": job["row"]["answer_canonical"],
                    "repetition_flags": repetition_flags(target)}
        if any(record.get(name) != value for name, value in {**scores, **metadata}.items()):
            raise ValueError("Pilot score or metadata changed; refusing resume")
        seen.add(key)
    return seen


def run_pilot(bundle, runner, runtime, output, pilot, scoring):
    plan = make_plan(bundle, pilot, scoring)
    jobs = jobs_from_plan(bundle, plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    identity = {"plan": plan, "runtime": runtime, "model_role": "untouched_base",
                "python": platform.python_version(), "packages_sha256": sha256("\n".join(packages).encode()),
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "code_files_sha256": {name: sha256((ROOT/name).read_bytes()) for name in SOURCE_FILES}}
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is running this pilot") from exc
        save_plan(output, plan)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Pilot resume identity changed; use a new output directory")
        else:
            if (output / "predictions.jsonl").exists():
                raise ValueError("Cannot resume predictions without identity")
            atomic_json(identity_path, identity)
        records = read_predictions(output / "predictions.jsonl")
        seen = validate_records(records, jobs, scoring)
        manifest_path = output / "result_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["prediction_count"] != len(jobs) or len(seen) != len(jobs) or set(manifest["files_sha256"]) != set(RESULT_FILES):
                raise ValueError("Pilot result manifest is incomplete")
            for name, digest in manifest["files_sha256"].items():
                if sha256((output/name).read_bytes()) != digest:
                    raise ValueError("Pilot result checksum mismatch")
            return json.loads((output / "summary.json").read_text())
        atomic_json(output / "source_data_manifest.json", bundle["manifest"])
        (output / "environment.txt").write_text("\n".join(packages)+"\n")
        print(f"Three-shot pilot: {len(seen)}/{len(jobs)} predictions saved", flush=True)
        try:
            with (output / "predictions.jsonl").open("a", encoding="utf-8") as sink:
                batch_size = pilot["batch_size"]
                for offset in range(0, len(jobs), batch_size):
                    batch = jobs[offset:offset+batch_size]
                    if all(j["key"] in seen for j in batch):
                        continue
                    # Regenerate an interrupted batch with its original composition.
                    # Never append records that were already flushed.
                    answers = runner.generate([j["prompt"] for j in batch])
                    if len(answers) != len(batch):
                        raise ValueError("Wrong number of pilot completions")
                    new_records = []
                    for job, answer in zip(batch, answers):
                        if job["key"] in seen:
                            continue
                        reference = job["row"]["answer_canonical"]
                        scores = score_target(answer["completion"], answer["finish_reason"], reference, scoring)
                        record = {**answer, **scores, "key": job["key"], "split": job["split"],
                                  "mode": job["mode"], "category": job["category"],
                                  "question": job["row"]["question"], "reference_answer": reference,
                                  "prompt_sha256": job["prompt_sha256"],
                                  "repetition_flags": repetition_flags(answer["completion"])}
                        new_records.append(record)
                    validate_records(new_records, jobs, scoring)
                    for record in new_records:
                        sink.write(json.dumps(record, ensure_ascii=False)+"\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                    records.extend(new_records)
                    seen.update(r["key"] for r in new_records)
                    write_report(output, records, plan)
                    print(f"Saved {len(seen)}/{len(jobs)} predictions ({batch[-1]['split']})", flush=True)
        finally:
            summary = write_report(output, records, plan)
        atomic_json(manifest_path, {"status": "pilot_complete", "prediction_count": len(records),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "peak_cuda_allocated_bytes_completing_process": runner.torch.cuda.max_memory_allocated()
                if str(getattr(runner, "device", "cpu")).startswith("cuda") else None,
            "files_sha256": {name: sha256((output/name).read_bytes()) for name in RESULT_FILES}})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/pilot_v2.json")
    parser.add_argument("--plan-only", action="store_true", help="Freeze IDs before loading model weights")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    pilot, scoring = load_protocol(args.protocol)
    bundle = load_bundle(args.bundle)
    plan = make_plan(bundle, pilot, scoring)
    save_plan(args.output, plan)
    if args.plan_only:
        print(json.dumps({"predictions": plan["prediction_count"], "scope": pilot["scope"],
            "comparison_contract_sha256": plan["comparison_contract_sha256"],
            "selection": str(args.output/"selection.json")}, indent=2))
        return
    from math_project.inference import load_runner
    from math_project.inference_v2 import BoundedRunner
    base, runtime = load_runner(bundle, args.device)
    runtime = {**runtime, "model_eos_ids": base.eos_ids, "pad_token_id": base.tokenizer.pad_token_id,
               "tokenizer_eos_id": base.tokenizer.eos_token_id, "stopping": "protocol_v2_per_sequence"}
    print(json.dumps(run_pilot(bundle, BoundedRunner(base, scoring), runtime, args.output, pilot, scoring), indent=2))


if __name__ == "__main__":
    main()
