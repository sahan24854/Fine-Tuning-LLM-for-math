"""Evaluate the untouched model; persist and resume predictions after every batch."""
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
from math_project.data import extract_final_answer, sha256
from math_project.evaluation import atomic_json, load_bundle, make_jobs, read_predictions, smoke_health_issues, validate_records, write_reports


def code_identity():
    files = ["math_project/data.py", "math_project/evaluation.py", "math_project/inference.py",
             "scripts/run_baseline.py", "scripts/verify_bundle.py"]
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "files_sha256": {name: sha256((ROOT / name).read_bytes()) for name in files}}


def evaluate(bundle, runner, runtime, output, stage, batch_size=4, smoke_count=20):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(bundle, stage, smoke_count)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    identity = {"schema_version": 1, "bundle_sha256": bundle["fingerprint"], "model": bundle["config"]["model"],
                "code": code_identity(), "runtime": runtime, "python": platform.python_version(),
                "stage": stage, "batch_size": batch_size, "packages_sha256": sha256("\n".join(packages).encode()),
                "job_keys_sha256": sha256(json.dumps([j["key"] for j in jobs]).encode()),
                "generation": bundle["config"]["generation"], "seed": bundle["config"]["seed"]}
    # OS releases this advisory lock even after a process crash.
    with (output / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this run") from exc
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Resume identity mismatch (data/code/hardware/settings). Use the original settings or a new output directory.")
        else:
            if (output / "predictions.jsonl").exists():
                raise ValueError("Predictions exist without run identity")
            atomic_json(identity_path, identity)
        records = read_predictions(output / "predictions.jsonl")
        seen = validate_records(records, jobs)
        print(f"{stage}: {len(seen)}/{len(jobs)} predictions already saved", flush=True)
        had_pending = len(seen) < len(jobs)
        if not had_pending and (output / "result_manifest.json").exists():
            completed = json.loads((output / "result_manifest.json").read_text())
            for name, digest in completed["files_sha256"].items():
                target = (output / name).resolve()
                if not target.is_relative_to(output.resolve()) or sha256(target.read_bytes()) != digest:
                    raise ValueError("Completed output checksum mismatch")
            # Keep original timing, memory statistics, and completion timestamp.
            return json.loads((output / "summary.json").read_text())
        (output / "environment.txt").write_text("\n".join(packages) + "\n")
        atomic_json(output / "data_manifest.json", bundle["manifest"])
        atomic_json(output / "config.resolved.json", bundle["config"])
        if len(seen) < len(jobs):
            (output / "result_manifest.json").unlink(missing_ok=True)
            # Warm up on validation, never test answers; exclude from timing and scores.
            from math_project.data import prompt
            runner.generate([prompt(bundle["splits"]["validation"][0]["question"])], max_new_tokens=8)
        try:
            with (output / "predictions.jsonl").open("a", encoding="utf-8") as sink:
                for offset in range(0, len(jobs), batch_size):
                    batch = jobs[offset:offset+batch_size]
                    if all(j["key"] in seen for j in batch):
                        continue
                    # Replay the original complete batch if a crash interrupted its write.
                    # This preserves padding and batch composition for resumed predictions.
                    answers = runner.generate([j["prompt"] for j in batch])
                    if len(answers) != len(batch):
                        raise ValueError("Model returned wrong number of completions")
                    for job, answer in zip(batch, answers):
                        if job["key"] in seen:
                            continue
                        extracted = extract_final_answer(answer["completion"])
                        category = None
                        if job["split"] == "arithmetic_test":
                            category = f"{job['row']['number_range']}/{job['row']['operation']}"
                        record = {**answer, "key": job["key"], "id": job["row"]["id"],
                                  "split": job["split"], "mode": job["mode"], "category": category,
                                  "prompt_sha256": job["prompt_sha256"],
                                  "reference_answer": job["row"]["answer_canonical"],
                                  "extracted_answer": extracted,
                                  "correct": extracted == job["row"]["answer_canonical"]}
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records.append(record)
                        seen.add(job["key"])
                    sink.flush()
                    os.fsync(sink.fileno())
                    if offset // batch_size % 10 == 0 or len(seen) == len(jobs):
                        print(f"Saved {len(seen)}/{len(jobs)} predictions ({job['split']}/{job['mode']})", flush=True)
                        write_reports(output, records, jobs, stage)
        finally:
            summary = write_reports(output, records, jobs, stage)
        if not summary["complete"]:
            raise RuntimeError("Incomplete evaluation; predictions remain available for resume")
        device = getattr(runner, "device", "cpu")
        peak = None
        if had_pending and str(device).startswith("cuda"):
            peak = runner.torch.cuda.max_memory_allocated()
        files = ["predictions.jsonl", "summary.json", "metrics.csv", "report.html", "identity.json",
                 "environment.txt", "data_manifest.json", "config.resolved.json"]
        atomic_json(output / "result_manifest.json", {
            "schema_version": 1, "status": "baseline_complete" if stage == "full" else "validation_smoke_complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "prediction_count": len(records), "peak_cuda_allocated_bytes": peak,
            "peak_cuda_memory_scope": "Completing process only; earlier interrupted sessions are not measured here. Null if no inference ran in this process.",
            "files_sha256": {name: sha256((output / name).read_bytes()) for name in files}})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke-count", type=int, default=20)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--smoke-output", type=Path, help="Required for full evaluation")
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    if args.stage == "full":
        if args.smoke_output is None:
            parser.error("Run validation smoke first and supply --smoke-output")
        smoke = json.loads((args.smoke_output / "summary.json").read_text())
        smoke_id = json.loads((args.smoke_output / "identity.json").read_text())
        smoke_manifest = json.loads((args.smoke_output / "result_manifest.json").read_text())
        for name, digest in smoke_manifest["files_sha256"].items():
            target = (args.smoke_output / name).resolve()
            if not target.is_relative_to(args.smoke_output.resolve()):
                raise ValueError("Smoke manifest path escapes output directory")
            if sha256(target.read_bytes()) != digest:
                raise ValueError("Smoke output checksum mismatch")
        if (not smoke["complete"] or smoke["stage"] != "smoke" or
                smoke_id["bundle_sha256"] != bundle["fingerprint"] or
                smoke_id["code"] != code_identity() or smoke_id["batch_size"] != args.batch_size):
            raise ValueError("Smoke output is incomplete or belongs to different data/code/settings")
        packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        if smoke_id["packages_sha256"] != sha256("\n".join(packages).encode()):
            raise ValueError("Installed packages changed since smoke test; use a consistent environment")
        issues = smoke_health_issues(smoke)
        if issues:
            raise ValueError("Full benchmark blocked by validation smoke health checks:\n" +
                             "\n".join(issues) + "\nInspect saved validation completions and termination before running a full benchmark. "
                             "These alarms do not prove the generation budget is too small; do not simply increase it.")
    from math_project.inference import load_runner
    runner, runtime = load_runner(bundle, args.device)
    if args.stage == "full" and smoke_id["runtime"] != runtime:
        raise ValueError("Hardware/runtime changed since smoke test; rerun smoke in a new output directory")
    result = evaluate(bundle, runner, runtime, args.output, args.stage, args.batch_size, args.smoke_count)
    print(json.dumps({"stage": args.stage, "complete": result["complete"],
                      "predictions": result["completed_predictions"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
