"""Generate bounded validation diagnostics for the original and selected adapter."""
import argparse
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
from math_project.comparison import verify_result, verify_runtime
from math_project.data import sha256
from math_project.evaluation import atomic_json, load_bundle, read_predictions
from math_project.protocol_v2 import repetition_flags, score_target
from math_project.validation_analysis import load_config, make_jobs, verify_training, write_review
from scripts.run_pilot import validate_records
from scripts.train_lora import file_hashes

SOURCES = ("math_project/data.py", "math_project/evaluation.py", "math_project/inference.py",
           "math_project/inference_v2.py", "math_project/protocol_v2.py", "math_project/validation_analysis.py",
           "math_project/comparison.py", "math_project/pilot.py", "scripts/verify_bundle.py",
           "scripts/train_lora.py", "scripts/run_pilot.py", "scripts/compare_pilot.py", "scripts/analyze_validation.py")
STAGE_FILES = ("identity.json", "predictions.jsonl", "environment.txt")


def initialize(output, bundle, training, config, scoring, jobs):
    output.mkdir(parents=True, exist_ok=True)
    selection = [{"key": j["key"], "prompt_sha256": j["prompt_sha256"],
                  "question_sha256": sha256(j["row"]["question"].encode()),
                  "reference_answer_sha256": sha256(j["row"]["answer_canonical"].encode()),
                  "reference_solution_sha256": sha256(j["row"]["solution"].encode())} for j in jobs]
    identity = {"config": config, "scoring": scoring, "selection": selection,
                "bundle_sha256": bundle["fingerprint"], "base_model": bundle["config"]["model"],
                "training_artifact_sha256": training["fingerprint"],
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,text=True).strip(),
                "code_files_sha256": {name: sha256((ROOT/name).read_bytes()) for name in SOURCES}}
    path = output/"analysis_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Validation analysis identity changed; use a new output directory")
        if json.loads((output/"selection.json").read_text()) != selection or json.loads((output/"training_summary.json").read_text()) != training["summary"]:
            raise ValueError("Saved validation selection or training summary changed")
    else:
        if (output/"base").exists() or (output/"tuned").exists():
            raise ValueError("Cannot recover analysis predictions without identity")
        atomic_json(path, identity)
        atomic_json(output/"selection.json", selection)
        atomic_json(output/"training_summary.json", training["summary"])
    return identity


def read_stage(output, role, jobs, scoring, require_complete=False):
    folder = output/role
    if (folder/"result_manifest.json").exists():
        manifest, _ = verify_result(folder, f"validation_{role}_complete", STAGE_FILES)
        rows = read_predictions(folder/"predictions.jsonl")
        if manifest["prediction_count"] != len(jobs) or len(rows) != len(jobs):
            raise ValueError("Validation stage count mismatch")
    elif require_complete:
        raise ValueError(f"Complete the {role} validation stage first")
    else:
        rows = read_predictions(folder/"predictions.jsonl")
    validate_records(rows, jobs, scoring)
    return rows


def run_stage(bundle, training, config, scoring, output, role, runner, runtime):
    if role not in ("base", "tuned"):
        raise ValueError("Unknown model role")
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    jobs = make_jobs(bundle,config)
    with (output/".analysis.lock").open("w") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another validation-analysis process is active") from exc
        common = initialize(output,bundle,training,config,scoring,jobs)
        packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
        identity = {"analysis_identity_sha256": sha256(json.dumps(common,sort_keys=True).encode()),
                    "role": role,"runtime": runtime,"python": platform.python_version(),
                    "packages_sha256": sha256("\n".join(packages).encode())}
        if role == "tuned":
            read_stage(output,"base",jobs,scoring,require_complete=True)
            source = json.loads((output/"base/identity.json").read_text())
            verify_runtime(source["runtime"],runtime)
            if any(source[name] != identity[name] for name in ("python","packages_sha256","analysis_identity_sha256")):
                raise ValueError("Base/tuned validation environments or inputs differ")
        folder = output/role
        folder.mkdir(exist_ok=True)
        path = folder/"identity.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError("Validation-stage resume identity changed")
        else:
            if (folder/"predictions.jsonl").exists():
                raise ValueError("Cannot resume predictions without model identity")
            atomic_json(path,identity)
        records = read_stage(output,role,jobs,scoring)
        if (folder/"result_manifest.json").exists():
            return {"role":role,"predictions":len(records),"complete":True}
        (folder/"environment.txt").write_text("\n".join(packages)+"\n")
        seen = {r["key"] for r in records}
        print(f"Validation {role}: {len(seen)}/{len(jobs)} predictions saved",flush=True)
        try:
            with (folder/"predictions.jsonl").open("a",encoding="utf-8") as sink:
                for offset in range(0,len(jobs),config["batch_size"]):
                    batch = jobs[offset:offset+config["batch_size"]]
                    if all(j["key"] in seen for j in batch):
                        continue
                    answers = runner.generate([j["prompt"] for j in batch])
                    if len(answers) != len(batch):
                        raise ValueError("Wrong number of validation completions")
                    fresh=[]
                    for j,a in zip(batch,answers):
                        if j["key"] in seen:
                            continue
                        reference=j["row"]["answer_canonical"]
                        fresh.append({**a,**score_target(a["completion"],a["finish_reason"],reference,scoring),
                            "key":j["key"],"split":"validation","mode":j["mode"],"category":None,
                            "question":j["row"]["question"],"reference_answer":reference,
                            "prompt_sha256":j["prompt_sha256"],"repetition_flags":repetition_flags(a["completion"])})
                    validate_records(fresh,jobs,scoring)
                    for row in fresh:
                        sink.write(json.dumps(row,ensure_ascii=False)+"\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                    records.extend(fresh)
                    seen.update(r["key"] for r in fresh)
                    print(f"Saved {len(seen)}/{len(jobs)} {role} validation predictions",flush=True)
        finally:
            models={role:records}
            other="tuned" if role=="base" else "base"
            models[other]=read_stage(output,other,jobs,scoring)
            write_review(output,models,jobs,config)
        atomic_json(folder/"result_manifest.json",{"status":f"validation_{role}_complete",
            "prediction_count":len(records),"files_sha256":{name:sha256((folder/name).read_bytes()) for name in STAGE_FILES}})
        return {"role":role,"predictions":len(records),"complete":True}


def finalize(bundle,training,config,scoring,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/".analysis.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs=make_jobs(bundle,config)
        common=initialize(output,bundle,training,config,scoring,jobs)
        if (output/"result_manifest.json").exists():
            verify_result(output,"validation_analysis_complete",("summary.json","report.html","review_template.csv","paired_predictions.jsonl"))
            return json.loads((output/"summary.json").read_text())
        rows={role:read_stage(output,role,jobs,scoring,require_complete=True) for role in ("base","tuned")}
        identities={role:json.loads((output/role/"identity.json").read_text()) for role in rows}
        expected=sha256(json.dumps(common,sort_keys=True).encode())
        if any(i["analysis_identity_sha256"] != expected for i in identities.values()):
            raise ValueError("Model-stage identity differs from analysis")
        verify_runtime(identities["base"]["runtime"],identities["tuned"]["runtime"])
        summary=write_review(output,rows,jobs,config)
        atomic_json(output/"result_manifest.json",{"status":"validation_analysis_complete",
            "prediction_count":summary["predictions"],
            "files_sha256":file_hashes(output,exclude=(".analysis.lock","result_manifest.json"))})
        return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--training",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/validation_analysis.json")
    parser.add_argument("--stage",choices=("verify","base","tuned","report"),required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    config,scoring=load_config(args.config)
    bundle=load_bundle(args.bundle)
    jobs=make_jobs(bundle,config)
    training=verify_training(bundle,args.training)
    if args.stage=="verify":
        print(json.dumps({"questions":config["questions"],"predictions_per_model":len(jobs),
            "total_predictions":2*len(jobs),"bundle_sha256":bundle["fingerprint"],
            "training_sha256":training["fingerprint"],"checkpoint":training["summary"]["best_checkpoint"],
            "interpretation":config["interpretation"]},indent=2))
        return
    if args.stage=="report":
        print(json.dumps(finalize(bundle,training,config,scoring,args.output),indent=2))
        return
    if args.device=="cuda":
        os.environ["CUDA_VISIBLE_DEVICES"]=os.environ.get("CUDA_VISIBLE_DEVICES","0").split(",")[0]
    if args.stage=="base":
        from math_project.inference import load_runner
        from math_project.inference_v2 import BoundedRunner
        base,runtime=load_runner(bundle,args.device)
        runtime.update(model_eos_ids=base.eos_ids,pad_token_id=base.tokenizer.pad_token_id,
            tokenizer_eos_id=base.tokenizer.eos_token_id,stopping="protocol_v2_per_sequence")
        runner=BoundedRunner(base,scoring)
    else:
        # Verify the base stage before loading adapter weights.
        read_stage(args.output,"base",jobs,scoring,require_complete=True)
        base_identity=json.loads((args.output/"base/identity.json").read_text())
        from scripts.compare_pilot import load_adapter_runner
        inputs={"adapter_path":training["path"]/"adapter","plan":{"scoring_protocol":scoring},
                "baseline_identity":base_identity}
        runner,runtime=load_adapter_runner(bundle,inputs,args.device)
    print(json.dumps(run_stage(bundle,training,config,scoring,args.output,args.stage,runner,runtime),indent=2))


if __name__=="__main__":
    main()
