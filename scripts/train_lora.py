"""Train and select a LoRA adapter using only original training/validation data."""
import argparse
from datetime import datetime, timezone
import fcntl
import html
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_project.data import sha256
from math_project.evaluation import atomic_json, load_bundle
from math_project.training import (ResponseCollator, datasets_for_training, response_loss,
                                   trainer_class, validate_config)

SOURCE_FILES = ["math_project/data.py", "math_project/evaluation.py", "math_project/inference.py",
                "math_project/training.py", "scripts/train_lora.py", "scripts/verify_bundle.py"]


def file_hashes(directory, exclude=()):
    return {str(p.relative_to(directory)): sha256(p.read_bytes()) for p in sorted(directory.rglob("*"))
            if p.is_file() and str(p.relative_to(directory)) not in exclude}


def verify_files(directory, manifest_name):
    directory = Path(directory)
    manifest = json.loads((directory/manifest_name).read_text())
    for name, digest in manifest["files_sha256"].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path.read_bytes()) != digest:
            raise ValueError(f"Training artifact checksum mismatch: {name}")
    return manifest


def verify_smoke(path, bundle, config, identity=None):
    manifest = verify_files(path, "result_manifest.json")
    summary = json.loads((Path(path)/"summary.json").read_text())
    source_identity = json.loads((Path(path)/"identity.json").read_text())
    if (manifest["status"] != "smoke_training_complete" or summary["stage"] != "smoke" or
            not summary["health_passed"] or source_identity["bundle_sha256"] != bundle["fingerprint"] or
            source_identity["training_config"] != config):
        raise ValueError("Full training requires a successful smoke run with the same bundle/config")
    if identity is not None:
        for name in ("runtime", "packages_sha256", "python", "code_commit", "code_files_sha256"):
            if source_identity[name] != identity[name]:
                raise ValueError(f"Smoke/full training identity differs: {name}")
    return summary


def run_training(bundle, base, runtime, config, output, stage, smoke_output=None):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    validate_config(config)
    train_rows, validation_rows, audit = datasets_for_training(bundle, base.tokenizer, config, stage)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    identity = {"training_config": config, "stage": stage, "bundle_sha256": bundle["fingerprint"],
        "model": bundle["config"]["model"], "runtime": runtime, "python": platform.python_version(),
        "packages_sha256": sha256("\n".join(packages).encode()),
        "data_audit_sha256": sha256(json.dumps(audit, sort_keys=True).encode()),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_files_sha256": {name: sha256((ROOT/name).read_bytes()) for name in SOURCE_FILES}}
    if stage == "full":
        if smoke_output is None:
            raise ValueError("Full training requires --smoke-output")
        verify_smoke(smoke_output, bundle, config, identity)
    with (output/".training.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is training in this directory") from exc
        identity_path = output/"identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Training resume identity changed; use a new output directory")
        else:
            if (output/"checkpoints").exists() or (output/"adapter").exists():
                raise ValueError("Cannot recover a training run without identity")
            atomic_json(identity_path, identity)
        if (output/"result_manifest.json").exists():
            verify_files(output, "result_manifest.json")
            return json.loads((output/"summary.json").read_text())
        atomic_json(output/"training_config.json", config)
        atomic_json(output/"data_audit.json", audit)
        atomic_json(output/"source_data_manifest.json", bundle["manifest"])
        (output/"environment.txt").write_text("\n".join(packages)+"\n")
        set_seed(config["seed"])
        model = get_peft_model(base.model, LoraConfig(task_type=TaskType.CAUSAL_LM,
            revision=bundle["config"]["model"]["revision"], **config["lora"]))
        model.config.use_cache = False
        trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
        if not trainable or any("lora_" not in name for name in trainable):
            raise ValueError("Only LoRA parameters may be trainable")
        if any(value.dtype != torch.float32 for value in trainable.values()):
            raise ValueError("LoRA optimizer parameters must remain FP32 for stable mixed precision")
        initial = {name: value.detach().cpu().clone() for name, value in trainable.items()}
        model.print_trainable_parameters()
        checkpoints = output/"checkpoints"
        previous = get_last_checkpoint(str(checkpoints)) if checkpoints.exists() else None
        if previous:
            verify_files(Path(previous), "checkpoint_manifest.json")
            saved_best = json.loads((Path(previous)/"trainer_state.json").read_text()).get("best_model_checkpoint")
            if saved_best and Path(saved_best) != Path(previous):
                verify_files(Path(saved_best), "checkpoint_manifest.json")
        cuda = str(base.device).startswith("cuda")
        args = TrainingArguments(output_dir=str(checkpoints),
            num_train_epochs=config["epochs"], max_steps=config["smoke"]["max_updates"] if stage == "smoke" else -1,
            per_device_train_batch_size=config["micro_batch_size"], per_device_eval_batch_size=config["eval_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
            warmup_ratio=config["warmup_ratio"], lr_scheduler_type="linear", optim="adamw_torch",
            max_grad_norm=config["max_grad_norm"], fp16=cuda, bf16=False, use_cpu=not cuda, tf32=False,
            gradient_checkpointing=config["gradient_checkpointing"],
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="steps", save_strategy="steps", eval_steps=config["checkpoint_every_updates"],
            save_steps=config["checkpoint_every_updates"], save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            logging_steps=10 if stage == "full" else 1, logging_nan_inf_filter=False,
            report_to="none", seed=config["seed"], data_seed=config["seed"], full_determinism=True,
            dataloader_num_workers=0, dataloader_pin_memory=cuda, remove_unused_columns=False,
            label_names=["labels"], prediction_loss_only=True, save_safetensors=True, disable_tqdm=True)
        if args.world_size != 1 or args.n_gpu > 1:
            raise ValueError("This first experiment supports one process and one visible GPU")

        class CheckpointContract(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                # Include the final model and epoch boundaries in selection, even
                # when total updates do not divide the checkpoint interval.
                if not any(row.get("step") == state.global_step and "eval_loss" in row
                           for row in state.log_history[-3:]):
                    control.should_evaluate = control.should_save = True
                return control

            def on_save(self, args, state, control, **kwargs):
                path = Path(args.output_dir)/f"checkpoint-{state.global_step}"
                atomic_json(path/"checkpoint_manifest.json", {
                    "files_sha256": file_hashes(path, exclude=("checkpoint_manifest.json",))})

        trainer = trainer_class()(model=model, args=args, train_dataset=train_rows,
            eval_dataset=validation_rows, data_collator=ResponseCollator(base.tokenizer.pad_token_id),
            processing_class=base.tokenizer, compute_loss_func=response_loss, callbacks=[CheckpointContract()])
        # Persist the base-model validation reference. Resume must not replace it
        # with a value measured after training has already begun.
        base_path = output/"base_validation.json"
        if base_path.exists():
            base_validation = json.loads(base_path.read_text())
        else:
            base_validation = trainer.evaluate(metric_key_prefix="base_validation")
            atomic_json(base_path, base_validation)
        result = trainer.train(resume_from_checkpoint=previous)
        selected = trainer.evaluate(metric_key_prefix="selected_validation")
        changed = any(not torch.equal(initial[name], value.detach().cpu()) for name, value in trainable.items())
        finite = all(bool(torch.isfinite(value).all()) for value in trainable.values())
        if not changed or not finite or not math.isfinite(result.training_loss):
            raise ValueError("Training health failed: require finite loss/adapter parameters and an actual LoRA update")
        if trainer.state.best_model_checkpoint is None:
            raise ValueError("No validation-selected checkpoint was saved")
        model.save_pretrained(output/"adapter", safe_serialization=True)
        base.tokenizer.save_pretrained(output/"tokenizer")
        trainer.save_state()
        history = trainer.state.log_history
        atomic_json(output/"training_history.json", history)
        summary = {"stage": stage, "status": "trained_adapter_not_test_evaluated", "health_passed": True,
            "base_model": bundle["config"]["model"], "bundle_sha256": bundle["fingerprint"],
            "train_rows": len(train_rows), "validation_rows": len(validation_rows),
            "optimizer_updates": trainer.state.global_step, "training_loss": result.training_loss,
            "base_validation_response_token_nll": base_validation["base_validation_loss"],
            "selected_validation_response_token_nll": selected["selected_validation_loss"],
            "selection_rule": config["checkpoint_selection"],
            "best_checkpoint": Path(trainer.state.best_model_checkpoint).name,
            "best_checkpoint_validation_loss": trainer.state.best_metric,
            "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "adapter_parameters_changed": changed, "effective_batch_size": audit["effective_batch_size"],
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "interpretation": "Validation teacher-forced response loss selects a checkpoint; it is not generated maths accuracy. The smoke adapter is only an engineering check. Test accuracy and paired pilot comparison belong to a later notebook."}
        atomic_json(output/"summary.json", summary)
        document = "<!doctype html><meta charset='utf-8'><title>LoRA training</title><style>body{font:16px system-ui;max-width:1000px;margin:2em auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        document += f"<h1>LoRA training — {stage}</h1><p>"+html.escape(summary["interpretation"])+"</p><pre>"+html.escape(json.dumps(summary, indent=2))+"</pre>"
        document += "<h2>Validation history</h2><table><tr><th>Update</th><th>Epoch</th><th>Response-token NLL</th></tr>"
        for row in history:
            if "eval_loss" in row:
                document += f"<tr><td>{row['step']}</td><td>{row.get('epoch')}</td><td>{row['eval_loss']:.6f}</td></tr>"
        (output/"report.html").write_text(document+"</table>", encoding="utf-8")
        atomic_json(output/"result_manifest.json", {"status": f"{stage}_training_complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "files_sha256": file_hashes(output, exclude=(".training.lock", "result_manifest.json"))})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/lora_training.json")
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--audit-only", action="store_true", help="Encode and verify data without loading model weights")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    bundle = load_bundle(args.bundle)
    if args.audit_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bundle["path"]/"tokenizer", local_files_only=True, trust_remote_code=False)
        _, _, audit = datasets_for_training(bundle, tokenizer, config, args.stage)
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output/"data_audit.json", audit)
        print(json.dumps(audit, indent=2))
        return
    if args.stage == "full":
        if args.smoke_output is None:
            parser.error("Full training requires --smoke-output")
        verify_smoke(args.smoke_output, bundle, config)
    # Trainer otherwise uses both Kaggle T4s through DataParallel. This first
    # experiment explicitly uses the first visible GPU and effective batch 32.
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    from math_project.inference import load_runner
    base, runtime = load_runner(bundle, args.device)
    print(json.dumps(run_training(bundle, base, runtime, config, args.output, args.stage, args.smoke_output), indent=2))


if __name__ == "__main__":
    main()
