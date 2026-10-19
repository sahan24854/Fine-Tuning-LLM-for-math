"""Run notebook 01's preparation pipeline outside the notebook kernel."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from math_project.data import (prepare_data, prompt, sha256, tokenize_and_audit, write_json, write_jsonl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--audit-sources-only", action="store_true",
                        help="Validate actual datasets without HF access; does NOT create baseline-ready artifacts")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Use a new directory; completed runs are immutable.")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    splits, audit = prepare_data(config, args.cache)
    if args.audit_sources_only:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "source_audit.json", {"status": "sources_only_not_baseline_ready", **audit})
        print(json.dumps(audit, indent=2))
        return

    # Import only on the full path. Dataset validation and unit tests need no ML stack.
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    # A supplied commit is immutable. 'main' is resolved once and recorded for later notebooks.
    revision = config["model"]["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = HfApi().model_info(config["model"]["id"], revision=revision).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Could not resolve an immutable model revision")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=revision, trust_remote_code=False)
    splits, demonstrations, excluded = tokenize_and_audit(splits, config, tokenizer, audit)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(args.output / f"{name}.jsonl", rows)
    write_jsonl(args.output / "excluded_training.jsonl", excluded)
    write_json(args.output / "demonstrations.json", demonstrations)
    write_json(args.output / "audit.json", audit)
    write_json(args.output / "config.requested.json", config)
    resolved = {**config, "model": {**config["model"], "revision": revision}}
    write_json(args.output / "config.resolved.json", resolved)
    tokenizer.save_pretrained(args.output / "tokenizer")
    write_json(args.output / "prompt_examples.json", {
        "zero_shot": prompt("What is 2 + 3?"),
        "few_shot": prompt("What is 2 + 3?", demonstrations),
        "note": "Only training demonstrations and a synthetic illustration are shown; no test answers are previewed."})
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    (args.output / "environment.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    hashes = {str(p.relative_to(args.output)): sha256(p.read_bytes())
              for p in sorted(args.output.rglob("*")) if p.is_file()}
    manifest = {"schema_version": 1, "status": "data_ready_not_model_evaluated",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "code_commit": git_sha, "working_tree_dirty": bool(git_status),
                "model": resolved["model"], "config_sha256": sha256(config_bytes),
                "sources": config["sources"], "seed": config["seed"],
                "python": platform.python_version(), "platform": platform.platform(),
                "counts": audit["final_counts"], "files_sha256": hashes,
                "next_step": "Use this bundle unchanged for baseline and adapted-model evaluation; no training or scoring has run."}
    # Manifest is written last and acts as a completion marker.
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"output": str(args.output), "counts": audit["final_counts"],
                      "excluded_training_rows": len(excluded), "code_commit": git_sha,
                      "model_revision": revision, "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
