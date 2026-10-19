"""Verify a notebook 01 bundle before using it in any later experiment."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["status"] != "data_ready_not_model_evaluated":
        raise ValueError("Incomplete or unsupported data bundle")
    for relative, expected in manifest["files_sha256"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError("Manifest path escapes bundle")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle)
    print(f"Verified {len(result['files_sha256'])} files; code commit {result['code_commit']}")
