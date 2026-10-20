"""Find one completed pilot or full-training artifact in Kaggle folders/ZIPs."""
import argparse
import json
from pathlib import Path
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.find_bundle import scan_input

STATUSES = {"pilot": "pilot_complete", "training": "full_training_complete"}


def unpack_results(path):
    destination = Path(tempfile.mkdtemp(prefix="math-results-"))
    with zipfile.ZipFile(path) as archive:
        items = archive.infolist()
        if len(items) > 20000 or sum(item.file_size for item in items) > 2_000_000_000:
            raise ValueError("Result ZIP exceeds the 2 GB / 20,000-file limit")
        for item in items:
            if not (destination/item.filename).resolve().is_relative_to(destination) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe path in result ZIP")
        archive.extractall(destination)
    return destination


def contains_result(path, status):
    try:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if Path(item.filename).name == "result_manifest.json" and item.file_size <= 10_000_000:
                    manifest = json.loads(archive.read(item))
                    if isinstance(manifest, dict) and manifest.get("status") == status:
                        return True
    except (OSError, ValueError, zipfile.BadZipFile):
        pass
    return False


def find_result(root, kind):
    status = STATUSES[kind]
    root = Path(root)
    if root.is_file():
        root = unpack_results(root) if root.suffix.lower() == ".zip" else root.parent
    files, sample = scan_input(root)
    candidates = {}
    for path in files:
        if path.name == "result_manifest.json":
            try:
                content = path.read_bytes()
                manifest = json.loads(content)
                if isinstance(manifest, dict) and manifest.get("status") == status:
                    # Identical copies of a saved artifact are interchangeable.
                    candidates.setdefault(content, path.parent.resolve())
            except (OSError, ValueError):
                continue
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if len(candidates) > 1:
        raise ValueError(f"Multiple {kind} results found. Set MATH_{'BASELINE' if kind == 'pilot' else 'TRAINING'}_PATH explicitly: {list(candidates.values())}")
    archives = sorted({p.resolve() for p in files if p.suffix.lower() == ".zip" and contains_result(p, status)})
    if len(archives) == 1:
        return find_result(unpack_results(archives[0]), kind)
    if len(archives) > 1:
        raise ValueError(f"Multiple {kind} ZIPs found; select one explicit input path: {archives}")
    raise ValueError(f"No completed {kind} result found under {root}. Attach the saved {'Notebook 02c' if kind == 'pilot' else 'Notebook 03'} output, including its manifest. Visible paths: {sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--kind", choices=STATUSES, required=True)
    args = parser.parse_args()
    try:
        print(find_result(args.root, args.kind))
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Result discovery failed: {exc}\n")
