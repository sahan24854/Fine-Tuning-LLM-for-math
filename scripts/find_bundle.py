"""Locate one Notebook 01 bundle under Kaggle inputs; support a standalone ZIP."""
import argparse
import json
from pathlib import Path
import stat
import tempfile
import zipfile


def unpack(archive):
    destination = Path(tempfile.mkdtemp(prefix="math-data-bundle-"))
    with zipfile.ZipFile(archive) as source:
        if sum(item.file_size for item in source.infolist()) > 200_000_000:
            raise ValueError("Data bundle ZIP exceeds 200 MB")
        for item in source.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Unsafe path in ZIP")
        source.extractall(destination)
    return destination


def find_bundle(root):
    root = Path(root)
    if root.is_file() and root.suffix == ".zip":
        root = unpack(root)
    candidates = []
    for path in root.rglob("manifest.json"):
        try:
            if json.loads(path.read_text()).get("status") == "data_ready_not_model_evaluated":
                candidates.append(path.parent)
        except (ValueError, OSError):
            continue
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("Multiple data bundles attached. Attach one Notebook 01 output, or set MATH_BUNDLE_PATH explicitly: " + str(candidates))
    archives = list(root.rglob("data-protocol-*.zip"))
    if len(archives) == 1:
        return find_bundle(unpack(archives[0]))
    raise ValueError("No unique Notebook 01 bundle found. Use Kaggle Add Input to attach its saved output (or its ZIP as a dataset).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(find_bundle(args.root))
