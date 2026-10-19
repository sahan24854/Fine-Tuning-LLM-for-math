"""Locate one Notebook 01 bundle under Kaggle inputs; support a standalone ZIP."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys
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


def scan_input(root):
    """Follow mount symlinks, deduplicate physical directories, and stop cycles."""
    pending, visited, files, sample, errors = [Path(root)], set(), [], [], []
    while pending:
        directory = pending.pop()
        try:
            physical = directory.resolve(strict=True)
            if physical in visited:
                continue
            visited.add(physical)
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda item: item.name):
                    path = Path(entry.path)
                    try:
                        if len(sample) < 30:
                            sample.append(str(path) + (" -> " + os.readlink(path) if entry.is_symlink() else ""))
                        if entry.is_dir(follow_symlinks=True):
                            pending.append(path)
                        elif entry.is_file(follow_symlinks=True):
                            files.append(path)
                        elif entry.is_symlink():
                            errors.append(f"Broken input link: {path}")
                    except OSError as exc:
                        errors.append(f"{path}: {exc}")
        except (OSError, RuntimeError) as exc:
            errors.append(f"{directory}: {exc}")
    if errors:
        raise ValueError("Could not inspect all attached input folders:\n" + "\n".join(errors[:10]))
    return files, sample


def archive_has_bundle(path):
    try:
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if Path(item.filename).name == "manifest.json" and item.file_size <= 1_000_000:
                    manifest = json.loads(archive.read(item))
                    if isinstance(manifest, dict) and manifest.get("status") == "data_ready_not_model_evaluated":
                        return True
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError):
        pass
    return False


def find_bundle(root):
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".zip":
        root = unpack(root)
    elif root.is_file() and root.name == "manifest.json":
        root = root.parent
    files, sample = scan_input(root)
    candidates = []
    rejected = []
    for path in files:
        if path.name != "manifest.json":
            continue
        try:
            manifest = json.loads(path.read_text())
            if isinstance(manifest, dict) and manifest.get("status") == "data_ready_not_model_evaluated":
                candidate = path.parent.resolve()
                if candidate not in candidates:
                    candidates.append(candidate)
            else:
                rejected.append(f"{path}: not a completed Notebook 01 manifest")
        except (ValueError, OSError) as exc:
            rejected.append(f"{path}: cannot read manifest ({exc})")
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("Multiple data bundles attached. Attach one Notebook 01 output, or set MATH_BUNDLE_PATH explicitly: " + str(candidates))
    # Uploaded archives can be renamed by the user or the platform.
    archives = sorted({p.resolve() for p in files if p.suffix.lower() == ".zip" and archive_has_bundle(p)})
    if len(archives) == 1:
        return find_bundle(unpack(archives[0]))
    if len(archives) > 1:
        raise ValueError("Multiple data bundle ZIPs attached; set MATH_BUNDLE_PATH to one: " + str(archives))
    details = "\nVisible input paths (first 30):\n" + ("\n".join(sample) or "(input directory is empty)")
    if rejected:
        details += "\nRejected manifests:\n" + "\n".join(rejected[:10])
    raise ValueError("No completed Notebook 01 bundle found under " + str(root) + ". "
                     "Attach the saved output files (manifest.json, data JSONL files and tokenizer), not only the .ipynb. "
                     "Symlinked folders were included in this search." + details)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    try:
        print(find_bundle(args.root))
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Bundle discovery failed: {exc}\n")
