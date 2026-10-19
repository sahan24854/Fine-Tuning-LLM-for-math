"""Create Notebook 01's isolated environment without requiring ensurepip or host pip."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
# Official PyPI pip 25.3 wheel, pinned by version, URL, and SHA-256.
PIP_WHEEL = "pip-25.3-py3-none-any.whl"
PIP_URL = "https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/" + PIP_WHEEL
PIP_SHA256 = "9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd"


def download_pip(directory):
    with urllib.request.urlopen(PIP_URL, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != PIP_SHA256:
        raise ValueError("pip wheel checksum mismatch; refusing to execute it")
    wheel = Path(directory) / PIP_WHEEL
    wheel.write_bytes(data)
    return wheel


def setup(environment):
    environment = Path(environment).resolve()
    if environment.exists() and any(environment.iterdir()):
        raise ValueError("Use a new, empty environment directory; existing environments are not overwritten")
    print(f"Creating isolated environment with Python {sys.version.split()[0]}", flush=True)
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(environment)], check=True)
    python = environment / "bin" / "python"
    # Run verified pip directly from its wheel to install itself offline into the
    # new environment. Neither ensurepip nor the notebook's global pip is needed.
    with tempfile.TemporaryDirectory(prefix="math-pip-bootstrap-") as temporary:
        wheel = download_pip(temporary)
        runner = "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module('pip', run_name='__main__')"
        subprocess.run([str(python), "-I", "-c", runner, str(wheel), "install",
                        "--disable-pip-version-check", "--no-index", "--no-deps", str(wheel)], check=True)
    subprocess.run([str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check",
                    "-r", str(ROOT / "requirements-data.txt")], check=True)
    subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True)
    print(f"Environment ready: {python}", flush=True)
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    setup(args.environment)
