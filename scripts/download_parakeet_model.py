#!/usr/bin/env python3
"""Download and extract the Parakeet TDT 0.6B v3 INT8 ONNX model.

Downloads to ~/.cache/vii-assistant/models/parakeet-tdt-0.6b-v3-int8/
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# TODO: Before release, host on a permanent URL or switch to HuggingFace Hub.
MODEL_URL = "https://blob.handy.computer/parakeet-v3-int8.tar.gz"
MODEL_DIR_NAME = "parakeet-tdt-0.6b-v3-int8"
CACHE_BASE = Path.home() / ".cache" / "vii-assistant" / "models"
MODEL_DIR = CACHE_BASE / MODEL_DIR_NAME

EXPECTED_FILES = [
    "nemo128.onnx",
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
]


def main() -> None:
    if MODEL_DIR.exists() and all((MODEL_DIR / f).exists() for f in EXPECTED_FILES):
        print(f"Model already present at {MODEL_DIR}")
        print("To re-download, remove the directory first:")
        print(f"  rm -rf {MODEL_DIR}")
        return

    print(f"Downloading {MODEL_URL} ...")
    CACHE_BASE.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "parakeet-v3-int8.tar.gz"

        result = subprocess.run(
            ["wget", "-O", str(tar_path), MODEL_URL],
            check=False,
        )
        if result.returncode != 0:
            # Fallback to curl
            result = subprocess.run(
                ["curl", "-L", "-o", str(tar_path), MODEL_URL],
                check=True,
            )

        print("Extracting ...")
        # Remove partial dir if it exists
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)

        MODEL_DIR.mkdir(parents=True)
        subprocess.run(
            ["tar", "xzf", str(tar_path), "-C", str(MODEL_DIR), "--strip-components=1"],
            check=True,
        )

    # Verify
    missing = [f for f in EXPECTED_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        print(f"WARNING: Missing files after extraction: {missing}")
        sys.exit(1)

    print(f"Model ready at {MODEL_DIR}")


if __name__ == "__main__":
    main()
