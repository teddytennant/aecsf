"""train.py smoke: exits 0 and prints a number."""

import subprocess
import sys
from pathlib import Path


def test_train_script_prints_a_number():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "train.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout + proc.stderr
    assert any(ch.isdigit() for ch in out), out
