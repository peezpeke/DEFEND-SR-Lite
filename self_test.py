import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    inp = root / "input"
    out = root / "output"
    inp.mkdir()

    a = np.random.default_rng(0).normal(0.5, 0.2, (128, 128)).astype(np.float32)
    a[0, 0] = np.nan
    a[0, 1] = np.inf
    np.save(inp / "sample.npy", a)

    subprocess.check_call([sys.executable, str(Path(__file__).parent / "run.py"), str(inp), str(out)])
    y = np.load(out / "sample.npy")

    assert y.shape == (256, 256), y.shape
    assert y.dtype == np.float32, y.dtype
    assert np.isfinite(y).all()
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0

print("SELF-TEST PASSED")
