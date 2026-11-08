"""V8 endpoint data split test: verify train/heldout split and VAE metadata."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.v8.common import STYLES, load_d5_latents  # noqa: E402


def main():
    train, val, meta = load_d5_latents(device="cpu", heldout=100)
    for s in STYLES:
        assert val[s].shape[0] == 100
        assert train[s].shape[0] > 0
        assert train[s].shape[1:] == (4, 64, 64)
    assert meta["scaling_factor"] == 0.18215
    print("V8 DATA SPLIT PASS:", {s: (int(val[s].shape[0]), int(train[s].shape[0])) for s in STYLES})


if __name__ == "__main__":
    main()
