"""Write runs/v10_local_4070/manifest.json (provenance: sha256 + hyperparams)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/v10_local_4070"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    man = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": "local RTX 4070 Laptop 8GB (~6GiB usable)",
        "tasks_sha256": sha256_of(REPO / "eval_protocol/tasks_750.json"),
        "s10_decoder_sha256": sha256_of(
            REPO / "deliver/checkpoints/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"),
        "pool": str(REPO / "weights_local/sd_pool_orig"),
        "bank_sha256": sha256_of(OUT / "bank/kde_bank.pt"),
        "hpsi_sha256": sha256_of(OUT / "bank/hpsi.pt"),
        "dsm_sha256": sha256_of(OUT / "bank/dsm.pt")
        if (OUT / "bank/dsm.pt").exists() else None,
        "hyperparams": {
            "sigma": 1.0, "tau": 2.0, "topk": 4096,
            "per_style_cap": 200_000, "hpsi_epochs": 40,
            "dsm_sigmas": [0.1, 0.3, 0.5, 1.0], "dsm_hidden": 448,
            "optim": {"eps": 0.10, "rho": 0.15, "etas": [1.0, 0.5, 0.25],
                      "content_gate": 1.2, "mask_frac": 0.25},
            "eval_caliber": "test750 full (2026-11-14 protocol correction; "
                            "readout on cross600; no dev preview split)",
        },
        "splits": json.loads((OUT / "splits_report.json").read_text()),
        "note": "latent-cache ablation on N2C cache; NOT end-to-end WEAVE latency",
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print("saved ->", OUT / "manifest.json")


if __name__ == "__main__":
    main()
