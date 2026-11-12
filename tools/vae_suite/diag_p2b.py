"""WEAVE-VAE V2 Diagnosis: P2.2/P2.3 only (gradient attribution, lean)."""
import sys
sys.path.insert(0, "/root/Workspace/WEAVE_VAE")

import json
from pathlib import Path
import numpy as np
import torch

from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.diag_p2_helpers import (build_desc, load_model,
                                             p2_2_gradient_conflict,
                                             p2_3_counterfactual_updates)

OUT_DIR = Path("/root/Workspace/WEAVE_VAE/runs_vae/v2_diagnosis")
MANIFEST = Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json")


def main():
    device = "cuda"
    desc_gpu, _ = build_desc()
    desc_gpu = desc_gpu.to(device)
    dataset = RawMemmapWikiArtDataset(MANIFEST, is_training=True)

    vae = load_model("checkpoint_step_001800.pt").to(device)
    gc = p2_2_gradient_conflict(vae, desc_gpu, dataset, device)
    del vae
    torch.cuda.empty_cache()

    cf = p2_3_counterfactual_updates(desc_gpu, dataset, device)

    (OUT_DIR / "04_gradient_conflict.json").write_text(json.dumps(gc, indent=2), encoding="utf-8")
    (OUT_DIR / "05_counterfactual_updates.json").write_text(json.dumps(cf, indent=2), encoding="utf-8")
    print("[Saved] 04 + 05")


if __name__ == "__main__":
    main()
