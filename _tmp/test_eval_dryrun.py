"""Benchmark / dry-run evaluation on one baseline (e.g. cut) to verify speed and metric parity."""
from __future__ import annotations

import csv
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths
import csd_model
from tools.eval_fast import MetricEngine, Timer, build_protocol
from tools.pseudo_gt.matching import encode_dino, load_dino_model
from utils.run_evaluation import _load_eval_image_tensor, to_lpips_input
from tools.v5.route_c import csd_prototypes

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
TEST_DIR = REPO_ROOT / "data" / "test"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load CSD
    t0 = time.time()
    csd, _, _ = csd_model.load_csd(device)
    proto = csd_prototypes()
    dino = load_dino_model(device=device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    print(f"Models loaded in {time.time()-t0:.2f}s")
    
    print("Testing source image load and precomputation...")
    test_images, requests, style_meta = build_protocol(TEST_DIR, max_src_samples=30, ref_pool="first")
    print(f"Loaded protocol: {len(test_images)} styles, {len(requests)} requests")
    
if __name__ == "__main__":
    main()
