import json
import re
from pathlib import Path

REPO_ROOT = Path("/mnt/i/Latent_Style/WEAVE")
RESULTS_DIR = REPO_ROOT / "results" / "D5-512"
TEST_DIR = REPO_ROOT / "data" / "test"

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]

# Index all source test images
test_images = {}
for s in STYLES:
    sdir = TEST_DIR / s
    if sdir.exists():
        for p in sdir.glob("*.jpg"):
            test_images[(s, p.stem)] = p
            test_images[p.stem] = p

print(f"Indexed {len(test_images)} source image entries in {TEST_DIR}")

# Methods to inspect
methods = [
    "adain", "aespa", "cut", "identity", "samam", "samst",
    "sdturbo", "seedream", "stylealigned", "styleid", "styleshot",
    "stytr2", "wct", "weave", "zstar"
]

def parse_filename(p: Path):
    stem = p.stem
    # 1. Check target style suffix
    tgt_style = None
    left = None
    for s in STYLES:
        for sep in ["__to__", "_to_"]:
            token = f"{sep}{s}"
            if stem.endswith(token):
                tgt_style = s
                left = stem[:-len(token)]
                break
        if tgt_style:
            break
            
    if not tgt_style or not left:
        return None
        
    # 2. Check source style prefix
    src_style = None
    core_stem = None
    for s in STYLES:
        for pref in [f"{s}__", f"{s}_"]:
            if left.startswith(pref):
                src_style = s
                core_stem = left[len(pref):]
                # Sometimes double prefix e.g. Early_Renaissance__Early_Renaissance__...
                if core_stem.startswith(pref):
                    core_stem = core_stem[len(pref):]
                break
        if src_style:
            break
            
    if not src_style:
        return None
        
    # Match with known test image
    src_path = test_images.get((src_style, core_stem)) or test_images.get(core_stem)
    if not src_path:
        # Try stripping leading/trailing tokens
        for k, cand_p in test_images.items():
            if isinstance(k, tuple):
                st, stem_cand = k
                if st == src_style and (stem_cand in core_stem or core_stem in stem_cand):
                    src_path = cand_p
                    break

    return {
        "file": p.name,
        "path": p,
        "src_style": src_style,
        "tgt_style": tgt_style,
        "core_stem": core_stem,
        "src_path": src_path,
        "cross": (src_style != tgt_style),
    }

print("\n--- Parsing Diagnostic Across 15 Baselines ---")
report = {}
for m in methods:
    mdir = RESULTS_DIR / m
    if not mdir.exists():
        print(f"{m:15s}: DIR_NOT_FOUND")
        continue
    imgs = sorted(list(mdir.glob("*.png")) + list(mdir.glob("*.jpg")))
    parsed = [parse_filename(p) for p in imgs]
    valid = [r for r in parsed if r is not None and r["src_path"] is not None]
    cross = [r for r in valid if r["cross"]]
    same = [r for r in valid if not r["cross"]]
    report[m] = {
        "total_imgs": len(imgs),
        "valid_pairs": len(valid),
        "cross_pairs": len(cross),
        "same_pairs": len(same),
        "unmatched": len(imgs) - len(valid),
    }
    sample_fail = [imgs[i].name for i, r in enumerate(parsed) if r is None or r["src_path"] is None][:2]
    print(f"{m:15s}: total={len(imgs):3d} | valid={len(valid):3d} (cross={len(cross):3d}, same={len(same):3d}) | unmatched={len(imgs)-len(valid)} | fail_sample={sample_fail}")
