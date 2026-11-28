"""Fixed split pairing and process-isolated embedding measurements (no 131-way CSD classifier)."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from exp.v36_art_foundation import CACHE_DIR, CKPT_DIR, ROOT
from exp.v37_editing import OUT, SEED, atomic_json

P_CKPTS = {"p1_30k": CKPT_DIR / "v36_main_p1_step030000.pt",
           "p1_42k": CKPT_DIR / "v36_main_p1_final.pt", "p2_50k": CKPT_DIR / "v36_main_p2_final.pt"}


def split_pack(split, n=0):
    key = {"seen_dev": "dev_info", "artist_ood": "art_info", "movement_ood": "mv_info"}[split]
    info = json.loads((CACHE_DIR / "v36_caches_ready.json").read_text())[key]
    recs = json.loads((CACHE_DIR / "curation_manifest.json").read_text())[split]
    rng = np.random.default_rng(SEED + 10)
    ci = rng.permutation(info["n"])[:n or info["n"]]
    styles = np.array([r["style"] for r in recs])
    ri, wi = [], []
    for i in ci:
        r = int(rng.choice(np.flatnonzero(styles != styles[i])))
        w = int(rng.choice(np.flatnonzero((styles != styles[i]) & (styles != styles[r]))))
        ri.append(r); wi.append(w)
    arrays = {k: np.load(ROOT / info["paths"][k], mmap_mode="r") for k in ("z_gt", "q32", "y16", "stat")}
    t = lambda k, ids: torch.from_numpy(np.array(arrays[k][ids])).float()
    return {"zc": t("z_gt", ci), "q": t("q32", ci), "y": t("y16", ci),
            "zr": t("z_gt", ri), "zw": t("z_gt", wi), "sr": t("stat", ri), "sw": t("stat", wi),
            "source_ids": ci.tolist(), "ref_ids": ri, "wrong_ids": wi,
            "eps": torch.randn((len(ci), 4, 64, 64), generator=torch.Generator().manual_seed(20261112))}


def save_latents(path, z, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.half().cpu(), "meta": meta}, path)


@torch.no_grad()
def measure_embeddings(directory):
    from src.utils import FeatureStack, decode_s10, load_s10_decoder
    dec, stack = load_s10_decoder("cuda"), FeatureStack("cuda")
    directory = Path(directory)
    packs = {}

    def features(z):
        out = {k: [] for k in ("dino", "csd", "clip")}
        for s in range(0, len(z), 16):
            rgb = decode_s10(dec, z[s:s + 16].cuda().float(), batch=16, output_cpu=True)
            images = [Image.fromarray((x.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)) for x in rgb]
            ff = stack.extract(images, batch=16)
            for k in out:
                v = ff[k]
                out[k].append(v if isinstance(v, np.ndarray) else v.cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    results = []
    for path in sorted(directory.glob("*.pt")):
        if path.name.endswith("_features.pt"):
            continue
        saved = directory / (path.stem + ".json")
        if saved.exists():
            results.append(json.loads(saved.read_text())); continue
        ck = torch.load(path, weights_only=False, map_location="cpu")
        m = ck["meta"]
        split, n = m["split"], m["n"]
        key = (split, n)
        if key not in packs:
            p = split_pack(split, n)
            packs[key] = {k: features(p[k]) for k in ("zc", "zr", "zw")}
        ff = features(ck["z"])
        pf = packs[key]
        cos = lambda k, ref: (ff[k] * pf[ref][k]).sum(-1)
        vals = {"dino_c": cos("dino", "zc"), "dino_s": cos("dino", "zr"),
                "clip_s": cos("clip", "zr"), "csd_tgt_cos": cos("csd", "zr"),
                "csd_margin_vs_wrong": cos("csd", "zr") - cos("csd", "zw"),
                "csd_2way_follow": (cos("csd", "zr") > cos("csd", "zw")).astype(float)}
        row = {**m, **{k: float(v.mean()) for k, v in vals.items()},
               "per_source": {k: v.tolist() for k, v in vals.items()},
               "metric_scope": "instance-reference embedding similarity; no Core5 classification"}
        atomic_json(saved, row)
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "per_source"}), flush=True)
    atomic_json(directory / "summary.json", results)
    return results
