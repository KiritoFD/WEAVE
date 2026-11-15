"""Score saved V15 raw payloads (rgb_u8) with the frozen DINO referee.

Run on the 3060 AFTER eval_arms:  python -m exp.v15.score_raw
Reads runs/v15/eval/raw/<arm>/t0_<level>/batch_*.pt, no resampling needed.
Writes runs/v15/eval/<arm>.json  {"rows": [...], "levels": {...}}  and prints
EVAL_SCORE_DONE.  CSD skipped (weights absent here); authoritative CSD/LPIPS
recompute happens locally on any downloaded subset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs/v15"
EVAL = RUNS / "eval"
RAW = EVAL / "raw"
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="",
                    help="comma-separated arm names; default = all")
    a = ap.parse_args()
    dev = a.device

    from src.utils._local_eval_refs import encode_dino, load_dino_model
    from PIL import Image

    tasks = {t["task_id"]: t for t in json.loads(
        (REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8")
    )["tasks"]}
    src = torch.load(REPO / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    src_dino = {s: torch.as_tensor(v, dtype=torch.float32)
                for s, v in src["raw_dino"].items()}
    refs = json.loads((REPO / "eval_protocol/references_v1.json")
                      .read_text(encoding="utf-8"))["references"]
    bank = torch.load(REPO / "eval_protocol/cache/references_v1_features.pt",
                      map_location="cpu", weights_only=False)
    rd = [torch.nn.functional.normalize(
        bank["styles"][s]["dino_cls"].float(), dim=-1).to(dev) for s in STYLES]
    ref_stems = {s: [r["stem"] for r in refs[s]] for s in STYLES}
    dino_model = load_dino_model(device=dev)

    arm_dirs = sorted(p for p in RAW.iterdir() if p.is_dir() and
                      (not a.arms or p.name in a.arms.split(",")))
    print(f"scoring arms ({dev}): {[p.name for p in arm_dirs]}", flush=True)
    for adir in arm_dirs:
        tag = adir.name
        rows = []
        t0w = time.time()
        with torch.inference_mode():
            for ldir in sorted(lev_dir for lev_dir in adir.iterdir()
                               if lev_dir.is_dir()):
                if not (ldir / "meta.json").exists():
                    print(f"  {tag} {ldir.name} SKIP (incomplete)", flush=True)
                    continue
                lev = float(ldir.name.split("_", 1)[1])
                batches = sorted(ldir.glob("batch_*.pt"))
                for bp in batches:
                    payload = torch.load(bp, map_location="cpu",
                                         weights_only=False)
                    u8 = payload["rgb_u8"]
                    imgs = [Image.fromarray(x) for x in
                            u8.permute(0, 2, 3, 1).numpy()]
                    df, _ = encode_dino(dino_model, imgs, batch=16,
                                        device=dev)
                    df = torch.nn.functional.normalize(df.float(), dim=-1)
                    gen = df.cpu().numpy()
                    group = [tasks[tid] for tid in payload["ids"]]
                    sd = torch.stack([src_dino[t["source_stem"]]
                                      for t in group])
                    dc = (gen * sd.numpy()).sum(-1)
                    sc = np.stack([(torch.as_tensor(gen).to(dev) @ p.T
                                    ).cpu().numpy() for p in rd], axis=1)
                    for i, t in enumerate(group):
                        stems = ref_stems[t["target_style"]]
                        sim = sc[i, t["target_style_id"]].copy()
                        for r, stem in enumerate(stems):
                            if stem == t["source_stem"]:
                                sim[r] = -1e9
                        rows.append({
                            "level": lev, "task_id": t["task_id"],
                            "src": t["source_stem"],
                            "tgt": int(t["target_style_id"]),
                            "cross": int(t["source_style_id"] !=
                                         t["target_style_id"]),
                            "dino_s": float(sim.max()),
                            "dino_c": float(dc[i]),
                            "style_top1": int(int(np.argmax(
                                sc[i].max(axis=1))) ==
                                int(t["target_style_id"]))})
                print(f"  {tag} {ldir.name} done {time.time()-t0w:.0f}s",
                      flush=True)
        rec = {}
        for lev in sorted({r["level"] for r in rows}):
            rr = [r for r in rows if r["level"] == lev]
            cr = [r for r in rr if r["cross"]]
            rec[f"t0={lev}"] = {
                "dino_s": round(float(np.mean([r["dino_s"] for r in rr])), 4),
                "dino_s_cross": round(float(np.mean([r["dino_s"] for r in cr])), 4),
                "dino_c": round(float(np.mean([r["dino_c"] for r in rr])), 4),
                "style_top1": round(float(np.mean([r["style_top1"] for r in rr])), 4),
                "style_top1_cross": round(float(np.mean(
                    [r["style_top1"] for r in cr])), 4),
                "n": len(rr), "n_cross": len(cr)}
        (EVAL / f"{tag}.json").write_text(json.dumps(
            {"rows": rows, "levels": rec}, indent=1))
        print(f"{tag}: " + " | ".join(
            f"L{k}: s={v['dino_s']}(x{v['dino_s_cross']}) c={v['dino_c']} "
            f"top1={v['style_top1']}(x{v['style_top1_cross']})"
            for k, v in rec.items()), flush=True)
    print("EVAL_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
