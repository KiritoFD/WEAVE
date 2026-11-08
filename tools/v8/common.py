"""WEAVE V8 shared utilities.

Strict design constraint from docs/bridge.md: V8 must not import any ScoreNet /
ConditionalScoreNet / score_diff module. This file only depends on frozen
Evaluator V2 protocols, VAE decode utilities, and standard evaluation caches.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402
from tools.evaluation_v2.features import EvaluatorModels, build_reference_features_cache  # noqa: E402
from tools.v5.route_c import decode_trt, load_trt, load_vae  # noqa: E402
from utils.run_evaluation import _load_eval_image_tensor  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
PROTOCOL_DIR = REPO_ROOT / "eval_protocol"
V8_ROOT = REPO_ROOT / "runs" / "v8"
DOCS_V8 = REPO_ROOT / "docs" / "v8"
D5_LATENT_ROOT = paths.asset_root() / "pseudo_gt_24h" / "orig_mean_fp16"


def ensure_dirs() -> None:
    V8_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_V8.mkdir(parents=True, exist_ok=True)
    (V8_ROOT / "cache").mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_d5_latents(device: str = "cuda", heldout: int = 100) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]:
    """Load scaled SD1.5-VAE posterior-mean D5 latents; first `heldout` are reserved."""
    train, val = {}, {}
    meta = {"root": str(D5_LATENT_ROOT), "heldout": heldout, "files": {}}
    for sid, s in enumerate(STYLES):
        p = D5_LATENT_ROOT / f"{sid:02d}_{s}.pt"
        if not p.exists():
            raise FileNotFoundError(p)
        d = torch.load(p, map_location="cpu", weights_only=False)
        x = d["latents"].float()
        if x.ndim != 4 or x.shape[1:] != (4, 64, 64):
            raise RuntimeError(f"Unexpected D5 latent shape for {s}: {tuple(x.shape)}")
        val[s] = x[:heldout].to(device)
        train[s] = x[heldout:].to(device)
        meta["files"][s] = {
            "path": str(p),
            "sha256": file_sha256(p),
            "count": int(x.shape[0]),
            "shape": list(x.shape),
            "scaling_factor": d.get("scaling_factor"),
            "vae": d.get("vae"),
        }
    common_sf = {v["scaling_factor"] for v in meta["files"].values()}
    common_vae = {v["vae"] for v in meta["files"].values()}
    if len(common_sf) != 1 or len(common_vae) != 1:
        raise RuntimeError("D5 latent caches disagree on VAE / scaling_factor")
    meta["scaling_factor"] = next(iter(common_sf))
    meta["vae"] = next(iter(common_vae))
    return train, val, meta


def test_source_stems() -> set[str]:
    p = PROTOCOL_DIR / "tasks_750.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {s["source_stem"] for s in d.get("sources", [])}


def load_frozen_tasks() -> List[dict]:
    d = json.loads((PROTOCOL_DIR / "tasks_750.json").read_text(encoding="utf-8"))
    tasks = d["tasks"]
    for t in tasks:
        if "source_rel_path" not in t:
            t["source_rel_path"] = f"data/test/{t['source_style']}/{t['source_filename']}"
    return tasks


def build_independent_val_tasks(per_style: int = 10) -> List[dict]:
    root = REPO_ROOT / "datasets" / "distinct5_512" / "train"
    banned = test_source_stems()
    tasks = []
    for sid, sname in enumerate(STYLES):
        sdir = root / sname
        chosen = []
        for p in sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png"))):
            stem = p.stem
            short = stem.split("__", 1)[1] if "__" in stem else stem
            if stem in banned or short in banned:
                continue
            chosen.append(p)
            if len(chosen) >= per_style:
                break
        if len(chosen) < per_style:
            raise RuntimeError(f"Only {len(chosen)} independent sources for {sname}")
        for p in chosen:
            for tid, tname in enumerate(STYLES):
                if tid == sid:
                    continue
                tasks.append({
                    "task_id": f"{sname}/{p.stem}/{tname}",
                    "source_stem": p.stem,
                    "source_style": sname,
                    "source_style_id": sid,
                    "target_style": tname,
                    "target_style_id": tid,
                    "source_rel_path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "is_cross": True,
                })
    return tasks


def build_source_latents(tasks: Sequence[dict], vae, device: str = "cuda") -> Dict[str, torch.Tensor]:
    rels = {}
    for t in tasks:
        rels[t["source_stem"]] = t["source_rel_path"]
    out = {}
    with torch.inference_mode():
        for stem, rel in rels.items():
            p = REPO_ROOT / rel
            img = _load_eval_image_tensor(p, size=512).unsqueeze(0).to(device)
            z = vae.encode((img * 2.0 - 1.0).half()).latent_dist.mean * vae.config.scaling_factor
            out[stem] = z.float().contiguous()
    return out


def batch_tensors(chunk, src_latents, device):
    z = torch.cat([src_latents[t["source_stem"]] for t in chunk], 0).to(device).float()
    s = torch.tensor([t["source_style_id"] for t in chunk], device=device, dtype=torch.long)
    t = torch.tensor([t["target_style_id"] for t in chunk], device=device, dtype=torch.long)
    return z, s, t


def load_eval_bundle(device: str = "cuda"):
    models = EvaluatorModels(device)
    ref_feats = build_reference_features_cache(models)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    return models, ref_feats, vae, trt


def source_cache_for_frozen() -> Dict[str, dict]:
    p = PROTOCOL_DIR / "cache" / "sources_paper_rgb_v1.pt"
    c = torch.load(p, map_location="cpu", weights_only=False)
    return {
        stem: {"csd": c["csd"][i].float(), "dino_cls": c["dino_cls"][i].float(), "clip": c["clip"][i].float()}
        for i, stem in enumerate(c["stems"])
    }


def _pil_to_lpips_tensor(im: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).to(device) / 127.5 - 1.0


def confusion_matrix(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(STYLES), len(STYLES)), dtype=np.float64)
    for t in range(len(STYLES)):
        m = tgt == t
        if m.any():
            for p in range(len(STYLES)):
                cm[t, p] = float((pred[m] == p).mean() * 100.0)
    return cm


def compute_batch_rows(z_gen, chunk, models, ref_feats, vae, trt, source_cache, *, latent_l2=None, nfe=None):
    """Decode generated latents to RGB and compute Evaluator V2 style metrics."""
    device = models.device
    B = len(chunk)
    z_gen = z_gen.float()
    for t in chunk:
        stem = t["source_stem"]
        if stem not in source_cache or "pil" not in source_cache[stem]:
            p = REPO_ROOT / t["source_rel_path"]
            pil = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
            if stem in source_cache:
                source_cache[stem]["pil"] = pil
            else:
                f = models.extract_image_features([pil])
                source_cache[stem] = {
                    "pil": pil,
                    "csd": f["csd"][0].float(),
                    "dino_cls": f["dino_cls"][0].float(),
                    "clip": f["clip"][0].float(),
                }

    with torch.inference_mode():
        img = decode_trt(trt, vae, z_gen.half(), batch=8)
        u8 = img.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu()
        pils = [Image.fromarray(u8[j].permute(1, 2, 0).numpy()) for j in range(B)]
        gen_feats = models.extract_image_features(pils)
        gt = torch.stack([_pil_to_lpips_tensor(im, device) for im in pils])
        st = torch.stack([_pil_to_lpips_tensor(source_cache[t["source_stem"]]["pil"], device) for t in chunk])
        lps = models.lpips(gt, st).flatten().cpu().tolist()

    proto = models.proto_csd
    gen_csd = gen_feats["csd"].numpy()
    scores = gen_csd @ proto.T
    preds = np.argmax(scores, axis=1)
    rows = []
    for j, t in enumerate(chunk):
        sid, tid = t["source_style_id"], t["target_style_id"]
        stem, tname = t["source_stem"], t["target_style"]
        src_scores = source_cache[stem]["csd"].numpy() @ proto.T
        s_target = float(scores[j, tid]); s_source = float(scores[j, sid])
        dT = s_target - float(src_scores[tid])
        dS = s_source - float(src_scores[sid])
        other = [k for k in range(len(STYLES)) if k != tid]
        margin = s_target - float(np.max(scores[j, other]))
        dino_c = float(torch.dot(gen_feats["dino_cls"][j].float(), source_cache[stem]["dino_cls"]).item())
        ref_cls = ref_feats["styles"][tname]["dino_cls"].to(device)
        dino_s = float((gen_feats["dino_cls"][j].to(device).unsqueeze(0) @ ref_cls.T).max().item())
        clip_s = float(torch.dot(gen_feats["clip"][j].to(device), ref_feats["clip_prototypes"][tname].to(device)).item())
        row = {
            "task_id": t["task_id"], "source_stem": stem,
            "source_style": t["source_style"], "target_style": tname,
            "source_style_id": sid, "target_style_id": tid,
            "is_cross": bool(t.get("is_cross", sid != tid)),
            "csd_dT": dT, "csd_dS": dS, "csd_dM": dT - dS,
            "target_score": s_target, "max_other_score": float(np.max(scores[j, other])),
            "target_margin": margin, "target_top1": int(preds[j] == tid),
            "pred_class": int(preds[j]), "dino_c": dino_c,
            "dino_s": dino_s, "clip_s": clip_s, "lpips": float(lps[j]),
        }
        if latent_l2 is not None:
            row["latent_l2"] = float(latent_l2[j])
        if nfe is not None:
            row["nfe"] = int(nfe)
        rows.append(row)
    return rows


def aggregate_rows(rows: Sequence[dict], only_cross: bool = False) -> dict:
    if only_cross:
        rows = [r for r in rows if r.get("is_cross", True)]
    if not rows:
        return {}
    dT = np.array([r["csd_dT"] for r in rows]); dS = np.array([r["csd_dS"] for r in rows])
    dM = np.array([r["csd_dM"] for r in rows]); top1 = np.array([r["target_top1"] for r in rows])
    margin = np.array([r["target_margin"] for r in rows]); dc = np.array([r["dino_c"] for r in rows])
    ds = np.array([r["dino_s"] for r in rows]); cs = np.array([r["clip_s"] for r in rows])
    lp = np.array([r["lpips"] for r in rows]); l2 = np.array([r.get("latent_l2", np.nan) for r in rows])
    preds = np.array([r["pred_class"] for r in rows]); tgts = np.array([r["target_style_id"] for r in rows])
    per_dT, per_top1, per_margin = {}, {}, {}
    for s in STYLES:
        sub = [r for r in rows if r["target_style"] == s]
        if sub:
            per_dT[s] = round(float(np.mean([r["csd_dT"] for r in sub])), 4)
            per_top1[s] = round(float(np.mean([r["target_top1"] for r in sub]) * 100.0), 1)
            per_margin[s] = round(float(np.mean([r["target_margin"] for r in sub])), 4)
    return {
        "n": len(rows), "mean_dT": round(float(dT.mean()), 4), "mean_dS": round(float(dS.mean()), 4),
        "mean_dM": round(float(dM.mean()), 4), "Target_Top1": round(float(top1.mean() * 100.0), 1),
        "Target_Margin": round(float(margin.mean()), 4), "DINO_C": round(float(dc.mean()), 4),
        "DINO_S": round(float(ds.mean()), 4), "CLIP_S": round(float(cs.mean()), 4),
        "LPIPS": round(float(lp.mean()), 4), "mean_latent_l2": round(float(np.nanmean(l2)), 4),
        "styles_improved": int(sum(v > 0 for v in per_dT.values())),
        "ukiyo_dT": per_dT.get("Ukiyo_e", 0.0), "ukiyo_top1": per_top1.get("Ukiyo_e", 0.0),
        "per_target_dT": per_dT, "per_target_top1": per_top1, "per_target_margin": per_margin,
        "confusion_matrix_pct": np.round(confusion_matrix(preds, tgts), 1).tolist(),
    }
