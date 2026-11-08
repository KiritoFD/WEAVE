"""Shared utilities for WEAVE V7 single-step distribution transport experiments.

Design goals:
- No dependency on test-set tuning: P0/P1 use an independent validation board built
  from the *train split* of distinct5_512, disjoint from the frozen 150 test sources.
- Deterministic per-task noise: all stochastic updates are seeded by task_id, so
  results are invariant to batch size / ordering / repeated evaluation.
- Keep image metrics from saved RGB output (or equivalent TRT-decoded PNG tensors),
  exactly matching the Evaluator V2 paper_rgb_v1 protocol.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import paths  # noqa: E402
from tools.evaluation_v2.features import EvaluatorModels, build_reference_features_cache  # noqa: E402
from utils.run_evaluation import _load_eval_image_tensor  # noqa: E402
from tools.v5.route_c import decode_trt, load_trt, load_vae  # noqa: E402
from tools.v6.models_v6 import ConditionalScoreNet  # noqa: E402

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
PROTOCOL_DIR = REPO_ROOT / "eval_protocol"
V6_ROOT = REPO_ROOT / "runs" / "v6"
V7_ROOT = REPO_ROOT / "runs" / "v7"
DOCS_V7 = REPO_ROOT / "docs" / "v7"
TEST_DIR = REPO_ROOT / "data" / "test"


def ensure_v7_dirs() -> None:
    V7_ROOT.mkdir(parents=True, exist_ok=True)
    DOCS_V7.mkdir(parents=True, exist_ok=True)
    (V7_ROOT / "cache").mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def task_seed(task_id: str, noise_idx: int = 0, global_seed: int = 42) -> int:
    h = hashlib.md5(f"{task_id}|{noise_idx}|{global_seed}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def test_source_stems() -> set[str]:
    p = PROTOCOL_DIR / "tasks_750.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text(encoding="utf-8"))
    stems = {s["source_stem"] for s in d.get("sources", [])}
    # also keep style-prefix-stripped variants for robust train/test disjointness
    extra = set()
    for s in stems:
        if "__" in s:
            extra.add(s.split("__", 1)[1])
        if "_" in s:
            extra.add(s.split("_", 1)[1])
    return stems | extra


def build_independent_val_tasks(per_style: int = 10) -> List[dict]:
    """Build 5 * per_style * 4 independent cross-style tasks from distinct5_512/train."""
    root = REPO_ROOT / "datasets" / "distinct5_512" / "train"
    if not root.exists():
        raise FileNotFoundError(f"Independent validation source root not found: {root}")

    banned = test_source_stems()
    tasks: List[dict] = []
    for sid, sname in enumerate(STYLES):
        sdir = root / sname
        candidates = sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")))
        chosen = []
        for p in candidates:
            stem = p.stem
            short = stem.split("__", 1)[1] if "__" in stem else stem
            if stem in banned or short in banned:
                continue
            chosen.append(p)
            if len(chosen) >= per_style:
                break
        if len(chosen) < per_style:
            raise RuntimeError(f"Only {len(chosen)} independent sources for {sname}, need {per_style}")

        for p in chosen:
            stem = p.stem
            rel = str(p.relative_to(REPO_ROOT)).replace("\\", "/")
            for tid, tname in enumerate(STYLES):
                if tid == sid:
                    continue
                tasks.append({
                    "task_id": f"{sname}/{stem}/{tname}",
                    "source_style": sname,
                    "source_style_id": sid,
                    "source_stem": stem,
                    "source_rel_path": rel,
                    "target_style": tname,
                    "target_style_id": tid,
                    "is_cross": True,
                })
    return tasks


def load_eval_bundle(device: str = "cuda") -> Tuple[EvaluatorModels, dict, torch.nn.Module, object]:
    models = EvaluatorModels(device)
    ref_feats = build_reference_features_cache(models)
    vae = load_vae(device)
    trt = load_trt(vae, device)
    return models, ref_feats, vae, trt


def load_score_model(ckpt_path: Optional[Path] = None, base_dim: int = 64, device: str = "cuda") -> ConditionalScoreNet:
    if ckpt_path is None:
        ckpt_path = V6_ROOT / "score_diff" / "score_diff_net.pt"
    model = ConditionalScoreNet(base_dim=base_dim, num_styles=len(STYLES)).to(device).eval()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Score checkpoint missing: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model


def build_source_latents(tasks: Sequence[dict], vae, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """VAE posterior-mean latents for unique independent validation sources."""
    unique: Dict[str, str] = {}
    for t in tasks:
        unique[t["source_stem"]] = t["source_rel_path"]

    out: Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for stem, rel in unique.items():
            p = REPO_ROOT / rel
            if not p.exists():
                raise FileNotFoundError(f"Independent validation source missing: {p}")
            img = _load_eval_image_tensor(p, size=512).unsqueeze(0).to(device)
            inp = (img * 2.0 - 1.0).half()
            z = vae.encode(inp).latent_dist.mean * vae.config.scaling_factor
            out[stem] = z.float().contiguous()
    return out


def batch_tensors(
    chunk: Sequence[dict],
    src_latents: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    z = torch.cat([src_latents[t["source_stem"]] for t in chunk], dim=0).to(device).float()
    s = torch.tensor([t["source_style_id"] for t in chunk], device=device, dtype=torch.long)
    t = torch.tensor([t["target_style_id"] for t in chunk], device=device, dtype=torch.long)
    task_ids = [t_["task_id"] for t_ in chunk]
    return z, s, t, task_ids


def _pil_to_lpips_tensor(im: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(im, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).to(device) / 127.5 - 1.0


def build_confusion_matrix(pred_classes: np.ndarray, tgt_classes: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(STYLES), len(STYLES)), dtype=np.float64)
    for t in range(len(STYLES)):
        mask = tgt_classes == t
        if mask.sum() == 0:
            continue
        for p in range(len(STYLES)):
            cm[t, p] = float((pred_classes[mask] == p).mean() * 100.0)
    return cm


def compute_batch_rows(
    z_gen: torch.Tensor,
    chunk: Sequence[dict],
    models: EvaluatorModels,
    ref_feats: dict,
    vae,
    trt,
    source_cache: Dict[str, dict],
    *,
    latent_l2: Optional[Sequence[float]] = None,
    extra: Optional[dict] = None,
) -> List[dict]:
    """Decode latents to RGB and compute V7 diagnostic rows under paper_rgb_v1."""
    device = models.device
    B = len(chunk)
    z_gen = z_gen.float()

    # Lazy source feature cache from raw RGB source images. If a precomputed
    # feature cache is supplied (Evaluator V2), only the missing PIL is filled.
    for t in chunk:
        stem = t["source_stem"]
        if stem in source_cache and "pil" in source_cache[stem]:
            continue
        p = REPO_ROOT / t["source_rel_path"]
        src_pil = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
        if stem in source_cache:
            source_cache[stem]["pil"] = src_pil
            continue
        feats = models.extract_image_features([src_pil])
        source_cache[stem] = {
            "pil": src_pil,
            "csd": feats["csd"][0].float(),
            "dino_cls": feats["dino_cls"][0].float(),
            "clip": feats["clip"][0].float(),
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
    gen_scores_all = gen_csd @ proto.T
    preds = np.argmax(gen_scores_all, axis=1)

    rows: List[dict] = []
    for j, t in enumerate(chunk):
        tid = t["target_style_id"]
        sid = t["source_style_id"]
        tname = t["target_style"]
        stem = t["source_stem"]

        src_csd = source_cache[stem]["csd"].numpy()
        src_scores_all = src_csd @ proto.T

        s_target = float(gen_scores_all[j, tid])
        s_source = float(gen_scores_all[j, sid])
        base_target = float(src_scores_all[tid])
        base_source = float(src_scores_all[sid])
        dT = s_target - base_target
        dS = s_source - base_source

        other = [k for k in range(len(STYLES)) if k != tid]
        max_other = float(np.max(gen_scores_all[j, other]))
        target_margin = s_target - max_other

        dino_c = float(torch.dot(gen_feats["dino_cls"][j].float(), source_cache[stem]["dino_cls"]).item())

        # Unified frozen reference-pool style metrics
        ref_cls = ref_feats["styles"][tname]["dino_cls"].to(device)
        dino_s = float((gen_feats["dino_cls"][j].to(device).unsqueeze(0) @ ref_cls.T).max().item())
        clip_proto = ref_feats["clip_prototypes"][tname].to(device)
        clip_s = float(torch.dot(gen_feats["clip"][j].to(device), clip_proto).item())

        row = {
            "task_id": t["task_id"],
            "source_stem": stem,
            "source_style": t["source_style"],
            "target_style": tname,
            "source_style_id": sid,
            "target_style_id": tid,
            "is_cross": bool(t.get("is_cross", True)),
            "csd_dT": dT,
            "csd_dS": dS,
            "csd_dM": dT - dS,
            "target_score": s_target,
            "max_other_score": max_other,
            "target_margin": target_margin,
            "target_top1": int(preds[j] == tid),
            "pred_class": int(preds[j]),
            "dino_c": dino_c,
            "dino_s": dino_s,
            "clip_s": clip_s,
            "lpips": float(lps[j]),
        }
        if latent_l2 is not None:
            row["latent_l2"] = float(latent_l2[j])
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def aggregate_rows(rows: Sequence[dict], only_cross: bool = False) -> dict:
    if only_cross:
        rows = [r for r in rows if r.get("is_cross", True)]
    if not rows:
        return {}
    dT = np.array([r["csd_dT"] for r in rows])
    dS = np.array([r["csd_dS"] for r in rows])
    dM = np.array([r["csd_dM"] for r in rows])
    top1 = np.array([r["target_top1"] for r in rows])
    margin = np.array([r["target_margin"] for r in rows])
    d_c = np.array([r["dino_c"] for r in rows])
    d_s = np.array([r.get("dino_s", np.nan) for r in rows])
    cl_s = np.array([r.get("clip_s", np.nan) for r in rows])
    lp = np.array([r["lpips"] for r in rows])
    l2 = np.array([r.get("latent_l2", np.nan) for r in rows])

    preds = np.array([r["pred_class"] for r in rows])
    tgts = np.array([r["target_style_id"] for r in rows])
    cm = build_confusion_matrix(preds, tgts)

    per_tgt_dT, per_tgt_top1, per_tgt_margin = {}, {}, {}
    for s in STYLES:
        sub = [r for r in rows if r["target_style"] == s]
        if sub:
            per_tgt_dT[s] = round(float(np.mean([r["csd_dT"] for r in sub])), 4)
            per_tgt_top1[s] = round(float(np.mean([r["target_top1"] for r in sub]) * 100.0), 1)
            per_tgt_margin[s] = round(float(np.mean([r["target_margin"] for r in sub])), 4)

    out = {
        "n": len(rows),
        "mean_dT": round(float(dT.mean()), 4),
        "mean_dS": round(float(dS.mean()), 4),
        "mean_dM": round(float(dM.mean()), 4),
        "Target_Top1": round(float(top1.mean() * 100.0), 1),
        "Target_Margin": round(float(margin.mean()), 4),
        "DINO_C": round(float(d_c.mean()), 4),
        "DINO_S": round(float(np.nanmean(d_s)), 4),
        "CLIP_S": round(float(np.nanmean(cl_s)), 4),
        "LPIPS": round(float(lp.mean()), 4),
        "mean_latent_l2": round(float(np.nanmean(l2)), 4),
        "styles_improved": int(sum(1 for v in per_tgt_dT.values() if v > 0)),
        "ukiyo_dT": per_tgt_dT.get("Ukiyo_e", 0.0),
        "ukiyo_top1": per_tgt_top1.get("Ukiyo_e", 0.0),
        "per_target_dT": per_tgt_dT,
        "per_target_top1": per_tgt_top1,
        "per_target_margin": per_tgt_margin,
        "confusion_matrix_pct": np.round(cm, 1).tolist(),
    }
    return out
