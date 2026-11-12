"""Shared RGB-referenced metrics for the frozen 750-task VAE swap evaluation."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/12h"))

import csd_model
from tools.pseudo_gt.matching import encode_dino
from transformers import AutoModel

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def read_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)


def pixels(pils):
    return torch.stack([torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1)
                        for im in pils]).float().div_(127.5).sub_(1)


def source_paths(tasks):
    test = ROOT / "deliver/testset"
    disk = {p.stem: p for p in (test / "data/test").rglob("*.jpg")}
    result = {}
    for t in tasks:
        stem = t["source_stem"]
        p = test / t["source_rel_path"]
        result[stem] = p if p.exists() else disk[stem]
    assert len(result) == 150
    return result


class Metrics:
    def __init__(self, tasks, paths, batch=16):
        import lpips
        self.batch = batch
        self.csd, _, _ = csd_model.load_csd("cuda")
        snaps = sorted((ROOT / "eval_cache/hf/models--facebook--dinov2-small/snapshots").iterdir())
        self.dino = AutoModel.from_pretrained(str(snaps[0]), local_files_only=True).cuda().eval()
        self.lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
        for model in [self.csd, self.dino, self.lpips]:
            model.requires_grad_(False)
        proto = torch.load(ROOT / "deliver/testset/refs/csd_proto.pt", map_location="cpu", weights_only=False)
        self.proto = np.stack([np.asarray(proto[s], dtype=np.float32) for s in STYLES])
        self.proto /= np.linalg.norm(self.proto, axis=1, keepdims=True)
        self.source = {}
        stems = sorted(paths)
        for i in range(0, len(stems), batch):
            chunk = stems[i:i + batch]
            pils = [read_rgb(paths[s]) for s in chunk]
            csd, dino = self.features(pils)
            rgb = pixels(pils)
            for j, stem in enumerate(chunk):
                self.source[stem] = (csd[j] @ self.proto.T, dino[j], rgb[j])
        print(f"[Reference] Cached {len(self.source)} RAW RGB sources; DINOv2-small", flush=True)

    @torch.inference_mode()
    def features(self, pils):
        csd = csd_model.feats_pils(self.csd, pils, batch=self.batch, device="cuda")
        csd /= np.linalg.norm(csd, axis=1, keepdims=True)
        dino, _ = encode_dino(self.dino, pils, batch=self.batch, device="cuda")
        return csd, F.normalize(dino.float(), dim=-1).cpu().numpy()

    @torch.inference_mode()
    def score(self, tasks, pils):
        csd, dino = self.features(pils)
        sources = [self.source[t["source_stem"]] for t in tasks]
        lp = self.lpips(pixels(pils).cuda(), torch.stack([s[2] for s in sources]).cuda())
        lp = lp.flatten().cpu().numpy()
        rows = []
        for i, t in enumerate(tasks):
            ti, si = t["target_style_id"], t["source_style_id"]
            scores = csd[i] @ self.proto.T
            pred = int(scores.argmax())
            row = {k: t[k] for k in ["task_id", "source_stem", "source_style", "target_style"]}
            row.update(is_cross=si != ti, target_top1=int(pred == ti), pred_class=pred,
                       target_margin=float(scores[ti] - np.delete(scores, ti).max()),
                       dT=float(scores[ti] - sources[i][0][ti]),
                       dS=float(scores[si] - sources[i][0][si]),
                       dino_c=float(dino[i] @ sources[i][1]), lpips=float(lp[i]))
            rows.append(row)
        return rows


def summarize(rows):
    assert len(rows) == len({r["task_id"] for r in rows}) == 750
    cross = [r for r in rows if r["is_cross"]]
    same = [r for r in rows if not r["is_cross"]]
    assert len(cross) == 600 and len(same) == 150
    keys = ["target_top1", "target_margin", "dT", "dS", "dino_c", "lpips"]
    out = {"cross600": {k: float(np.mean([r[k] for r in cross])) for k in keys},
           "same150": {k: float(np.mean([r[k] for r in same])) for k in keys}}
    out["per_target_top1"] = {s: float(np.mean([r["target_top1"] for r in cross
                                                       if r["target_style"] == s])) for s in STYLES}
    non_min = [r for r in cross if r["target_style"] != "Minimalism"]
    out["minimalism_offtarget"] = float(np.mean([r["pred_class"] == 2 for r in non_min]))
    return out


def compare(rows, base):
    """Paired source-cluster bootstrap, preserving repeated cluster draws."""
    bmap = {r["task_id"]: r for r in base}
    cross = [r for r in rows if r["is_cross"]]
    stems = sorted({r["source_stem"] for r in cross})
    keys = ["target_top1", "target_margin", "dT", "dino_c", "lpips"]
    delta = np.array([[np.mean([r[k] - bmap[r["task_id"]][k] for r in cross
                               if r["source_stem"] == stem]) for k in keys] for stem in stems])
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(stems), size=(2000, len(stems)))
    boot = delta[draws].mean(axis=1)
    return {k: {"mean": float(delta[:, j].mean()),
                "ci95": np.percentile(boot[:, j], [2.5, 97.5]).tolist()}
            for j, k in enumerate(keys)}


def write_results(out, result):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
