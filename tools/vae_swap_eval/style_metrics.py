"""DINO-S / CLIP-S with the evaluator-v2 frozen, test-disjoint references."""
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, CLIPProcessor

from metrics import ROOT, STYLES, pixels
from tools.pseudo_gt.matching import encode_dino

REFS = ROOT / "deliver/vae_ranked_eval/vae_style_reference_subset.pt"


class StyleMetrics:
    def __init__(self, cache, batch=16):
        self.cache, self.batch = cache, batch
        cache.mkdir(parents=True, exist_ok=True)
        self.rows_buffer = []
        self.dino_buffer, self.clip_buffer, self.stats_buffer = [], [], []
        dino_path = sorted((ROOT / "eval_cache/hf/models--facebook--dinov2-small/snapshots").iterdir())[0]
        clip_path = sorted((ROOT / "eval_cache/hf/models--openai--clip-vit-base-patch32/snapshots").iterdir())[0]
        self.dino = AutoModel.from_pretrained(str(dino_path), local_files_only=True).cuda().eval()
        self.clip = CLIPModel.from_pretrained(str(clip_path), local_files_only=True).cuda().eval()
        self.processor = CLIPProcessor.from_pretrained(str(clip_path), local_files_only=True)
        self.dino.requires_grad_(False)
        self.clip.requires_grad_(False)
        refs = torch.load(REFS, map_location="cpu", weights_only=False)
        self.ref_dino = [F.normalize(refs["styles"][s]["dino_cls"].float(), dim=-1).cuda()
                         for s in STYLES]
        self.proto_clip = F.normalize(torch.stack([refs["clip_prototypes"][s].float()
                                                  for s in STYLES]), dim=-1).cuda()
        self.source_scores = {}
        self.csd = None
        self.ref_csd = None
        self.metadata = {
            "dino_model": str(dino_path), "clip_model": str(clip_path),
            "reference_cache_sha256": hashlib.sha256(REFS.read_bytes()).hexdigest(),
            "dino_s": "max cosine to 30 frozen target-style DINOv2-small CLS references",
            "clip_s": "cosine to normalized mean of 30 frozen target-style CLIP ViT-B/32 references",
            "reference_source": "eval_protocol/cache/references_v1_features.pt (3060)",
            "style_score_vectors": "DINO max-reference / CLIP prototype cosine against ALL five styles",
        }

    def start_cell(self):
        self.rows_buffer = []
        self.dino_buffer, self.clip_buffer, self.stats_buffer = [], [], []

    @torch.inference_mode()
    def extract(self, pils):
        dino, _ = encode_dino(self.dino, pils, batch=self.batch, device="cuda")
        dino = F.normalize(dino.float(), dim=-1)
        clip = self.clip.get_image_features(**self.processor(images=pils, return_tensors="pt").to("cuda"))
        if hasattr(clip, "pooler_output"):
            clip = clip.pooler_output
        clip = F.normalize(clip.float(), dim=-1)
        x = pixels(pils).cuda().add(1).mul(0.5)
        dx, dy = x[..., 1:] - x[..., :-1], x[..., 1:, :] - x[..., :-1, :]
        edge = 0.5 * (dx.square().mean((1, 2, 3)) + dy.square().mean((1, 2, 3)))
        variance = x.var((2, 3), unbiased=False).mean(1)
        stats = torch.stack([edge, variance, edge / variance.clamp_min(1e-8)], dim=1)
        return dino, clip, stats

    @torch.inference_mode()
    def score_features(self, tasks, dino, clip, stats):
        ds = torch.stack([(dino @ refs.T).max(dim=1).values for refs in self.ref_dino], dim=1)
        cs = clip @ self.proto_clip.T
        ds, cs, stat = [a.float().cpu().numpy() for a in [ds, cs, stats]]
        rows = []
        for i, t in enumerate(tasks):
            ti, si = t["target_style_id"], t["source_style_id"]
            row = {k: t[k] for k in ["task_id", "source_stem", "source_style", "target_style"]}
            row.update(is_cross=si != ti, dino_scores=ds[i].tolist(), clip_scores=cs[i].tolist(),
                       edge_energy=float(stat[i, 0]), spatial_variance=float(stat[i, 1]),
                       normalized_edge=float(stat[i, 2]))
            for name, scores in [("dino", ds[i]), ("clip", cs[i])]:
                row[name + "_s"] = float(scores[ti])
                row[name + "_margin"] = float(scores[ti] - np.delete(scores, ti).max())
                row[name + "_top1"] = int(scores.argmax() == ti)
                if t["source_stem"] in self.source_scores:
                    source = self.source_scores[t["source_stem"]][name]
                    row[name + "_dT"] = float(scores[ti] - source[ti])
            rows.append(row)
        return rows

    def score(self, tasks, pils):
        dino, clip, stats = self.extract(pils)
        rows = self.score_features(tasks, dino, clip, stats)
        self.dino_buffer.append(dino.cpu().numpy())
        self.clip_buffer.append(clip.cpu().numpy())
        self.stats_buffer.append(stats.cpu().numpy())
        self.rows_buffer.extend(rows)
        return rows

    def save_cell(self, name):
        np.savez_compressed(self.cache / (name + ".npz"),
                            dino=np.concatenate(self.dino_buffer), clip=np.concatenate(self.clip_buffer),
                            stats=np.concatenate(self.stats_buffer),
                            task_ids=np.array([r["task_id"] for r in self.rows_buffer]))
        self.start_cell()

    def read_cell(self, name, tasks):
        p = self.cache / (name + ".npz")
        if not p.exists():
            return None
        d = np.load(p)
        assert d["task_ids"].tolist() == [t["task_id"] for t in tasks]
        rows = []
        for i in range(0, len(tasks), self.batch):
            sl = slice(i, i+self.batch)
            rows.extend(self.score_features(tasks[sl], torch.from_numpy(d["dino"][sl]).cuda(),
                         torch.from_numpy(d["clip"][sl]).cuda(), torch.from_numpy(d["stats"][sl]).cuda()))
        return rows

    def enable_csd(self):
        if self.csd is not None:
            return
        import csd_model
        self.csd, _, _ = csd_model.load_csd("cuda")
        self.csd.requires_grad_(False)
        d = torch.load(ROOT / "deliver/testset/refs/csd_proto.pt", map_location="cpu", weights_only=False)
        self.ref_csd = np.stack([np.asarray(d[s], dtype=np.float32) for s in STYLES])
        self.ref_csd /= np.linalg.norm(self.ref_csd, axis=1, keepdims=True)

    def csd_rows(self, tasks, pils):
        import csd_model
        self.enable_csd()
        feats = csd_model.feats_pils(self.csd, pils, batch=self.batch, device="cuda")
        scores = feats @ self.ref_csd.T
        return [{"task_id": t["task_id"], "csd_scores": score.tolist(),
                 "target_top1": int(score.argmax() == t["target_style_id"]),
                 "target_margin": float(score[t["target_style_id"]] - np.delete(score, t["target_style_id"]).max())}
                for t, score in zip(tasks, scores)]
