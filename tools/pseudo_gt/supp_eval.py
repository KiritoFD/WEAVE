"""补充测量 P1/P2/P3：Pattern 对照、构造消融、六维筛选的逐样本记录。

每个配置都从同一批 300 张源图、同一供体、同一批“已通过内容匹配”的格子出发，
只改变空间排列/覆盖率/融合/频带；每条记录落盘 8 项 mask 测量与完整指标，
供后续分层与配对差 CI 分析使用。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

import csd_model  # type: ignore  # noqa: E402
import paths  # noqa: E402

from tools.pseudo_gt import construct as C  # noqa: E402
from tools.pseudo_gt import pattern_masks as PM  # noqa: E402
from tools.pseudo_gt import q0_quality as Q  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, _load_rgb, _to_tensor, load_manifest  # noqa: E402
from tools.pseudo_gt.masks import Rect, build_patch_layout, rasterize, subband_mask  # noqa: E402
from tools.pseudo_gt.matching import Match, encode_dino, match_rect  # noqa: E402
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar  # noqa: E402

CONFIGS: List[str] = [
    "R64", "G64", "C64", "S64", "F64", "R128", "G128", "C128",
    "L1", "L2", "L3", "L4", "C12", "C37", "C50",
    "FEATHER_4", "FEATHER_8", "FEATHER_16", "FEATHER_EFF",
    "LL_0", "LL_50", "LL_100",
    "A_P5", "CTRL_none", "CTRL_rand", "CTRL_same", "CTRL_destyle", "CTRL_seed",
]


# ---------------------------------------------------------------- per-cell maps
def cell_qmap(sp: torch.Tensor, dp: torch.Tensor, size: int) -> Tuple[np.ndarray, np.ndarray]:
    n = PM.cells_of(size)
    qm = np.full((n, n), -2.0, dtype=np.float64)
    mm = np.zeros((n, n), dtype=np.float64)
    for cy in range(n):
        for cx in range(n):
            m = match_rect(sp, dp, Rect(cx * size, cy * size, size, size), min_q=-2.0)
            if m is not None:
                qm[cy, cx] = m.q
                mm[cy, cx] = m.mutual_ratio
    return qm, mm


def cells_matches(sp: torch.Tensor, dp: torch.Tensor, cells, size: int) -> List[Match]:
    out = []
    for cx, cy in cells:
        m = match_rect(sp, dp, Rect(cx * size, cy * size, size, size), min_q=-2.0)
        if m is not None:
            out.append(m)
    return out


# ---------------------------------------------------------------- image stats
def _gray(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float64).mean(-1) / 255.0


def _box_blur(x: np.ndarray, k: int = 3) -> np.ndarray:
    pad = k // 2
    p = np.pad(x, pad, mode="edge")
    out = np.zeros_like(x)
    for dy in range(k):
        for dx in range(k):
            out += p[dy : dy + x.shape[0], dx : dx + x.shape[1]]
    return out / (k * k)


def texture_saliency(src_img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """8×8（64px 格）纹理复杂度与显著度；两者独立计算。"""
    g = _gray(np.asarray(src_img.convert("RGB")))
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    grad = np.zeros_like(g)
    grad[:, 1:] += gx
    grad[1:, :] += gy
    texture = grad.reshape(8, 64, 8, 64).mean(axis=(1, 3))

    small = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((128, 128), Image.BILINEAR), dtype=np.float64) / 255.0
    Fq = np.fft.fft2(small)
    logamp = np.log(np.abs(Fq) + 1e-8)
    residual = logamp - _box_blur(logamp, 3)
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(Fq)))) ** 2
    sal = _box_blur(sal, 3)
    sal8 = sal.reshape(8, 16, 8, 16).mean(axis=(1, 3))
    sal8 = (sal8 - sal8.min()) / max(float(sal8.max() - sal8.min()), 1e-9)
    return texture, sal8


# ---------------------------------------------------------------- geometry helpers
def _pixel_mask(rects: Sequence[Rect]) -> torch.Tensor:
    return rasterize(rects, feather=False) > 0.5


def _token_mask(rects: Sequence[Rect]) -> torch.Tensor:
    m = torch.zeros(16, 16, dtype=torch.bool)
    for r in rects:
        m[r.y // 32 : (r.y + r.h) // 32, r.x // 32 : (r.x + r.w) // 32] = True
    return m


def _sub_mask(rects: Sequence[Rect]) -> torch.Tensor:
    return subband_mask(rects, feather=False) > 0.5


def _latent_mask(rects: Sequence[Rect], latent_size: int = 64) -> torch.Tensor:
    """latent 分辨率（64×64）上的注入区掩码，用于“掩码外变化量”。"""
    m = torch.zeros(latent_size, latent_size, dtype=torch.bool)
    for r in rects:
        m[r.y // 8 : (r.y + r.h) // 8, r.x // 8 : (r.x + r.w) // 8] = True
    return m


def seam_stats(image: torch.Tensor, rects: Sequence[Rect]) -> Tuple[float, float, float]:
    """(边界平均梯度, 内部平均梯度, 比值 R_seam)。"""
    gray = image.mean(dim=1, keepdim=True) if image.dim() == 4 else image.mean(dim=0, keepdim=True)[None]
    gy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
    gx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    g = torch.zeros_like(gray)
    g[:, :, 1:, :] += gy
    g[:, :, :, 1:] += gx
    boundary = torch.zeros_like(gray)
    for r in rects:
        boundary[:, :, r.y : r.y + r.h, max(r.x - 1, 0) : r.x + 1] = 1
        boundary[:, :, r.y : r.y + r.h, r.x + r.w - 1 : min(r.x + r.w + 1, 512)] = 1
        boundary[:, :, max(r.y - 1, 0) : r.y + 1, r.x : r.x + r.w] = 1
        boundary[:, :, r.y + r.h - 1 : min(r.y + r.h + 1, 512), r.x : r.x + r.w] = 1
    inner = torch.ones_like(gray)
    for r in rects:
        inner[:, :, r.y + 2 : r.y + r.h - 2, r.x + 2 : r.x + r.w - 2] = 0
    b = float((g * boundary).sum() / boundary.sum().clamp_min(1.0))
    i = float((g * inner).sum() / inner.sum().clamp_min(1.0))
    return b, i, b / max(i, 1e-6)


def structure_corr(gen: np.ndarray, src: np.ndarray, mask: torch.Tensor) -> float:
    """patch 内部梯度结构的余弦相似度（局部几何是否被破坏）。"""
    g_gen = _gray(gen)
    g_src = _gray(src)
    mag = np.zeros_like(g_gen)
    mag[:, 1:] += np.abs(np.diff(g_gen, axis=1))
    mag[1:, :] += np.abs(np.diff(g_gen, axis=0))
    ref = np.zeros_like(g_src)
    ref[:, 1:] += np.abs(np.diff(g_src, axis=1))
    ref[1:, :] += np.abs(np.diff(g_src, axis=0))
    inner = F.max_pool2d(mask.float()[None, None], 5, stride=1, padding=2)[0, 0] > 0.5
    a = mag[inner.numpy()].ravel()
    b = ref[inner.numpy()].ravel()
    if a.size < 8:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-9))


def band_parts(z_src: torch.Tensor, z_h: torch.Tensor, rects: Sequence[Rect]) -> dict:
    """u = z1 − z0 的四带能量、注入内外分布与 R_active。"""
    ll_s, lh_s, hl_s, hh_s = dwt2_haar(z_src)
    ll_h, lh_h, hl_h, hh_h = dwt2_haar(z_h)
    sm = _sub_mask(rects)
    e, ein, eout = {}, 0.0, 0.0
    for name, s, h in (("LL", ll_s, ll_h), ("LH", lh_s, lh_h), ("HL", hl_s, hl_h), ("HH", hh_s, hh_h)):
        u = (h - s).detach()
        e[name] = float((u ** 2).sum().item())
        e_inside = float((u[:, :, sm] ** 2).sum().item())
        eoutside = float((u[:, :, ~sm] ** 2).sum().item())
        ein += e_inside
        eout += eoutside
    total = sum(e.values())
    active = e["LL"] + e["LH"] + e["HL"]
    return {
        "bands": {k: (v / total if total > 0 else 0.0) for k, v in e.items()},
        "R_active": (active / total) if total > 0 else 0.0,
        "inside_frac": (ein / total) if total > 0 else 0.0,
        "outside_frac": (eout / total) if total > 0 else 0.0,
    }


def rgb_adain(src: torch.Tensor, donor: torch.Tensor, matches: Sequence[Match]) -> torch.Tensor:
    """颜色/亮度扰乱：只搬 patch 内 RGB 均值方差，不搬结构。"""
    out = src.clone()
    for m in matches:
        s = src[:, :, m.src.y : m.src.y + m.src.h, m.src.x : m.src.x + m.src.w]
        d = donor[:, :, m.donor.y : m.donor.y + m.donor.h, m.donor.x : m.donor.x + m.donor.w]
        mu_s = s.mean(dim=(-1, -2), keepdim=True)
        mu_d = d.mean(dim=(-1, -2), keepdim=True)
        sd_s = s.var(dim=(-1, -2), keepdim=True, unbiased=False).sqrt()
        sd_d = d.var(dim=(-1, -2), keepdim=True, unbiased=False).sqrt()
        out[:, :, m.src.y : m.src.y + m.src.h, m.src.x : m.src.x + m.src.w] = \
            (s - mu_s) / (sd_s + 1e-6) * (sd_d + 1e-6) + mu_d
    return out


# ---------------------------------------------------------------- scoring
class Scorer:
    def __init__(self):
        self.metric = Q.Metrics()
        self.protos = None

    def score(self, z_list: Sequence[torch.Tensor], src_img: Image.Image, proto_ids: Sequence[str],
              proto_mat: np.ndarray) -> dict:
        z = torch.cat(list(z_list), 0)
        pils = self.metric.decode(z)
        csd = csd_model.feats_pils(self.metric.csd, pils, batch=len(pils))
        cls, pat = encode_dino(self.metric.dino, pils, batch=len(pils))
        arr_src = np.asarray(src_img.convert("RGB"), dtype=np.uint8)
        src_t = (torch.from_numpy(arr_src.astype(np.float32)).permute(2, 0, 1) / 127.5 - 1)[None].to("cuda")
        gen_t = torch.stack([
            torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1).to("cuda") / 127.5 - 1
            for im in pils
        ])
        with torch.inference_mode():
            lp = self.metric.lpips(gen_t, src_t.expand(len(pils), -1, -1, -1)).flatten()
        scores = csd @ proto_mat.T
        return {
            "pils": pils,
            "csd5": scores,
            "argmax": scores.argmax(1).tolist(),
            "cls": cls,
            "pat": pat,
            "lpips": [float(v) for v in lp],
            "arr_src": arr_src,
        }


def choose_donor(sp, dino, src_style, tgt_style, src_id, cands, seed):
    best = None
    for c in cands:
        di = dino[tgt_style]["index"].get(c)
        if di is None:
            continue
        dp = F.normalize(dino[tgt_style]["patches"][di].float().to("cuda"), dim=-1)
        qm, mm = cell_qmap(sp, dp, 64)
        qual = PM._qualified(qm, mm)
        if int(qual.sum()) < 16:
            continue
        order = np.argsort(-qm, axis=None)[:16]
        score = float(np.mean([qm[j // 8, j % 8] for j in order]))
        if best is None or score > best[0]:
            best = (score, c, qm, mm)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(paths.project_root()) / "runs/p2/supp_records.jsonl"))
    ap.add_argument("--meta", default=str(Path(paths.project_root()) / "runs/p2/supp_meta.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=Q.SEED)
    ap.add_argument("--n-per-style", type=int, default=Q.N_PER_STYLE)
    ap.add_argument("--configs", default="")
    args = ap.parse_args()

    Q.SEED, Q.N_PER_STYLE = args.seed, args.n_per_style
    run_cfgs = CONFIGS
    if args.configs:
        wanted = {x.strip() for x in args.configs.split(",") if x.strip()}
        run_cfgs = [c for c in CONFIGS if c in wanted]
        if not run_cfgs:
            raise SystemExit(f"未匹配配置: {sorted(wanted)}")

    t0 = time.time()
    items = load_manifest()
    test_stems = {p.stem for p in (REPO_ROOT / "data/test").rglob("*.jpg")}
    test_stems |= {p.stem for p in (REPO_ROOT / "datasets/distinct5_512/test").rglob("*.jpg")}
    sources = Q.shared_sources(items, test_stems)
    if args.limit:
        sources = sources[: args.limit]
    id_map = {it["id"]: it for it in items}
    lat, dino, retrieval = Q.load_latents(), Q.load_dino(), Q.load_retrieval()
    print(f"[supp] {len(sources)} 源图, {len(run_cfgs)} 配置", flush=True)

    sc = Scorer()
    protos = Q.build_prototypes(sc.metric, lat, {it["id"] for it in sources})
    proto_ids = list(STYLES)
    proto_mat = np.stack([protos[s] for s in proto_ids])
    t_proto = time.time()

    out = open(args.out, "w", encoding="utf-8")
    meta = {"n_sources": len(sources), "seed": Q.SEED, "configs": run_cfgs,
            "feasible": {c: 0 for c in run_cfgs}, "infeasible": {c: 0 for c in run_cfgs},
            "reasons": {c: {} for c in run_cfgs}, "seconds": {}}
    t_cfg: Dict[str, float] = {c: 0.0 for c in run_cfgs}

    for k, it in enumerate(sources):
        src_style = it["style"]
        tgt_style = Q.target_style_for(src_style)
        src_img = _load_rgb(it["path"])
        src_rgb = _to_tensor(src_img)[None]
        sp = F.normalize(dino[src_style]["patches"][dino[src_style]["index"][it["id"]]].to("cuda").float(), dim=-1)
        ret = retrieval[src_style]
        cands = ret["by_target"][tgt_style]["donor_ids"][ret["source_ids"].index(it["id"])]
        cands = [c for c in cands if dino[tgt_style]["index"].get(c) is not None][:8]
        if len(cands) < 2:
            cls_s = dino[src_style]["cls"][dino[src_style]["index"][it["id"]]].float()
            topv, topi = (dino[tgt_style]["cls"].float() @ cls_s).topk(min(8, len(dino[tgt_style]["index"])))
            cands = [list(dino[tgt_style]["index"])[j] for j in topi.tolist()]
        rng = random.Random(args.seed * 100003 + k)
        chosen = choose_donor(sp, dino, src_style, tgt_style, it["id"], cands, args.seed)
        if chosen is None:
            for c in run_cfgs:
                meta["infeasible"][c] += 1
            continue
        _, donor_id, q64, m64 = chosen
        dp = F.normalize(dino[tgt_style]["patches"][dino[tgt_style]["index"][donor_id]].float().to("cuda"), dim=-1)
        q128, m128 = cell_qmap(sp, dp, 128)
        donor_rgb = _to_tensor(_load_rgb(id_map[donor_id]["path"]))[None]
        texture, sal8 = texture_saliency(src_img)

        z_src = lat[src_style]["latents"][lat[src_style]["index"][it["id"]]][None].to("cuda")
        csd_src = csd_model.feats_pils(sc.metric.csd, [src_img], batch=1)[0]
        base_scores = csd_src @ proto_mat.T
        src_cls = encode_dino(sc.metric.dino, [src_img], batch=1)[0][0]
        base_T = float(base_scores[proto_ids.index(tgt_style)])
        base_S = float(base_scores[proto_ids.index(src_style)])
        src_pat = dino[src_style]["patches"][dino[src_style]["index"][it["id"]]].float()
        src_pat_n = F.normalize(src_pat, dim=-1).cuda()

        plans = []  # (config, h_rgb, rects, alpha_sum, extra)
        maskm_by: Dict[str, dict] = {}
        for cfg in run_cfgs:
            if cfg in PM.PATTERN_SPECS or cfg in PM.ABLATION_SPECS:
                size = (PM.PATTERN_SPECS.get(cfg) or PM.ABLATION_SPECS[cfg])["size"]
                qm, mm = (q64, m64) if size == 64 else (q128, m128)
                cells = PM.select_cells(cfg, qmap=qm, mmap=mm, texture=texture, saliency=sal8,
                                        rng=random.Random(rng.random()))
                if cells is None:
                    meta["infeasible"][cfg] += 1
                    continue
                rects = PM.rects_from_cells(cells, size)
                matches = cells_matches(sp, dp, cells, size)
                if len(matches) < len(cells):
                    meta["infeasible"][cfg] += 1
                    continue
                h = C.construct_A(src_rgb, donor_rgb, matches)
                plans.append((cfg, h, rects, 1.0, PM.mask_metrics(cells, size, qmap=qm, saliency=sal8)))
            elif cfg.startswith("FEATHER_"):
                cells = PM.select_cells("R128", qmap=q128, mmap=m128, rng=random.Random(rng.random()))
                if cells is None:
                    meta["infeasible"][cfg] += 1
                    continue
                rects = PM.rects_from_cells(cells, 128)
                matches = cells_matches(sp, dp, cells, 128)
                if len(matches) < len(cells):
                    meta["infeasible"][cfg] += 1
                    continue
                if cfg == "FEATHER_EFF":
                    a0 = rasterize(rects, feather=True, feather_width=16.0)
                    scale = 0.25 / max(float(a0.mean()), 1e-6)
                    h = C.construct_feather(src_rgb, donor_rgb, matches, feather_width=16.0, alpha_scale=scale)
                    alpha_sum = float(rasterize(rects, feather=True, feather_width=16.0, alpha_scale=scale).mean())
                else:
                    w = float(cfg.split("_")[1])
                    h = C.construct_feather(src_rgb, donor_rgb, matches, feather_width=w)
                    alpha_sum = float(rasterize(rects, feather=True, feather_width=w).mean())
                plans.append((cfg, h, rects, alpha_sum, PM.mask_metrics(cells, 128, qmap=q128, saliency=sal8)))
            elif cfg == "A_P5":
                rects = build_patch_layout("P5", seed=args.seed)
                cells = [(r.x // 128, r.y // 128) for r in rects]
                matches = cells_matches(sp, dp, cells, 128)
                if len(matches) < len(cells):
                    meta["infeasible"][cfg] += 1
                    continue
                h = C.construct_A(src_rgb, donor_rgb, matches)
                plans.append((cfg, h, rects, 1.0, PM.mask_metrics(cells, 128, qmap=q128, saliency=sal8)))
            elif cfg == "CTRL_rand":
                cells = PM.select_cells("R64", qmap=q64, mmap=m64, rng=random.Random(rng.random()))
                if cells is None:
                    meta["infeasible"][cfg] += 1
                    continue
                other = [s for s in STYLES if s not in (src_style, tgt_style)]
                rs = random.Random(rng.random()).choice(other)
                other_ids = [x for x in dino[rs]["index"]]
                did = random.Random(rng.random()).choice(other_ids)
                drgb = _to_tensor(_load_rgb(id_map[did]["path"]))[None]
                rrng = random.Random(rng.random())
                rects = PM.rects_from_cells(cells, 64)
                matches = [Match(src=r, donor=Rect(rrng.randrange(0, 449, 16), rrng.randrange(0, 449, 16), 64, 64),
                                 q=0.0, mutual_ratio=0.0) for r in rects]
                plans.append((cfg, C.construct_A(src_rgb, drgb, matches), rects, 1.0,
                              PM.mask_metrics(cells, 64, qmap=q64, saliency=sal8)))
            elif cfg == "CTRL_same":
                same = [x for x in dino[src_style]["index"] if x != it["id"]]
                did = random.Random(rng.random()).choice(same)
                dp2 = F.normalize(dino[src_style]["patches"][dino[src_style]["index"][did]].float().to("cuda"), dim=-1)
                q2, m2 = cell_qmap(sp, dp2, 64)
                cells = PM.select_cells("R64", qmap=q2, mmap=m2, rng=random.Random(rng.random()))
                if cells is None:
                    meta["infeasible"][cfg] += 1
                    continue
                drgb = _to_tensor(_load_rgb(id_map[did]["path"]))[None]
                rects = PM.rects_from_cells(cells, 64)
                matches = cells_matches(sp, dp2, cells, 64)
                plans.append((cfg, C.construct_A(src_rgb, drgb, matches), rects, 1.0,
                              PM.mask_metrics(cells, 64, qmap=q2, saliency=sal8)))
            elif cfg == "CTRL_destyle":
                cells = PM.select_cells("R64", qmap=q64, mmap=m64, rng=random.Random(rng.random()))
                if cells is None:
                    meta["infeasible"][cfg] += 1
                    continue
                rects = PM.rects_from_cells(cells, 64)
                matches = cells_matches(sp, dp, cells, 64)
                plans.append((cfg, rgb_adain(src_rgb, donor_rgb, matches), rects, 1.0,
                              PM.mask_metrics(cells, 64, qmap=q64, saliency=sal8)))
            elif cfg == "CTRL_seed":
                cells = PM.select_cells("R64", qmap=q64, mmap=m64,
                                        rng=random.Random(rng.random() + 0.5))
                if cells is None:
                    meta["infeasible"][cfg] += 1
                    continue
                rects = PM.rects_from_cells(cells, 64)
                matches = cells_matches(sp, dp, cells, 64)
                plans.append((cfg, C.construct_A(src_rgb, donor_rgb, matches), rects, 1.0,
                              PM.mask_metrics(cells, 64, qmap=q64, saliency=sal8)))
            elif cfg == "CTRL_none":
                plans.append((cfg, src_rgb.clone(), [], 0.0, {
                    "coverage_actual": 0.0, "component_count": 0, "boundary_length": 0,
                    "mean_patch_distance": 0.0,
                    "quadrant_coverage": {"q_tl": 0, "q_tr": 0, "q_bl": 0, "q_br": 0},
                    "spatial_periodicity": 0.0, "match_q_mean": 0.0, "salient_region_fraction": 0.0,
                }))
            else:
                continue  # LL 组在 z 之后构造

        # 2) 编码所有 RGB 构造
        z_by = {}
        for s in range(0, len(plans), 8):
            chunk = plans[s : s + 8]
            zs = sc.metric.encode(torch.cat([c[1] for c in chunk], 0))
            for i, (cfg, _, _, _, _) in enumerate(chunk):
                z_by[cfg] = zs[i : i + 1]
        maskm_by = {p[0]: p[4] for p in plans}

        # 3) LL 插值（复用 R128 硬几何）
        if any(c in run_cfgs for c in ("LL_0", "LL_50", "LL_100")) and "R128" in z_by:
            zh = z_by["R128"]
            ll_x, lh_x, hl_x, hh_x = dwt2_haar(z_src)
            ll_a, lh_a, hl_a, hh_a = dwt2_haar(zh)
            ll_rects = [p[2] for p in plans if p[0] == "R128"][0]
            for cfg, a in (("LL_0", 0.0), ("LL_50", 0.5), ("LL_100", 1.0)):
                if cfg not in run_cfgs:
                    continue
                z = idwt2_haar(ll_x + a * (ll_a - ll_x), lh_a, hl_a, hh_a)
                z_by[cfg] = z
                plans.append((cfg, None, ll_rects, 1.0, maskm_by["R128"]))

        # 4) 打分（每批 8 张，避免 12GB 显存峰值）
        score_items = [p for p in plans if p[0] in z_by]
        token_mask_cache: Dict[tuple, torch.Tensor] = {}
        pixel_mask_cache: Dict[tuple, torch.Tensor] = {}
        sub_mask_cache: Dict[tuple, torch.Tensor] = {}
        latent_mask_cache: Dict[tuple, torch.Tensor] = {}
        for s in range(0, len(score_items), 8):
            chunk = score_items[s : s + 8]
            t_s = time.time()
            res = sc.score([z_by[p[0]] for p in chunk], src_img, proto_ids, proto_mat)
            dt = time.time() - t_s
            for j, (cfg, h, rects, alpha_sum, maskm) in enumerate(chunk):
                gen = res["pils"][j]
                arr = np.asarray(gen, dtype=np.uint8)
                tm = pm = sm = lm = None
                if rects:
                    key = tuple((r.x, r.y, r.w, r.h) for r in rects)
                    if key not in token_mask_cache:
                        token_mask_cache[key] = _token_mask(rects)
                        pixel_mask_cache[key] = _pixel_mask(rects)
                        sub_mask_cache[key] = _sub_mask(rects)
                        latent_mask_cache[key] = _latent_mask(rects)
                    tm, pm, sm = token_mask_cache[key], pixel_mask_cache[key], sub_mask_cache[key]
                    lm = latent_mask_cache[key]
                bm, im, seam = seam_stats(
                    torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1)[None], rects
                ) if rects else (0.0, 0.0, 0.0)
                parts = band_parts(z_src, z_by[cfg], rects)
                dino_cos = float(res["cls"][j] @ src_cls)
                patch_cos = float(
                    (res["pat"][j] * src_pat_n).sum(-1)[tm.view(-1).cuda()].mean()
                ) if tm is not None else 0.0
                struct = structure_corr(arr, res["arr_src"], pm) if pm is not None else 0.0
                diff = (z_by[cfg] - z_src).abs()
                if lm is not None:
                    w_out = (~lm).float()[None, None].to(diff.device)
                    outside = float((diff * w_out).sum() / (w_out.sum() * diff.shape[1]).clamp_min(1.0))
                else:
                    outside = float(diff.mean())
                csd5 = np.asarray(res["csd5"][j], dtype=np.float64)
                record = {
                    "config": cfg,
                    "source_id": it["id"],
                    "source_style": src_style,
                    "target_style": tgt_style,
                    "donor_id": donor_id,
                    "mask": maskm,
                    "alpha_sum": float(alpha_sum),
                    "coverage": maskm["coverage_actual"],
                    "q_mean": maskm["match_q_mean"],
                    "csd5": [float(v) for v in csd5],
                    "csd5_base": [float(v) for v in base_scores],
                    "csd_T": float(csd5[proto_ids.index(tgt_style)]),
                    "csd_S": float(csd5[proto_ids.index(src_style)]),
                    "csd_T_base": base_T,
                    "csd_S_base": base_S,
                    "csd_donor": float(csd5[proto_ids.index(tgt_style)]),
                    "argmax_style": proto_ids[int(np.argmax(csd5))],
                    "argmax_base": proto_ids[int(np.argmax(base_scores))],
                    "dino_c": dino_cos,
                    "dino_patch_cos": patch_cos,
                    "structure_corr": struct,
                    "lpips": res["lpips"][j],
                    "seam_ratio": seam,
                    "seam_boundary": bm,
                    "seam_interior": im,
                    "outside_change": outside,
                    "bands": parts["bands"],
                    "R_active": parts["R_active"],
                    "band_inside_frac": parts["inside_frac"],
                    "band_outside_frac": parts["outside_frac"],
                    "batch_seconds": dt / max(len(chunk), 1),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta["feasible"][cfg] += 1
                t_cfg[cfg] = t_cfg.get(cfg, 0.0) + dt / max(len(chunk), 1)
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(sources)}  {time.time()-t0:.0f}s", flush=True)

    out.close()
    meta["seconds"] = {k: round(v, 1) for k, v in t_cfg.items() if v > 0}
    meta["total_seconds"] = round(time.time() - t0, 1)
    meta["proto_seconds"] = round(t_proto - t0, 1)
    Path(args.meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[supp] done {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
