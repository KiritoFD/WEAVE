"""Q0：28 组伪GT构造的 300 样本质量研究（gt-build §3）。

共享匹配集：5 风格 × 60 源图，固定 seed；同一批源图/供体/patch 位置供所有配置共用。
"""
from __future__ import annotations

import argparse
import json
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

import paths  # noqa: E402
import csd_model  # type: ignore  # noqa: E402
import trt_vae_decoder  # noqa: E402

from tools.pseudo_gt import construct as C  # noqa: E402
from tools.pseudo_gt.build_assets import STYLES, STYLE_IDS, load_manifest, _load_rgb, pgt_root, work_root  # noqa: E402
from tools.pseudo_gt.cache import SampleRecord, make_sample_id  # noqa: E402
from tools.pseudo_gt.matching import (  # noqa: E402
    DEVICE,
    assert_cuda,
    encode_dino,
    load_dino_model,
    match_layout,
)
from tools.pseudo_gt.masks import E_SPECS, PATCH_SPECS, build_patch_layout  # noqa: E402
from tools.legacy_utils.inference import download_vae_with_fallback  # noqa: E402

SPECS_A = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
SPECS_BCD = ["P2", "P5", "P7", "P9"]
CONFIGS: List[Tuple[str, str]] = (
    [("A", s) for s in SPECS_A]
    + [("B", s) for s in SPECS_BCD]
    + [("C", s) for s in SPECS_BCD]
    + [("D", s) for s in SPECS_BCD]
    + [("E", s) for s in E_SPECS]
)
DIAGNOSTIC: List[Tuple[str, str]] = [("D_HF", "P5")]

N_PER_STYLE = 60
SEED = 2024


# ---------------------------------------------------------------- assets
def load_latents() -> Dict[str, dict]:
    out = {}
    for style in STYLES:
        p = pgt_root() / "orig_mean_fp16" / f"{STYLE_IDS[style]:02d}_{style}.pt"
        payload = torch.load(p, map_location="cpu", weights_only=False)
        out[style] = {
            "latents": payload["latents"].contiguous(),  # 保持 fp16，用时再转 float
            "ids": payload["ids"],
            "index": {sid: i for i, sid in enumerate(payload["ids"])},
        }
        del payload
    return out


def load_dino() -> Dict[str, dict]:
    out = {}
    for style in STYLES:
        payload = torch.load(pgt_root() / "dino_224" / f"{style}.pt", map_location="cpu", weights_only=False)
        out[style] = {
            "cls": F.normalize(payload["cls"].float(), dim=-1),
            "patches": payload["patches"],  # 保持 fp16 原值；用时再转 float 并归一化（fp16 归一化会出 nan）
            "index": {sid: i for i, sid in enumerate(payload["ids"])},
        }
        del payload
    return out


def load_retrieval() -> dict:
    return json.loads((pgt_root() / "retrieval" / "top8.json").read_text(encoding="utf-8"))["styles"]


def shared_sources(items: List[dict], test_stems: set[str]) -> List[dict]:
    """每风格 60 张训练池源图（排除 test），固定 seed。"""
    rng = random.Random(SEED)
    out = []
    for style in STYLES:
        pool = [it for it in items if it["style"] == style and it["split"] == "train"]
        pool = [it for it in pool if it["stem"] not in test_stems]
        pool.sort(key=lambda it: it["id"])
        out.extend(rng.sample(pool, N_PER_STYLE))
    return out


def target_style_for(style: str) -> str:
    return STYLES[(STYLES.index(style) + 1) % len(STYLES)]


# ---------------------------------------------------------------- metrics
class Metrics:
    def __init__(self):
        from lpips import LPIPS

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        self.vae = download_vae_with_fallback("stabilityai/sd-vae-ft-ema", device=DEVICE)
        self.dino = load_dino_model()
        self.lpips = LPIPS(net="alex", verbose=False).to(DEVICE).eval()
        self.csd, self.csd_ck, self.csd_msg = csd_model.load_csd("cuda")
        # TRT 解码器（batch=8 引擎），比 eager VAE 解码快数倍；无引擎时自动回退 torch
        self.trt = trt_vae_decoder.TrtVaeDecoder(REPO_ROOT / "runs" / "trt" / "engines",
                                                 self.vae, precision="fp16", device=DEVICE)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> List[Image.Image]:
        z = z.to(DEVICE, torch.float16).contiguous(memory_format=torch.channels_last)
        assert_cuda(z, "decode latent")
        n = z.shape[0]
        parts = []
        for s in range(0, n, 8):
            chunk = z[s:s + 8]
            if chunk.shape[0] < 8:
                chunk = torch.cat([chunk, chunk[-1:].expand(8 - chunk.shape[0], -1, -1, -1)], 0)
            parts.append(self.trt.decode(chunk)[: min(8, n - s)])
        pixels = torch.cat(parts, 0)
        assert_cuda(pixels, "vae decode output")
        arr = (pixels.float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        return [Image.fromarray(a) for a in arr]

    def score(self, z_gen: torch.Tensor, src_imgs: List[Image.Image], proto_t: np.ndarray, proto_s: np.ndarray, base: dict) -> dict:
        """批量打分。src_imgs 与 z_gen 同批（同源图重复即可）。返回列表。"""
        gen = self.decode(z_gen)
        cg = csd_model.feats_pils(self.csd, gen, batch=len(gen))
        cls_g, _ = encode_dino(self.dino, gen, batch=len(gen))
        def to_t(im):
            return (torch.from_numpy(np.array(im, dtype=np.float32)).permute(2, 0, 1).to(DEVICE) / 127.5 - 1)
        gen_t = torch.stack([to_t(im) for im in gen])
        src_t = torch.stack([to_t(im) for im in src_imgs])
        with torch.inference_mode():
            lp = self.lpips(gen_t, src_t).flatten()
        return {
            "csd_T": [float(x @ proto_t) for x in cg],
            "csd_S": [float(x @ proto_s) for x in cg],
            "dino_c": [float(x @ base["cls_s"]) for x in cls_g],
            "lpips": [float(v) for v in lp],
            "gen": gen,
        }

    def source_baseline(self, src_img: Image.Image, proto_t: np.ndarray, proto_s: np.ndarray) -> dict:
        cs = csd_model.feats_pils(self.csd, [src_img], batch=1)[0]
        cls_s, _ = encode_dino(self.dino, [src_img], batch=1)
        return {"csd_T": float(cs @ proto_t), "csd_S": float(cs @ proto_s), "cls_s": cls_s[0]}

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.to(DEVICE, torch.float16).clamp(0, 1) * 2 - 1
        rgb = rgb.contiguous(memory_format=torch.channels_last)
        assert_cuda(rgb, "vae encode input")
        # 模型权重是普通张量（requires_grad=True），encode 必须在 inference_mode 下跑；
        # 返回前 clone 一次，把 latent 从 inference tensor 转成普通张量，避免后续保存报错。
        with torch.inference_mode():
            z = self.vae.encode(rgb).latent_dist.mean * self.vae.config.scaling_factor
        return z.clone()


def build_prototypes(metric: Metrics, lat: Dict[str, dict], src_ids: set[str]) -> Dict[str, np.ndarray]:
    """冻结训练池风格原型（排除 300 源图），缓存一次。"""
    path = pgt_root() / "csd_proto.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    protos = {}
    rng = random.Random(SEED + 1)
    for style in STYLES:
        payload = torch.load(pgt_root() / "orig_mean_fp16" / f"{STYLE_IDS[style]:02d}_{style}.pt", map_location="cpu", weights_only=False)
        ids = [i for i in payload["ids"] if i not in src_ids]
        pick = rng.sample(ids, min(500, len(ids)))
        idx = [payload["ids"].index(i) for i in pick]
        feats = []
        for s in range(0, len(idx), 16):
            z = payload["latents"][idx[s : s + 16]].float()
            pils = metric.decode(z)
            feats.append(csd_model.feats_pils(metric.csd, pils, batch=len(pils)))
        f = np.concatenate(feats, 0).astype(np.float32).mean(0)
        protos[style] = f / np.linalg.norm(f)
        print(f"[proto] {style}: {len(idx)} 张", flush=True)
    torch.save(protos, path)
    return protos


# ---------------------------------------------------------------- matching
def pick_donor(
    src_style: str,
    src_id: str,
    tgt_style: str,
    donor_ids: Sequence[str],
    dino: Dict[str, dict],
    rects_sets: Dict[str, list],
) -> Tuple[Optional[str], Dict[str, list], Dict[str, float]]:
    """在 top-8 候选中选 patch 匹配质量最高的供体（对各 spec 取平均 q）。"""
    src_pat = F.normalize(dino[src_style]["patches"][dino[src_style]["index"][src_id]].to(DEVICE).float(), dim=-1)
    best = (None, {}, {}, 0.0)
    for donor_id in donor_ids:
        dpat = F.normalize(dino[tgt_style]["patches"][dino[tgt_style]["index"][donor_id]].to(DEVICE).float(), dim=-1)
        matches, qs = {}, {}
        ok = True
        for name, rects in rects_sets.items():
            m, q = match_layout(src_pat, dpat, rects)
            if m is None:
                ok = False
                break
            matches[name] = m
            qs[name] = q
        if not ok:
            continue
        avg = float(np.mean(list(qs.values())))
        if avg > best[3]:
            best = (donor_id, matches, qs, avg)
    return best[0], best[1], best[2]


def main() -> None:
    global SEED, N_PER_STYLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(paths.project_root()) / "runs/p2/pgt_q0.json"))
    ap.add_argument("--limit", type=int, default=0, help="调试用：只跑前 N 个源图")
    ap.add_argument("--seed", type=int, default=SEED, help="源图采样与布局种子（用于跨 seed 复核）")
    ap.add_argument("--n-per-style", type=int, default=N_PER_STYLE, help="每风格源图数")
    ap.add_argument("--configs", default="", help="逗号分隔的子集，如 A_P5,A_P2,A_P7,E7")
    args = ap.parse_args()
    SEED, N_PER_STYLE = args.seed, args.n_per_style
    run_cfgs = CONFIGS + DIAGNOSTIC
    if args.configs:
        wanted = {x.strip() for x in args.configs.split(",") if x.strip()}
        run_cfgs = [(m, s) for m, s in run_cfgs
                    if f"{m}_{s}" in wanted or (m == "E" and s in wanted)]
        if not run_cfgs:
            raise SystemExit(f"--configs 未匹配任何配置: {sorted(wanted)}")
    t_start = time.time()

    items = load_manifest()
    repo_root = Path(__file__).resolve().parents[2]
    test_stems = {p.stem for p in (repo_root / "data/test").rglob("*.jpg")}
    test_stems |= {p.stem for p in (repo_root / "datasets/distinct5_512/test").rglob("*.jpg")}
    sources = shared_sources(items, test_stems)
    if args.limit:
        sources = sources[: args.limit]
    print(f"[q0] 源图 {len(sources)}（每风格 {N_PER_STYLE}）", flush=True)

    lat, dino, retrieval = load_latents(), load_dino(), load_retrieval()
    metric = Metrics()
    protos = build_prototypes(metric, lat, {it["id"] for it in sources})

    # 共享匹配：每个源图一次，所有配置复用
    lat_index = {it["id"]: lat[it["style"]]["index"][it["id"]] for it in sources}
    plans = []
    need_specs = set(SPECS_A) | set(E_SPECS)
    rects_sets = {name: build_patch_layout(name, seed=SEED) for name in need_specs}
    t_match = time.time()
    for k, it in enumerate(sources):
        tgt = target_style_for(it["style"])
        ret = retrieval[it["style"]]
        cands = ret["by_target"][tgt]["donor_ids"][ret["source_ids"].index(it["id"])]
        cands = [c for c in cands if dino[tgt]["index"].get(c) is not None][:8]
        if len(cands) < 2:
            cls_s = dino[it["style"]]["cls"][dino[it["style"]]["index"][it["id"]]].float()
            topv, topi = (dino[tgt]["cls"].float() @ cls_s).topk(min(8, len(dino[tgt]["index"])))
            ids = list(dino[tgt]["index"])
            cands = [ids[j] for j in topi.tolist()]
        donor_id, matches, qs = pick_donor(it["style"], it["id"], tgt, cands, dino, rects_sets)
        if donor_id is None:
            plans.append(None)
            continue
        plans.append(
            {
                "source": it,
                "target_style": tgt,
                "donor_id": donor_id,
                "matches": matches,
                "qs": qs,
                "z_src": lat[it["style"]]["latents"][lat_index[it["id"]]][None].float().to(DEVICE),
                "z_don": lat[tgt]["latents"][lat[tgt]["index"][donor_id]][None].float().to(DEVICE),
            }
        )
        if (k + 1) % 20 == 0:
            print(f"  match {k+1}/{len(sources)}  {time.time()-t_match:.1f}s", flush=True)
    print(f"[q0] 共享匹配完成 {sum(1 for p in plans if p)}/{len(plans)}  {time.time()-t_match:.1f}s", flush=True)

    rows: Dict[str, List[dict]] = {f"{m}_{s}": [] for m, s in run_cfgs}
    failed: Dict[str, int] = {k: 0 for k in rows}
    errs: Dict[str, str] = {}
    id_map = {it["id"]: it for it in items}

    for name, plan in enumerate(plans):
        if plan is None:
            for key in rows:
                failed[key] += 1
            continue
        src_it = plan["source"]
        src_img = _load_rgb(src_it["path"])
        src_rgb = _to_tensor(src_img)[None]
        donor_rgb = _to_tensor(_load_rgb(id_map[plan["donor_id"]]["path"]))[None]
        proto_t = protos[plan["target_style"]]
        proto_s = protos[src_it["style"]]
        base = metric.source_baseline(src_img, proto_t, proto_s)
        todo, rgb_todo = [], []
        for method, spec in run_cfgs:
            if spec not in plan["matches"]:
                failed[f"{method}_{spec}"] += 1
                continue
            matches = plan["matches"][spec]
            try:
                if method in {"A", "B", "E"}:
                    if method == "A":
                        rgb = C.construct_A(src_rgb, donor_rgb, matches)
                    elif method == "B":
                        rgb = C.construct_B(src_rgb, donor_rgb, matches)
                    else:
                        rgb = C.construct_E(src_rgb, donor_rgb, matches, feather=E_SPECS[spec]["feather"])
                    rgb_todo.append((method, spec, matches, rgb))
                else:
                    z = C.construct_C(plan["z_src"], plan["z_don"], matches) if method == "C" \
                        else C.construct_D(plan["z_src"], plan["z_don"], matches, keep_ll=(method != "D_HF"))
                    todo.append((method, spec, matches, z))
            except Exception as exc:  # 记录失败但不中断
                failed[f"{method}_{spec}"] += 1
                errs.setdefault(f"{method}_{spec}", f"{type(exc).__name__}: {exc}")
                rows[f"{method}_{spec}"].append({"error": f"{type(exc).__name__}: {exc}"})
        # RGB 类构造按 8 张一批做 VAE 编码
        for s in range(0, len(rgb_todo), 8):
            chunk = rgb_todo[s:s + 8]
            try:
                zs = metric.encode(torch.cat([c[3] for c in chunk], 0))
                for i, (method, spec, matches, _) in enumerate(chunk):
                    todo.append((method, spec, matches, zs[i:i + 1]))
            except Exception as exc:
                for method, spec, _, _ in chunk:
                    failed[f"{method}_{spec}"] += 1
                    errs.setdefault(f"{method}_{spec}", f"encode {type(exc).__name__}: {exc}")
        # 打分按 8 张一批（TRT decoder 引擎固定 batch 8）
        for s in range(0, len(todo), 8):
            chunk = todo[s:s + 8]
            try:
                z = torch.cat([c[3] for c in chunk], 0)
                sc = metric.score(z, [src_img] * len(chunk), proto_t, proto_s, base)
            except Exception as exc:
                for method, spec, _, _ in chunk:
                    failed[f"{method}_{spec}"] += 1
                    errs.setdefault(f"{method}_{spec}", f"score {type(exc).__name__}: {exc}")
                continue
            for j, (method, spec, matches, zj) in enumerate(chunk):
                seam = C.seam_gradient(
                    torch.from_numpy(np.array(sc["gen"][j], dtype=np.float32) / 255.0).permute(2, 0, 1)[None],
                    [m.src for m in matches],
                )
                bands = C.band_energy_share(plan["z_src"], zj)
                rows[f"{method}_{spec}"].append(
                    {
                        "source_id": src_it["id"],
                        "source_style": src_it["style"],
                        "target_style": plan["target_style"],
                        "donor_id": plan["donor_id"],
                        "q": plan["qs"][spec],
                        "csd_dT": sc["csd_T"][j] - base["csd_T"],
                        "csd_dS": sc["csd_S"][j] - base["csd_S"],
                        "csd_dM": (sc["csd_T"][j] - base["csd_T"]) - (sc["csd_S"][j] - base["csd_S"]),
                        "dino_c": sc["dino_c"][j],
                        "lpips": sc["lpips"][j],
                        "seam": seam,
                        "bands": bands,
                        "coverage": float(sum(m.src.area for m in matches) / (512 ** 2)),
                    }
                )
        if (name + 1) % 10 == 0:
            print(f"  build {name+1}/{len(plans)}  {time.time()-t_start:.1f}s", flush=True)

    out = summarise(rows, failed, len(plans))
    out["errors"] = errs
    print("[q0] errors:", json.dumps(errs, ensure_ascii=False), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: {
        "n": v.get("n"), "failed": v.get("failed"), "pass_rate": v.get("pass_rate"),
        "csd_dT": (v.get("csd_dT") or {}).get("mean"),
        "csd_dM": (v.get("csd_dM") or {}).get("mean"),
        "dino_c": (v.get("dino_c") or {}).get("mean"),
        "lpips": (v.get("lpips") or {}).get("mean"),
    } for k, v in out["configs"].items()}, ensure_ascii=False, indent=1))
    print(f"[q0] done {time.time()-t_start:.1f}s -> {args.out}", flush=True)


def _strat_bootstrap(by_src_style: Dict[str, List[float]], n_boot: int = 2000, seed: int = SEED) -> dict:
    """以源图为重采样单位、按源风格分层的 95% bootstrap CI。"""
    rng = np.random.default_rng(seed)
    styles = sorted(by_src_style)
    strata = [np.asarray(by_src_style[s], dtype=np.float64) for s in styles if by_src_style[s]]
    if not strata:
        return {"lo": None, "hi": None}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        vals = [rng.choice(st, size=len(st), replace=True) for st in strata]
        boots[b] = np.concatenate(vals).mean()
    return {"lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}


def summarise(rows: Dict[str, List[dict]], failed: Dict[str, int], n_src: int) -> dict:
    out = {"n_source": n_src, "seed": SEED, "configs": {}}
    for key, vals in rows.items():
        valid = [v for v in vals if "error" not in v]
        if not valid:
            out["configs"][key] = {"n": 0, "failed": failed[key], "pass_rate": 0.0}
            continue
        def stat(field: str) -> dict:
            arr = np.asarray([v[field] for v in valid], dtype=np.float64)
            by_style: Dict[str, List[float]] = {}
            for v in valid:
                by_style.setdefault(v["source_style"], []).append(float(v[field]))
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "frac_pos": float((arr > 0).mean()),
                "ci": _strat_bootstrap(by_style),
            }
        rec = {
            "n": len(valid),
            "failed": failed[key],
            "pass_rate": round(len(valid) / max(n_src, 1), 4),
            "csd_dT": stat("csd_dT"),
            "csd_dS": stat("csd_dS"),
            "csd_dM": stat("csd_dM"),
            "dino_c": stat("dino_c"),
            "lpips": stat("lpips"),
            "q": stat("q"),
            "seam": stat("seam"),
            "coverage": stat("coverage"),
            "band_LL": float(np.mean([v["bands"]["LL"] for v in valid])),
            "band_LH": float(np.mean([v["bands"]["LH"] for v in valid])),
            "band_HL": float(np.mean([v["bands"]["HL"] for v in valid])),
            "band_HH": float(np.mean([v["bands"]["HH"] for v in valid])),
        }
        out["configs"][key] = rec
    return out


def _to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0


if __name__ == "__main__":
    main()
