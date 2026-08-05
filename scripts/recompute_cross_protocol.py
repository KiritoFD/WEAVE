"""Recompute D5-512 / P2A-256 metrics under all vs cross (off-diagonal) protocol.

Uses existing generated images under results/{D5-512,P256}/<method>/.
When per-sample metrics.csv with CLIP/LPIPS/DINO already exist, re-aggregates only.
Otherwise runs local GPU evaluation (DINOv2 + CLIP + LPIPS).

Output: results/cross_protocol/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Paper main-table methods (folder name -> display name)
D5_METHODS = {
    "identity": "Identity",
    "tgt": "Target Style",
    "sdturbo": "SD-Turbo",
    "stylealigned": "StyleAligned",
    "zstar": "Z-STAR",
    "styleshot": "StyleShot",
    "styleid": "StyleID",
    "cut": "CUT",
    "samst": "SaMST",
    "samam": "SaMam",
    "stytr2": "StyTR-2",
    "aespa": "AesPA-Net",
    "seedream": "Seedream 4.5",
    "wct": "Latent-WCT",  # folder may be pixel WCT; still report
    "weave": "WEAVE",
    "weave_oriented_e4": "WEAVE (oriented e4)",
}

P2A_METHODS = dict(D5_METHODS)

D5_STYLES = [
    "Early_Renaissance",
    "Impressionism",
    "Minimalism",
    "Rococo",
    "Ukiyo_e",
]
P2A_STYLES = ["photo", "cezanne", "monet", "vangogh", "Hayao"]


def mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def parse_d5_name(name: str) -> tuple[str, str, str] | None:
    """Parse D5 filename to (src_style, src_stem, tgt_style)."""
    stem = Path(name).stem
    # native: {src_style}_{src_stem}_to_{tgt}
    # baseline: {src_style}__{src_stem}__to__{tgt}
    # adain-like: {src_style}__{src_style}__{stem}__to__{tgt}
    if "__to__" in stem:
        left, tgt = stem.split("__to__", 1)
        parts = left.split("__")
        if len(parts) >= 2:
            # could be src_style__stem or src_style__src_style__stem
            if parts[0] in D5_STYLES:
                src = parts[0]
                if len(parts) >= 3 and parts[1] == src:
                    src_stem = "__".join(parts[2:])
                else:
                    src_stem = "__".join(parts[1:])
                if tgt in D5_STYLES:
                    return src, src_stem, tgt
        return None
    if "_to_" in stem:
        # prefer last occurrence of style after _to_
        for tgt in D5_STYLES:
            suffix = f"_to_{tgt}"
            if stem.endswith(suffix):
                left = stem[: -len(suffix)]
                for src in D5_STYLES:
                    if left.startswith(src + "_"):
                        return src, left[len(src) + 1 :], tgt
                    if left.startswith(src + "__"):
                        rest = left[len(src) + 2 :]
                        if rest.startswith(src + "__"):
                            rest = rest[len(src) + 2 :]
                        return src, rest, tgt
    return None


def parse_p2a_name(name: str) -> tuple[str, str, str] | None:
    stem = Path(name).stem
    # Formats:
    #   Hayao_0_to_cezanne
    #   Hayao__0__to__cezanne
    #   photo_00012_to_vangogh
    if "__to__" in stem:
        left, tgt = stem.split("__to__", 1)
        left = left.replace("__", "_")
    elif "_to_" in stem:
        left, tgt = stem.rsplit("_to_", 1)
    else:
        return None
    tgt_norm = None
    for s in P2A_STYLES:
        if s.lower() == tgt.lower() or s == tgt:
            tgt_norm = s
            break
    if tgt_norm is None:
        return None
    for s in P2A_STYLES:
        if left == s or left.lower() == s.lower():
            return s, left, tgt_norm
        pref = s + "_"
        if left.startswith(pref) or left.lower().startswith(s.lower() + "_"):
            # strip first prefix only
            if left.startswith(pref):
                return s, left[len(pref) :], tgt_norm
            # case-insensitive
            idx = left.lower().find("_")
            return s, left[idx + 1 :], tgt_norm
    return None


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    # prefer flat images; skip nested duplicates of same tree
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if imgs:
        return sorted(imgs)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_existing_metrics(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        return None
    need = {"src_style", "tgt_style"}
    if not need.issubset(rows[0].keys()):
        return None
    return rows


def aggregate(rows: list[dict], metric_keys: list[str]) -> dict:
    all_r = rows
    cross = [r for r in rows if r.get("src_style") != r.get("tgt_style")]
    out = {"n_all": len(all_r), "n_cross": len(cross)}
    for k in metric_keys:
        if k not in rows[0]:
            continue
        def vals(rs):
            xs = []
            for r in rs:
                try:
                    xs.append(float(r[k]))
                except Exception:
                    pass
            return xs
        va, vc = vals(all_r), vals(cross)
        out[f"all_{k}"] = mean(va)
        out[f"cross_{k}"] = mean(vc)
    return out


class MetricEngine:
    def __init__(self, device: str, dino_cache: str = "", allow_network: bool = False):
        self.device = device
        self.dino = None
        self.clip = None
        self.clip_proc = None
        self.lpips = None
        self.dino_cache = dino_cache
        self.allow_network = allow_network
        self.tf_dino = T.Compose(
            [
                T.Resize(224, interpolation=Image.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.tf_lpips = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    def _ensure_dino(self):
        if self.dino is not None:
            return
        from transformers import AutoModel
        import os

        os.environ["HF_HUB_OFFLINE"] = "0" if self.allow_network else "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "0" if self.allow_network else "1"
        name = "facebook/dinov2-small"
        kwargs = {}
        if self.dino_cache:
            # try local snapshot
            hub = Path(self.dino_cache) / "hub" / "models--facebook--dinov2-small" / "snapshots"
            if hub.exists():
                revs = [p for p in hub.iterdir() if p.is_dir()]
                if revs:
                    name = str(revs[0])
            else:
                kwargs["cache_dir"] = self.dino_cache
        print(f"[INFO] load DINO {name}")
        self.dino = AutoModel.from_pretrained(name, **kwargs).to(self.device).eval()

    def _ensure_clip(self):
        if self.clip is not None:
            return
        from transformers import CLIPModel, CLIPProcessor
        import os

        os.environ["HF_HUB_OFFLINE"] = "0" if self.allow_network else "1"
        local = ROOT / "runs" / "cache" / "manual_clip" / "openai-clip-vit-base-patch32"
        name = str(local) if local.exists() else "openai/clip-vit-base-patch32"
        print(f"[INFO] load CLIP {name}")
        self.clip = CLIPModel.from_pretrained(name).to(self.device).eval()
        try:
            self.clip_proc = CLIPProcessor.from_pretrained(name)
        except Exception:
            self.clip_proc = None
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

    def _ensure_lpips(self):
        if self.lpips is not None:
            return
        import lpips

        print("[INFO] load LPIPS alex")
        self.lpips = lpips.LPIPS(net="alex").to(self.device).eval()

    @torch.inference_mode()
    def dino_cls(self, paths: list[Path], bs: int = 8) -> torch.Tensor:
        self._ensure_dino()
        feats = []
        for i in range(0, len(paths), bs):
            batch = paths[i : i + bs]
            px = torch.stack([self.tf_dino(Image.open(p).convert("RGB")) for p in batch]).to(self.device)
            out = self.dino(px)
            feats.append(F.normalize(out.last_hidden_state[:, 0].float(), dim=-1).cpu())
            del px, out
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def clip_emb(self, paths: list[Path], bs: int = 8) -> torch.Tensor:
        self._ensure_clip()
        feats = []
        for i in range(0, len(paths), bs):
            batch = paths[i : i + bs]
            imgs = [Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC) for p in batch]
            x = torch.stack([T.ToTensor()(im) for im in imgs]).to(self.device)
            x = (x - self.clip_mean) / self.clip_std
            # Some transformers builds return BaseModelOutputWithPooling from get_image_features
            # where pooler_output is already the projected 512-d image embed.
            out = self.clip.get_image_features(pixel_values=x)
            if torch.is_tensor(out):
                e = out
            elif hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
                e = out.pooler_output
            else:
                vis = self.clip.vision_model(pixel_values=x)
                e = self.clip.visual_projection(vis.pooler_output)
            e = F.normalize(e.float(), dim=-1)
            feats.append(e.cpu())
            del x, e
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return torch.cat(feats, dim=0)

    @torch.inference_mode()
    def lpips_pair(self, gen_paths: list[Path], src_paths: list[Path], bs: int = 4) -> list[float]:
        self._ensure_lpips()
        out = []
        for i in range(0, len(gen_paths), bs):
            g = gen_paths[i : i + bs]
            s = src_paths[i : i + bs]
            gt = torch.stack([self.tf_lpips(Image.open(p).convert("RGB")) for p in g]).to(self.device)
            st = torch.stack([self.tf_lpips(Image.open(p).convert("RGB")) for p in s]).to(self.device)
            # lpips expects [-1,1]
            gt = gt * 2 - 1
            st = st * 2 - 1
            d = self.lpips(gt, st).view(-1).detach().cpu().tolist()
            out.extend(float(x) for x in d)
            del gt, st, d
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        return out


def build_rows_from_images(
    method_dir: Path,
    board: str,
    test_dir: Path,
) -> list[dict]:
    styles = D5_STYLES if board == "D5-512" else P2A_STYLES
    parse = parse_d5_name if board == "D5-512" else parse_p2a_name
    imgs = list_images(method_dir)
    rows = []
    for p in imgs:
        parsed = parse(p.name)
        if parsed is None:
            # try parent-relative path
            parsed = parse(p.name)
        if parsed is None:
            continue
        src, stem, tgt = parsed
        # source image path in test set
        src_path = None
        style_dir = test_dir / src
        if not style_dir.is_dir():
            # case variants for Hayao
            for cand in test_dir.iterdir():
                if cand.is_dir() and cand.name.lower() == src.lower():
                    style_dir = cand
                    src = cand.name
                    break
        if style_dir.is_dir():
            cands = [c for c in style_dir.iterdir() if c.is_file() and c.suffix.lower() in IMAGE_EXTS]

            def _strip_style_prefix(cs: str) -> str:
                if cs.startswith(src + "__"):
                    return cs[len(src) + 2 :]
                if cs.startswith(src + "_"):
                    return cs[len(src) + 1 :]
                return cs

            # 1) exact stem match after stripping style prefix
            for c in cands:
                cs2 = _strip_style_prefix(c.stem)
                if cs2 == stem or c.stem == stem or c.name == stem:
                    src_path = c
                    break
            # 2) stem is a contiguous suffix/prefix of candidate (or vice versa), min length 8
            if src_path is None and len(stem) >= 6:
                for c in cands:
                    cs2 = _strip_style_prefix(c.stem)
                    if stem in cs2 or cs2 in stem:
                        # require substantial overlap
                        if min(len(stem), len(cs2)) >= 8 or stem == cs2:
                            src_path = c
                            break
            # 3) P2A numeric ids: stem "0" / "00057" etc.
            if src_path is None and re.fullmatch(r"\d+", stem or ""):
                for c in cands:
                    if c.stem == stem or c.stem.lstrip("0") == stem.lstrip("0") or c.stem == stem.zfill(5):
                        src_path = c
                        break
        rows.append(
            {
                "src_style": src,
                "tgt_style": tgt,
                "src_image": src_path.name if src_path else stem,
                "src_path": str(src_path) if src_path else "",
                "gen_image": p.name,
                "gen_path": str(p),
            }
        )
    return rows


def evaluate_method(
    engine: MetricEngine,
    rows: list[dict],
    test_dir: Path,
    styles: list[str],
    max_refs: int,
    bs: int,
) -> list[dict]:
    # map style -> ref paths
    style_refs: dict[str, list[Path]] = {}
    for s in styles:
        sd = test_dir / s
        if not sd.is_dir():
            for cand in test_dir.iterdir():
                if cand.is_dir() and cand.name.lower() == s.lower():
                    sd = cand
                    break
        if sd.is_dir():
            refs = sorted([p for p in sd.iterdir() if p.suffix.lower() in IMAGE_EXTS])[:max_refs]
            style_refs[s] = refs

    gen_paths = [Path(r["gen_path"]) for r in rows]
    src_paths = []
    for r in rows:
        if r.get("src_path"):
            src_paths.append(Path(r["src_path"]))
        else:
            # fallback identity content loss if missing
            src_paths.append(Path(r["gen_path"]))

    print(f"  DINO on {len(gen_paths)} gens + sources + refs...")
    gen_cls = engine.dino_cls(gen_paths, bs=bs)
    # unique sources
    uniq_src = list(dict.fromkeys(src_paths))
    src_cls_all = engine.dino_cls(uniq_src, bs=bs)
    src_idx = {p: i for i, p in enumerate(uniq_src)}

    ref_cls: dict[str, torch.Tensor] = {}
    for s, refs in style_refs.items():
        if refs:
            ref_cls[s] = engine.dino_cls(refs, bs=bs)

    print("  CLIP...")
    gen_clip = engine.clip_emb(gen_paths, bs=bs)
    # style prototypes: mean CLIP of refs
    proto = {}
    for s, refs in style_refs.items():
        if refs:
            emb = engine.clip_emb(refs, bs=bs)
            proto[s] = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1).squeeze(0)

    print("  LPIPS...")
    lpips_vals = engine.lpips_pair(gen_paths, src_paths, bs=max(1, bs // 2))

    out = []
    for i, r in enumerate(rows):
        sp = src_paths[i]
        dino_c = float((gen_cls[i] * src_cls_all[src_idx[sp]]).sum().item())
        tgt = r["tgt_style"]
        # match proto keys case-insensitively
        ref = None
        for k, v in ref_cls.items():
            if k == tgt or k.lower() == tgt.lower():
                ref = v
                break
        if ref is not None:
            dino_s = float((ref @ gen_cls[i]).max().item())
        else:
            dino_s = 0.0
        pr = None
        for k, v in proto.items():
            if k == tgt or k.lower() == tgt.lower():
                pr = v
                break
        clip_s = float((gen_clip[i] * pr).sum().item()) if pr is not None else 0.0
        out.append(
            {
                **{k: r[k] for k in ("src_style", "tgt_style", "src_image", "gen_image")},
                "clip_style": clip_s,
                "content_lpips": float(lpips_vals[i]),
                "dino_style": dino_s,
                "dino_content": dino_c,
            }
        )
    return out


def process_board(
    board: str,
    results_root: Path,
    test_dir: Path,
    methods: dict[str, str],
    styles: list[str],
    engine: MetricEngine | None,
    out_dir: Path,
    max_refs: int,
    bs: int,
    reuse_only: bool,
) -> dict:
    board_out = out_dir / board
    board_out.mkdir(parents=True, exist_ok=True)
    summary = {}

    # also try known existing metrics locations for D5
    extra_metrics = {}
    if board == "D5-512":
        base = ROOT / "eval_outputs" / "d5_750_baselines"
        mapping = {
            "weave": base / "weave" / "metrics.csv",
            "samam": base / "samam" / "metrics.csv",
            "cut": base / "cut" / "metrics.csv",
            "aespa": base / "aespa" / "metrics.csv",
            "stytr2": base / "stytr2" / "metrics.csv",
            "seedream": base / "seedream" / "metrics.csv",
        }
        for k, p in mapping.items():
            if p.is_file():
                extra_metrics[k] = p

    for folder, display in methods.items():
        method_dir = results_root / folder
        print(f"\n[{board}] {display} ({folder})")
        rows = None
        source = None

        # 1) existing metrics next to method
        for cand in [
            method_dir / "metrics.csv",
            method_dir / "eval" / "metrics.csv",
            extra_metrics.get(folder),
        ]:
            if cand and Path(cand).is_file():
                rows = load_existing_metrics(Path(cand))
                if rows and ("dino_style" in rows[0] or "dino_s" in rows[0] or "clip_style" in rows[0]):
                    source = str(cand)
                    # normalize dino key names
                    norm = []
                    for r in rows:
                        rr = dict(r)
                        if "dino_s" in rr and "dino_style" not in rr:
                            rr["dino_style"] = rr["dino_s"]
                        if "dino_c" in rr and "dino_content" not in rr:
                            rr["dino_content"] = rr["dino_c"]
                        norm.append(rr)
                    rows = norm
                    break

        if rows is None and not reuse_only:
            if not method_dir.is_dir():
                print("  skip: no folder")
                continue
            if engine is None:
                print("  skip: no engine and no metrics")
                continue
            built = build_rows_from_images(method_dir, board, test_dir)
            # filter to rows with resolvable sources when possible
            ok = [r for r in built if r.get("src_path")]
            if not ok:
                print(f"  skip: parsed={len(built)} but 0 resolvable source paths")
                continue
            if len(ok) < len(built):
                print(f"  warn: using {len(ok)}/{len(built)} rows with resolvable sources (drop unmatched)")
            print(f"  eval {len(ok)} pairs on GPU...")
            rows = evaluate_method(engine, ok, test_dir, styles, max_refs, bs)
            source = "gpu_eval"
            # write per-sample
            mpath = board_out / f"{folder}_metrics.csv"
            keys = list(rows[0].keys())
            with mpath.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        elif rows is None:
            print("  skip: reuse_only and no metrics")
            continue

        # normalize metric keys present
        metric_keys = [k for k in ["clip_style", "content_lpips", "dino_style", "dino_content"] if k in rows[0]]
        agg = aggregate(rows, metric_keys)
        agg["display"] = display
        agg["folder"] = folder
        agg["source"] = source
        summary[folder] = agg
        print(
            f"  n_all={agg['n_all']} n_cross={agg['n_cross']} "
            + " ".join(
                f"{k.replace('dino_style','DINO-S').replace('dino_content','DINO-C').replace('clip_style','CLIP-S').replace('content_lpips','LPIPS')}:"
                f"all={agg.get('all_'+k):.4f}/cross={agg.get('cross_'+k):.4f}"
                for k in metric_keys
                if agg.get("all_" + k) is not None
            )
        )

    (board_out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_markdown(out_dir: Path, d5: dict, p2a: dict) -> None:
    lines = []
    lines.append("# All vs Cross Protocol Metrics (D5-512 & P2A-256)\n")
    lines.append("Generated by `scripts/recompute_cross_protocol.py`.\n")
    lines.append("**Protocol**\n")
    lines.append("- **all**: mean over all ordered content–style pairs (includes identity diagonal `src_style == tgt_style`).\n")
    lines.append("- **cross**: mean over off-diagonal pairs only (`src_style != tgt_style`).\n")
    lines.append("- DINO-S: max cosine between generated CLS and target-style reference pool.\n")
    lines.append("- DINO-C: cosine between generated CLS and source CLS.\n")
    lines.append("- CLIP-S / LPIPS: same as paper main table when available.\n")

    def table_for(title: str, summary: dict):
        lines.append(f"\n## {title}\n")
        lines.append("| Method | n_all | n_cross | CLIP-S all | CLIP-S cross | LPIPS all | LPIPS cross | DINO-S all | DINO-S cross | DINO-C all | DINO-C cross | source |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for folder, agg in sorted(summary.items(), key=lambda x: x[1].get("display", x[0])):
            def f(k):
                v = agg.get(k)
                return f"{v:.4f}" if isinstance(v, float) else "—"
            lines.append(
                f"| {agg.get('display', folder)} | {agg.get('n_all','—')} | {agg.get('n_cross','—')} | "
                f"{f('all_clip_style')} | {f('cross_clip_style')} | {f('all_content_lpips')} | {f('cross_content_lpips')} | "
                f"{f('all_dino_style')} | {f('cross_dino_style')} | {f('all_dino_content')} | {f('cross_dino_content')} | "
                f"{agg.get('source','—')} |"
            )

    table_for("D5-512", d5)
    table_for("P2A-256", p2a)
    lines.append("\n## Notes\n")
    lines.append("- Identity/TGT rows are anchors; for Identity, cross still measures content image vs other styles' style affinity.\n")
    lines.append("- Prefer **cross** for ranking style transfer strength without diagonal self-style inflation of DINO-S/CLIP-S.\n")
    lines.append("- Paper Table 1 historically used **all** averages; this document reports both.\n")
    (out_dir / "CROSS_PROTOCOL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=ROOT / "results")
    ap.add_argument("--d5_test", type=Path, default=ROOT / "data" / "test")
    ap.add_argument(
        "--p2a_test",
        type=Path,
        default=ROOT / "archives" / "datasets_legacy_20260721" / "p2a_test" / "test",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "cross_protocol")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_refs", type=int, default=30)
    ap.add_argument("--allow_network", action="store_true")
    ap.add_argument("--dino_cache", default=str(ROOT / "runs" / "cache" / "hf"))
    ap.add_argument("--reuse_only", action="store_true", help="Only re-aggregate existing metrics.csv")
    ap.add_argument("--boards", nargs="+", default=["D5-512", "P256"], choices=["D5-512", "P256"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    engine = None
    if not args.reuse_only:
        engine = MetricEngine(args.device, dino_cache=args.dino_cache, allow_network=args.allow_network)

    d5_sum, p2a_sum = {}, {}
    if "D5-512" in args.boards:
        d5_sum = process_board(
            "D5-512",
            args.results_root / "D5-512",
            args.d5_test,
            D5_METHODS,
            D5_STYLES,
            engine,
            args.out,
            args.max_refs,
            args.batch_size,
            args.reuse_only,
        )
    if "P256" in args.boards:
        p2a_sum = process_board(
            "P2A-256",
            args.results_root / "P256",
            args.p2a_test,
            P2A_METHODS,
            P2A_STYLES,
            engine,
            args.out,
            args.max_refs,
            args.batch_size,
            args.reuse_only,
        )

    write_markdown(args.out, d5_sum, p2a_sum)
    (args.out / "all_summary.json").write_text(
        json.dumps({"D5-512": d5_sum, "P2A-256": p2a_sum}, indent=2), encoding="utf-8"
    )
    print("\nWROTE", args.out / "CROSS_PROTOCOL_REPORT.md")


if __name__ == "__main__":
    main()
