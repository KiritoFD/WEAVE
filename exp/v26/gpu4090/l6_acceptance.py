r"""V26-L L6 验收补齐 (docs/1120 §8 L6 清单的遗留项).

缺口与补法:
  (1) 20 条有向迁移路线矩阵 — 从 measured eval.json 逐任务行聚合 5×5 (源×目标)
      Top-1 (20 off-diagonal 有向路线 + 5 same), 无需重解码;
  (2) 源图聚类配对 bootstrap CI — 源图 DINO 特征 k-means (k=10) 聚类, 簇级重采样
      2000 次, 配对差 Top-1(A)−Top-1(B) 的 95% CI (整簇重采样保持簇内相关);
  (3) 难例对比图 — S0 cross 任务按 target_margin 最低取 12 例, 源|输出|参考 三联 PNG;
  (4) S0 第二/三种子 — pool 已产出 (S0_vgg_global_seed23/seed45_t45, 完整 test750),
      本模块纳入 CI 与复验表, 不重训.

用法:  python -m exp.v26.gpu4090.l6_acceptance --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from exp.v26 import OUT, ROOT, sanitize_cuda_env, save_json, utc

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
ARMS = ("S0_vgg_global", "F2_global_teacher", "S0_vgg_global_seed23", "S0_vgg_global_seed45")
K_CLUSTERS = 10
N_BOOT = 2000
BOOT_SEED = 20266123


def _rows(arm: str) -> dict[str, dict]:
    p = OUT / "measured" / f"{arm}_t45" / "eval.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r["task_id"]: r for r in d["rows"]}


def route_matrix(arms_rows: dict[str, dict]) -> dict:
    """5×5 Top-1 矩阵 (行=源画派, 列=目标画派); 20 条有向迁移 = off-diagonal."""
    out = {}
    for arm, rows in arms_rows.items():
        m = [[{"n": 0, "top1": 0} for _ in range(5)] for _ in range(5)]
        for r in rows.values():
            cell = m[r["src_style_id"]][r["tgt_style_id"]]
            cell["n"] += 1
            cell["top1"] += int(r["target_top1"])
        mat = [[round(c["top1"] / max(c["n"], 1), 4) for c in row] for row in m]
        ns = [[c["n"] for c in row] for row in m]
        cross = [mat[i][j] for i in range(5) for j in range(5) if i != j]
        out[arm] = {"matrix": mat, "n": ns, "directed_top1_mean": round(sum(cross) / len(cross), 4),
                    "directed_min": min(cross), "directed_argmin":
                    [(STYLES[i], STYLES[j]) for i in range(5) for j in range(5) if i != j
                     and mat[i][j] == min(cross)][0]}
    return out


def _kmeans(feats: torch.Tensor, k: int, iters: int = 25, seed: int = 20266124) -> torch.Tensor:
    """轻量 k-means (cos 相似, k-means++ 初始化) → 簇 id (N,)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.nn.functional.normalize(feats.float(), dim=-1)
    idx0 = torch.randperm(x.shape[0], generator=g)[0]
    centers = [x[idx0]]
    for _ in range(k - 1):
        d2 = 1.0 - torch.stack([x @ c for c in centers], 1).max(dim=1).values
        centers.append(x[torch.multinomial(d2.clamp_min(1e-6), 1, generator=g)])
    C = torch.stack(centers, 0)
    for _ in range(iters):
        assign = (x @ C.T).argmax(1)
        for j in range(k):
            sel = x[assign == j]
            if len(sel):
                C[j] = torch.nn.functional.normalize(sel.mean(0), dim=-1)
    return (x @ C.T).argmax(1)


def cluster_bootstrap_ci(arms_rows: dict[str, dict], pairs: list[tuple[str, str]],
                         task_clusters: dict[str, int]) -> dict:
    """簇级配对 bootstrap: 整簇重采样 → 配对 Top-1 差的 95% CI (保持簇内相关)."""
    g = torch.Generator().manual_seed(BOOT_SEED)
    cl_ids = sorted(set(task_clusters.values()))
    cl_to_idx = {c: i for i, c in enumerate(cl_ids)}
    rows_by_cluster = {arm: {c: [] for c in cl_ids} for arm in arms_rows}
    for arm, rows in arms_rows.items():
        for tid, r in rows.items():
            rows_by_cluster[arm][task_clusters[tid]].append(int(r["target_top1"]))
    out = {}
    for a, b in pairs:
        diffs = []
        for _ in range(N_BOOT):
            pick = [cl_ids[int(torch.randint(0, len(cl_ids), (1,), generator=g))]
                    for _ in range(len(cl_ids))]
            sa = sum(sum(rows_by_cluster[a][c]) for c in pick)
            na = sum(len(rows_by_cluster[a][c]) for c in pick)
            sb = sum(sum(rows_by_cluster[b][c]) for c in pick)
            nb = sum(len(rows_by_cluster[b][c]) for c in pick)
            diffs.append(sa / max(na, 1) - sb / max(nb, 1))
        d = torch.tensor(diffs)
        q = torch.quantile(d, torch.tensor([0.025, 0.5, 0.975]))
        out[f"{a}_minus_{b}"] = {
            "mean_diff": round(float(d.mean()), 4),
            "ci95": [round(float(q[0]), 4), round(float(q[2]), 4)],
            "p_gt0": round(float((d > 0).float().mean()), 4),
            "note": "整簇重采样 2000 次; Top-1(A)−Top-1(B), >0 = A 更优",
        }
    return out


@torch.no_grad()
def hard_examples_grid(arm: str, k: int = 12, device: str = "cuda") -> dict:
    """难例三联图: target_margin 最低的 k 个 cross 任务 (源|输出|参考).

    参考列 = 推理时真实使用的 SG-5 core exemplar (c21 缓存, 与评测协议一致).
    """
    from exp.v18 import tasks_and_sources
    from exp.v26.sr_cfg import TAU0
    from src.utils import decode_s10, load_s10_decoder
    from PIL import Image, ImageDraw

    rows = {tid: r for tid, r in _rows(arm).items() if r["is_cross"]}
    worst = sorted(rows.values(), key=lambda r: r["target_margin"])[:k]
    tasks, z_src_750, _ = tasks_and_sources()
    by_stem = {t["source_stem"]: t for t in tasks}
    dec = load_s10_decoder(device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    z_ref5 = c21["sg5"]["z"].float()  # (5, 4, 64, 64) — 5 画派 core exemplar
    rgb_ref5 = decode_s10(dec, z_ref5, batch=5)
    out_dir = OUT / "gpu4090" / "hard_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    cell = 256
    grid = Image.new("RGB", (3 * cell + 20, k * cell + 10 * (k + 1)), (20, 20, 20))
    dr = ImageDraw.Draw(grid)
    for i, r in enumerate(worst):
        t = by_stem[r["src_stem"]]
        z_src = z_src_750[tasks.index(t)].to(device).float()
        src_im = decode_s10(dec, z_src[None], batch=1)[0]
        src_png = Image.fromarray((src_im.permute(1, 2, 0).cpu().numpy() * 255)
                                  .round().clip(0, 255).astype("uint8")).resize((cell, cell))
        out_png = None
        for entry in json.loads((OUT / "raw" / f"{arm}_t45" / "meta.json")
                                .read_text(encoding="utf-8"))["batches"]:
            payload = torch.load(OUT / "raw" / f"{arm}_t45" / entry["file"],
                                 map_location="cpu", weights_only=False)
            if r["task_id"] in payload["ids"]:
                j = list(payload["ids"]).index(r["task_id"])
                out_png = Image.fromarray(payload["rgb_u8"][j].permute(1, 2, 0).numpy()).resize((cell, cell))
                break
        grid.paste(src_png, (5, i * (cell + 10)))
        if out_png:
            grid.paste(out_png, (cell + 10, i * (cell + 10)))
        ref_im = Image.fromarray((rgb_ref5[r["tgt_style_id"]].permute(1, 2, 0).cpu().numpy() * 255)
                                 .round().clip(0, 255).astype("uint8")).resize((cell, cell))
        grid.paste(ref_im, (2 * cell + 15, i * (cell + 10)))
        dr.text((6, i * (cell + 10) + 2),
                f"{r['src_style'][:6]}->{r['tgt_style'][:6]} m={r['target_margin']:.3f}",
                fill=(255, 255, 0))
    png_p = out_dir / f"hard_examples_{arm}.png"
    grid.save(png_p)
    return {"png": str(png_p), "k": k,
            "worst_routes": sorted({f"{r['src_style']}->{r['tgt_style']}" for r in worst})}


def run_acceptance(pool: str, device: str = "cuda") -> dict:
    arms_rows = {a: _rows(a) for a in ARMS}
    ns = {a: len(r) for a, r in arms_rows.items()}
    assert all(n == 750 for n in ns.values()), f"行数非 750: {ns}"

    # 源图聚类 (DINO 内容特征, 750 源图 → k=10)
    src = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt",
                     map_location="cpu", weights_only=False)
    tasks = json.loads((ROOT / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]
    stems = [t["source_stem"] for t in tasks]
    feats = torch.stack([torch.as_tensor(src["raw_dino"][s]) for s in stems])
    clusters = _kmeans(feats, K_CLUSTERS)
    task_clusters = {t["task_id"]: int(c) for t, c in zip(tasks, clusters)}

    routes = route_matrix(arms_rows)
    ci = cluster_bootstrap_ci(arms_rows, [
        ("S0_vgg_global", "F2_global_teacher"),
        ("S0_vgg_global", "S0_vgg_global_seed23"),
        ("S0_vgg_global", "S0_vgg_global_seed45"),
        ("F2_global_teacher", "S0_vgg_global_seed23"),
    ], task_clusters)
    hard = hard_examples_grid("S0_vgg_global", k=12, device=device)

    out = {"task": "V26-L L6 acceptance: route matrix + cluster-paired CI + hard examples",
           "updated_utc": utc(), "n_rows_per_arm": ns,
           "route_matrix": routes, "cluster_bootstrap_ci": ci,
           "cluster_k": K_CLUSTERS, "n_boot": N_BOOT, "hard_examples": hard}
    save_json(OUT / "gpu4090" / "l6_acceptance.json", out)
    print("[Accept] route directed mean:", {a: routes[a]["directed_top1_mean"] for a in routes})
    for k, v in ci.items():
        print(f"[Accept CI] {k}: mean={v['mean_diff']} ci95={v['ci95']} p>0={v['p_gt0']}")
    print(f"[Accept] hard examples → {hard['png']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_acceptance(args.pool, device=args.device)


if __name__ == "__main__":
    main()
