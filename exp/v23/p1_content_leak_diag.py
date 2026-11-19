"""V23 P1 第二组 (4090): 定位真正的内容破坏位置（0%, 25%, 50%, 75%, 100% 逐步轨迹诊断）。

在代表性 Cross 任务上对比 Peak1500 (g=0.45/wc=1.0, g=0.50/wc=1.5) 与 T1 (g=0.50/wc=1.4):
1. 解码每一步即时终点预测 z1_hat(t) = z_t + (1 - tau_t) * v_guided（以及起始锚点 z_anchor / 最终 z_out）；
2. 用冻结 DINOv2-small 测量每一步的 DINO-C（全局语义）与 16x16 spatial patch 对齐余弦（中尺度结构定位）；
3. 用冻结 CSD 测量每一步的目标画派概率与 Top-1；
4. 保存代表性轨迹拼图 PNG，直观定位“内容在起点、前半段还是后半段被破坏”。

用法 (4090):
  python -m exp.v23.p1_content_leak_diag --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from . import OUT, EVAL_SEED, ROOT, sanitize_cuda_env, save_json, utc
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore, build_content_condition, effective_gate_components, make_initial_state, predict_guided_velocity
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from tools.pseudo_gt.matching import encode_dino

PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"
T1_CKPT = ROOT / "runs/v22/tmatrix/slf_T1.pt"


def _select_eval_indices(tasks: list[dict], per_pair: int = 2) -> list[int]:
    """从 20 个跨风格有向对中各选 per_pair 个样本 = 40 个代表性跨风格任务。"""
    picked = {}
    for idx, t in enumerate(tasks):
        s, tg = t["source_style_id"], t["target_style_id"]
        if s == tg:
            continue
        k = (s, tg)
        picked.setdefault(k, [])
        if len(picked[k]) < per_pair:
            picked[k].append(idx)
    return [i for k in sorted(picked) for i in picked[k]]


@torch.no_grad()
def _extract_dino_patches_and_csd(imgs01: torch.Tensor, stack, dino_model, device: str = "cuda"):
    """返回 (dino_cls (B,384), dino_patch_16x16 (B,256,384), csd_logits_5 (B,5))。"""
    pils = [Image.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8")) for x in imgs01]
    fo = stack.extract(pils, batch=20)
    csd_feats = torch.as_tensor(fo["csd"], dtype=torch.float32)
    protos = F.normalize(torch.as_tensor(stack.csd_prototypes, dtype=torch.float32), dim=-1)
    csd_5 = csd_feats @ protos.T  # (B, 5)
    cls_t, patches_t = encode_dino(dino_model, pils, batch=20, device=device)
    return F.normalize(cls_t.float().cpu(), dim=-1), F.normalize(patches_t.float().cpu(), dim=-1), csd_5


@torch.no_grad()
def trace_model(core: PrismCore, name: str, ctx: dict, sel: list[int],
                g_val: float, wc_val: float, stack, dino_model, device: str = "cuda") -> dict:
    from src.utils import decode_s10

    tasks = ctx["tasks"]
    sel_t = torch.tensor(sel, dtype=torch.long)
    tid_b = torch.tensor([tasks[i]["target_style_id"] for i in sel], device=device)
    z_anc_b = ctx["z_t4"][sel_t].to(device).float()
    eps_b = ctx["eps_750"][sel_t].to(device).float()
    y16_b = ctx["y16_750"][sel_t].to(device).float()
    d16_b = ctx["d16_750"][sel_t].to(device).float()

    gate = GateConfig(mode="fixed", fixed_g=g_val)
    cfg = CFGConfig(branches="full", rescale_phi=0.48)
    solver = dict(D5_SOLVER, wc_fig=wc_val, wc_abs=wc_val)
    steps = int(solver["steps"])  # 10

    u_style, dsd_ref = core.build_style_condition(d16_b, tid_b, mode="core8")
    g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_ref, gate, float(core.bank["mid_dsd"]))
    lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
    content = build_content_condition(y16_b, d16_b, g_layout)
    tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
    z = make_initial_state(eps_b, z_anc_b, tau_v)

    # 关键里程碑 step index: 起点anchor, step=0(0% z1_hat), step=2(25%), step=5(50%), step=7(75%), final(100%)
    snap_z1 = {"anchor": z_anc_b.clone()}
    dt = (1.0 - tau_v).view(-1, 1, 1, 1) / float(steps)

    for i in range(steps):
        prog1 = i / float(steps)
        tv1 = tau_v + prog1 * (1.0 - tau_v)
        v1 = predict_guided_velocity(core.model, z, tv1, content, tid_b, u_style, lscale, g_guid, cfg, solver, prog1)
        if i in (0, 2, 5, 7):
            tag = {0: "p00", 2: "p25", 5: "p50", 7: "p75"}[i]
            snap_z1[tag] = (z + (1.0 - tv1).view(-1, 1, 1, 1) * v1).clamp(-6.0, 6.0)
        zm = z + 0.5 * dt * v1
        prog2 = (i + 0.5) / float(steps)
        v2 = predict_guided_velocity(core.model, zm, tau_v + prog2 * (1.0 - tau_v), content, tid_b, u_style, lscale, g_guid, cfg, solver, prog2)
        z = z + dt * v2
    snap_z1["p100"] = z.clone()

    src_cls, src_patch = ctx["src_dino_cls"], ctx["src_dino_patch"]
    dec = ctx["dec"]
    stats = {}
    vis_imgs = {}
    for tag, z_snap in snap_z1.items():
        rgb = decode_s10(dec, z_snap, batch=16).clamp(0.0, 1.0)
        cls_f, pat_f, csd5 = _extract_dino_patches_and_csd(rgb, stack, dino_model, device=device)
        dino_c = float((cls_f * src_cls).sum(dim=-1).mean())
        patch_cos = float((pat_f * src_patch).sum(dim=-1).mean())
        row = {"dino_c": round(dino_c, 4), "patch_cos_16x16": round(patch_cos, 4)}
        if csd5 is not None:
            pred = csd5.argmax(dim=-1)
            row["csd_top1"] = round(float((pred == tid_b.cpu()).float().mean()), 4)
        stats[tag] = row
        vis_imgs[tag] = rgb[:5].cpu()  # 取前5张作视觉轨迹拼图
        print(f"[TRAJ {name:16s} {tag:6s}] DINO-C={row['dino_c']:.4f} PatchCos={row['patch_cos_16x16']:.4f} Top1={row.get('csd_top1', -1):.4f}", flush=True)

    # 保存视觉轨迹拼图 (5行 × 7列: Src, Anchor, p00, p25, p50, p75, p100)
    grid_dir = OUT / "traj_diag"
    grid_dir.mkdir(parents=True, exist_ok=True)
    cols = ["src", "anchor", "p00", "p25", "p50", "p75", "p100"]
    vis_imgs["src"] = ctx["src_rgb_5"]
    thumb = 224
    canvas = Image.new("RGB", (len(cols) * thumb, 5 * thumb), "white")
    for r_i in range(5):
        for c_i, ck in enumerate(cols):
            arr = (vis_imgs[ck][r_i].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
            im = Image.fromarray(arr).resize((thumb, thumb), Image.Resampling.BILINEAR)
            canvas.paste(im, (c_i * thumb, r_i * thumb))
    canvas.save(grid_dir / f"traj_{name}.jpg", quality=88)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)
    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))

    from src.utils import FeatureStack, decode_s10, load_s10_decoder
    dec = load_s10_decoder(args.device)
    stack = FeatureStack(args.device)
    dino_model = stack.dino

    sel = _select_eval_indices(tasks, per_pair=2)
    sel_t = torch.tensor(sel, dtype=torch.long)
    src_rgb = decode_s10(dec, z_src_750[sel_t].to(args.device).float(), batch=16).clamp(0.0, 1.0)
    src_cls, src_patch, _ = _extract_dino_patches_and_csd(src_rgb, stack, dino_model, device=args.device)

    ctx = {
        "tasks": tasks, "z_src_750": z_src_750, "y16_750": y16_750, "d16_750": d16_750,
        "eps_750": eps_750, "z_t4": z_t4, "dec": dec,
        "src_dino_cls": src_cls, "src_dino_patch": src_patch, "src_rgb_5": src_rgb[:5].cpu(),
    }

    report = {}
    core_pk = PrismCore(bundle, bank, PEAK_CKPT, device=args.device)
    report["Peak1500_g45_wc10"] = trace_model(core_pk, "Peak_g45_wc10", ctx, sel, 0.45, 1.0, stack, dino_model, device=args.device)
    report["Peak1500_g50_wc15"] = trace_model(core_pk, "Peak_g50_wc15", ctx, sel, 0.50, 1.5, stack, dino_model, device=args.device)
    del core_pk
    torch.cuda.empty_cache()

    core_t1 = PrismCore(bundle, bank, T1_CKPT, device=args.device)
    report["T1_g50_wc14"] = trace_model(core_t1, "T1_g50_wc14", ctx, sel, 0.50, 1.4, stack, dino_model, device=args.device)

    save_json(OUT / "p1_content_leak_diag.json", {"updated_utc": utc(), "n_tasks": len(sel), "trajectories": report})
    print(f"[TRAJ] Saved -> {OUT / 'p1_content_leak_diag.json'}", flush=True)


if __name__ == "__main__":
    main()
