r"""V26-L 16% 硬天花板根因排查 (最小诊断清单, 用户 2026-11-22 指令).

背景: F2/S0 全配样卡死 16.0-16.8% (7.5× 步数/4× 数据/换损失/换采样/加注意力全部无效),
教师→学生信号流失 60%→16% (<1/3 吸收). 四项逐一排查:

  D1  教师直测 (决定性): l1_teacher_v22dual_crossstyle.pt 的教师 latent 直接
      S10 解码 → CSD 裁判 Top-1. 600 配对 (每目标画派 120).
      假设: 教师生成从源图锚 τ0=0.45 出发 (build_teacher L123), 按 V22 锚点消融
      (源锚 20.5% vs PRISM-1 锚 57.17%) 教师天花板 ≈ 20% → 学生 16% 完全被解释.
  D1b 锚点-质量前沿: 96 训练内容 × τ0 ∈ {0.0, 0.15, 0.45} 教师重生成 →
      Top-1 + DINO-C(对源) — 教师本身能走多远, 内容代价多大.
  D2  闭集原型探针: S0 × 96 测试任务, u_glob/u_style 从单参考 (SG-5 exemplar)
      换成闭集五画派均值原型纤维 → 原型也卡 16% ⇒ 教师质量/锚定/锚点;
      原型冲 50% ⇒ 单图纤维实例噪声.
  D3  CFG 审计并入 D4 静态清单 (训练 drop_style 恒零).
  D4  静态审计: 锚定项清单.

注意: 教师配方 (build_teacher) 的 u_style = core exemplar 纤维的 StyleFiberEncoder
嵌入, u_glob = 795d 归一化纤维 — 无 RefAdapter; 评测协议 (sr_cfg) 的 u_glob 同为
795d (fib_n_sg5), u_style = ad(u_sg5, u_src).

用法:  python -m exp.v26.gpu4090.l7_diag_16pct --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse

import torch

from exp.v26 import OUT, ROOT, sanitize_cuda_env, save_json, utc

STYLES = ("Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e")
N_D1_PER_STYLE = 120
N_PROBE = 96
EVAL_SEED = 20261185


def _pil(x: torch.Tensor):
    from PIL import Image as PILImage
    return PILImage.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))


def _csd_top1(stack, pil_imgs, want: list[int]) -> list[int]:
    feats = stack.extract(pil_imgs, batch=24)
    cs = feats["csd"] @ stack.csd_prototypes.T
    return cs.argmax(1).tolist()


# ------------------------------------------------------------- D1 ----

@torch.no_grad()
def d1_teacher_direct(device: str = "cuda") -> dict:
    """现缓存教师 latent → S10 → CSD Top-1 (每目标画派 120 配对, 共 600)."""
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from src.utils import FeatureStack, load_s10_decoder

    pool = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")
    build_or_load_v19_foundation(pool, device=device)  # 保持缓存路径一致
    tdata = torch.load(ROOT / "runs/v26/cache/l1_teacher_v22dual_crossstyle.pt",
                       map_location="cpu", weights_only=False)
    z_t, valid = tdata["z"], tdata["valid"]
    print(f"[D1] teacher cache z{tuple(z_t.shape)} valid={int(valid.sum())} "
          f"verify_color_diff={tdata.get('verify_crossstyle', {}).get('mean_color_diff')}", flush=True)
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    g = torch.Generator().manual_seed(20266201)
    lat, want, styles = [], [], []
    for t in range(5):
        pool_i = (valid[:, t]).nonzero(as_tuple=True)[0]
        pool_i = pool_i[torch.randperm(len(pool_i), generator=g)[:N_D1_PER_STYLE]]
        for i in pool_i.tolist():
            lat.append(z_t[i, t])
            want.append(t)
            styles.append(t)
    pred_all = []
    for st in range(0, len(lat), 24):
        chunk = torch.stack(lat[st:st + 24]).to(device).float()
        rgb = decode_s10_stub(dec, chunk)
        pred_all += _csd_top1(stack, [_pil(x) for x in rgb], want[st:st + 24])
    per_style = {STYLES[t]: round(sum(1 for p, s in zip(pred_all, styles)
                                      if s == t and p == t) / N_D1_PER_STYLE, 4) for t in range(5)}
    out = {"arm": "teacher_direct", "n_pairs": len(lat),
           "top1_overall": round(sum(int(p == w) for p, w in zip(pred_all, want)) / len(lat), 4),
           "top1_per_target_style": per_style}
    print(f"[D1] teacher direct Top-1 = {out['top1_overall']} per={per_style}", flush=True)
    return out


def decode_s10_stub(dec, z):
    from src.utils import decode_s10
    return decode_s10(dec, z, batch=24).clamp(0.0, 1.0)


# ------------------------------------------------------------- D1b ----

@torch.no_grad()
def _regen_teacher_batched(bundle, c21, tau0: float, idxs: list[int],
                           device: str = "cuda") -> torch.Tensor:
    """build_teacher 同配方重生成 (仅改锚点强度); 按目标画派分组批量."""
    from exp.v24.p0_o1_o3_value import DUAL_CKPT, SLFUNetV24OneShot
    from exp.v19.fiber_duality import StyleFiberEncoder
    from exp.v26.sr_cfg import integrate_sr, sr_velocity
    from exp.v20.slf_flow import coarse_y16_lcn

    tr = bundle["v18_cache"]["train"]
    z_all, sid_all = tr["z"], tr["sid"].long()
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
    fm.load_state_dict(torch.load(DUAL_CKPT, map_location="cpu", weights_only=False)["ema"], strict=False)
    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_all = fiber_enc.encode(bundle["fnd"]["tr_fib"].to(device).float())
    fib_tr = bundle["fnd"]["tr_fib"].float()
    fib_n_all = ((fib_tr.to(device) - fib_tr.mean(0, keepdim=True).to(device))
                 / fib_tr.std(0, keepdim=True).clamp_min(1e-4).to(device))
    core_idx = [c.clone() for c in c21["train"]["core_idx_per_style"]]
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)
    d16_all = tr["dino_16"]
    g = torch.Generator().manual_seed(20261185)
    wj = 2.2
    out_by_i: dict[int, torch.Tensor] = {}
    for t in range(5):
        idx_t = [i for i in idxs if (int(sid_all[i]) + 1) % 5 == t]
        if not idx_t:
            continue
        core_t = core_idx[t].to(u_all.device)
        for st in range(0, len(idx_t), 24):
            chunk = idx_t[st:st + 24]
            B = len(chunk)
            zs = z_all[chunk].to(device).float()
            y16 = coarse_y16_lcn(((zs * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()
            d16 = d16_all[chunk].to(device).float()
            pick = torch.randint(0, len(core_t), (B,), generator=g)
            u_style = u_all[core_t[pick]]
            u_glob = fib_n_all[core_t[pick]]
            eps = torch.randn(B, 4, 64, 64, generator=g).to(device)
            z = (1.0 - tau0) * eps + tau0 * zs
            slot = torch.zeros(B, dtype=torch.long, device=device)
            f0 = torch.zeros(B, dtype=torch.bool, device=device)
            f1 = torch.ones(B, dtype=torch.bool, device=device)

            def vel_fn(zz, prog):
                tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
                out = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                         slot.repeat(2), torch.cat([u_style, u_style], 0), tv.repeat(2),
                         drop_style=torch.cat([f0, f1], 0),
                         drop_content=torch.zeros(2 * B, dtype=torch.bool, device=device),
                         layout_scale=torch.ones(2 * B, device=device),
                         d16_ref=None, v16_ref=None, u_glob=u_glob.repeat(2, 1)).float()
                v_cond, v_unc = out.chunk(2, 0)
                return v_unc + wj * (v_cond - v_unc)

            z_gen = integrate_sr(vel_fn, z, tau0, 10, "rk2").cpu()
            for j, i in enumerate(chunk):
                out_by_i[i] = z_gen[j]
    del fm, fiber_enc
    torch.cuda.empty_cache()
    return torch.stack([out_by_i[i] for i in idxs], 0)


@torch.no_grad()
def d1b_tau0_sweep(bundle, c21, device: str = "cuda") -> dict:
    """96 训练内容 × τ0 ∈ {0.0, 0.15, 0.45}: 教师重生成 → Top-1 + DINO-C 对源."""
    from src.utils import FeatureStack, load_s10_decoder

    tr = bundle["v18_cache"]["train"]
    sid_all = tr["sid"].long()
    g = torch.Generator().manual_seed(20266202)
    pool_i = torch.arange(len(sid_all))[torch.randperm(len(sid_all), generator=g)[:N_PROBE]].tolist()
    want = [(int(sid_all[i]) + 1) % 5 for i in pool_i]
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    # 源 DINO (解码 z_src, 一次)
    from src.utils import decode_s10
    z_src = tr["z"][pool_i].to(device).float()
    src_dino = stack.extract([_pil(x) for x in decode_s10(dec, z_src, batch=24).clamp(0.0, 1.0)],
                             batch=24)["dino"]
    out = {}
    for tau0 in (0.0, 0.15, 0.45):
        z_gen = _regen_teacher_batched(bundle, c21, tau0, pool_i, device=device)
        pred_all, dc_all = [], []
        for st in range(0, len(pool_i), 24):
            rgb = decode_s10(dec, z_gen[st:st + 24].to(device).float(), batch=24).clamp(0.0, 1.0)
            feats = stack.extract([_pil(x) for x in rgb], batch=24)
            cs = feats["csd"] @ stack.csd_prototypes.T
            pred_all += cs.argmax(1).tolist()
            dc_all += torch.diagonal(feats["dino"] @ src_dino[st:st + 24].T).tolist()
        out[f"tau0_{tau0}"] = {
            "top1": round(sum(int(p == w) for p, w in zip(pred_all, want)) / len(pred_all), 4),
            "dino_c_vs_src": round(sum(dc_all) / len(dc_all), 4)}
        print(f"[D1b] tau0={tau0}: {out[f'tau0_{tau0}']}", flush=True)
    return out


# ------------------------------------------------------------- D2 ----

@torch.no_grad()
def d2_prototype_probe(pool: str, device: str = "cuda") -> dict:
    """S0 × 96 测试任务: exemplar (协议口径) vs 闭集原型纤维 (Top-1 对比)."""
    from exp.v18 import tasks_and_sources
    from exp.v19.fiber_duality import StyleFiberEncoder
    from exp.v26.sr_cfg import TAU0, WJ, integrate_sr, load_sr_env, sr_velocity
    from src.utils import FeatureStack, load_s10_decoder

    env = load_sr_env(pool, device=device, ckpt_path=OUT / "oneshot_val" / "S0_vgg_global.pt")
    fm, ad = env["fm"], env["ad"]
    fib_mean, fib_std = env["fib_mean"].to(device), env["fib_std"].to(device)
    fib_tr = env["bundle"]["fnd"]["tr_fib"].float()
    sid_tr = env["bundle"]["v18_cache"]["train"]["sid"].long()
    enc = StyleFiberEncoder(in_dim=fib_tr.shape[1], dim=128).to(device).eval()
    enc.load_state_dict(env["bundle"]["fnd"]["fiber_state"])
    fib_proto = torch.stack([fib_tr.to(device)[sid_tr == s].mean(0) for s in range(5)])
    fib_n_proto = (fib_proto - fib_mean) / fib_std
    u_proto = enc.encode(fib_n_proto)  # (5, 128)
    del enc
    tasks, z_src_750, _ = tasks_and_sources()
    cross = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:N_PROBE]
    sel = torch.tensor(cross)
    tid = torch.tensor([tasks[i]["target_style_id"] for i in cross])
    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)
    from exp.v20.slf_flow import coarse_y16_lcn
    y16_750 = coarse_y16_lcn(((z_src_750.float().to(device) * w_l).sum(1, keepdim=True)
                              + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    d16_750 = env["bundle"]["test_struct"]["dino_16"]
    out = {}
    for tag, u_glob_ref, u_style_ref in (
            ("exemplar", env["fib_n_sg5"], env["u_sg5"]),
            ("prototype", fib_n_proto, u_proto)):
        pred_all = []
        for st in range(0, len(sel), 24):
            sl = slice(st, min(st + 24, len(sel)))
            B = sl.stop - sl.start
            zs = z_src_750[sel[sl]].to(device).float()
            y16 = y16_750[sl].to(device).float()
            d16 = d16_750[sel[sl]].to(device).float()
            u_glob_b = u_glob_ref.to(device)[tid[sl]]
            u_style_b = ad(u_style_ref.to(device)[tid[sl]], env["u_src_750"][sel[sl]].to(device))
            z = (1.0 - TAU0) * eps_750[sel[sl]].to(device).float() + TAU0 * zs

            def vel_fn(zz, prog):
                tv = torch.full((B,), TAU0 + prog * (1.0 - TAU0), device=device)
                v, _ = sr_velocity(fm, ad, u_style_b, None, u_glob_b, None,
                                   d16, None, None, None, None, y16, zz, tv,
                                   mode="baseline", w=0.0, wj=WJ, device=device)
                return v

            z_gen = integrate_sr(vel_fn, z, TAU0, 10, "rk2")
            rgb = decode_s10(dec, z_gen, batch=24).clamp(0.0, 1.0)
            pred_all += _csd_top1(stack, [_pil(x) for x in rgb], tid[sl].tolist())
        want = tid.tolist()
        out[tag] = round(sum(int(p == w) for p, w in zip(pred_all, want)) / len(pred_all), 4)
        print(f"[D2] {tag}: Top-1={out[tag]}", flush=True)
    return out


# ------------------------------------------------------------- D4 ----

def d4_static_audit() -> list[str]:
    return [
        "教师锚点: build_teacher L123 z=(1-0.45)*eps+0.45*z_src — 45% 源图锚 (V22 消融: 源锚 20.5% vs PRISM-1 锚 57.17%)",
        "学生 CFG dropout: train_l1_arm drop_style=torch.zeros — 全程无 Style Dropout → 推理 2-branch CFG 的 uncond 分支 OOD",
        "学生锚定项: flow-MSE 轨迹起点 z_init=源锚混合 (L314) + swap 损失 λ_sw=0.5 λ_c=1.0 (L390) — 源图内容项双路锚定",
        "swap 损失以 1 步预览 decode (L372) — 弱轨迹下格内排序分位数趋坍缩 (S4 失效的另一技术原因, 用户 2026-11-22 分析)",
    ]


def run_diag(device: str = "cuda") -> dict:
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    pool = str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16")
    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    out = {"task": "V26-L 16% hard-ceiling root-cause diagnostics", "updated_utc": utc()}
    out["D1_teacher_direct"] = d1_teacher_direct(device=device)
    out["D1b_tau0_sweep"] = d1b_tau0_sweep(bundle, c21, device=device)
    try:
        out["D2_prototype_probe"] = d2_prototype_probe(pool, device=device)
    except Exception as exc:
        out["D2_prototype_probe"] = {"error": repr(exc)[:300]}
    out["D4_static_audit"] = d4_static_audit()
    save_json(OUT / "gpu4090" / "l7_diag_16pct.json", out)
    print(f"[DIAG] saved → {OUT / 'gpu4090' / 'l7_diag_16pct.json'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sanitize_cuda_env()
    run_diag(device=args.device)


if __name__ == "__main__":
    main()
