"""V23 P1 第四组 (4090): 零初始化 32×32 残差内容注入 (h32 + gamma * A(c32), gamma_init=0)。

严格遵循 V23 §1.5 设计:
- 新模块初始输出严格为零（加入前后原模型输出逐位一致, max_diff == 0）;
- 不替换原有 ResBlock, 仅在 32×32 注入轻量化残差卷积块;
- 先单训新模块 400 updates, 再部分主干联合微调 1000 updates;
- 设同起点(Peak1500)、同预算(1000 updates)、同学习率的无 S32 对照臂 (S32_ctrl)。

用法 (4090):
  python -m exp.v23.p1_s32_zero --pool ../pseudo_gt_24h/orig_mean_fp16
"""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, EVAL_SEED, ROOT, RawWriterV23, autocast, config_hash, export_raw_pngs, file_sha, git_sha, read_json, run_measure_v23, sanitize_cuda_env, save_json, seed_all, utc, validate_raw_meta
from exp.v22.arch_s32 import build_y32_pair
from exp.v22.core import CFGConfig, D5_SOLVER, GateConfig, PrismCore, _guidance_curves, build_content_condition, effective_gate_components, make_initial_state, rk2_integrate
from exp.v22.t4 import build_or_load_t4_latents_750
from exp.v18 import tasks_and_sources
from exp.v18.ccf_model import sinusoidal_time_emb
from exp.v19.fiber_duality import build_or_load_v19_foundation
from exp.v20.run_v20 import get_test750_y16_and_d16
from exp.v20.slf_flow import SLFUNet, coarse_y16_lcn, project_latent_y16_lcn
from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from src.utils import load_s10_decoder

PEAK_CKPT = ROOT / "runs/v22/tmatrix/slf_T4Ssnap1500.pt"


class SLFUNetS32Zero(SLFUNet):
    """SLFUNet + 零初始化 32×32 残差注入: u32 = u32 + gamma * A_phi(c32), gamma_init=0."""

    def __init__(self, base_dim: int = 144, u_dim: int = 128, cond_dim: int = 288, s32_ch: int = 2):
        super().__init__(base_dim=base_dim, u_dim=u_dim, cond_dim=cond_dim)
        w1 = base_dim * 2
        self.s32_gamma = nn.Parameter(torch.zeros(1))
        self.s32_adapter = nn.Sequential(
            nn.Conv2d(w1 + s32_ch, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, w1, 3, padding=1),
        )
        nn.init.zeros_(self.s32_adapter[-1].weight)
        nn.init.zeros_(self.s32_adapter[-1].bias)

    def forward(self, z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, *,
                drop_style=None, drop_content=None, layout_scale=None,
                y32_struct: torch.Tensor | None = None):
        B = z_tau.shape[0]
        dev = z_tau.device
        if drop_style is None:
            drop_style = torch.zeros(B, dtype=torch.bool, device=dev)
        if drop_content is None:
            drop_content = torch.zeros(B, dtype=torch.bool, device=dev)
        cmask4 = (~drop_content).float().view(B, 1, 1, 1)
        smask2 = (~drop_style).float().view(B, 1)
        if layout_scale is None:
            lscale = (~drop_content).float()
        else:
            lscale = layout_scale.to(dev).float().view(B).clamp(0.0, 1.0) * (~drop_content).float()
        lscale4 = lscale.view(B, 1, 1, 1)

        d_feat = self.dino_proj(dino_16 * cmask4) * lscale4
        s16 = torch.cat([y16_lcn * lscale4, d_feat], dim=1)

        oh = F.one_hot(style_id.clamp(0, self.N_STYLE - 1), self.N_STYLE).float() * smask2
        uv = u_vivid * smask2
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([drop_style.float(), 1.0 - lscale], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, oh, uv, flags], dim=-1))

        h64 = self.in_conv(z_tau)
        for blk in self.enc64:
            h64 = blk(h64, cond)
        h32 = self.down32(h64)
        for blk in self.enc32:
            h32 = blk(h32, cond)
        h16 = self.down16(h32)
        h16 = self.mid1(h16, s16, cond)
        h16 = self.mid_attn(h16)
        h16 = self.mid2(h16, s16, cond)

        u32 = self.up32(h16.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u32 = self.skip32(torch.cat([u32, h32], dim=1))
        if y32_struct is not None:
            c32 = y32_struct * cmask4 * lscale4
            u32 = u32 + self.s32_gamma * self.s32_adapter(torch.cat([u32, c32], dim=1))
        for blk in self.dec32:
            u32 = blk(u32, cond)

        u64 = self.up64(u32.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1))
        u64 = self.skip64(torch.cat([u64, h64], dim=1))
        for blk in self.dec64:
            u64 = blk(u64, cond)
        return self.out_conv(F.silu(self.out_norm(u64)))


def _assert_zero_init_exact(base_state: dict, device: str):
    m0 = SLFUNet(base_dim=144, u_dim=128).to(device).eval()
    m0.load_state_dict(base_state, strict=True)
    m1 = SLFUNetS32Zero(base_dim=144, u_dim=128).to(device).eval()
    m1.load_state_dict(base_state, strict=False)
    with torch.no_grad():
        z = torch.randn(2, 4, 64, 64, device=device)
        y16 = torch.randn(2, 1, 16, 16, device=device)
        d16 = torch.randn(2, 384, 16, 16, device=device)
        y32 = torch.randn(2, 2, 32, 32, device=device)
        sid = torch.tensor([0, 3], device=device)
        uv = torch.randn(2, 128, device=device)
        tau = torch.tensor([0.4, 0.7], device=device)
        out0 = m0(z, y16, d16, sid, uv, tau)
        out1 = m1(z, y16, d16, sid, uv, tau, y32_struct=y32)
        diff = float((out0 - out1).abs().max())
        assert diff == 0.0, f"S32Zero 初始输出必须与原模型逐位一致, diff={diff}"
    print(f"[V23 S32Zero] Zero-init exact invariance verified (max_diff={diff})", flush=True)


def train_s32_arm(bundle: dict, core_bank: dict, use_s32: bool,
                  warmup_steps: int = 400, joint_steps: int = 1000, device: str = "cuda") -> Path:
    tag = "S32Zero" if use_s32 else "S32Ctrl"
    out_p = OUT / "s32_zero" / f"slf_{tag}.pt"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        print(f"[V23 {tag}] Reusing {out_p}", flush=True)
        return out_p

    seed_all(20261500)
    tr = bundle["v18_cache"]["train"]
    w_luma = bundle["v18_cache"]["luma_proj"]["w"].to(device).float()
    b_luma = bundle["v18_cache"]["luma_proj"]["b"].to(device).float()
    z_all, sid_all = tr["z"], tr["sid"].long()
    d16_all = tr["dino_16"]
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()
    y32_all = build_y32_pair(tr["y_64"].float()).half()
    u_exact_all = core_bank["u_exact_all"]

    all_cls_idx = [(sid_all == s).nonzero(as_tuple=True)[0] for s in range(5)]
    core_cls_idx = [ci.long() for ci in core_bank["core_idx_per_style"]]

    base_state = torch.load(PEAK_CKPT, map_location="cpu", weights_only=False)["ema"]
    if use_s32:
        _assert_zero_init_exact(base_state, device)

    model = SLFUNetS32Zero(base_dim=144, u_dim=128).to(device)
    model.load_state_dict(base_state, strict=False)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    micro = 120
    # 阶段1 (仅 use_s32=True): 单训 s32 模块 400 updates
    phases = [("warmup", warmup_steps, 1e-4)] if use_s32 else []
    phases.append(("joint", joint_steps, 4e-5))

    for ph_name, n_steps, peak_lr in phases:
        if ph_name == "warmup":
            for n, p in model.named_parameters():
                p.requires_grad_(n.startswith("s32_"))
        else:
            unfreeze = ("mid1.", "mid_attn.", "mid2.", "up32.", "skip32.", "dec32.",
                        "up64.", "skip64.", "dec64.", "out_norm.", "out_conv.") + (("s32_",) if use_s32 else ())
            for n, p in model.named_parameters():
                p.requires_grad_(n.startswith(unfreeze))

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=peak_lr, betas=(0.9, 0.99), weight_decay=1e-4)

        for step in range(1, n_steps + 1):
            prog = step / max(1, n_steps)
            lr = (peak_lr * 0.2) + 0.5 * (peak_lr * 0.8) * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)

            per_cls = micro // 5
            cur_pools = core_cls_idx if (step % 5 != 0) else all_cls_idx
            idx = torch.cat([ci[torch.randint(0, len(ci), (per_cls,))] for ci in cur_pools], dim=0)
            B = len(idx)

            z1 = z_all[idx].to(device, non_blocking=True).float()
            sid = sid_all[idx].to(device, non_blocking=True)
            d16 = d16_all[idx].to(device, non_blocking=True).float()
            y16 = y16_all[idx].to(device, non_blocking=True).float()
            y32 = y32_all[idx].to(device, non_blocking=True).float()
            uv = u_exact_all[idx].to(device, non_blocking=True).float()

            d16_aug = d16 + 0.08 * torch.randn_like(d16)
            d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
            y16_aug = (y16 * (0.85 + 0.30 * torch.rand(B, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

            u_r = torch.rand(B, device=device)
            ds = u_r < 0.12
            dc = (u_r >= 0.12) & (u_r < 0.42)
            both = u_r >= 0.95
            ds, dc = ds | both, dc | both
            lscale = torch.ones(B, device=device)

            z0 = torch.randn_like(z1)
            tau = torch.rand(B, device=device)
            zt = (1.0 - tau).view(B, 1, 1, 1) * z0 + tau.view(B, 1, 1, 1) * z1
            target_v = z1 - z0

            with autocast(device):
                pred_v = model(zt, y16_aug, d16_aug, sid, uv, tau,
                               drop_style=ds, drop_content=dc, layout_scale=lscale,
                               y32_struct=(y32 if use_s32 else None))
                fm_loss = F.mse_loss(pred_v.float(), target_v)
                z1_hat = (zt + (1.0 - tau).view(B, 1, 1, 1) * pred_v.float()).clamp(-5.0, 5.0)
                y16_hat = project_latent_y16_lcn(z1_hat, w_luma, b_luma)
                w_tau = ((tau > 0.25) & (~dc)).float()
                st_loss = (((y16_hat - y16).abs().mean(dim=(1, 2, 3))) * w_tau).sum() / w_tau.sum().clamp_min(1.0)
                loss = fm_loss + 0.10 * st_loss

            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            with torch.no_grad():
                for pe_w, pw in zip(ema.parameters(), model.parameters()):
                    pe_w.mul_(0.996).add_(pw, alpha=0.004)

            if step == 1 or step % 250 == 0 or step == n_steps:
                print(f"[V23 {tag}:{ph_name}] {step:4d}/{n_steps} fm={fm_loss.item():.4f} gamma={float(model.s32_gamma.detach()):.4f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"ema": ema.state_dict(), "use_s32": use_s32}, out_p)
    print(f"[V23 {tag}] Saved {out_p} in {(time.time()-t0)/60:.1f} min", flush=True)
    return out_p


@torch.no_grad()
def eval_s32_arm(bundle: dict, bank: dict, ckpt_p: Path, use_s32: bool, ctx: dict, device: str = "cuda") -> list[dict]:
    model = SLFUNetS32Zero(base_dim=144, u_dim=128).to(device).eval()
    model.load_state_dict(torch.load(ckpt_p, map_location="cpu", weights_only=False)["ema"], strict=True)
    core = PrismCore(bundle, bank, model=model, device=device)

    tasks, z_src_750 = ctx["tasks"], ctx["z_src_750"]
    sid_all = torch.tensor([t["source_style_id"] for t in tasks])
    tid_all = torch.tensor([t["target_style_id"] for t in tasks])
    same_all = sid_all == tid_all
    tag = "S32Zero" if use_s32 else "S32Ctrl"
    res = []

    for g_val, wc_val, wp_tag in ((0.45, 1.0, "hiS"), (0.50, 1.5, "hiC")):
        name = f"{tag}_{wp_tag}"
        cfg_hash = config_hash({"name": name, "ckpt": file_sha(ckpt_p)[:12], "g": g_val, "wc": wc_val})
        if (OUT / "raw" / name / "meta.json").exists() and validate_raw_meta(read_json(OUT / "raw" / name / "meta.json")):
            s = read_json(OUT / f"measured/{name}/summary_v23.json")
        else:
            gate = GateConfig(mode="fixed", fixed_g=g_val)
            cfg = CFGConfig(branches="full", rescale_phi=0.48)
            solver = dict(D5_SOLVER, wc_fig=wc_val, wc_abs=wc_val)
            wr = RawWriterV23(name, dec=ctx["dec"], device=device)
            t0 = time.time()
            for st in range(0, 750, 25):
                sl = slice(st, min(st + 25, 750))
                d16_b = ctx["d16_750"][sl].to(device).float()
                y16_b = ctx["y16_750"][sl].to(device).float()
                y32_b = ctx["y32_750"][sl].to(device).float() if use_s32 else None
                tid_b = tid_all[sl].to(device)
                u_style, dsd_ref = core.build_style_condition(d16_b, tid_b, mode="core8")
                g_tau, g_layout, g_guid, _ = effective_gate_components(dsd_ref, gate, float(bank["mid_dsd"]))
                lscale = (solver["lscale_abs"] + (1.0 - solver["lscale_abs"]) * g_layout).clamp(0.15, 1.0)
                content = build_content_condition(y16_b, d16_b, g_layout)
                tau_v = (solver["tau_min"] + (solver["tau_hi"] - solver["tau_min"]) * g_tau).clamp(0.0, 0.78)
                z = make_initial_state(ctx["eps_750"][sl].to(device).float(), ctx["z_t4"][sl].to(device).float(), tau_v)

                def vel_fn(zz, prog):
                    tv = tau_v + prog * (1.0 - tau_v)
                    B = zz.shape[0]
                    f0 = torch.zeros(B, dtype=torch.bool, device=device)
                    f1 = torch.ones(B, dtype=torch.bool, device=device)
                    ls0 = torch.zeros(B, device=device)
                    z4, y4, d4 = zz.repeat(4, 1, 1, 1), content["y_in"].repeat(4, 1, 1, 1), content["d_in"].repeat(4, 1, 1, 1)
                    s4, u4, t4 = tid_b.repeat(4), u_style.repeat(4, 1), tv.repeat(4)
                    ds = torch.cat([f0, f0, f1, f1], 0)
                    dc = torch.cat([f0, f1, f0, f1], 0)
                    l4 = torch.cat([lscale, ls0, lscale, ls0], 0)
                    y32_4 = y32_b.repeat(4, 1, 1, 1) if y32_b is not None else None
                    with autocast(device):
                        v_sc, v_s0, v_0c, v_00 = model(
                            z4, y4, d4, s4, u4, t4, drop_style=ds, drop_content=dc, layout_scale=l4, y32_struct=y32_4
                        ).float().chunk(4, 0)
                    wc, ws_j, ws_p = _guidance_curves(g_guid, solver, prog)
                    vr = v_00 + wc * (v_0c - v_00) + ws_j * (v_sc - v_0c) + ws_p * (v_s0 - v_00)
                    gf4 = g_guid.view(B, 1, 1, 1)
                    s_ref = (gf4 * v_sc + (1.0 - gf4) * v_s0).flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                    s_raw = vr.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                    return 0.48 * vr * (s_ref / s_raw) + 0.52 * vr

                z_gen = rk2_integrate(vel_fn, z, tau_v, int(solver["steps"]))
                z_out = torch.where(same_all[sl].view(-1, 1, 1, 1).to(device), z_src_750[sl].to(device).float(), z_gen)
                wr.write_batch(z_out, tasks[sl])
            meta = wr.finalize(cfg_hash=cfg_hash, git_sha=git_sha(), ckpt_sha256=file_sha(ckpt_p),
                               tasks_sha256=file_sha(ROOT / "eval_protocol/tasks_750.json"),
                               seed=EVAL_SEED, g=g_val, wc=wc_val, wall_sec=round(time.time() - t0, 1))
            assert validate_raw_meta(meta)
            s = run_measure_v23(name, device=device)
            export_raw_pngs(OUT / f"raw/{name}", OUT / f"png_750/{name}", max_workers=8)
        c = s["cross600"]
        res.append({"arm": name, "csd_top1": round(c["csd_top1"], 5), "dino_s": round(c["dino_s"], 5),
                    "dino_c": round(c["dino_c"], 5), "style_affinity": round(c["style_affinity"], 5),
                    "lpips": round(c["lpips"], 5)})
    del model, core
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device=args.device)

    ck_zero = train_s32_arm(bundle, bank, use_s32=True, device=args.device)
    ck_ctrl = train_s32_arm(bundle, bank, use_s32=False, device=args.device)

    tasks, z_src_750, _ = tasks_and_sources()
    z_t4, _ = build_or_load_t4_latents_750(bundle, device=args.device)
    y16_750, d16_750 = get_test750_y16_and_d16(bundle)
    if "y_64" in bundle["test_struct"]:
        y64_750 = bundle["test_struct"]["y_64"].float()
    else:
        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        y64_750 = ((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
    y32_750 = build_y32_pair(y64_750).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED))
    dec = load_s10_decoder(args.device)
    ctx = {"tasks": tasks, "z_src_750": z_src_750, "z_t4": z_t4, "y16_750": y16_750,
           "d16_750": d16_750, "y32_750": y32_750, "eps_750": eps_750, "dec": dec}

    results = []
    results.extend(eval_s32_arm(bundle, bank, ck_zero, True, ctx, device=args.device))
    results.extend(eval_s32_arm(bundle, bank, ck_ctrl, False, ctx, device=args.device))

    save_json(OUT / "p1_s32_zero.json", {"updated_utc": utc(), "results": results})
    print("\n=== V23 P1 第四组（S32Zero vs 同预算对照）汇总 ===", flush=True)
    for r in results:
        print(f"{r['arm']:16s}: Top1={r['csd_top1']*100:5.2f}% DINO-S={r['dino_s']:.4f} DINO-C={r['dino_c']:.4f} LPIPS={r['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
