"""V22-P2-1b RefAdapter-v2：异图重建目标（假设检验）。

P2-1 判负根因 = 重建式目标（重建内容图自身）使 u_style 冗余。v2 把目标换成**参考图**：
  结构条件（y16/d16）= 内容图 z_i；目标 = 参考图 z_ref（同画派异图）；u_style = A(u_ref, u_src)。
  现在 u_style 是唯一携带 z_ref 信息的通道 → 模型被迫从 u_style 读风格。
评测：1-shot SG-5 同协议（结构=内容图，u_style=adapter(u_ref,u_src)，纯噪声起点）。
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, ROOT, autocast, sanitize_cuda_env, save_json, seed_all, utc
from .ref_adapter import RefAdapter, RefAdapterModel, ZeroOnehotSLFUNet
from exp.v19.fiber_duality import build_or_load_v19_foundation


def train_ref_adapter_v2(bundle: dict, c21: dict, updates: int = 2000,
                         micro: int = 24, device: str = "cuda") -> list[float]:
    seed_all(20261221)
    tr = bundle["v18_cache"]["train"]
    z_all = tr["z"]
    sid_all = tr["sid"].long()
    d16_all = tr["dino_16"]
    from exp.v20.slf_flow import coarse_y16_lcn
    y16_all = coarse_y16_lcn(tr["y_64"].float()).half()

    pack = RefAdapterModel(
        fm_ckpt=OUT / "attn_flow/prism_dual_branch.pt",
        fiber_state=bundle["fnd"]["fiber_state"],
        fiber_in_dim=bundle["fnd"]["tr_fib"].shape[1],
        device=device,
    )
    nbr_idx = c21["train"]["nbr_idx"]

    opt = torch.optim.AdamW(pack.adapter.parameters(), lr=2e-4, weight_decay=1e-2)
    losses = []
    t0 = time.time()

    for step in range(1, updates + 1):
        prog = step / max(1, updates)
        lr = 5e-5 + 0.5 * (2e-4 - 5e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad(set_to_none=True)

        idx = torch.randint(0, len(z_all), (micro,))
        perm_nbr = torch.rand(micro, 12).topk(1, dim=1).indices
        ref_idx = nbr_idx[idx[:, None], perm_nbr].squeeze(1)

        # 结构 = 内容图；目标 = 参考图（u_style 必须携带 z_ref 信息）
        z_struct = z_all[idx].to(device).float()
        z_target = z_all[ref_idx].to(device).float()
        sid = sid_all[ref_idx].to(device)  # 目标画派 = 参考图画派
        d16 = d16_all[idx].to(device).float()
        y16 = y16_all[idx].to(device).float()

        u_src = pack.u_from_fiber(bundle["fnd"]["tr_fib"][idx]).detach()
        u_ref = pack.u_from_fiber(bundle["fnd"]["tr_fib"][ref_idx]).detach()
        u_style = pack.adapter(u_ref, u_src)

        d16_aug = d16 + 0.08 * torch.randn_like(d16)
        d16_aug = d16_aug - d16_aug.mean(dim=(2, 3), keepdim=True)
        y16_aug = (y16 * (0.85 + 0.30 * torch.rand(micro, 1, 1, 1, device=device)) + 0.04 * torch.randn_like(y16)).clamp(-1.1, 1.1)

        z0 = torch.randn_like(z_target)
        tau = torch.rand(micro, device=device)
        zt = (1.0 - tau).view(-1, 1, 1, 1) * z0 + tau.view(-1, 1, 1, 1) * z_target
        target_v = z_target - z0

        with autocast(device):
            pred_v = pack.fm(zt, y16_aug, d16_aug, sid, u_style, tau,
                             drop_style=torch.zeros(micro, dtype=torch.bool, device=device),
                             drop_content=torch.zeros(micro, dtype=torch.bool, device=device),
                             layout_scale=torch.ones(micro, device=device))
            loss = F.mse_loss(pred_v.float(), target_v)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(pack.adapter.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

        if step == 1 or step % 400 == 0 or step == updates:
            print(f"[V22 RefV2] {step:4d}/{updates} loss={loss.item():.4f} alpha={pack.adapter.alpha.item():.3f} ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"adapter": pack.adapter.state_dict(), "updates": updates, "target": "z_ref(异图重建)"},
               OUT / "ref_adapter/ref_adapter_v2.pt")
    print(f"[V22 RefV2] Saved in {(time.time()-t0)/60:.1f} min", flush=True)
    return losses


@torch.no_grad()
def eval_ref_adapter_v2_1shot(bundle: dict, c21: dict, device: str = "cuda") -> list[dict]:
    from exp.v18 import tasks_and_sources
    from exp.v20.slf_flow import coarse_y16_lcn
    from src.utils import decode_s10, load_s10_decoder
    from src.utils.raw_io import RawWriter, export_raw_pngs
    from src.eval.runner import run_measure_subprocess

    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)

    pack = RefAdapterModel(
        fm_ckpt=OUT / "attn_flow/prism_dual_branch.pt",
        fiber_state=bundle["fnd"]["fiber_state"],
        fiber_in_dim=bundle["fnd"]["tr_fib"].shape[1],
        device=device,
    )
    from .ref_adapter import RefAdapter
    ad = RefAdapter(dim=128).to(device)
    ad.load_state_dict(torch.load(OUT / "ref_adapter/ref_adapter_v2.pt", map_location="cpu", weights_only=False)["adapter"])
    pack.adapter = ad

    fib_750 = bundle["fnd"]["test750_fib"].float()
    u_src_750 = pack.u_from_fiber(fib_750).cpu()

    if "y_64" in bundle["test_struct"]:
        y16_750 = coarse_y16_lcn(bundle["test_struct"]["y_64"].float()).half()
    else:
        w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
        b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
        y64_750 = ((z_src_750.float() * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)
        y16_750 = coarse_y16_lcn(y64_750).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(20261185))
    u_sg5 = c21["sg5"]["u"].float()

    results = []
    for arm_name in ("P2B_1shot_v2_vsc",):
        wr = RawWriter(arm_name, raw_root=OUT / "raw", kind="v22")
        t0 = time.time()
        for st in range(0, 750, 25):
            sl = slice(st, min(st + 25, 750))
            grp = tasks[sl]
            B = len(grp)
            zs = z_src_750[sl].to(device).float()
            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
            same = sid_p == tid_t

            u_ref_b = u_sg5[tid_t.cpu()].to(device)
            u_src_b = u_src_750[sl].to(device)
            u_style = pack.adapter(u_ref_b, u_src_b)

            y16 = y16_750[sl].to(device).float()
            d16 = bundle["test_struct"]["dino_16"][sl].to(device).float()
            z = eps_750[sl].to(device).float()
            f0 = torch.zeros(B, dtype=torch.bool, device=device)
            l1 = torch.ones(B, device=device)
            sid0 = torch.zeros(B, dtype=torch.long, device=device)
            steps = 10
            dt = 1.0 / steps

            def vel(zz, prog):
                with autocast(device):
                    return pack.fm(zz, y16, d16, sid0, u_style, prog * torch.ones(B, device=device),
                                   drop_style=f0, drop_content=f0, layout_scale=l1).float()

            for i in range(steps):
                v1 = vel(z, i / steps)
                zm = z + 0.5 * dt * v1
                v2 = vel(zm, (i + 0.5) / steps)
                z = z + dt * v2
            z_out = torch.where(same.view(B, 1, 1, 1), zs, z)
            wr.append([t["task_id"] for t in grp],
                      decode_s10(dec, z_out, batch=16).clamp(0.0, 1.0), z_out.cpu())
        wall = round(time.time() - t0, 1)
        wr.finish(arm=arm_name, wall_sec=wall)
        print(f"[V22 RefV2] {arm_name} done in {wall}s", flush=True)
        run_measure_subprocess(OUT / "raw" / arm_name, OUT / "measured" / arm_name, arm_name=arm_name, device=device)
        export_raw_pngs(OUT / f"raw/{arm_name}", OUT / f"png_750/{arm_name}", max_workers=8)
        results.append({"arm": arm_name, "wall_sec": wall})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--micro", type=int, default=24)
    args = ap.parse_args()

    sanitize_cuda_env()
    bundle = build_or_load_v19_foundation(args.pool, device=args.device)
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt", map_location="cpu", weights_only=False)

    if not (OUT / "ref_adapter/ref_adapter_v2.pt").exists():
        train_ref_adapter_v2(bundle, c21, updates=args.updates, micro=args.micro, device=args.device)
    res = eval_ref_adapter_v2_1shot(bundle, c21, device=args.device)
    save_json(OUT / "p2b_1shot_v2_results.json", {"updated_utc": utc(), "arms": res})


if __name__ == "__main__":
    main()
