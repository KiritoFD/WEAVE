"""V18 Track A & Track C: Calibrated Target-Source LD4+ & Pure 1-Step B4+DINO Amortization.

Track A1 (LD4+):
  Fixes two bottlenecks in V9 LD4:
    1. Per-sample LocalStyleEnergy (fixes batch-mean broadcast dilution) +
       holdout-calibrated R1 class bias in candidate acceptance proxy (cures
       R1's -1.18 logit bias against Ukiyo-e / Minimalism).
    2. RMS-normalized Target - 0.4*Source SPD-preconditioned R1 direction
       (`r1_ts`), actively erasing residual source-style features while pushing
       toward the target style manifold.

Track A2 (FastStudent — 0.42M Pure 1-Step Feed-Forward Amortization):
  Generates 2,500 clean balanced cross-style pairs (B4 start -> N2C -> LD4+
  endpoint) and trains a zero-identity target-source DINO-Layout Residual
  Adapter so inference is a SINGLE forward pass (B4 + Adapter + S10, zero
  test-time R1/energy gradients!).
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import OUT, RawWriterV18, autocast, save_json, seed_all, tasks_and_sources


class CalibratedLD4Plus:
    """Upgraded LD4+ operator usable on N2C latents, B4 latents, or CCF latents."""

    def __init__(self, r1_bias: torch.Tensor | None = None, device: str = "cuda"):
        from src.infer import load_critic, load_c3unet, canonicalizer
        from exp.v8.guidance import SPDHead
        from exp.v9_local.local_style import LocalStyleEnergy, BETA, MU
        from exp.v9_local.direction import DirectionComputer
        from . import ROOT

        class PerSampleLSE(LocalStyleEnergy):
            def loss(self, z, tid):
                et, en = self.energies(z, tid)
                return (et + BETA * F.softplus(MU + et - en)).mean(1)

        self.device = device
        self.critic = load_critic(device)
        ck = torch.load(ROOT / "runs/v8/train/N2C/step_04000.pt", map_location="cpu", weights_only=False)
        self.head = SPDHead(rank=int(ck.get("rank", 0))).to(device).eval()
        self.head.load_state_dict(ck["head"])
        self.head.requires_grad_(False)

        self.lse = PerSampleLSE(str(ROOT / "runs/v9_local_4070/bank/style_bank.pt"), device)
        self.dcomp = DirectionComputer(self.critic, self.head, self.lse, device)
        self.b4, _ = load_c3unet(ROOT / "runs/v6/P1_B4_distill50/P1_B4_distill50.pt", device)
        self.canon = canonicalizer(device)
        self.r1_bias = r1_bias.to(device).float() if r1_bias is not None else torch.zeros(5, device=device)

    @torch.no_grad()
    def calibrate_on_holdout(self, z_holdout: torch.Tensor, sid_holdout: torch.Tensor):
        """Fit class-prior margin bias on clean holdout latents (never Test750 / CSD)."""
        logits_all = []
        for st in range(0, len(z_holdout), 64):
            logits_all.append(self.critic(z_holdout[st:st + 64].to(self.device).float()).float().cpu())
        logits = torch.cat(logits_all, dim=0)  # (N, 5)
        mean_logit = logits.mean(dim=0)        # classes with inflated prior logits get subtracted
        self.r1_bias = (-(mean_logit - mean_logit.mean())).to(self.device)
        save_json(OUT / "ld4plus_r1_calibration.json", {
            "mean_holdout_logit": mean_logit.tolist(),
            "applied_bias": self.r1_bias.cpu().tolist(),
        })
        print("[V18 LD4+] R1 logit bias:", [round(x, 4) for x in self.r1_bias.cpu().tolist()], flush=True)

    def calibrated_margin(self, z: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        q = self.critic(z.float()).float() + self.r1_bias.unsqueeze(0)
        oh = F.one_hot(tid, 5).bool()
        others = q.masked_fill(oh, -1e9)
        return q.gather(1, tid[:, None]).squeeze(1) - others.max(dim=1).values

    @torch.no_grad()
    def run_b4_and_n2c(self, z_src: torch.Tensor, tid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run B4 one-shot -> 2-step SPD N2C unroll on arbitrary source latents."""
        from exp.v8.guidance import normalize, project_rms, target_margin

        c = self.canon(z_src.float(), mode="c1_gentle", augment=False)
        z0 = self.b4(c, 0.0, c, tid).detach()
        cur = z0.clone()
        etas = [2.0, 1.5, 1.25, 1.0, 0.75, 0.5]
        d_gen = (z0 - z_src).square().flatten(1).mean(1).clamp_min(1e-8)

        for _ in range(2):
            with torch.enable_grad():
                zr = cur.detach().clone().requires_grad_(True)
                g = torch.autograd.grad(target_margin(self.critic, zr.float(), tid).sum(), zr)[0].detach()
            diag, u = self.head(c, cur, tid, g)
            ug = (u * g.unsqueeze(1)).sum(2)
            delta = diag * g + (u * ug.unsqueeze(2)).sum(1)
            step = normalize(delta)

            m_cur = target_margin(self.critic, cur.float(), tid)
            chosen = cur.clone()
            rem = torch.ones(cur.shape[0], dtype=torch.bool, device=cur.device)
            for mult in etas:
                cand = project_rms(cur + 0.08 * mult * step, z0, 0.25)
                cost = (cand - z0).square().flatten(1).mean(1) / d_gen
                ok = rem & (cost <= 0.15) & (target_margin(self.critic, cand.float(), tid) > m_cur)
                if ok.any():
                    chosen[ok] = cand[ok]
                    rem[ok] = False
            cur = chosen
        same = torch.zeros(len(z_src), dtype=torch.bool, device=z_src.device)
        return c, z0, cur

    def refine(
        self,
        z_start: torch.Tensor,
        z_anchor: torch.Tensor,
        z_src: torch.Tensor,
        sid: torch.Tensor,
        tid: torch.Tensor,
        *,
        eps: float = 0.16,
        rho: float = 0.20,
        k_steps: int = 3,
        mask_frac: float = 0.30,
        src_push_alpha: float = 0.35,
        w_r1: float = 0.35,
        w_lse: float = 1.0,
        etas: tuple[float, ...] = (0.35, 0.65, 1.0, 1.35),
    ) -> torch.Tensor:
        """Multi-direction masked solver with RMS-normalized Target - alpha*Source R1."""
        from exp.v8.guidance import normalize, project_rms
        from exp.v9_local.solver import cost_tot

        z = z_start.clone()
        b = z.shape[0]
        dev = z.device
        same = sid == tid

        with torch.no_grad():
            mask = self.dcomp.region_mask(z, z_anchor, tid, frac=mask_frac) if mask_frac < 0.999 else None

        active = ~same.clone()
        for _ in range(k_steps):
            if not active.any():
                break
            idx = active.nonzero(as_tuple=True)[0]
            zi, z0i, zsi = z[idx], z_anchor[idx], z_src[idx]
            si, ti = sid[idx], tid[idx]
            mi = mask[idx] if mask is not None else 1.0

            # Compute candidate directions in RMS-normalized scale (||d||_RMS = 1)
            with torch.enable_grad():
                base_dirs = self.dcomp.directions(zi, z0i, ti, "mix")
                zr = zi.detach().clone().requires_grad_(True)
                obj_ts = self.calibrated_margin(zr, ti) - src_push_alpha * self.calibrated_margin(zr, si)
                g_ts = torch.autograd.grad(obj_ts.sum(), zr)[0].detach()
            with torch.no_grad():
                diag, u = self.head(z0i, zi.detach(), ti, g_ts)
                ug = (u * g_ts.unsqueeze(1)).sum(2)
                gr_ts = diag * g_ts + (u * ug.unsqueeze(2)).sum(1)
                r1_ts = normalize(gr_ts)

                cand_dirs = {
                    "r1": base_dirs["r1"],
                    "local": base_dirs["local"],
                    "mix": base_dirs["mix"],
                    "r1_ts": r1_ts,
                    "mix_ts": normalize(0.55 * r1_ts + 0.45 * base_dirs["local"]),
                }
                # Scale step basis to per-image feasible budget radius so outer steps don't saturate clip
                denom = (z0i - zsi).flatten(1).square().mean(1).clamp_min(1e-4)
                step_scale = (eps * denom).sqrt().clamp(0.05, rho).view(-1, 1, 1, 1)

                def _proxy(zc, t_vec):
                    return w_r1 * self.calibrated_margin(zc, t_vec) - w_lse * self.lse.loss(zc, t_vec)

                best_p = _proxy(zi, ti)
                best_z = zi.clone()
                accepted = torch.zeros(len(idx), dtype=torch.bool, device=dev)

                for eta in etas:
                    for d_tensor in cand_dirs.values():
                        step = d_tensor * mi * step_scale
                        cand = project_rms(zi + eta * step, z0i, rho)
                        feas = cost_tot(cand, z0i, zsi) <= eps
                        if not feas.any():
                            continue
                        fi = feas.nonzero(as_tuple=True)[0]
                        p_val = _proxy(cand[feas], ti[feas])
                        better = p_val > best_p[fi]
                        if better.any():
                            sel = fi[better]
                            best_z[sel] = cand[sel]
                            best_p[sel] = p_val[better]
                            accepted[sel] = True

                for j, orig_i in enumerate(idx.tolist()):
                    if accepted[j]:
                        z[orig_i] = best_z[j]
                    else:
                        active[orig_i] = False

        with torch.no_grad():
            z[same] = z_start[same]
        return z.detach()


class FastDINOStudent(nn.Module):
    """0.44M Single-Step Feed-Forward Corrector over B4: exact zero when sid == tid."""

    def __init__(self, width: int = 80, blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(8 + 4, width, 3, padding=1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        self.dino_proj = nn.Sequential(
            nn.Conv2d(384, width, 1),
            nn.GroupNorm(16, width),
            nn.SiLU(inplace=True),
        )
        self.style_emb = nn.Embedding(5, width)
        self.film = nn.ModuleList([nn.Linear(width, width * 2) for _ in range(blocks)])
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(16, width),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GroupNorm(16, width),
                nn.SiLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
            )
            for _ in range(blocks)
        ])
        self.out = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        z_b4: torch.Tensor,
        c_canon: torch.Tensor,
        c_64: torch.Tensor,
        dino_16: torch.Tensor,
        sid: torch.Tensor,
        tid: torch.Tensor,
    ) -> torch.Tensor:
        # Exact target-source difference embedding: strictly 0 when sid == tid
        cond = self.style_emb(tid.clamp(0, 4)) - self.style_emb(sid.clamp(0, 4))
        d_up = F.interpolate(self.dino_proj(dino_16), size=(64, 64), mode="bilinear", align_corners=False)
        h = self.stem(torch.cat([z_b4, c_canon, c_64], dim=1)) + d_up
        for conv, fc in zip(self.convs, self.film):
            g, b = fc(cond).chunk(2, dim=-1)
            h_mod = h * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
            h = h + conv(h_mod)
        delta = self.out(h)
        same = (sid == tid).view(-1, 1, 1, 1)
        return torch.where(same, torch.zeros_like(delta), delta)


def run_track_a_and_amortize(cache_dict: dict, test_struct: dict, device: str = "cuda"):
    """Run A0 (LD4 repro), A1 (LD4+ sweep), and A2 (FastDINOStudent 1-step amortization)."""
    from src.utils import load_s10_decoder, decode_s10

    op = CalibratedLD4Plus(device=device)
    op.calibrate_on_holdout(cache_dict["holdout"]["z"], cache_dict["holdout"]["sid"])
    tasks, z_src_750, n2c_cache = tasks_and_sources()
    z0_750 = n2c_cache["z0"].float()
    zn2_750 = n2c_cache["z_n2"].float()

    dec = load_s10_decoder(device)

    # ---- A0: Standard V9 LD4 on full 750 (3060 same-session anchor) ----
    if not (OUT / "raw/V18_A0_LD4_repro/meta.json").exists():
        from exp.v9_local.solver import solve_batch
        w_a0 = RawWriterV18("V18_A0_LD4_repro")
        t0 = time.time()
        for st in range(0, 750, 16):
            sl = slice(st, min(st + 16, 750))
            grp = tasks[sl]
            sid = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid = torch.tensor([t["target_style_id"] for t in grp], device=device)
            zn = zn2_750[sl].to(device)
            z0 = z0_750[sl].to(device)
            zs = z_src_750[sl].to(device)
            with torch.enable_grad():
                dirs = op.dcomp.directions(zn, z0, tid, "mix")
            with torch.no_grad():
                mask = op.dcomp.region_mask(zn, z0, tid, frac=0.25)
                z_out, _ = solve_batch(zn, z0, zs, sid, tid, [dirs], mask, op.critic, op.lse, eps=0.10, k_steps=2, rho=0.15)
                z_out[sid == tid] = zs[sid == tid]
                rgb = decode_s10(dec, z_out, batch=8)
            w_a0.append([t["task_id"] for t in grp], rgb, z_out)
        w_a0.finish(note="V9 LD4 exact 750 anchor on 3060", wall_sec=round(time.time() - t0, 1))

    # ---- A1: Upgraded LD4+ (calibrated R1 + Target-Source RMS dir + eps=0.16, K=3) ----
    for arm_tag, eps_v, rho_v, k_v, mfrac in [
        ("V18_A1_LD4plus_e10", 0.10, 0.16, 3, 0.30),
        ("V18_A1b_LD4plus_e18", 0.18, 0.22, 3, 0.35),
    ]:
        if (OUT / f"raw/{arm_tag}/meta.json").exists():
            continue
        wr = RawWriterV18(arm_tag)
        t0 = time.time()
        for st in range(0, 750, 16):
            sl = slice(st, min(st + 16, 750))
            grp = tasks[sl]
            sid = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid = torch.tensor([t["target_style_id"] for t in grp], device=device)
            zn = zn2_750[sl].to(device)
            z0 = z0_750[sl].to(device)
            zs = z_src_750[sl].to(device)
            z_out = op.refine(zn, z0, zs, sid, tid, eps=eps_v, rho=rho_v, k_steps=k_v, mask_frac=mfrac)
            with torch.no_grad():
                z_out[sid == tid] = zs[sid == tid]
                rgb = decode_s10(dec, z_out, batch=8)
            wr.append([t["task_id"] for t in grp], rgb, z_out)
        wr.finish(eps=eps_v, rho=rho_v, k_steps=k_v, wall_sec=round(time.time() - t0, 1))
        print(f"[V18 LD4+] {arm_tag} generated in {time.time()-t0:.1f}s", flush=True)

    # ---- A2: Build 2,500 clean balanced B4 -> LD4+ pairs & train 1-step FastDINOStudent ----
    pairs_path = OUT / "cache/b4_ld4plus_pairs.pt"
    if not pairs_path.exists():
        t0 = time.time()
        tr = cache_dict["train"]
        # Pick 100 per style = 500 balanced clean paintings x 5 targets = 2,500 pairs
        g_p = torch.Generator().manual_seed(20261118)
        sel_idx = torch.cat([
            (tr["sid"] == s).nonzero(as_tuple=True)[0][torch.randperm(int((tr["sid"] == s).sum()), generator=g_p)[:100]]
            for s in range(5)
        ])
        z_s = tr["z"][sel_idx].repeat_interleave(5, dim=0)
        sid_p = tr["sid"][sel_idx].repeat_interleave(5, dim=0)
        tid_p = torch.arange(5).repeat(len(sel_idx))

        z_b4_list, c_can_list, delta_list = [], [], []
        for st in range(0, len(z_s), 25):
            sl = slice(st, min(st + 25, len(z_s)))
            zb = z_s[sl].to(device)
            sb = sid_p[sl].to(device)
            tb = tid_p[sl].to(device)
            c_can, z0_b, zn2_b = op.run_b4_and_n2c(zb, tb)
            z_end = op.refine(zn2_b, z0_b, zb, sb, tb, eps=0.16, rho=0.20, k_steps=3, mask_frac=0.32)
            with torch.no_grad():
                z_end[sb == tb] = z0_b[sb == tb]
            z_b4_list.append(z0_b.cpu().half())
            c_can_list.append(c_can.cpu().half())
            delta_list.append((z_end - z0_b).cpu().half())
            if (st // 25) % 20 == 0:
                print(f"[V18 Pairs] {st+len(zb)}/{len(z_s)} ({time.time()-t0:.0f}s)", flush=True)

        torch.save(
            {
                "sel_idx": sel_idx,
                "sid": sid_p,
                "tid": tid_p,
                "z_b4": torch.cat(z_b4_list, dim=0),
                "c_canon": torch.cat(c_can_list, dim=0),
                "delta": torch.cat(delta_list, dim=0),
            },
            pairs_path,
        )
        print(f"[V18 Pairs] Saved 2,500 B4->LD4+ pairs in {time.time()-t0:.1f}s", flush=True)

    student_ckpt = OUT / "a2_student/fast_student.pt"
    if not student_ckpt.exists():
        student_ckpt.parent.mkdir(parents=True, exist_ok=True)
        pairs = torch.load(pairs_path, map_location="cpu", weights_only=False)
        tr = cache_dict["train"]
        sel_idx = pairs["sel_idx"]
        c64_src = tr["c_64"][sel_idx]      # (500, 4, 64, 64)
        dino_src = tr["dino_16"][sel_idx]  # (500, 384, 16, 16)

        seed_all(20261118)
        net = FastDINOStudent(width=80, blocks=4).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=4e-4, weight_decay=0.01)
        updates = 6000
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=updates, pct_start=0.06)
        gen = torch.Generator().manual_seed(20261118)

        # Train primarily on cross-style pairs (same-style is exact 0 by architecture!)
        cross_idx = (pairs["sid"] != pairs["tid"]).nonzero(as_tuple=True)[0]
        t0 = time.time()
        net.train()
        for step in range(1, updates + 1):
            pick = cross_idx[torch.randint(len(cross_idx), (32,), generator=gen)]
            src_row = pick // 5
            zb4 = pairs["z_b4"][pick].to(device).float()
            cc = pairs["c_canon"][pick].to(device).float()
            dt_gt = pairs["delta"][pick].to(device).float()
            sb = pairs["sid"][pick].to(device).long()
            tb = pairs["tid"][pick].to(device).long()
            c64 = c64_src[src_row].to(device).float()
            d16 = dino_src[src_row].to(device).float()

            with autocast(device):
                pred = net(zb4, cc, c64, d16, sb, tb).float()

            mse = F.mse_loss(pred, dt_gt)
            valid = dt_gt.flatten(1).norm(dim=1) > 1e-4
            cos_loss = (1.0 - F.cosine_similarity(pred[valid].flatten(1), dt_gt[valid].flatten(1), dim=1)).mean() if valid.any() else 0.0
            lf_pen = F.avg_pool2d(pred - dt_gt, 4).square().mean()
            loss = mse + 0.15 * cos_loss + 0.10 * lf_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if step == 1 or step % 500 == 0 or step == updates:
                with torch.no_grad():
                    cos_v = float(F.cosine_similarity(pred[valid].flatten(1), dt_gt[valid].flatten(1), dim=1).mean().item())
                    rms_ratio = float(pred.flatten(1).norm(dim=1).mean() / dt_gt.flatten(1).norm(dim=1).clamp_min(1e-6).mean())
                print(f"[V18 Student] {step:4d}/{updates} mse={mse.item():.5f} cos={cos_v:.4f} rmsRatio={rms_ratio:.3f} {time.time()-t0:.0f}s", flush=True)

        torch.save({"model": net.cpu().state_dict(), "width": 80, "blocks": 4, "n_params": net.n_params()}, student_ckpt)
        del net, opt
        torch.cuda.empty_cache()

    # Evaluate FastDINOStudent (1-step pure feed-forward!) + Wrong-Label Control on Test750
    st_ck = torch.load(student_ckpt, map_location="cpu", weights_only=False)
    student = FastDINOStudent(width=st_ck["width"], blocks=st_ck["blocks"]).to(device).eval()
    student.load_state_dict(st_ck["model"])

    for arm_tag, is_wrong in [("V18_A2_FastStudent", False), ("V18_A2_FastStudent_wrong", True)]:
        if (OUT / f"raw/{arm_tag}/meta.json").exists():
            continue
        wr = RawWriterV18(arm_tag)
        t0 = time.time()
        with torch.no_grad():
            for st in range(0, 750, 32):
                sl = slice(st, min(st + 32, 750))
                grp = tasks[sl]
                zs = z_src_750[sl].to(device)
                sid = torch.tensor([t["source_style_id"] for t in grp], device=device)
                tid = torch.tensor([t["target_style_id"] for t in grp], device=device)
                used_tid = (tid + 2) % 5 if is_wrong else tid
                c_can = op.canon(zs, mode="c1_gentle", augment=False)
                z_b4 = op.b4(c_can, 0.0, c_can, used_tid)
                c64 = test_struct["c_64"][sl].to(device).float()
                d16 = test_struct["dino_16"][sl].to(device).float()
                with autocast(device):
                    delta = student(z_b4, c_can, c64, d16, sid, used_tid).float()
                z_final = torch.where((sid == tid).view(-1, 1, 1, 1), zs, z_b4 + delta)
                rgb = decode_s10(dec, z_final, batch=8)
                wr.append([t["task_id"] for t in grp], rgb, z_final)
        wr.finish(wrong_label=is_wrong, pure_feedforward=True, wall_sec=round(time.time() - t0, 1))
        print(f"[V18 Student Eval] {arm_tag} 750 generated in {time.time()-t0:.1f}s (pure 1-step FF!)", flush=True)

    del student, op, dec
    torch.cuda.empty_cache()


def run_ccf_plus_ld4(source_ccf_arm: str, out_arm_name: str, cache_dict: dict, device: str = "cuda"):
    """Track C: Apply 2 lightweight masked LD4+ steps on top of a CCF-generated raw directory."""
    if (OUT / f"raw/{out_arm_name}/meta.json").exists():
        return
    from src.utils import load_s10_decoder, decode_s10
    from . import read_json

    op = CalibratedLD4Plus(device=device)
    op.calibrate_on_holdout(cache_dict["holdout"]["z"], cache_dict["holdout"]["sid"])
    dec = load_s10_decoder(device)

    tasks, z_src_750, _ = tasks_and_sources()
    src_raw = OUT / "raw" / source_ccf_arm
    meta = read_json(src_raw / "meta.json")
    wr = RawWriterV18(out_arm_name)
    t0 = time.time()

    offset = 0
    for b_entry in meta["batches"]:
        b_data = torch.load(src_raw / b_entry["file"], map_location="cpu", weights_only=False)
        z_ccf = b_data["z_final"].float().to(device)
        n = len(z_ccf)
        grp = tasks[offset:offset + n]
        zs = z_src_750[offset:offset + n].to(device)
        sid = torch.tensor([t["source_style_id"] for t in grp], device=device)
        tid = torch.tensor([t["target_style_id"] for t in grp], device=device)
        z_ref = op.refine(z_ccf, z_ccf, zs, sid, tid, eps=0.10, rho=0.15, k_steps=2, mask_frac=0.28)
        with torch.no_grad():
            rgb = decode_s10(dec, z_ref, batch=8)
        wr.append(b_entry["ids"], rgb, z_ref)
        offset += n

    wr.finish(base_ccf_arm=source_ccf_arm, wall_sec=round(time.time() - t0, 1))
    print(f"[V18 Track C] {out_arm_name} generated in {time.time()-t0:.1f}s", flush=True)
    del op, dec
    torch.cuda.empty_cache()
