"""V18 Track B: De-Styled Structural-Content Flow U-Net (CCFUNet).

Integrates from PURE GAUSSIAN NOISE (tau0 = 0.0) so zero source-style pixel/latent
statistics pass through z_tau0. Spatial layout is locked via multi-scale SPADE on
palette-free C_64 + spatially-whitened grayscale DINOv2 layout P_16; target style
is injected via Sub-Mode Prototype AdaLN with Style-CFG / Dual-CFG.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import EVAL_SEED, OUT, RawWriterV18, autocast, save_json, seed_all, tasks_and_sources
from .structure import augment_structure, project_latent_lcn


def sinusoidal_time_emb(tau: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half
    )
    args = tau.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class SPADEStyleBlock(nn.Module):
    """Spatial structure modulation (SPADE) + global style/time modulation (AdaLN-FiLM)."""

    def __init__(self, ch: int, struct_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(16, ch), ch, affine=False)
        self.spade = nn.Sequential(
            nn.Conv2d(struct_ch, 64, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, ch * 2, 3, padding=1),
        )
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(16, ch), ch, affine=False)
        self.film = nn.Linear(cond_dim, ch * 2)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, s_map: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        gc, bc = self.spade(s_map).chunk(2, dim=1)
        h = self.conv1(self.act(h * (1.0 + gc) + bc))

        h = self.norm2(h)
        gs, bs = self.film(cond).chunk(2, dim=-1)
        h = self.conv2(self.act(h * (1.0 + gs[:, :, None, None]) + bs[:, :, None, None]))
        return x + h


class BottleneckAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xf = x.flatten(2).transpose(1, 2)
        xn = self.norm(xf)
        out, _ = self.attn(xn, xn, xn, need_weights=False)
        return x + out.transpose(1, 2).reshape(B, C, H, W)


class CCFUNet(nn.Module):
    """De-Styled Structural-Content Flow U-Net (64 -> 32 -> 16 -> 32 -> 64)."""

    N_STYLE = 5

    def __init__(self, base_dim: int = 128, proto_dim: int = 48, cond_dim: int = 256):
        super().__init__()
        self.base_dim = base_dim
        self.proto_dim = proto_dim
        w0, w1, w2 = base_dim, base_dim * 2, base_dim * 3
        dino_ch = 64

        self.dino_proj = nn.Sequential(
            nn.Conv2d(384, 128, 1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, dino_ch, 1),
        )
        # Condition vector: sin_time(64) + onehot(5) + proto(48) + drop_flags(2) = 119
        self.cond_mlp = nn.Sequential(
            nn.Linear(64 + self.N_STYLE + proto_dim + 2, cond_dim),
            nn.SiLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(inplace=True),
        )

        s64_ch = 4
        s32_ch = 4 + dino_ch
        s16_ch = 4 + dino_ch

        self.in_conv = nn.Conv2d(4 + 4, w0, 3, padding=1)
        self.enc0 = SPADEStyleBlock(w0, s64_ch, cond_dim)
        self.down1 = nn.Conv2d(w0, w1, 3, stride=2, padding=1)
        self.enc1a = SPADEStyleBlock(w1, s32_ch, cond_dim)
        self.enc1b = SPADEStyleBlock(w1, s32_ch, cond_dim)
        self.down2 = nn.Conv2d(w1, w2, 3, stride=2, padding=1)
        self.enc2 = SPADEStyleBlock(w2, s16_ch, cond_dim)

        self.mid_attn = BottleneckAttention(w2, heads=8)
        self.mid_blk = SPADEStyleBlock(w2, s16_ch, cond_dim)

        self.up2 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.dec1_fuse = nn.Conv2d(w1 * 2, w1, 1)
        self.dec1a = SPADEStyleBlock(w1, s32_ch, cond_dim)
        self.dec1b = SPADEStyleBlock(w1, s32_ch, cond_dim)

        self.up1 = nn.ConvTranspose2d(w1, w0, 2, stride=2)
        self.dec0_fuse = nn.Conv2d(w0 * 2, w0, 1)
        self.dec0a = SPADEStyleBlock(w0, s64_ch, cond_dim)
        self.dec0b = SPADEStyleBlock(w0, s64_ch, cond_dim)

        self.out_norm = nn.GroupNorm(16, w0)
        self.out_conv = nn.Conv2d(w0, 4, 3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init residual second convs for clean gradient flow at step 0
        for m in self.modules():
            if isinstance(m, SPADEStyleBlock):
                nn.init.zeros_(m.conv2.weight)
                nn.init.zeros_(m.conv2.bias)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        z_tau: torch.Tensor,
        c_64: torch.Tensor,
        dino_16: torch.Tensor,
        style_id: torch.Tensor,
        proto: torch.Tensor,
        tau: torch.Tensor,
        drop_style: torch.Tensor | bool = False,
        drop_content: torch.Tensor | bool = False,
    ) -> torch.Tensor:
        B = z_tau.shape[0]
        dev = z_tau.device

        if isinstance(drop_style, bool):
            ds = torch.full((B,), drop_style, dtype=torch.bool, device=dev)
        else:
            ds = drop_style.to(dev).bool()
        if isinstance(drop_content, bool):
            dc = torch.full((B,), drop_content, dtype=torch.bool, device=dev)
        else:
            dc = drop_content.to(dev).bool()

        keep_s = (~ds).float()
        keep_c = (~dc).float()

        c64 = c_64 * keep_c.view(B, 1, 1, 1)
        d16_proj = self.dino_proj(dino_16) * keep_c.view(B, 1, 1, 1)
        d32_proj = F.interpolate(d16_proj, size=(32, 32), mode="bilinear", align_corners=False)

        s64 = c64
        s32 = torch.cat([F.avg_pool2d(c64, 2), d32_proj], dim=1)
        s16 = torch.cat([F.avg_pool2d(c64, 4), d16_proj], dim=1)

        oh = F.one_hot(style_id.clamp(0, self.N_STYLE - 1), self.N_STYLE).float() * keep_s.view(B, 1)
        pr = proto.float() * keep_s.view(B, 1)
        temb = sinusoidal_time_emb(tau, 64)
        flags = torch.stack([ds.float(), dc.float()], dim=-1)
        cond = self.cond_mlp(torch.cat([temb, oh, pr, flags], dim=-1))

        h0 = self.enc0(self.in_conv(torch.cat([z_tau, c64], dim=1)), s64, cond)
        h1 = self.enc1b(self.enc1a(self.down1(h0), s32, cond), s32, cond)
        h2 = self.enc2(self.down2(h1), s16, cond)

        hm = self.mid_blk(self.mid_attn(h2), s16, cond)

        u1 = self.dec1_fuse(torch.cat([self.up2(hm), h1], dim=1))
        u1 = self.dec1b(self.dec1a(u1, s32, cond), s32, cond)

        u0 = self.dec0_fuse(torch.cat([self.up1(u1), h0], dim=1))
        u0 = self.dec0b(self.dec0a(u0, s64, cond), s64, cond)

        return self.out_conv(F.silu(self.out_norm(u0)))


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.998):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train_ccf(
    cache_dict: dict,
    out_dir: Path | None = None,
    *,
    base_dim: int = 128,
    updates: int = 8000,
    micro: int = 64,
    accum: int = 2,
    lr: float = 3e-4,
    lam_struct: float = 0.12,
    seed: int = 20261118,
    device: str = "cuda",
) -> Path:
    out_dir = Path(out_dir or (OUT / "ccf_main"))
    ckpt_final = out_dir / "ccf_final.pt"
    if ckpt_final.exists():
        print(f"[V18 CCF] Reusing trained {ckpt_final}", flush=True)
        return ckpt_final

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)

    tr = cache_dict["train"]
    bank = cache_dict["style_bank"]
    w_luma = cache_dict["luma_proj"]["w"].to(device).float()
    b_luma = cache_dict["luma_proj"]["b"].to(device).float()

    # Index buckets per style for exact 1:1:1:1:1 balanced sampling every microbatch
    buckets = [(tr["sid"] == s).nonzero(as_tuple=True)[0] for s in range(5)]
    mean_protos = bank["mean_protos"].float()    # (5, 48)
    sample_protos = bank["sample_proto"].float() # (N, 48)

    model = CCFUNet(base_dim=base_dim).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    n_params = model.n_params()
    print(f"[V18 CCF] CCFUNet base_dim={base_dim} params={n_params/1e6:.2f}M updates={updates}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=updates, pct_start=0.06)
    gen = torch.Generator().manual_seed(seed)

    # Resume if interrupted
    start_step = 0
    latest_path = out_dir / "ccf_latest.pt"
    if latest_path.exists():
        ck = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_step = int(ck["step"])
        print(f"[V18 CCF] Resumed from step {start_step}", flush=True)

    curves = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model.train()

    for step in range(start_step + 1, updates + 1):
        opt.zero_grad(set_to_none=True)
        loss_fm_acc = 0.0
        loss_st_acc = 0.0

        for _ in range(accum):
            # Exact 5-style balanced microbatch
            cls_seq = torch.randperm(micro, generator=gen) % 5
            idxs = []
            for s in cls_seq.tolist():
                bk = buckets[s]
                pick = bk[int(torch.randint(len(bk), (1,), generator=gen).item())]
                idxs.append(int(pick))
            idx_t = torch.tensor(idxs, dtype=torch.long)

            z_real = tr["z"][idx_t].to(device).float()
            c_64 = augment_structure(tr["c_64"][idx_t].to(device).float(), gen=gen)
            dino_16 = tr["dino_16"][idx_t].to(device).float()
            sid = tr["sid"][idx_t].to(device).long()

            # Sub-mode vs class-mean vs unconditional style conditioning
            u_mode = torch.rand(micro, generator=gen)
            proto = torch.where(
                (u_mode < 0.55).unsqueeze(1),
                sample_protos[idx_t],
                mean_protos[sid.cpu()],
            ).to(device)
            drop_style = (u_mode >= 0.85).to(device)
            drop_content = (torch.rand(micro, generator=gen) < 0.10).to(device)

            # Logit-normal / uniform hybrid timestep sampling emphasizing mid-to-early flow
            tau = torch.rand(micro, generator=gen).to(device)
            eps = torch.randn(z_real.shape, generator=gen).to(device)
            ta = tau.view(-1, 1, 1, 1)
            z_tau = (1.0 - ta) * eps + ta * z_real
            v_star = z_real - eps

            with autocast(device):
                v_pred = model(
                    z_tau, c_64, dino_16, sid, proto, tau,
                    drop_style=drop_style, drop_content=drop_content,
                ).float()

            loss_fm = F.mse_loss(v_pred, v_star)

            # Endpoint structural consistency on predicted z_1 = z_tau + (1 - tau) * v_pred
            valid_st = (tau >= 0.25) & (~drop_content)
            if valid_st.any() and lam_struct > 0:
                z1_hat = z_tau[valid_st] + (1.0 - ta[valid_st]) * v_pred[valid_st]
                lcn_hat = project_latent_lcn(z1_hat, w_luma, b_luma)
                # Compare against un-blurred clean LCN_fine target
                c_clean = tr["c_64"][idx_t].to(device).float()[valid_st, 0:1]
                loss_st = F.l1_loss(lcn_hat, c_clean)
            else:
                loss_st = torch.zeros((), device=device)

            loss = loss_fm + lam_struct * loss_st
            (loss / accum).backward()
            loss_fm_acc += float(loss_fm.item()) / accum
            loss_st_acc += float(loss_st.item()) / accum

        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient at CCF step {step}")
        opt.step()
        sched.step()
        # Warm EMA faster in early 500 steps, then 0.998
        ema_decay = min(0.998, (1.0 + step) / (10.0 + step))
        update_ema(ema, model, decay=ema_decay)

        if step == 1 or step % 200 == 0 or step == updates:
            rec = {
                "step": step,
                "loss_fm": round(loss_fm_acc, 5),
                "loss_struct": round(loss_st_acc, 5),
                "grad_norm": round(grad_norm, 4),
                "lr": round(sched.get_last_lr()[0], 7),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "sec": round(time.time() - t0, 1),
            }
            curves.append(rec)
            print(
                f"[V18 CCF] {step:4d}/{updates} fm={rec['loss_fm']:.4f} "
                f"st={rec['loss_struct']:.4f} peak={rec['peak_gb']}GB {rec['sec']:.0f}s",
                flush=True,
            )

        if step % 2000 == 0 or step == updates:
            payload = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "base_dim": base_dim,
                "style_bank": {
                    "mean_protos": bank["mean_protos"],
                    "vivid_protos": bank["vivid_protos"],
                    "submode_centers": bank["submode_centers"],
                },
            }
            tmp = latest_path.with_suffix(".tmp")
            torch.save(payload, tmp)
            tmp.replace(latest_path)
            save_json(out_dir / "train_curves.json", curves)

    torch.save(
        {
            "ema": ema.cpu().state_dict(),
            "model": model.cpu().state_dict(),
            "base_dim": base_dim,
            "updates": updates,
            "n_params": n_params,
            "style_bank": {
                "mean_protos": bank["mean_protos"],
                "vivid_protos": bank["vivid_protos"],
                "submode_centers": bank["submode_centers"],
            },
        },
        ckpt_final,
    )
    del model, ema, opt
    torch.cuda.empty_cache()
    return ckpt_final


@torch.no_grad()
def sample_ccf_batch(
    model: CCFUNet,
    c_64: torch.Tensor,
    dino_16: torch.Tensor,
    tid: torch.Tensor,
    proto: torch.Tensor,
    eps: torch.Tensor,
    *,
    steps: int = 24,
    w_style: float = 2.5,
    w_content: float = 1.0,
    tau0: float = 0.0,
    z_src: torch.Tensor | None = None,
) -> torch.Tensor:
    """Euler integration from tau0 (default 0.0 = pure Gaussian noise!) with Style/Dual-CFG."""
    dev = c_64.device
    if tau0 <= 0.0 or z_src is None:
        z = eps.to(dev).float()
    else:
        z = tau0 * z_src.to(dev).float() + (1.0 - tau0) * eps.to(dev).float()

    dt = (1.0 - tau0) / steps
    B = z.shape[0]
    for k in range(steps):
        t_val = tau0 + k * dt
        tau = torch.full((B,), t_val, device=dev, dtype=torch.float32)
        with autocast(dev):
            v_cs = model(z, c_64, dino_16, tid, proto, tau, drop_style=False, drop_content=False).float()
            if w_style != 1.0 or w_content != 1.0:
                v_cu = model(z, c_64, dino_16, tid, proto, tau, drop_style=True, drop_content=False).float()
                if w_content != 1.0:
                    v_uu = model(z, c_64, dino_16, tid, proto, tau, drop_style=True, drop_content=True).float()
                    v = v_uu + w_content * (v_cu - v_uu) + w_style * (v_cs - v_cu)
                else:
                    v = v_cu + w_style * (v_cs - v_cu)
            else:
                v = v_cs
        z = z + dt * v
    return z


@torch.no_grad()
def generate_ccf_arm_750(
    arm_name: str,
    ckpt_path: Path,
    test_struct: dict,
    *,
    w_style: float = 2.5,
    w_content: float = 1.0,
    proto_mode: str = "mean",  # "mean" | "vivid"
    wrong_label: bool = False,
    steps: int = 24,
    tau0: float = 0.0,
    device: str = "cuda",
) -> Path:
    """Generate all 750 tasks with CCFUNet (EMA), S10-decode, save raw payload, and release VRAM."""
    raw_meta = OUT / "raw" / arm_name / "meta.json"
    if raw_meta.exists():
        print(f"[V18 CCF Gen] Reusing raw/{arm_name}", flush=True)
        return raw_meta.parent

    from src.utils import load_s10_decoder, decode_s10

    tasks, z_src_750, _ = tasks_and_sources()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CCFUNet(base_dim=int(ck["base_dim"])).to(device).eval()
    model.load_state_dict(ck["ema"])
    bank = ck["style_bank"]
    protos_table = bank["vivid_protos"] if proto_mode == "vivid" else bank["mean_protos"]
    protos_table = protos_table.to(device).float()

    dec = load_s10_decoder(device)
    g = torch.Generator().manual_seed(EVAL_SEED)
    eps_all = torch.randn(z_src_750.shape, generator=g)

    writer = RawWriterV18(arm_name)
    t0 = time.time()
    for st in range(0, 750, 32):
        sl = slice(st, min(st + 32, 750))
        grp = tasks[sl]
        tid = torch.tensor([t["target_style_id"] for t in grp], device=device, dtype=torch.long)
        used_tid = (tid + 2) % 5 if wrong_label else tid
        proto = protos_table[used_tid]
        c64 = test_struct["c_64"][sl].to(device).float()
        d16 = test_struct["dino_16"][sl].to(device).float()
        z_out = sample_ccf_batch(
            model, c64, d16, used_tid, proto, eps_all[sl],
            steps=steps, w_style=w_style, w_content=w_content,
            tau0=tau0, z_src=z_src_750[sl],
        )
        rgb = decode_s10(dec, z_out, batch=8)
        writer.append([t["task_id"] for t in grp], rgb, z_out)

    out_path = writer.finish(
        w_style=w_style,
        w_content=w_content,
        proto_mode=proto_mode,
        wrong_label=wrong_label,
        steps=steps,
        tau0=tau0,
        wall_sec=round(time.time() - t0, 1),
    )
    print(f"[V18 CCF Gen] {arm_name} generated in {time.time()-t0:.1f}s", flush=True)
    del model, dec
    torch.cuda.empty_cache()
    return out_path
