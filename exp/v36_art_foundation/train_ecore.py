"""Train V36 Open-Ref 1-NFE Reference E-Core strictly on Seen-Train (131 movements, Zero Core-5, Zero DINO).

Combines three complementary self-supervised / cross-painting objectives:
  - Task A (Proxy -> Real Painting): destylized proxy -> real Seen-Train target (same-artist / cropped ref),
    supervised by Latent Smooth-L1 + Haar DWT high-freq + Differentiable S10 VGG-16 Gram/SWD/Oklab/Content.
  - Task B (Cross-Painting Reference Swap): real painting A + two distinct Seen-Train style refs (R1, R2),
    supervised via Differentiable S10 VGG-16 Contrastive Swap Margin + Gram/SWD + Oklab + Grayscale Content.
  - Task C (Self-Identity Stabilization): self-reference reconstruction across low-to-mid lambda in [0.10, 0.60].
"""
from __future__ import annotations

import argparse
import copy
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from exp.v35_cleave_one.prepare_data import _encode_rgb_scaled, _load_sd_vae_encoder
from exp.v36_art_foundation import CACHE_DIR, CKPT_DIR, LOG_DIR, MEAS_DIR, ROOT, SEED_V36
from exp.v36_art_foundation.ecore_losses import ECorePerceptualCritic, latent_proxy_reconstruction_loss
from exp.v36_art_foundation.ecore_model import ReferenceECoreNet
from exp.v36_art_foundation.proxy import destylize_proxy_v36
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, utc


def _atomic_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class _ECoreSeenTrainPool:
    """Fast reader over Seen-Train memmaps (zero Core-5 overlap)."""

    def __init__(self, *, train_subset: int = 0, seed: int = SEED_V36 + 50):
        self.meta = torch.load(CACHE_DIR / "seen_train_index.pt", map_location="cpu", weights_only=False)
        self.priors = torch.load(CACHE_DIR / "v36_seen_train_priors.pt", map_location="cpu", weights_only=False)
        N = int(self.meta["n_seen"])
        self.N = N
        self.z_gt = np.array(np.lib.format.open_memmap(ROOT / self.meta["z_gt_rel"], mode="r", dtype=np.float16, shape=(N, 4, 64, 64)))
        self.q32 = np.array(np.lib.format.open_memmap(ROOT / self.meta["q32_rel"], mode="r", dtype=np.float16, shape=(N, 32, 16, 16)))
        self.stat = np.array(np.lib.format.open_memmap(ROOT / self.meta["stat_rel"], mode="r", dtype=np.float16, shape=(N, 1561)))
        self.stat_crop = np.array(np.lib.format.open_memmap(ROOT / self.meta["stat_crop_rel"], mode="r", dtype=np.float16, shape=(N, 1561)))
        self.artist_nbrs = self.meta["artist_nbrs"].numpy()

        valid_idx = self.meta["valid_idx"].numpy()
        probs = self.meta["sample_probs"].numpy()
        if train_subset > 0 and train_subset < len(valid_idx):
            rng_s = np.random.default_rng(seed)
            valid_idx = np.sort(rng_s.choice(valid_idx, size=train_subset, replace=False))
            mask = np.zeros(N, dtype=bool)
            mask[valid_idx] = True
            bad = ~mask[np.maximum(0, self.artist_nbrs)]
            self.artist_nbrs = np.where((self.artist_nbrs >= 0) & (~bad), self.artist_nbrs, -1)
        probs_v = probs[valid_idx]
        self.valid_idx = valid_idx
        self.probs = probs_v / probs_v.sum()
        self.has_nbr = self.artist_nbrs[:, 0] >= 0

    def sample_indices(self, B: int, rng: np.random.Generator, *, balanced_frac: float = 0.35) -> np.ndarray:
        n_bal = int(round(B * balanced_frac))
        n_unif = B - n_bal
        iu = rng.choice(self.valid_idx, size=n_unif, replace=False) if n_unif > 0 else np.empty(0, dtype=np.int64)
        ib = rng.choice(self.valid_idx, size=n_bal, replace=True, p=self.probs) if n_bal > 0 else np.empty(0, dtype=np.int64)
        return np.concatenate([iu, ib])

    def sample_task_a(self, B: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        idx_c = self.sample_indices(B, rng, balanced_frac=0.25)
        idx_r = idx_c.copy()
        for k, ci in enumerate(idx_c):
            if self.has_nbr[ci] and rng.random() < 0.70:
                row = self.artist_nbrs[ci]
                vn = row[row >= 0]
                if len(vn) > 0:
                    idx_r[k] = int(vn[rng.integers(0, len(vn))])
        return idx_c, idx_r

    def sample_task_b_triplets(self, B: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_c = self.sample_indices(B, rng, balanced_frac=0.30)
        idx_a = self.sample_indices(B, rng, balanced_frac=0.50)
        idx_b = self.sample_indices(B, rng, balanced_frac=0.50)
        # Ensure non-degenerate distinct A/B references
        same_ab = idx_a == idx_b
        if same_ab.any():
            idx_b[same_ab] = self.sample_indices(int(same_ab.sum()), rng, balanced_frac=0.50)
        return idx_c, idx_a, idx_b


@torch.no_grad()
def _eval_seen_dev_ecore(
    ema_ec: ReferenceECoreNet,
    priors_gpu: dict[str, torch.Tensor],
    s10_dec,
    critic: ECorePerceptualCritic,
    dev_pack: dict[str, torch.Tensor],
    *,
    device: str = "cuda",
) -> dict[str, float]:
    ema_ec.eval()
    zc = dev_pack["zc"].to(device)
    q32 = dev_pack["q32"].to(device)
    zra = dev_pack["zra"].to(device)
    zrb = dev_pack["zrb"].to(device)
    sta = dev_pack["sta"].to(device)
    stb = dev_pack["stb"].to(device)
    lam = torch.ones(zc.shape[0], device=device)
    eps = dev_pack["eps"].to(device)

    za_out = ema_ec(zc, zra, q32, sta, lam, priors_gpu, noise=eps)
    zb_out = ema_ec(zc, zrb, q32, stb, lam, priors_gpu, noise=eps)

    def _chunks(z_in, fn):
        res = []
        for s in range(0, z_in.shape[0], 8):
            rgb = decode_s10(s10_dec, z_in[s:s + 8], batch=min(8, z_in.shape[0] - s), output_cpu=False)
            res.append(fn(rgb))
        return torch.cat(res, dim=0)

    vxa = _chunks(za_out, critic.style_signature)
    vxb = _chunks(zb_out, critic.style_signature)
    vra = _chunks(zra, critic.style_signature)
    vrb = _chunks(zrb, critic.style_signature)
    ca = _chunks(za_out, critic.vgg.gray_content_vec)
    cb = _chunks(zb_out, critic.vgg.gray_content_vec)
    cgt = _chunks(zc, critic.vgg.gray_content_vec)

    daa = (vxa * vra).sum(-1)
    dab = (vxa * vrb).sum(-1)
    dbb = (vxb * vrb).sum(-1)
    dba = (vxb * vra).sum(-1)
    follow = 0.5 * ((daa > dab).float().mean().item() + (dbb > dba).float().mean().item())
    margin = 0.5 * ((daa - dab).mean().item() + (dbb - dba).mean().item())
    c_l1 = 0.5 * (F.l1_loss(ca, cgt).item() + F.l1_loss(cb, cgt).item())
    out_std = float(za_out.float().flatten(1).std(1).mean().item())
    return {
        "dev_vgg_follow_2way": round(follow, 4),
        "dev_vgg_style_margin": round(margin, 5),
        "dev_gray_content_l1": round(c_l1, 5),
        "dev_draft_latent_std": round(out_std, 4),
    }


def train_reference_ecore(
    *,
    steps: int = 6000,
    batch_proxy: int = 6,
    batch_swap: int = 4,
    batch_id: int = 4,
    lr: float = 3.0e-4,
    tag: str = "v36_ecore",
    train_subset: int = 0,
    save_every: int = 1000,
    diag_every: int = 500,
    resume: bool = True,
    device: str = "cuda",
) -> Path:
    sanitize_cuda_env()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    seed_all(SEED_V36 + 2026)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pool = _ECoreSeenTrainPool(train_subset=train_subset)
    priors_gpu = {
        k: v.to(device).float() if torch.is_tensor(v) else v
        for k, v in pool.priors.items()
    }

    s10_dec = load_s10_decoder(device)
    vae_enc = _load_sd_vae_encoder(device)
    critic = ECorePerceptualCritic(device=device)

    model = ReferenceECoreNet(pool.priors["stat_mean"], pool.priors["stat_std"]).to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    # Fixed Seen-Dev pack (32 works) for clean monitoring
    ready = read_json(CACHE_DIR / "v36_caches_ready.json")
    dev_i = ready["dev_info"]
    nd = int(dev_i["n"])
    mm_dz = np.lib.format.open_memmap(ROOT / dev_i["paths"]["z_gt"], mode="r", dtype=np.float16, shape=(nd, 4, 64, 64))
    mm_dq = np.lib.format.open_memmap(ROOT / dev_i["paths"]["q32"], mode="r", dtype=np.float16, shape=(nd, 32, 16, 16))
    mm_ds = np.lib.format.open_memmap(ROOT / dev_i["paths"]["stat"], mode="r", dtype=np.float16, shape=(nd, 1561))
    ic = np.arange(0, min(32, nd))
    dev_pack = {
        "zc": torch.from_numpy(np.array(mm_dz[ic])).float(),
        "q32": torch.from_numpy(np.array(mm_dq[ic])).float(),
        "zra": torch.from_numpy(np.array(mm_dz[(ic + 123) % nd])).float(),
        "zrb": torch.from_numpy(np.array(mm_dz[(ic + 317) % nd])).float(),
        "sta": torch.from_numpy(np.array(mm_ds[(ic + 123) % nd])).float(),
        "stb": torch.from_numpy(np.array(mm_ds[(ic + 317) % nd])).float(),
        "eps": torch.randn((len(ic), 4, 64, 64), generator=torch.Generator().manual_seed(SEED_V36 + 99)),
    }

    latest_p = CKPT_DIR / f"{tag}_latest.pt"
    final_p = CKPT_DIR / f"{tag}_final.pt"
    if final_p.exists() and resume:
        print(f"[{tag}] Final E-Core checkpoint already exists: {final_p}", flush=True)
        return final_p

    start_step = 0
    history: list[dict] = []
    rng = np.random.default_rng(SEED_V36 + 888)
    if resume and latest_p.exists():
        ck = torch.load(latest_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["ecore"], strict=True)
        ema.load_state_dict(ck["ecore_ema"], strict=True)
        opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        history = list(ck.get("history", []))
        if "np_rng_state" in ck:
            rng.bit_generator.state = ck["np_rng_state"]
        print(f"[{tag}] Resumed E-Core from step {start_step}/{steps}", flush=True)

    print(f"[{tag}] Start Reference E-Core | params={n_params/1e6:.2f}M | N_train={len(pool.valid_idx)} | steps={start_step}->{steps}", flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    warmup = min(250, max(30, steps // 15))
    gen_proxy = torch.Generator(device=device).manual_seed(SEED_V36 + 444)

    for step in range(start_step + 1, steps + 1):
        if step <= warmup:
            lr_m = step / float(warmup)
        else:
            frac = float(step - warmup) / float(max(1, steps - warmup))
            lr_m = 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * frac))
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_m

        model.train()
        opt.zero_grad(set_to_none=True)

        # ---- Sample indices & single batched no_grad S10 decode for Task A + Task B ----
        ia_c, ia_r = pool.sample_task_a(batch_proxy, rng)
        ib_c, ib_ra, ib_rb = pool.sample_task_b_triplets(batch_swap, rng)

        z_gt_a = torch.from_numpy(pool.z_gt[ia_c]).to(device).float()
        q32_a = torch.from_numpy(pool.q32[ia_c]).to(device).float()
        z_ref_a = torch.from_numpy(pool.z_gt[ia_r]).to(device).float()
        st_ref_a = torch.from_numpy(pool.stat_crop[ia_r]).to(device).float()
        lam_a = torch.empty(batch_proxy, device=device).uniform_(0.35, 1.05).clamp_max(1.0)

        z_src_b = torch.from_numpy(pool.z_gt[ib_c]).to(device).float()
        q32_b = torch.from_numpy(pool.q32[ib_c]).to(device).float()
        z_ra = torch.from_numpy(pool.z_gt[ib_ra]).to(device).float()
        z_rb = torch.from_numpy(pool.z_gt[ib_rb]).to(device).float()
        st_ra = torch.from_numpy(pool.stat_crop[ib_ra]).to(device).float()
        st_rb = torch.from_numpy(pool.stat_crop[ib_rb]).to(device).float()
        lam_b = torch.empty(batch_swap, device=device).uniform_(0.40, 1.05).clamp_max(1.0)

        with torch.no_grad():
            z_all = torch.cat([z_gt_a, z_ref_a, z_src_b, z_ra, z_rb], dim=0)
            x_all = decode_s10(s10_dec, z_all, batch=z_all.shape[0], output_cpu=False)
            n0 = batch_proxy
            n1 = n0 + batch_proxy
            n2 = n1 + batch_swap
            n3 = n2 + batch_swap
            x_gt_a, x_ref_a = x_all[:n0], x_all[n0:n1]
            x_src_b, x_ra, x_rb = x_all[n1:n2], x_all[n2:n3], x_all[n3:]
            px_a = destylize_proxy_v36(x_gt_a, gen=gen_proxy)
            with bf16_autocast(device):
                z_px_a = _encode_rgb_scaled(vae_enc, px_a).float()

        with bf16_autocast(device):
            z_pred_a = model(z_px_a, z_ref_a, q32_a, st_ref_a, lam_a, priors_gpu, noise=torch.randn_like(z_px_a))
        l_lat_a = latent_proxy_reconstruction_loss(z_pred_a, z_gt_a, lam_a)
        l_px_a, px_logs = critic.forward_proxy_pixel(s10_dec, z_pred_a, x_gt_a, x_ref_a, use_ckpt=False)
        loss_a = 0.70 * l_lat_a + 0.35 * l_px_a
        loss_a.backward()

        eps_b = torch.randn_like(z_src_b)
        with bf16_autocast(device):
            z_out_a = model(z_src_b, z_ra, q32_b, st_ra, lam_b, priors_gpu, noise=eps_b)
            z_out_b = model(z_src_b, z_rb, q32_b, st_rb, lam_b, priors_gpu, noise=eps_b)
        l_swap, sw_logs = critic.forward_swap_pair(s10_dec, z_out_a, z_out_b, x_src_b, x_ra, x_rb, lam_b, use_ckpt=False)
        (0.90 * l_swap).backward()

        # ---- Task C: Self-Identity Anchor (every step, lightweight latent-only) ----
        ii_c = pool.sample_indices(batch_id, rng, balanced_frac=0.20)
        z_id = torch.from_numpy(pool.z_gt[ii_c]).to(device).float()
        q32_id = torch.from_numpy(pool.q32[ii_c]).to(device).float()
        st_id = torch.from_numpy(pool.stat[ii_c]).to(device).float()
        lam_id = torch.empty(batch_id, device=device).uniform_(0.10, 0.70)
        with bf16_autocast(device):
            z_out_id = model(z_id, z_id, q32_id, st_id, lam_id, priors_gpu, noise=torch.randn_like(z_id))
        l_id = 0.30 * latent_proxy_reconstruction_loss(z_out_id, z_id, lam_id, is_identity=torch.ones(batch_id, dtype=torch.bool, device=device))
        l_id.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), [p.detach() for p in model.parameters()], 0.005)

        if step % 100 == 0 or step == start_step + 1:
            dt = time.perf_counter() - t0
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f"[{tag} step {step}/{steps}] lat_a={l_lat_a.item():.4f} px_gram={px_logs['l_px_gram']:.4f} "
                f"swap={sw_logs['l_swap']:.4f} follow={sw_logs['swap_follow']:.3f} id={l_id.item():.4f} "
                f"| {step/max(1e-3, dt):.2f} upd/s | vram={vram_gb:.2f}GB",
                flush=True,
            )

        if step % diag_every == 0 or step == steps:
            drow = _eval_seen_dev_ecore(ema, priors_gpu, s10_dec, critic, dev_pack, device=device)
            drow["step"] = step
            history.append(drow)
            save_json(LOG_DIR / f"{tag}_diag.json", history)
            print(f"[{tag} Seen-Dev @{step}] {drow}", flush=True)

        if step % save_every == 0 or step == steps:
            _atomic_save(
                {
                    "step": step,
                    "params": n_params,
                    "ecore": model.state_dict(),
                    "ecore_ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "np_rng_state": rng.bit_generator.state,
                    "history": history,
                },
                latest_p,
            )
            if step % (save_every * 2) == 0 or step == steps:
                _atomic_save(
                    {"step": step, "params": n_params, "ecore_ema": ema.state_dict(), "history": history},
                    CKPT_DIR / f"{tag}_step{step:05d}.pt",
                )

    _atomic_save(
        {"step": steps, "params": n_params, "ecore_ema": ema.state_dict(), "history": history},
        final_p,
    )
    del s10_dec, vae_enc, critic, model, ema, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--tag", default="v36_ecore")
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    train_reference_ecore(
        steps=a.steps,
        tag=a.tag,
        train_subset=a.train_subset,
        resume=not a.no_resume,
        device=a.device,
    )


if __name__ == "__main__":
    main()
