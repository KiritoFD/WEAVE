"""V36 Foundation Training Engine (Stage P1 131-Movement Pretraining + Stage P2 Transfer Strengthening).

Strictly Zero-DINO, zero SD-VAE/ConvNeXt/S10/VGG loaded during training: reads pure SSD memmaps.
Supports:
  - Pre-flight micro-batch throughput & VRAM profiler (4/8/16) with WDDM paging safety (<= 9.5GB)
  - 200-step sanity/smoke verification
  - Exact two-stage T0/T1/T2/T3 schedule + 80/20 uniform/sqrt sampling
  - Atomic checkpointing every 2,500 updates + Seen-Dev diagnostics every 5,000 updates
  - Optional --train-subset 20000 for 20K data-scale control ablation
"""
from __future__ import annotations

import argparse
import gc
import math
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp.v29_biglittle.phase1_train import _blur_or_drop_y16
from exp.v36_art_foundation import CACHE_DIR, CKPT_DIR, LOG_DIR, ROOT, SEED_V36
from exp.v36_art_foundation.models import FoundationPCoreNet, StyleMLPEncoder
from src.utils.runtime import bf16_autocast, read_json, sanitize_cuda_env, save_json, seed_all, utc


STAGE_RATIOS: dict[str, tuple[float, float, float, float]] = {
    # (T0: self+crop, T1: same-artist+crop, T2: proxy->real+crop, T3: proxy->real+same-artist)
    "p1": (0.30, 0.40, 0.15, 0.15),
    "p2": (0.10, 0.30, 0.30, 0.30),
}


def pseudo_huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    diff = pred.float() - target.float()
    d2 = delta * delta
    return (torch.sqrt(diff.square() + d2) - delta).mean()


@torch.no_grad()
def _ema_update(ema: nn.Module, src: nn.Module, decay: float = 0.999) -> None:
    ema_p = [p.data for p in ema.parameters()]
    src_p = [p.data for p in src.parameters()]
    if ema_p:
        torch._foreach_lerp_(ema_p, src_p, 1.0 - decay)


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class _MemmapDataset:
    def __init__(self, *, train_subset: int = 0, seed: int = SEED_V36):
        self.meta = torch.load(CACHE_DIR / "seen_train_index.pt", map_location="cpu", weights_only=False)
        self.priors = torch.load(CACHE_DIR / "v36_seen_train_priors.pt", map_location="cpu", weights_only=False)
        N = int(self.meta["n_seen"])
        self.N = N
        self.z_gt = np.lib.format.open_memmap(ROOT / self.meta["z_gt_rel"], mode="r", dtype=np.float16, shape=(N, 4, 64, 64))
        self.q32 = np.lib.format.open_memmap(ROOT / self.meta["q32_rel"], mode="r", dtype=np.float16, shape=(N, 32, 16, 16))
        self.q32_p0 = np.lib.format.open_memmap(ROOT / self.meta["q32_p0_rel"], mode="r", dtype=np.float16, shape=(N, 32, 16, 16))
        self.q32_p1 = np.lib.format.open_memmap(ROOT / self.meta["q32_p1_rel"], mode="r", dtype=np.float16, shape=(N, 32, 16, 16))
        # Preload lightweight arrays (~593 MB total for 76k rows) in RAM to avoid small random memmap page faults
        self.y16 = np.array(np.lib.format.open_memmap(ROOT / self.meta["y16_rel"], mode="r", dtype=np.float16, shape=(N, 1, 16, 16)))
        self.y16_p0 = np.array(np.lib.format.open_memmap(ROOT / self.meta["y16_p0_rel"], mode="r", dtype=np.float16, shape=(N, 1, 16, 16)))
        self.y16_p1 = np.array(np.lib.format.open_memmap(ROOT / self.meta["y16_p1_rel"], mode="r", dtype=np.float16, shape=(N, 1, 16, 16)))
        self.stat = np.array(np.lib.format.open_memmap(ROOT / self.meta["stat_rel"], mode="r", dtype=np.float16, shape=(N, 1561)))
        self.stat_crop = np.array(np.lib.format.open_memmap(ROOT / self.meta["stat_crop_rel"], mode="r", dtype=np.float16, shape=(N, 1561)))

        self.artist_nbrs = self.meta["artist_nbrs"].numpy()
        valid_idx = self.meta["valid_idx"].numpy()
        probs = self.meta["sample_probs"].numpy()

        if train_subset > 0 and train_subset < len(valid_idx):
            rng_s = np.random.default_rng(seed + 77)
            valid_idx = np.sort(rng_s.choice(valid_idx, size=train_subset, replace=False))
            mask = np.zeros(N, dtype=bool)
            mask[valid_idx] = True
            # Mask out neighbors outside subset
            bad = ~mask[np.maximum(0, self.artist_nbrs)]
            self.artist_nbrs = np.where((self.artist_nbrs >= 0) & (~bad), self.artist_nbrs, -1)
            probs = probs[valid_idx]
            probs = probs / probs.sum()
        else:
            probs = probs[valid_idx]
            probs = probs / probs.sum()

        self.valid_idx = valid_idx
        self.probs = probs
        self.has_nbr = (self.artist_nbrs[:, 0] >= 0)

    def sample_batch_fp16(
        self,
        B: int,
        ratios: tuple[float, float, float, float],
        rng: np.random.Generator,
        epoch_cursor: dict,
        *,
        pin: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (z_gt, y16, q32, stat_ref) as float16 (optionally pinned) tensors for zero-stall async H2D."""
        n_unif = int(round(B * 0.80))
        n_w = B - n_unif
        cur = epoch_cursor["pos"]
        perm = epoch_cursor["perm"]
        if cur + n_unif > len(perm):
            perm = rng.permutation(self.valid_idx)
            epoch_cursor["perm"] = perm
            cur = 0
        idx_u = perm[cur:cur + n_unif]
        epoch_cursor["pos"] = cur + n_unif
        idx_w = rng.choice(self.valid_idx, size=n_w, replace=True, p=self.probs) if n_w > 0 else np.empty(0, dtype=np.int64)
        idx_c = np.concatenate([idx_u, idx_w])

        r_type = rng.random(B)
        c0, c1, c2 = ratios[0], ratios[0] + ratios[1], ratios[0] + ratios[1] + ratios[2]
        t_kind = np.where(r_type < c0, 0, np.where(r_type < c1, 1, np.where(r_type < c2, 2, 3)))
        no_n = ~self.has_nbr[idx_c]
        t_kind = np.where((t_kind == 1) & no_n, 0, t_kind)
        t_kind = np.where((t_kind == 3) & no_n, 2, t_kind)

        use_proxy = (t_kind >= 2)
        use_nbr = (t_kind == 1) | (t_kind == 3)

        idx_r = idx_c.copy()
        for k in np.flatnonzero(use_nbr):
            nbr_row = self.artist_nbrs[idx_c[k]]
            valid_n = nbr_row[nbr_row >= 0]
            if len(valid_n) > 0:
                idx_r[k] = int(valid_n[rng.integers(0, len(valid_n))])

        z_np = np.array(self.z_gt[idx_c], dtype=np.float16)
        if use_proxy.any():
            p0_m = use_proxy & (rng.random(B) >= 0.5)
            p1_m = use_proxy & (~p0_m)
            y_np = self.y16[idx_c].copy()
            q_np = np.array(self.q32[idx_c], dtype=np.float16)
            if p0_m.any():
                idx_p0 = idx_c[p0_m]
                y_np[p0_m] = self.y16_p0[idx_p0]
                q_np[p0_m] = np.array(self.q32_p0[idx_p0], dtype=np.float16)
            if p1_m.any():
                idx_p1 = idx_c[p1_m]
                y_np[p1_m] = self.y16_p1[idx_p1]
                q_np[p1_m] = np.array(self.q32_p1[idx_p1], dtype=np.float16)
        else:
            y_np = self.y16[idx_c].copy()
            q_np = np.array(self.q32[idx_c], dtype=np.float16)

        use_cr = (rng.random(B) < 0.75) | (~use_nbr)
        st_np = np.where(use_cr[:, None], self.stat_crop[idx_r], self.stat[idx_r]).astype(np.float16, copy=False)

        t_z = torch.from_numpy(z_np)
        t_y = torch.from_numpy(y_np)
        t_q = torch.from_numpy(q_np)
        t_st = torch.from_numpy(st_np)
        if pin and torch.cuda.is_available():
            t_z, t_y, t_q, t_st = t_z.pin_memory(), t_y.pin_memory(), t_q.pin_memory(), t_st.pin_memory()
        return t_z, t_y, t_q, t_st

    def sample_batch(
        self,
        B: int,
        ratios: tuple[float, float, float, float],
        rng: np.random.Generator,
        epoch_cursor: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample float32 (z_gt, y16, q32, stat_ref) synchronously."""
        t_z, t_y, t_q, t_st = self.sample_batch_fp16(B, ratios, rng, epoch_cursor, pin=False)
        return t_z.float(), t_y.float(), t_q.float(), t_st.float()


class _AsyncBatchPrefetcher:
    """Overlap CPU memmap sampling & pinned fp16 tensor preparation with GPU compute."""

    def __init__(
        self,
        ds: _MemmapDataset,
        micro: int,
        ratios: tuple[float, float, float, float],
        rng: np.random.Generator,
        epoch_cursor: dict,
        *,
        qsize: int = 6,
    ) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=qsize)
        self._stop = threading.Event()
        self._err: BaseException | None = None

        def _worker() -> None:
            try:
                while not self._stop.is_set():
                    item = ds.sample_batch_fp16(micro, ratios, rng, epoch_cursor, pin=True)
                    while not self._stop.is_set():
                        try:
                            self._q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
            except BaseException as ex:
                self._err = ex
                self._stop.set()

        self._th = threading.Thread(target=_worker, daemon=True)
        self._th.start()

    def next_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        while True:
            if self._err is not None:
                raise RuntimeError(f"Prefetcher worker failed: {self._err}") from self._err
            try:
                z1, y16, q32, st_r = self._q.get(timeout=0.5)
                return (
                    z1.to(device, non_blocking=True).float(),
                    y16.to(device, non_blocking=True).float(),
                    q32.to(device, non_blocking=True).float(),
                    st_r.to(device, non_blocking=True).float(),
                )
            except queue.Empty:
                continue

    def close(self) -> None:
        self._stop.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._th.join(timeout=2.0)


def _lr_at_step(step: int, total_steps: int, base_lr: float, warmup: int = 500, min_lr: float = 2e-5) -> float:
    if step <= warmup:
        return base_lr * (float(step) / float(max(1, warmup)))
    decay_start = int(0.85 * total_steps)
    if step <= decay_start:
        return base_lr
    frac = (step - decay_start) / float(max(1, total_steps - decay_start))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, frac)))


@torch.no_grad()
def run_seen_dev_diagnostics(
    pcore_ema: nn.Module,
    style_enc_ema: nn.Module,
    *,
    step: int,
    stage: str,
    device: str = "cuda",
) -> dict[str, float]:
    """Evaluate strictly on Seen-Dev (ZERO Core-5 OOD touched!):
    1) Flow velocity loss on content-recon & pure-noise art generation
    2) Cross-work reference responsiveness ||v(ref_a) - v(ref_b)|| / cosine separation
    """
    ready = read_json(CACHE_DIR / "v36_caches_ready.json")
    dev_info = ready["dev_info"]
    N_d = int(dev_info["n"])
    mm_z = np.lib.format.open_memmap(ROOT / dev_info["paths"]["z_gt"], mode="r", dtype=np.float16, shape=(N_d, 4, 64, 64))
    mm_y = np.lib.format.open_memmap(ROOT / dev_info["paths"]["y16"], mode="r", dtype=np.float16, shape=(N_d, 1, 16, 16))
    mm_q = np.lib.format.open_memmap(ROOT / dev_info["paths"]["q32"], mode="r", dtype=np.float16, shape=(N_d, 32, 16, 16))
    mm_st = np.lib.format.open_memmap(ROOT / dev_info["paths"]["stat"], mode="r", dtype=np.float16, shape=(N_d, 1561))

    idx = np.linspace(0, N_d - 1, min(48, N_d), dtype=np.int64)
    idx_b = np.roll(idx, 19)
    z1 = torch.from_numpy(np.array(mm_z[idx])).to(device).float()
    y16 = torch.from_numpy(np.array(mm_y[idx])).to(device).float()
    q32 = torch.from_numpy(np.array(mm_q[idx])).to(device).float()
    st_a = torch.from_numpy(np.array(mm_st[idx])).to(device).float()
    st_b = torch.from_numpy(np.array(mm_st[idx_b])).to(device).float()

    pcore_ema.eval()
    style_enc_ema.eval()
    rng_g = torch.Generator(device=device).manual_seed(SEED_V36 + 999)
    eps = torch.randn(z1.shape, device=device, generator=rng_g)

    with bf16_autocast(device):
        ea = style_enc_ema(st_a)
        eb = style_enc_ema(st_b)
        # 1. Conditional velocity error at t=0.35
        t35 = torch.full((len(idx),), 0.35, device=device)
        zt = 0.65 * eps + 0.35 * z1
        va = pcore_ema(zt, y16, q32, ea, t35)
        vb = pcore_ema(zt, y16, q32, eb, t35)
        v_null_s = pcore_ema(zt, y16, q32, torch.zeros_like(ea), t35, drop_style=torch.ones(len(idx), dtype=torch.bool, device=device))

        # 2. Pure-noise uncond/cond 6-step ODE check
        z_gen = eps.clone()
        dt = 1.0 / 6.0
        for i in range(6):
            ti = torch.full((len(idx),), i * dt, device=device)
            v_sc = pcore_ema(z_gen, y16, q32, ea, ti)
            v_0c = pcore_ema(z_gen, y16, q32, torch.zeros_like(ea), ti, drop_style=torch.ones(len(idx), dtype=torch.bool, device=device))
            z_gen = z_gen + dt * (v_0c + 2.5 * (v_sc - v_0c))

    res = {
        "step": step,
        "stage": stage,
        "dev_phuber_t35": round(float(pseudo_huber_loss(va, z1 - eps).item()), 5),
        "dev_style_delta_l2": round(float((va.float() - vb.float()).square().mean().sqrt().item()), 4),
        "dev_cfg_delta_l2": round(float((va.float() - v_null_s.float()).square().mean().sqrt().item()), 4),
        "dev_gen_recon_l1": round(float(F.l1_loss(z_gen.float(), z1).item()), 4),
        "dev_gen_latent_std": round(float(z_gen.float().std().item()), 4),
        "dev_gt_latent_std": round(float(z1.float().std().item()), 4),
    }
    print(f"[V36 Seen-Dev {stage} step {step}] {res}", flush=True)
    return res


def _train_accum_step(
    prefetcher: _AsyncBatchPrefetcher,
    pcore: nn.Module,
    style_enc: nn.Module,
    micro: int,
    accum: int,
    device: str,
) -> torch.Tensor:
    """Run one effective batch (accum x micro) without per-micro CPU-GPU syncs."""
    loss_acc_t = torch.zeros((), device=device, dtype=torch.float32)
    for _ in range(accum):
        z1, y16, q32, st_r = prefetcher.next_batch(device)
        y16_aug, _ = _blur_or_drop_y16(y16, drop_prob=0.40)
        q32_aug = q32 + 0.05 * torch.randn_like(q32)
        q32_aug = q32_aug - q32_aug.mean(dim=(2, 3), keepdim=True)

        dc = torch.rand(micro, device=device) < 0.30
        ds_m = torch.rand(micro, device=device) < 0.15
        lscale = torch.where(dc, torch.zeros(micro, device=device), 0.25 + 0.75 * torch.rand(micro, device=device))
        gamma = torch.where(
            torch.rand(micro, device=device) < 0.40,
            torch.ones(micro, device=device),
            torch.empty(micro, device=device).uniform_(0.15, 1.0),
        )
        t = torch.rand(micro, device=device).clamp(0.02, 0.98)
        eps = torch.randn_like(z1)
        zt = (1.0 - t.view(-1, 1, 1, 1)) * eps + t.view(-1, 1, 1, 1) * z1

        with bf16_autocast(device):
            e_s = style_enc(st_r)
            v_pred = pcore(
                zt, y16_aug, q32_aug, e_s, t,
                drop_style=ds_m, drop_content=dc, layout_scale=lscale, gamma=gamma,
            )
            l_fm = pseudo_huber_loss(v_pred, z1 - eps, delta=0.05)

        (l_fm / float(accum)).backward()
        loss_acc_t = loss_acc_t + l_fm.detach() / float(accum)
    return loss_acc_t


def profile_micro_batch(
    ds: _MemmapDataset,
    *,
    candidates: tuple[int, ...] = (32, 48),
    steps_per_cand: int = 12,
    eff_batch: int = 96,
    device: str = "cuda",
) -> dict:
    """Fast real-prefetch profiler selecting highest samples/s config under <= 9.6GB alloc (~10.8GB card) without WDDM spill."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    results = []
    for ckpt_mode in (False,):
        for micro in candidates:
            if eff_batch % micro != 0:
                continue
            accum = eff_batch // micro
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            prefetcher = None
            try:
                pcore = FoundationPCoreNet(
                    base_dim=208, style_dim=256, cond_dim=384,
                    use_gamma=True, normalize_style=True, per_layer_adagn=True,
                    use_checkpoint=ckpt_mode,
                ).to(device).train()
                senc = StyleMLPEncoder(1561, 512, 256, ds.priors["stat_mean"], ds.priors["stat_std"]).to(device).train()
                all_params = list(pcore.parameters()) + list(senc.parameters())
                opt = torch.optim.AdamW([
                    {"params": list(pcore.parameters()), "lr": 2e-4},
                    {"params": list(senc.parameters()), "lr": 3e-4},
                ], betas=(0.9, 0.999), weight_decay=0.01)

                rng = np.random.default_rng(SEED_V36 + 1)
                cur = {"pos": 0, "perm": rng.permutation(ds.valid_idx)}
                prefetcher = _AsyncBatchPrefetcher(ds, micro, STAGE_RATIOS["p1"], rng, cur, qsize=6)
                t0 = 0.0
                for st in range(1, steps_per_cand + 4):
                    if st == 4:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    _train_accum_step(prefetcher, pcore, senc, micro, accum, device)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    opt.step()
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                resv_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                sps = (steps_per_cand * eff_batch) / max(1e-4, dt)
                sec_per_upd = dt / steps_per_cand
                row = {
                    "use_checkpoint": ckpt_mode,
                    "micro": micro,
                    "accum": accum,
                    "eff_batch": eff_batch,
                    "peak_vram_gb": round(peak_gb, 3),
                    "resv_vram_gb": round(resv_gb, 3),
                    "sec_per_update": round(sec_per_upd, 4),
                    "samples_per_sec": round(sps, 2),
                    "est_steps_22h": int(22 * 3600 / sec_per_upd),
                    "est_steps_6h": int(6 * 3600 / sec_per_upd),
                    "ok_9_5gb": peak_gb <= 9.6,
                }
                results.append(row)
                print(f"[V36 Profile] ckpt={ckpt_mode} micro={micro}x{accum} -> {row['samples_per_sec']} samp/s ({row['sec_per_update']}s/upd), alloc={row['peak_vram_gb']}GB resv={row['resv_vram_gb']}GB", flush=True)
                del pcore, senc, opt, all_params
            except RuntimeError as ex:
                results.append({"use_checkpoint": ckpt_mode, "micro": micro, "eff_batch": eff_batch, "error": str(ex)[:120], "ok_9_5gb": False})
                print(f"[V36 Profile] ckpt={ckpt_mode} micro={micro} OOM/ERR: {ex}", flush=True)
                for _obj_name in ("pcore", "senc", "opt", "all_params"):
                    if _obj_name in locals():
                        del locals()[_obj_name]
                gc.collect()
                torch.cuda.empty_cache()
            finally:
                if prefetcher is not None:
                    prefetcher.close()

    valid = [r for r in results if r.get("ok_9_5gb")]
    best = max(valid, key=lambda r: r["samples_per_sec"]) if valid else results[0]
    out = {"utc": utc(), "candidates": results, "best": best}
    save_json(LOG_DIR / "profile_micro.json", out)
    return out


def train_foundation(
    *,
    stage: str = "p1",
    steps: int = 30000,
    micro: int = 48,
    eff_batch: int = 96,
    use_checkpoint: bool = False,
    use_compile: bool = False,
    resume_from: str = "",
    tag: str = "v36_main",
    train_subset: int = 0,
    save_every: int = 2500,
    diag_every: int = 5000,
    max_hours: float = 0.0,
    device: str = "cuda",
) -> Path:
    sanitize_cuda_env()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    seed_all(SEED_V36 + (1000 if stage == "p2" else 0))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    assert eff_batch % micro == 0
    accum = eff_batch // micro
    ds = _MemmapDataset(train_subset=train_subset)

    pcore = FoundationPCoreNet(
        base_dim=208, style_dim=256, cond_dim=384,
        use_gamma=True, normalize_style=True, per_layer_adagn=True,
        use_checkpoint=use_checkpoint,
    ).to(device).train()
    style_enc = StyleMLPEncoder(1561, 512, 256, ds.priors["stat_mean"], ds.priors["stat_std"]).to(device).train()

    lr_p = 2e-4 if stage == "p1" else 1e-4
    lr_s = 3e-4 if stage == "p1" else 1.5e-4
    opt = torch.optim.AdamW([
        {"params": list(pcore.parameters()), "lr": lr_p},
        {"params": list(style_enc.parameters()), "lr": lr_s},
    ], betas=(0.9, 0.999), weight_decay=0.01)

    pcore_ema = FoundationPCoreNet(
        base_dim=208, style_dim=256, cond_dim=384,
        use_gamma=True, normalize_style=True, per_layer_adagn=True,
        use_checkpoint=False,
    ).to(device).eval()
    style_enc_ema = StyleMLPEncoder(1561, 512, 256, ds.priors["stat_mean"], ds.priors["stat_std"]).to(device).eval()
    pcore_ema.load_state_dict(pcore.state_dict())
    style_enc_ema.load_state_dict(style_enc.state_dict())
    for p in list(pcore_ema.parameters()) + list(style_enc_ema.parameters()):
        p.requires_grad_(False)

    rng = np.random.default_rng(SEED_V36 + (777 if stage == "p2" else 333))
    epoch_cursor = {"pos": 0, "perm": rng.permutation(ds.valid_idx)}

    start_step = 0
    diag_history: list[dict] = []
    latest_p = CKPT_DIR / f"{tag}_{stage}_latest.pt"
    final_p = CKPT_DIR / f"{tag}_{stage}_final.pt"
    if final_p.exists() and not resume_from:
        print(f"[V36 {tag} {stage}] Final checkpoint already exists: {final_p}", flush=True)
        return final_p

    if latest_p.exists():
        ck = torch.load(latest_p, map_location="cpu", weights_only=False)
        pcore.load_state_dict(ck["pcore"], strict=True)
        style_enc.load_state_dict(ck["style_enc"], strict=True)
        pcore_ema.load_state_dict(ck["pcore_ema"], strict=True)
        style_enc_ema.load_state_dict(ck["style_enc_ema"], strict=True)
        opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        diag_history = list(ck.get("diag_history", []))
        if "np_rng_state" in ck:
            rng.bit_generator.state = ck["np_rng_state"]
        print(f"[V36 {tag} {stage}] Resumed atomically from step {start_step}/{steps}", flush=True)
    elif resume_from and Path(resume_from).exists():
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        pcore.load_state_dict(ck["pcore_ema"], strict=True)
        style_enc.load_state_dict(ck["style_enc_ema"], strict=True)
        pcore_ema.load_state_dict(ck["pcore_ema"], strict=True)
        style_enc_ema.load_state_dict(ck["style_enc_ema"], strict=True)
        print(f"[V36 {tag} {stage}] Initialized stage weights from {resume_from}", flush=True)

    pcore_fwd: nn.Module = torch.compile(pcore) if (use_compile and hasattr(torch, "compile")) else pcore
    ratios = STAGE_RATIOS[stage]
    n_params = sum(p.numel() for p in pcore.parameters()) + sum(p.numel() for p in style_enc.parameters())
    print(
        f"[V36 {tag} {stage}] Start | total_params={n_params/1e6:.2f}M | N_train={len(ds.valid_idx)} | "
        f"steps={start_step}->{steps} | micro={micro}x{accum} | ckpt={use_checkpoint} | compile={use_compile} | ratios={ratios}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    warmup = min(500, max(50, steps // 20))
    last_step = start_step
    all_params = list(pcore.parameters()) + list(style_enc.parameters())
    prefetcher = _AsyncBatchPrefetcher(ds, micro, ratios, rng, epoch_cursor, qsize=6)

    try:
        for step in range(start_step + 1, steps + 1):
            opt.param_groups[0]["lr"] = _lr_at_step(step, steps, lr_p, warmup=warmup, min_lr=2e-5)
            opt.param_groups[1]["lr"] = _lr_at_step(step, steps, lr_s, warmup=warmup, min_lr=2e-5)

            pcore.train()
            style_enc.train()
            opt.zero_grad(set_to_none=True)
            loss_acc_t = _train_accum_step(prefetcher, pcore_fwd, style_enc, micro, accum, device)

            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            _ema_update(pcore_ema, pcore, 0.999)
            _ema_update(style_enc_ema, style_enc, 0.999)
            last_step = step

            elapsed = time.perf_counter() - t_start
            if step == start_step + 1 or step == start_step + 10 or step == start_step + 50 or step % 250 == 0 or step == steps:
                loss_val = float(loss_acc_t.item())
                done_n = max(1, step - start_step)
                rate = (done_n * eff_batch) / max(1e-3, elapsed)
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                eta_h = ((steps - step) * elapsed / done_n) / 3600.0
                print(
                    f"[V36 {tag} {stage} {step}/{steps}] loss={loss_val:.4f} | {rate:.1f} samp/s | "
                    f"exposed={step * eff_batch:,} ({step * eff_batch / len(ds.valid_idx):.1f} ep) | "
                    f"peak={peak_gb:.2f}GB | ETA={eta_h:.2f}h",
                    flush=True,
                )

            if diag_every > 0 and (step % diag_every == 0 or step == steps):
                opt.zero_grad(set_to_none=True)
                diag = run_seen_dev_diagnostics(pcore_ema, style_enc_ema, step=step, stage=stage, device=device)
                diag_history.append(diag)
                save_json(LOG_DIR / f"{tag}_{stage}_diag.json", diag_history)

            if (save_every > 0 and step % save_every == 0) or step == steps or (max_hours > 0 and elapsed / 3600.0 >= max_hours):
                state = {
                    "tag": tag,
                    "stage": stage,
                    "step": step,
                    "steps_target": steps,
                    "eff_batch": eff_batch,
                    "utc": utc(),
                    "pcore": pcore.state_dict(),
                    "style_enc": style_enc.state_dict(),
                    "pcore_ema": pcore_ema.state_dict(),
                    "style_enc_ema": style_enc_ema.state_dict(),
                    "opt": opt.state_dict(),
                    "np_rng_state": rng.bit_generator.state,
                    "diag_history": diag_history,
                }
                _atomic_torch_save(state, latest_p)
                if step % save_every == 0:
                    _atomic_torch_save(state, CKPT_DIR / f"{tag}_{stage}_step{step:06d}.pt")
                if max_hours > 0 and elapsed / 3600.0 >= max_hours:
                    print(f"[V36 {tag} {stage}] Reached max_hours={max_hours:.2f}h cleanly at step {step}", flush=True)
                    break
    finally:
        prefetcher.close()

    final_state = {
        "tag": tag,
        "stage": stage,
        "step": last_step,
        "eff_batch": eff_batch,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3),
        "wall_hours": round((time.perf_counter() - t_start) / 3600.0, 3),
        "utc": utc(),
        "pcore_ema": pcore_ema.state_dict(),
        "style_enc_ema": style_enc_ema.state_dict(),
        "diag_history": diag_history,
    }
    _atomic_torch_save(final_state, final_p)
    print(f"[V36 {tag} {stage}] Saved final -> {final_p}", flush=True)
    return final_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--auto-steps", action="store_true", help="Read profile_micro.json to pick micro/ckpt & steps fitting --hours")
    ap.add_argument("--stage", default="p1", choices=["p1", "p2"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--hours", type=float, default=22.0)
    ap.add_argument("--micro", type=int, default=48)
    ap.add_argument("--eff-batch", type=int, default=96)
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume-from", default="")
    ap.add_argument("--tag", default="v36_main")
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--diag-every", type=int, default=5000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    sanitize_cuda_env()
    if a.profile_only:
        ds = _MemmapDataset(train_subset=a.train_subset)
        profile_micro_batch(ds, eff_batch=a.eff_batch, device=a.device)
        return

    micro = a.micro
    eff_batch = a.eff_batch
    use_ckpt = not a.no_checkpoint
    steps = a.steps
    if a.auto_steps and (LOG_DIR / "profile_micro.json").exists():
        prof = read_json(LOG_DIR / "profile_micro.json")["best"]
        micro = int(prof["micro"])
        eff_batch = int(prof.get("eff_batch", eff_batch))
        use_ckpt = bool(prof["use_checkpoint"])
        sec_u = float(prof["sec_per_update"])
        est = int(a.hours * 3600.0 / max(0.05, sec_u))
        # Clamp inside reasonable target envelope
        steps = max(2000, min(60000 if a.stage == "p1" else 20000, est))
        print(f"[V36 Auto] Using micro={micro}, eff_batch={eff_batch}, ckpt={use_ckpt}, steps={steps} for {a.hours}h budget", flush=True)

    train_foundation(
        stage=a.stage,
        steps=steps,
        micro=micro,
        eff_batch=eff_batch,
        use_checkpoint=use_ckpt,
        use_compile=a.compile,
        resume_from=a.resume_from,
        tag=a.tag,
        train_subset=a.train_subset,
        save_every=a.save_every,
        diag_every=a.diag_every,
        max_hours=a.hours * 1.05 if a.auto_steps else 0.0,
        device=a.device,
    )


if __name__ == "__main__":
    main()
