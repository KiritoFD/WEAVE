"""V5 Branch B1: Latent Style Encoder & Oracle Re-test
1. Train a lightweight LatentStyleEncoder (E_s: z -> h_s, ~1.2M params) directly on 15.5k D5 real art latents
   using CrossEntropy + Supervised Contrastive Loss.
2. Verify its style classification and cluster separation on holdout validation.
3. Freeze E_s and re-run Latent Oracle using E_s feature distance: L_style = 1 - cos(E_s(z'), h_t*).
4. Evaluate with independent CSD (dT, dS, dM) and DINO-C to decide between M1 (Route A) and Route C.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))

import paths
import csd_model
from tools.legacy_root.wavelet import dwt2_haar, idwt2_haar
from tools.pseudo_gt.matching import load_dino_model, encode_dino

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
STYLE_IDS = {name: i for i, name in enumerate(STYLES)}


class ResidualBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(min(8, out_c), out_c)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(min(8, out_c), out_c)
        self.act2 = nn.SiLU()

        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_c), out_c),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + res)


class LatentStyleEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, embed_dim: int = 256, num_styles: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.layer1 = ResidualBlock(64, 128, stride=2)   # 32x32
        self.layer2 = ResidualBlock(128, 256, stride=2)  # 16x16
        self.layer3 = ResidualBlock(256, 256, stride=2)  # 8x8
        self.layer4 = ResidualBlock(256, 256, stride=1)  # 8x8

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embed_dim)
        self.head = nn.Linear(embed_dim, num_styles)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(z)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        flat = self.pool(h).flatten(1)
        emb_raw = self.proj(flat)
        emb_norm = F.normalize(emb_raw, dim=-1)
        logits = self.head(emb_raw)
        return emb_norm, logits


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    sim = emb @ emb.t() / temperature
    mask = labels[:, None] == labels[None, :]
    mask.fill_diagonal_(False)
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(logits)
    log_prob = logits - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
    pos = exp * mask
    denom = pos.sum(dim=1)
    valid = denom > 0
    if not bool(valid.any()):
        return emb.new_zeros(())
    return -(log_prob * mask).sum(dim=1)[valid].div(denom[valid]).mean()


def train_style_encoder(
    orig_dir: Path,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Tuple[LatentStyleEncoder, dict]:
    print("[b1] Loading all real art latents from orig_mean_fp16...", flush=True)
    all_latents = []
    all_labels = []

    for sid, style in enumerate(STYLES):
        p = orig_dir / f"{sid:02d}_{style}.pt"
        data = torch.load(p, map_location="cpu", weights_only=False)
        lats = data["latents"].float()  # (N, 4, 64, 64)
        all_latents.append(lats)
        all_labels.append(torch.full((lats.shape[0],), sid, dtype=torch.long))

    X = torch.cat(all_latents, dim=0)
    Y = torch.cat(all_labels, dim=0)
    N = X.shape[0]
    print(f"[b1] Total real art latents: {N} across 5 styles.", flush=True)

    # 80/20 train/val split
    torch.manual_seed(42)
    perm = torch.randperm(N)
    n_val = int(0.2 * N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    model = LatentStyleEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = X_train.shape[0]
    steps_per_epoch = int(np.ceil(n_train / batch_size))

    print(f"[b1] Training LatentStyleEncoder for {epochs} epochs (batch={batch_size}, {steps_per_epoch} steps/ep)...", flush=True)
    t_start = time.time()

    for ep in range(epochs):
        model.train()
        ep_perm = torch.randperm(n_train)
        total_loss, total_ce, total_sc = 0.0, 0.0, 0.0
        correct, total = 0, 0

        for s in range(0, n_train, batch_size):
            batch_idx = ep_perm[s:s + batch_size]
            xb = X_train[batch_idx].to(device)
            yb = Y_train[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            emb, logits = model(xb)

            loss_ce = F.cross_entropy(logits, yb)
            loss_sc = supcon_loss(emb, yb)
            loss = loss_ce + 0.3 * loss_sc

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)
            total_ce += loss_ce.item() * len(batch_idx)
            total_sc += loss_sc.item() * len(batch_idx)

            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += len(batch_idx)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for s in range(0, n_val, batch_size):
                xb = X_val[s:s + batch_size].to(device)
                yb = Y_val[s:s + batch_size].to(device)
                _, logits = model(xb)
                val_correct += (logits.argmax(dim=-1) == yb).sum().item()
                val_total += len(yb)
        val_acc = val_correct / val_total

        print(f"  Ep {ep+1:02d}/{epochs:02d} | Loss: {total_loss/total:.4f} (CE: {total_ce/total:.4f}, SC: {total_sc/total:.4f}) | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%", flush=True)

    print(f"[b1] Style Encoder trained in {time.time() - t_start:.2f}s | Final Val Acc: {val_acc*100:.2f}%", flush=True)

    # Compute per-style prototype centers h_t*
    model.eval()
    prototypes = {}
    with torch.inference_mode():
        for sid, style in enumerate(STYLES):
            mask = (Y == sid)
            lat_style = X[mask]
            embs = []
            for s in range(0, lat_style.shape[0], batch_size):
                xb = lat_style[s:s + batch_size].to(device)
                e, _ = model(xb)
                embs.append(e.cpu())
            mean_e = torch.cat(embs, dim=0).mean(dim=0)
            prototypes[style] = F.normalize(mean_e, dim=0)

    train_stats = {
        "final_train_acc": float(train_acc),
        "final_val_acc": float(val_acc),
        "train_time_sec": round(time.time() - t_start, 2),
    }
    return model, prototypes, train_stats


def spatial_self_similarity(z: torch.Tensor) -> torch.Tensor:
    p = F.adaptive_avg_pool2d(z, (16, 16))
    B, C, H, W = p.shape
    flat = p.permute(0, 2, 3, 1).reshape(B, H * W, C)
    norm = F.normalize(flat, dim=-1, eps=1e-6)
    return torch.bmm(norm, norm.transpose(1, 2))


def gradient_loss(z: torch.Tensor, z_src: torch.Tensor) -> torch.Tensor:
    dx = z[:, :, :, 1:] - z[:, :, :, :-1]
    dy = z[:, :, 1:, :] - z[:, :, :-1, :]
    dx_s = z_src[:, :, :, 1:] - z_src[:, :, :, :-1]
    dy_s = z_src[:, :, 1:, :] - z_src[:, :, :-1, :]
    cos_x = F.cosine_similarity(dx, dx_s, dim=1, eps=1e-6)
    cos_y = F.cosine_similarity(dy, dy_s, dim=1, eps=1e-6)
    return (1.0 - cos_x.mean()) + (1.0 - cos_y.mean())


def run_oracle_b1_arm(
    model: LatentStyleEncoder,
    prototypes: Dict[str, torch.Tensor],
    z_src: torch.Tensor,
    sim_src: torch.Tensor,
    tgt_style_ids: torch.Tensor,
    lam_struct: float,
    num_steps: int = 64,
    lr: float = 0.05,
    device: str = "cuda",
) -> torch.Tensor:
    """Optimize z' = z + delta using LatentStyleEncoder representation loss."""
    proto_mat = torch.stack([prototypes[s] for s in STYLES]).to(device)  # (5, 256)
    delta = torch.zeros_like(z_src, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)
        z_rec = z_src + delta

        # Style loss from LatentStyleEncoder
        emb, _ = model(z_rec)  # (B, 256)
        tgt_protos = proto_mat[tgt_style_ids]  # (B, 256)
        l_style = (1.0 - (emb * tgt_protos).sum(dim=-1)).mean()

        if lam_struct > 0.0:
            sim_z = spatial_self_similarity(z_rec)
            l_selfsim = F.mse_loss(sim_z, sim_src)
            l_grad = gradient_loss(z_rec, z_src)
            l_struct = l_selfsim + 0.2 * l_grad
            loss = l_style + lam_struct * l_struct
        else:
            loss = l_style

        loss.backward()
        optimizer.step()

    z_final = (z_src + delta).detach()
    del delta, optimizer
    torch.cuda.empty_cache()
    return z_final


def decode_latents_to_pils(vae, z: torch.Tensor, batch_size: int = 8) -> List[Image.Image]:
    B = z.shape[0]
    pils = []
    scaling = float(vae.config.scaling_factor)
    with torch.inference_mode():
        for s in range(0, B, batch_size):
            chunk = (z[s:s + batch_size] / scaling).half()
            img_g = vae.decode(chunk).sample.clamp(-1.0, 1.0)
            arr_g = ((img_g.permute(0, 2, 3, 1).cpu().float().numpy() + 1.0) * 127.5).astype(np.uint8)
            pils.extend([Image.fromarray(a) for a in arr_g])
    return pils


def evaluate_latents(
    z_final: torch.Tensor,
    tasks: List[dict],
    proto_mat: np.ndarray,
    csd: torch.nn.Module,
    dino: torch.nn.Module,
    vae: torch.nn.Module,
    csd_src: np.ndarray,
    dino_src: np.ndarray,
    device: str = "cuda",
) -> List[dict]:
    pils_gen = decode_latents_to_pils(vae, z_final, batch_size=8)
    with torch.inference_mode():
        csd_gen = csd_model.feats_pils(csd, pils_gen, batch=32, device=device)
        dino_gen_t, _ = encode_dino(dino, pils_gen, batch=32, device=device)
        dino_gen = dino_gen_t.float().cpu().numpy()

    results = []
    for i, t in enumerate(tasks):
        fg = csd_gen[i]
        fs = csd_src[i]
        tgt_id = t["tgt_style_id"]
        src_id = t["src_style_id"]

        cos_tgt_g = float(fg @ proto_mat[tgt_id])
        cos_tgt_s = float(fs @ proto_mat[tgt_id])
        cos_src_g = float(fg @ proto_mat[src_id])
        cos_src_s = float(fs @ proto_mat[src_id])

        dT = cos_tgt_g - cos_tgt_s
        dS = cos_src_g - cos_src_s
        dM = dT - dS
        dino_c = float(dino_gen[i] @ dino_src[i])

        results.append({
            "task_id": t["task_id"],
            "src_style": t["src_style"],
            "tgt_style": t["tgt_style"],
            "dT": dT,
            "dS": dS,
            "dM": dM,
            "dino_c": dino_c,
            "cos_tgt_g": cos_tgt_g,
            "cos_tgt_s": cos_tgt_s,
        })
    return results


def main():
    print("=== WEAVE V5 Branch B1: Latent Style Encoder & Oracle Re-test ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = REPO_ROOT / "runs" / "v5" / "b1_style_encoder"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train LatentStyleEncoder
    orig_dir = paths.asset_root() / "pseudo_gt_24h" / "orig_mean_fp16"
    style_encoder, prototypes, train_stats = train_style_encoder(
        orig_dir=orig_dir,
        epochs=12,
        batch_size=256,
        lr=1e-3,
        device=device,
    )

    # Save trained encoder checkpoint
    torch.save({
        "model": style_encoder.state_dict(),
        "prototypes": {k: v.cpu() for k, v in prototypes.items()},
        "train_stats": train_stats,
    }, out_dir / "latent_style_encoder.pt")
    print(f"[b1] Saved latent_style_encoder.pt", flush=True)

    # 2. Re-test Latent Oracle under B1
    tasks = json.loads((REPO_ROOT / "runs" / "v5" / "v5_oracle100_tasks.json").read_text(encoding="utf-8"))
    src_latents = torch.load(REPO_ROOT / "runs" / "v5" / "cache" / "eval_src_latents.pt", map_location="cpu")

    proto_p = paths.asset_root() / "pseudo_gt_24h" / "csd_proto.pt"
    proto_dict = torch.load(proto_p, map_location="cpu", weights_only=False)
    proto_mat = np.stack([proto_dict[s] for s in STYLES])

    print("[b1] Loading CSD, DINO, and VAE (fp16) for independent evaluation...", flush=True)
    csd, ck, msg = csd_model.load_csd(device)
    dino = load_dino_model(device=device)

    candidates = [
        paths.asset_root() / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
        REPO_ROOT / "eval_cache" / "hf" / "modelscope" / "stabilityai_sd-vae-ft-ema" / "models" / "stabilityai--sd-vae-ft-ema" / "snapshots",
    ]
    vae_dir = None
    for c in candidates:
        if c.exists():
            snaps = list(c.glob("*"))
            if snaps and (snaps[0] / "config.json").exists():
                vae_dir = snaps[0]
                break

    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float16, local_files_only=True).to(device).eval()

    z_src_list = [src_latents[t["src_stem"]] for t in tasks]
    z_src = torch.stack(z_src_list).to(device=device, dtype=torch.float32)
    sim_src = spatial_self_similarity(z_src)
    tgt_style_ids = torch.tensor([t["tgt_style_id"] for t in tasks], device=device, dtype=torch.long)

    pils_src = decode_latents_to_pils(vae, z_src, batch_size=8)
    with torch.inference_mode():
        csd_src = csd_model.feats_pils(csd, pils_src, batch=32, device=device)
        dino_src_t, _ = encode_dino(dino, pils_src, batch=32, device=device)
        dino_src = dino_src_t.float().cpu().numpy()

    # Matrix for B1
    matrix_configs = [
        {"name": "B1_unconstrained", "lam": 0.0},
        {"name": "B1_weak", "lam": 0.5},
        {"name": "B1_strong", "lam": 2.0},
    ]

    report = {
        "train_stats": train_stats,
        "oracle_b1_matrix": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"\n[b1] Running Oracle Re-test across {len(matrix_configs)} arms...", flush=True)

    for m in matrix_configs:
        name = m["name"]
        print(f"\n>>> Oracle-B1 Arm: {name} (lambda={m['lam']}) <<<", flush=True)
        t0 = time.time()
        z_opt = run_oracle_b1_arm(
            model=style_encoder,
            prototypes=prototypes,
            z_src=z_src,
            sim_src=sim_src,
            tgt_style_ids=tgt_style_ids,
            lam_struct=m["lam"],
            num_steps=64,
            lr=0.05,
            device=device,
        )
        opt_time = time.time() - t0

        vram_mib = torch.cuda.max_memory_allocated(0) / (1024**2) if torch.cuda.is_available() else 0.0
        print(f"[b1] {name} optimized in {opt_time:.2f}s | Peak VRAM: {vram_mib:.1f} MiB", flush=True)

        eval_rows = evaluate_latents(
            z_final=z_opt,
            tasks=tasks,
            proto_mat=proto_mat,
            csd=csd,
            dino=dino,
            vae=vae,
            csd_src=csd_src,
            dino_src=dino_src,
            device=device,
        )

        dTs = [r["dT"] for r in eval_rows]
        dSs = [r["dS"] for r in eval_rows]
        dMs = [r["dM"] for r in eval_rows]
        dino_cs = [r["dino_c"] for r in eval_rows]

        summary = {
            "name": name,
            "lambda_struct": m["lam"],
            "opt_time_sec": round(opt_time, 2),
            "peak_vram_mib": round(vram_mib, 1),
            "mean_dT": float(np.mean(dTs)),
            "std_dT": float(np.std(dTs)),
            "mean_dS": float(np.mean(dSs)),
            "mean_dM": float(np.mean(dMs)),
            "mean_dino_c": float(np.mean(dino_cs)),
            "frac_dT_positive": float(np.mean([x > 0 for x in dTs])),
            "frac_dM_positive": float(np.mean([x > 0 for x in dMs])),
        }
        report["oracle_b1_matrix"][name] = summary
        print(f"[{name} RESULT] dT: {summary['mean_dT']:+.4f} (frac>0: {summary['frac_dT_positive']*100:.1f}%) | dS: {summary['mean_dS']:+.4f} | dM: {summary['mean_dM']:+.4f} | DINO-C: {summary['mean_dino_c']:.4f}", flush=True)

    # Decision on B1 vs Route C
    # Success threshold: dT >= +0.03 and DINO-C >= 0.78
    best_arm = None
    best_dT = -999.0
    for name, s in report["oracle_b1_matrix"].items():
        if name == "B1_unconstrained":
            continue
        if s["mean_dino_c"] >= 0.78 and s["mean_dT"] > best_dT:
            best_dT = s["mean_dT"]
            best_arm = name

    b1_success = bool(best_arm is not None and report["oracle_b1_matrix"][best_arm]["mean_dT"] >= 0.03)
    next_decision = "M1_Training_with_StyleEncoder" if b1_success else "Route_C_Real_Art_Distribution_Flow_Prior"

    report["decision"] = {
        "best_arm": best_arm,
        "best_arm_dT": report["oracle_b1_matrix"][best_arm]["mean_dT"] if best_arm else None,
        "best_arm_dino_c": report["oracle_b1_matrix"][best_arm]["mean_dino_c"] if best_arm else None,
        "b1_success": b1_success,
        "next_decision": next_decision,
    }

    print("\n=======================================================", flush=True)
    print(f"=== B1 Decision: {next_decision} (B1 Success={b1_success}, Best Arm={best_arm}) ===", flush=True)
    print("=======================================================", flush=True)

    out_file = out_dir / "b1_report.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[b1] Complete! Saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
