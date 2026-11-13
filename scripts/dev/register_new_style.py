"""Freeze the D5 trunk and register one new style memory entry per Other5 style.

Usage:
    python register_new_style.py \
        --base_ckpt code/checkpoints/weave_epoch_0004.pt \
        --style_name Cubism \
        --train_img_dir F:/datasets/r5/train_reg64/Cubism \
        --output_ckpt F:/datasets/r5/registered/Cubism.pt \
        --num_steps 500 \
        --batch_size 4 \
        --lr 1e-3 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config_schema import ExperimentConfig, load_config, merge_config_dicts
from flow import FlowMatchingObjective
from model import build_model_from_config
from style_families import prune_state_dict_for_tokenizer_family
from tools.legacy_utils.inference import encode_image, load_vae
from tools.legacy_utils.training import strip_compile_prefix


def _pil_to_tensor(pil_img):
    arr = np.array(pil_img.convert("RGB").resize((256, 256))).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


class ImageLatentDataset(Dataset):
    """VAE-encode images on-the-fly and serve as latent tensors."""

    def __init__(self, image_dir: str, vae, device: str, seed: int = 0):
        self.device = device
        self.vae = vae
        self.paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if not self.paths:
            raise ValueError(f"No images in {image_dir}")
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        pil = Image.open(self.paths[idx]).convert("RGB")
        img_t = _pil_to_tensor(pil)
        with torch.no_grad():
            latent = encode_image(self.vae, img_t, device=self.device).squeeze(0)
        return latent


def load_base_checkpoint(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_config = ckpt.get("config", {}) or {}
    config = ExperimentConfig.from_mapping(raw_config)
    # apply raw values back to dataclass (some fields may be missing in schema)
    for section_name in ("model", "bridge", "training", "data", "checkpoint"):
        raw_section = raw_config.get(section_name, {})
        section_obj = getattr(config, section_name, None)
        if not isinstance(raw_section, dict) or section_obj is None:
            continue
        for key, value in raw_section.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    state_dict = strip_compile_prefix(ckpt["model_state_dict"])
    state_dict, _ = prune_state_dict_for_tokenizer_family(
        state_dict,
        tokenizer_family=str(getattr(config.model, "tokenizer_family", "legacy_factorized")),
        contract_family=str(getattr(config.model, "contract_family", "legacy")),
        style_injection_mode=str(getattr(config.model, "style_injection_mode", "none")),
        proximal_mode=str(getattr(config.model, "proximal_mode", "off")),
        style_delta_mode=str(getattr(config.model, "style_delta_mode", "none")),
        output_appearance_alignment_mode=str(getattr(config.model, "output_appearance_alignment_mode", "none")),
    )
    model = build_model_from_config(config.model, bridge_cfg=config.bridge, use_checkpointing=False).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model, config, ckpt


def expand_style_memory(model, new_num_styles: int):
    """Expand style_memory to new_num_styles, copying old entries and zero-init new."""
    conditioner = model.style_conditioner
    old_mem = conditioner.style_memory  # [old_styles, tokens, dino_dim]
    old_n, tokens, dim = old_mem.shape
    if new_num_styles <= old_n:
        return
    new_mem = torch.zeros(new_num_styles, tokens, dim, device=old_mem.device, dtype=old_mem.dtype)
    new_mem[:old_n] = old_mem.data.detach()
    # small random init for new entries
    torch.manual_seed(42)
    new_mem[old_n:].normal_(std=0.02)
    conditioner.style_memory = torch.nn.Parameter(new_mem)
    conditioner.num_styles = new_num_styles
    model.num_styles = new_num_styles


def freeze_all_except_new_memory(model, new_style_id: int):
    """Freeze everything; only the new style memory entry is trainable."""
    for p in model.parameters():
        p.requires_grad = False
    model.style_conditioner.style_memory.requires_grad = True
    # optionally only train the single entry
    # model.style_conditioner.style_memory.data[:new_style_id].requires_grad = False
    # (the whole tensor is still trainable; we mask gradients below)


def register_style(
    base_ckpt: str,
    train_img_dir: str,
    output_ckpt: str,
    new_style_id: int,
    num_steps: int,
    batch_size: int,
    lr: float,
    device: str,
    vae_model: str = "ema",
    hf_cache_dir: str = "",
):
    print(f"[Register] Loading base checkpoint: {base_ckpt}")
    model, config, ckpt = load_base_checkpoint(base_ckpt, device)
    old_num_styles = int(model.num_styles)
    new_num_styles = old_num_styles + 1
    expand_style_memory(model, new_num_styles)
    freeze_all_except_new_memory(model, new_style_id)

    print(f"  old styles: {old_num_styles}, new total: {new_num_styles}, trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("[Register] Loading VAE")
    vae = load_vae(device, model_id=vae_model, cache_dir=hf_cache_dir or None)
    vae.eval()

    print("[Register] Building dataset")
    ds = ImageLatentDataset(train_img_dir, vae, device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    loss_fn = FlowMatchingObjective(config)
    optimizer = torch.optim.Adam([model.style_conditioner.style_memory], lr=lr, betas=(0.9, 0.999))

    # Gradient mask: only update the new entry
    def _hook(grad):
        mask = torch.zeros_like(grad)
        mask[new_style_id] = 1.0
        return grad * mask

    model.style_conditioner.style_memory.register_hook(_hook)

    model.train()
    losses = []
    t0 = time.perf_counter()
    print(f"[Register] Training {num_steps} steps")
    step = 0
    epoch = 0
    while step < num_steps:
        epoch += 1
        for batch in loader:
            target = batch.to(device)
            # content: a different random sample from the same style pool
            perm = torch.randperm(target.shape[0])
            content = target[perm]
            target_style_id = torch.full((target.shape[0],), new_style_id, dtype=torch.long, device=device)
            metrics = loss_fn.compute(
                model,
                content=content,
                target_style=target,
                target_style_id=target_style_id,
            )
            loss = metrics["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            step += 1
            if step % 50 == 0:
                print(f"  step {step}/{num_steps} loss={np.mean(losses[-50:]):.6f}")
            if step >= num_steps:
                break
    wall = time.perf_counter() - t0
    print(f"[Register] Done in {wall:.1f}s, final loss={np.mean(losses[-50:]):.6f}")

    # Save checkpoint
    out_path = Path(output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_state_dict = model.state_dict()
    # patch config
    new_config = ckpt.get("config", {})
    new_config["model"]["num_styles"] = new_num_styles
    new_ckpt = {
        "epoch": ckpt.get("epoch", 4),
        "global_step": ckpt.get("global_step", 0),
        "model_state_dict": new_state_dict,
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "loss_state_dict": {},
        "config": new_config,
        "metrics": ckpt.get("metrics", {}),
    }
    torch.save(new_ckpt, out_path)
    print(f"[Register] Saved: {out_path}")
    return out_path, wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", default="code/checkpoints/weave_epoch_0004.pt")
    parser.add_argument("--style_name", required=True)
    parser.add_argument("--train_img_dir", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--new_style_id", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae_model", default="ema")
    parser.add_argument("--hf_cache_dir", default="")
    args = parser.parse_args()
    register_style(
        base_ckpt=args.base_ckpt,
        train_img_dir=args.train_img_dir,
        output_ckpt=args.output_ckpt,
        new_style_id=args.new_style_id,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        vae_model=args.vae_model,
        hf_cache_dir=args.hf_cache_dir,
    )


if __name__ == "__main__":
    main()
