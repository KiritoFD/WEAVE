"""Memory-only (and full-field) style registration for Other5 on the WEAVE D5 checkpoint.

This implements the experiment: freeze the WEAVE trunk, LL/LH/HL heads, target-HF
route, VAE, and the existing D5 memory rows (0-4); then train ONLY a new memory
entry per held-out Other5 style (rows 5-9, ~R^{256x384} ~= 98K params each).

Usage:
  # 1) Encode N=30 registration images/style (disjoint from the 30 eval images)
  python scripts/register_other5.py prepare --n 30

  # 2a) Memory-only registration (rows 5-9 trainable, trunk + rows 0-4 frozen)
  python scripts/register_other5.py train --mode memory --steps 2000

  # 2b) Full-field fine-tuning (entire model trainable)
  python scripts/register_other5.py train --mode full --steps 3000

  # smoke test (validates plumbing with random latents, 3 steps)
  python scripts/register_other5.py train --mode memory --smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_schema import ExperimentConfig, ModelConfig, BridgeConfig  # noqa: E402
from model import build_model_from_config  # noqa: E402
from flow import FlowMatchingObjective  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG_JSON = ROOT / "config.json"
D5_CKPT = ROOT / "archive" / "runs_submission" / "canonical_oriented_epoch4" / "epoch_0004.pt"
WIKIART = Path(r"F:\wikiart\wikiart")
OTHER5_TEST = Path(r"G:\GitHub\Latent_Style\Dataset\other5_512\test")
REG_LATENT_DIR = Path(r"G:\GitHub\Latent_Style\Dataset\other5_reg_latents_ema")

# Other5 styles (from other5_selection.json) and their target memory ids (5..9)
OTHER5_STYLES = [
    "Abstract_Expressionism",
    "Art_Nouveau_Modern",
    "Cubism",
    "Expressionism",
    "Symbolism",
]
NUM_BASE = 5          # D5 styles -> memory rows 0..4
NUM_OTHER5 = 5
NUM_STYLES = NUM_BASE + NUM_OTHER5   # 10


def load_config(num_styles: int = NUM_STYLES) -> ExperimentConfig:
    raw = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    model_cfg = ModelConfig.from_mapping(raw["model"])
    bridge_cfg = BridgeConfig.from_mapping(raw.get("bridge", {}))
    model_cfg.num_styles = num_styles
    # The D5 submission checkpoint enables the target-HF subband route; the root
    # config.json does not. Enable it so the architecture matches the checkpoint
    # exactly (otherwise 57 target_latent_hf_subband_* keys are dropped).
    model_cfg.target_latent_hf_subband_fusion_enabled = True
    exp = ExperimentConfig(model=model_cfg, bridge=bridge_cfg)
    return exp


def build_expanded_model(exp: ExperimentConfig) -> torch.nn.Module:
    """Build a num_styles=10 WEAVE model, load D5 trunk + copy memory rows 0-4,
    initialize rows 5-9 with the same 0.02*randn scale used by StyleConditioner."""
    model = build_model_from_config(exp.model, bridge_cfg=exp.bridge)
    if not D5_CKPT.exists():
        raise FileNotFoundError(f"D5 checkpoint not found: {D5_CKPT}")
    state = torch.load(D5_CKPT, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"]
    mem = sd["style_conditioner.style_memory"]  # (5, 256, 384)
    new_mem = torch.randn(NUM_STYLES, mem.shape[1], mem.shape[2], dtype=mem.dtype) * 0.02
    new_mem[:NUM_BASE] = mem
    sd["style_conditioner.style_memory"] = new_mem
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("[load] unexpected keys:", unexpected[:20])
    return model


def freeze_for_mode(model: torch.nn.Module, mode: str):
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "memory":
        sm = model.style_conditioner.style_memory
        sm.requires_grad_(True)

        def _zero_base_rows(grad):
            grad[:NUM_BASE] = 0.0
            return grad

        sm.register_hook(_zero_base_rows)
        trainable = [sm]
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(mode)
    n_train = sum(p.numel() for p in trainable)
    print(f"[freeze] mode={mode} trainable_params={n_train}")
    return trainable


# --------------------------------------------------------------------------- #
# Registration latent preparation
# --------------------------------------------------------------------------- #
def prepare_latents(n: int = 30) -> None:
    from tools.legacy_utils.inference import load_vae, encode_image

    REG_LATENT_DIR.mkdir(parents=True, exist_ok=True)
    # exclude the 30 eval-test images per style
    excluded: dict[str, set[str]] = {}
    for style in OTHER5_STYLES:
        d = OTHER5_TEST / style
        excluded[style] = {p.stem for p in d.glob("*.jpg")} | {p.stem for p in d.glob("*.png")}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(device, model_id="ema")
    vae.to(device)
    rng = random.Random(1234)

    for i, style in enumerate(OTHER5_STYLES):
        sid = NUM_BASE + i
        src_dir = WIKIART / style
        all_imgs = [p for p in src_dir.glob("*.jpg") if p.stem not in excluded[style]]
        rng.shuffle(all_imgs)
        chosen = all_imgs[:n]
        if len(chosen) < n:
            raise RuntimeError(f"Only {len(chosen)} disjoint registration images for {style}")
        latents = []
        for p in chosen:
            from PIL import Image
            img = Image.open(p).convert("RGB").resize((512, 512))
            import torchvision.transforms as T
            t = T.ToTensor()(img).unsqueeze(0).to(device)
            with torch.no_grad():
                lat = encode_image(vae, t, device)
            latents.append(lat.cpu())
        stack = torch.cat(latents, dim=0)
        out = REG_LATENT_DIR / f"reg_{sid:02d}_{style}.pt"
        torch.save(
            {"schema": 1, "subdir": style, "style_id": sid, "count": stack.shape[0],
             "files": [p.name for p in chosen], "latents": stack},
            out,
        )
        print(f"[prepare] {style} id={sid} -> {out} ({stack.shape})")


def load_reg_latents() -> list[torch.Tensor]:
    latents: list[torch.Tensor] = []
    for i in range(NUM_OTHER5):
        sid = NUM_BASE + i
        style = OTHER5_STYLES[i]
        p = REG_LATENT_DIR / f"reg_{sid:02d}_{style}.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing registration latent {p}; run `prepare` first.")
        payload = torch.load(p, map_location="cpu", weights_only=False)
        latents.append(payload["latents"].float())
    return latents


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(mode: str, steps: int, lr: float, batch: int, smoke: bool,
          n_reg: int = 30) -> None:
    """Memory-only (or full) registration.

    `n_reg` subsamples the pre-encoded 30-image registration pool per style to the
    first `n_reg` latents (deterministic nested subsets: 5 c 10 c 30). Used to plot
    the registration image-count curve (1/5/10/30). n=1 is degenerate (content==target
    pairs) and collapses toward identity, which bounds the minimum useful count.
    """
    exp = load_config(NUM_STYLES)
    model = build_expanded_model(exp)
    trainable = freeze_for_mode(model, mode)
    objective = FlowMatchingObjective(exp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    if smoke:
        # synthetic latents to validate the plumbing
        latents = [torch.randn(8, 4, 64, 64) for _ in range(NUM_OTHER5)]
        steps = 3
        batch = 2
    else:
        latents = load_reg_latents()
        if n_reg is not None and n_reg < latents[0].shape[0]:
            latents = [t[:n_reg].contiguous() for t in latents]
        for t in latents:
            print(f"[data] reg latent shape={tuple(t.shape)}")

    rng = random.Random(0)
    for step in range(steps):
        si = rng.randrange(NUM_OTHER5)
        sid = NUM_BASE + si
        pool = latents[si]
        n = pool.shape[0]
        ci = torch.randperm(n)[:batch]
        ti = torch.randperm(n)[:batch]
        while ti.equal(ci) and n > 1:
            ti = torch.randperm(n)[:batch]
        content = pool[ci].to(device)
        target = pool[ti].to(device)
        b = content.shape[0]
        tgt_ids = torch.full((b,), sid, dtype=torch.long, device=device)
        out = objective.compute(model, content=content, target_style=target, target_style_id=tgt_ids)
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            print(f"[step {step:05d}] loss={float(loss.detach()):.4f} t_mean={float(out['t_mean']):.3f}")

    out_dir = ROOT / "archive" / "exp" / "other5_reg" / f"{mode}_n{n_reg}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "epoch_reg.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "mode": mode, "num_styles": NUM_STYLES,
         "other5_styles": OTHER5_STYLES, "n_reg": n_reg},
        ckpt,
    )
    print(f"[save] {ckpt}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--n", type=int, default=30)
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--mode", choices=["memory", "full"], default="memory")
    p_tr.add_argument("--steps", type=int, default=2000)
    p_tr.add_argument("--lr", type=float, default=2e-4)
    p_tr.add_argument("--batch", type=int, default=4)
    p_tr.add_argument("--n", type=int, default=30, dest="n_reg",
                      help="subsampling size of the pre-encoded 30-image pool (image-count curve)")
    p_tr.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.action == "prepare":
        prepare_latents(args.n)
    elif args.action == "train":
        train(args.mode, args.steps, args.lr, args.batch, args.smoke, n_reg=args.n_reg)


if __name__ == "__main__":
    main()
