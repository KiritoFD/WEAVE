"""Infra probe: real throughput + peak memory for joint Encoder+Decoder training.

The plan forbids reusing the decoder-only 24 img/s figure: with the Encoder
unfrozen we must backprop through it at 512px, which is a completely different
memory profile.

Each (unfreeze mode, batch, grad-ckpt) runs in its OWN subprocess.

Modes:
  frozen  : encoder requires_grad=False (reference)
  partial : last two down_blocks + mid_block + conv_out + quant_conv trainable
  full    : whole encoder trainable

Loss = L1 + 0.5*LPIPS@512 + beta*KL, i.e. the actual experiment-A objective.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/vae_suite")

import torch

OUT = Path("/root/Workspace/WEAVE_VAE/runs/vae_style_vae")
OUT.mkdir(parents=True, exist_ok=True)
A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
S10_CKPT = "/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"


def set_trainable(vae, mode):
    for p in vae.encoder.parameters():
        p.requires_grad = False
    if mode == "frozen":
        return ["encoder frozen"]
    names = []
    if mode == "partial":
        mods = [("down_blocks.2", vae.encoder.down_blocks[2]),
                ("down_blocks.3", vae.encoder.down_blocks[3]),
                ("mid_block", vae.encoder.mid_block),
                ("conv_norm_out", vae.encoder.conv_norm_out),
                ("conv_out", vae.encoder.conv_out),
                ("quant_conv", vae.quant_conv)]
    else:  # full
        mods = [("encoder", vae.encoder), ("quant_conv", vae.quant_conv)]
    for n, m in mods:
        for p in m.parameters():
            p.requires_grad = True
        names.append(n)
    return names


def run_one(mode, bs, grad_ckpt, iters, warmup):
    from diffusers import AutoencoderKL
    from tools.vae_suite.decoder_prune import build_decoder, StudentDecoder
    import lpips

    torch.set_num_threads(4)
    dev = "cuda"
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev)
    vae.train()
    ck = torch.load(S10_CKPT, map_location="cpu", weights_only=False)
    dec = build_decoder(ck["meta"]["widths_forward"])
    dec.load_state_dict(ck["state_dict"])
    student = StudentDecoder(vae.post_quant_conv, dec).to(dev)
    for p in student.post_quant_conv.parameters():
        p.requires_grad = False          # shared with vae, keep frozen (4->4)
    student.train()

    trained = set_trainable(vae, mode)
    if grad_ckpt:
        vae.encoder.gradient_checkpointing = True

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False

    groups = [{"params": [p for p in student.decoder.parameters() if p.requires_grad],
               "lr": 2e-5}]
    enc_ps = [p for p in vae.encoder.parameters() if p.requires_grad]
    if enc_ps:
        groups.append({"params": enc_ps, "lr": 1e-5})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    x = torch.randn(bs, 3, 512, 512, device=dev) * 0.3

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            post = vae.encode(x)
            z = post.latent_dist.sample() if mode != "frozen" else post.latent_dist.mean
            y = student(z)
            l1 = (y.float() - x).abs().mean()
            l_lp = lp(y.float(), x).mean()
            kl = 0.5 * (-post.latent_dist.logvar - 1 + post.latent_dist.logvar.exp()
                        + post.latent_dist.mean ** 2).mean()
            loss = l1 + 0.5 * l_lp + 1e-6 * kl
        loss.backward()
        opt.step()
        return float(loss)

    try:
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import time
        t0 = time.time()
        for _ in range(iters):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        n_tr = sum(p.numel() for p in vae.encoder.parameters() if p.requires_grad)
        res = {"mode": mode, "bs": bs, "grad_ckpt": grad_ckpt, "ok": True,
               "trained": trained,
               "enc_trainable_M": round(n_tr / 1e6, 3),
               "allocated_GB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 3),
               "sec_per_update": round(dt, 3),
               "img_per_sec": round(bs / dt, 2)}
    except torch.cuda.OutOfMemoryError:
        res = {"mode": mode, "bs": bs, "grad_ckpt": grad_ckpt, "ok": False, "oom": True}
    print("RESULT " + json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default="")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    if args.child:
        m, bs, gc = args.child.split(":")
        run_one(m, int(bs), gc == "1", args.iters, args.warmup)
        return

    grid = []
    for mode in ["frozen", "partial", "full"]:
        for bs in [1, 2, 4, 6, 8]:
            grid.append((mode, bs, False))
    # gradient checkpointing is the escape hatch if encoder backward OOMs
    for bs in [4, 8, 12]:
        grid.append(("full", bs, True))

    results = []
    for mode, bs, gc in grid:
        cmd = [sys.executable, __file__, "--child", f"{mode}:{bs}:{1 if gc else 0}",
               "--iters", str(args.iters), "--warmup", str(args.warmup)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            r = {"mode": mode, "bs": bs, "grad_ckpt": gc, "ok": False,
                 "error": (p.stderr or "")[-300:]}
            print(f"  {mode:<8} bs={bs:<3} gc={int(gc)} ERROR", flush=True)
        else:
            r = json.loads(line[0][7:])
            if r.get("ok"):
                print(f"  {r['mode']:<8} bs={r['bs']:<3} gc={int(r['grad_ckpt'])} "
                      f"alloc={r['allocated_GB']:>6.2f}G res={r['reserved_GB']:>6.2f}G "
                      f"{r['sec_per_update']:>6.2f}s/upd {r['img_per_sec']:>6.2f} img/s "
                      f"(enc_train={r['enc_trainable_M']:.1f}M)", flush=True)
            else:
                print(f"  {r['mode']:<8} bs={r['bs']:<3} gc={int(r['grad_ckpt'])} OOM",
                      flush=True)
        results.append(r)

    (OUT / "joint_infra_bench.json").write_text(json.dumps(results, indent=2))
    print(f"\n[Saved] {OUT/'joint_infra_bench.json'}")


if __name__ == "__main__":
    main()
