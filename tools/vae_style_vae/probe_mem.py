"""Where does stage-B memory actually go? One update, phase-by-phase peaks."""
import sys

sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
sys.path.insert(0, "/root/Workspace/WEAVE_VAE/tools/12h")

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL

from tools.vae_style_vae.data import MANIFEST, StyleImageDS, load_split
from tools.vae_style_vae.losses import StyleVAELoss, crop_consistency, edit_loss, supcon
from tools.vae_style_vae.model import StyleHead, StyleVAE, style_content
from tools.vae_suite.decoder_prune import build_decoder

A0_DIR = "/root/Workspace/WEAVE_VAE/runs_vae/A0/final_diffusers_vae"
S10_CKPT = "/root/Workspace/WEAVE_VAE/runs/vae_decoder_prune/s10_p5_lowlr/decoder.pt"
dev = "cuda"


def peak():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2)


def main():
    man, tr, va = load_split(MANIFEST)
    ds = StyleImageDS(tr, man)
    vae = AutoencoderKL.from_pretrained(A0_DIR, torch_dtype=torch.float32).to(dev)
    ck = torch.load(S10_CKPT, map_location="cpu", weights_only=False)
    dec = build_decoder(ck["meta"]["widths_forward"])
    dec.load_state_dict(ck["state_dict"])
    model = StyleVAE(vae, dec.to(dev)).to(dev)
    for p in model.post_quant_conv.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.quant_conv.parameters():
        p.requires_grad = True
    model.train()

    import lpips
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad = False
    crit = StyleVAELoss(lpips_fn=lp, lam_p=0.5, beta=1e-4, kl_reduction="elem")

    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ckpt = (len(sys.argv) > 2 and sys.argv[2] == "1")
    x = torch.stack([ds[i]["x"] for i in range(bs)]).to(dev)
    sid = torch.tensor([ds.recs[ds.indices[i]]["style_id"] for i in range(bs)],
                       dtype=torch.long, device=dev)
    head = StyleHead(int(sid.max()) + 2).to(dev)

    print(f"bs={bs} ckpt={ckpt}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    print(f"  after data          {peak():>6.2f} GB", flush=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        post = model.encode(x, use_ckpt=ckpt)
    print(f"  after encode fwd    {peak():>6.2f} GB", flush=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        z = post.sample()
        y = model.decode(z)
    print(f"  after decode fwd    {peak():>6.2f} GB", flush=True)

    tot, parts = crit(x, y, post)
    print(f"  after recon loss    {peak():>6.2f} GB  (l1={parts['l1']:.4f})",
          flush=True)

    svec, cvec, _, _ = style_content(z.float())
    l_ce = F.cross_entropy(head(svec), sid)
    l_sc = supcon(svec, sid)
    l_sty = l_ce + l_sc
    tot2 = tot + 0.1 * l_sty
    print(f"  after style loss    {peak():>6.2f} GB", flush=True)

    tot2.backward()
    print(f"  after backward      {peak():>6.2f} GB", flush=True)

    # crop branch on top
    torch.cuda.reset_peak_memory_stats()
    xc = F.interpolate(x, (448, 448), mode="bilinear", align_corners=False)
    crop = xc[:, :, 32:32 + 384, 32:32 + 384]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        sc_, _, _, _ = style_content(model.encode(crop, use_ckpt=ckpt).mean.float())
    print(f"  crop branch alone   {peak():>6.2f} GB", flush=True)


if __name__ == "__main__":
    main()
