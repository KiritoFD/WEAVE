"""Content-cluster bootstrap CIs for the three headline claims.

Cluster unit = DINOv2 content cluster (k=25) over the 1000 diagnostic query
latents; five-style outputs of the same content are never treated as
independent samples.

B1  S3  top-1 style retrieval (swd distances, fixed projections)
B2  S4  dino_s drop caused by patch_shuffle (identity minus shuffled)
B3  S13 per-image dino_s gain after constrained latent ascent (swd, offset +1)

Usage: python3 tools/audit/g_boot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import audit_core as C
import latent_losses as L
from b_grad import ref_pools, perturb
from d_optim import rms_proj
import s2_confound as S2


def cluster_ids(tokens: torch.Tensor, k: int = 25, iters: int = 30) -> torch.Tensor:
    torch.manual_seed(0)
    x = F.normalize(tokens.float(), dim=1)
    cent = x[torch.randperm(x.shape[0])[:k]].clone()
    a = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(iters):
        d = torch.cdist(x, cent)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                cent[j] = F.normalize(x[m].mean(0), dim=0)
    return a


def boot_ci(vals: torch.Tensor, groups: torch.Tensor, *, n_boot: int = 300,
            seed: int = 0) -> dict:
    g = groups.reshape(-1)
    uq = g.unique()
    idx = {int(u): (g == u).nonzero().squeeze(1) for u in uq}
    gen = torch.Generator().manual_seed(seed)
    reps = []
    for _ in range(n_boot):
        pick = uq[torch.randint(len(uq), (len(uq),), generator=gen)]
        sel = torch.cat([idx[int(p)] for p in pick])
        reps.append(float(vals[sel].mean()))
    r = torch.tensor(reps)
    return {"mean": float(vals.mean()), "lo": float(r.quantile(0.025)),
            "hi": float(r.quantile(0.975)), "n_clusters": int(len(uq)),
            "n_images": int(vals.numel())}


def per_image_dino_s(img: torch.Tensor, pools: dict) -> torch.Tensor:
    """[N,3,H,W] -> [N,5] max cosine to each style's reference set."""
    q = C.quantize01(img)
    gd = C.dino_cls(q)
    return torch.stack([(gd @ pools["dino_cls"][s].T).max(1).values for s in range(5)], 1)


def main() -> None:
    torch.manual_seed(0)
    ds = C.load_dataset()
    split = C.make_split(1000, n_ref=800, n_query=200, seed=0)
    z_ref = torch.stack([ds.latents[s][split.ref[s]] for s in range(5)])
    z_q = torch.stack([ds.latents[s][split.query[s]] for s in range(5)]).to("cuda")
    tg = L.Targets.build(z_ref, device="cuda")
    loss = L.LatentStyleLoss(tg)
    pools = ref_pools("cuda")
    out = {}

    # ---- content clusters on the 1000 query images -----------------------
    tok = S2.load_dino_tokens(ds)                      # [5,1000,768]
    qtok = torch.cat([tok[s][split.query[s]] for s in range(5)])
    grp = cluster_ids(qtok, k=25)

    # ---- B1: top-1 style retrieval with swd ---------------------------
    gproj = torch.Generator().manual_seed(0)
    proj = torch.randn(11, 32, generator=gproj)
    proj = proj / proj.norm(dim=0, keepdim=True)
    ref_patch = [C.feat_patch(z_ref[k][:100].cpu()).reshape(-1, 11).float() for k in range(5)]
    z_flat = z_q.cpu().reshape(-1, 4, 64, 64)
    D = torch.zeros(z_flat.shape[0], 5)
    for i in range(z_flat.shape[0]):
        q = C.feat_patch(z_flat[i:i + 1]).reshape(-1, 11).float()
        g = torch.Generator().manual_seed(0)
        q = q[torch.randperm(q.shape[0], generator=g)[:256]]
        for k in range(5):
            D[i, k] = C.swd2_proj(q, ref_patch[k], proj=proj, n_q=48)
    pred = D.argmin(1)
    truth = torch.arange(5).repeat_interleave(z_q.shape[1])
    hit = (pred.to(truth.device) == truth).float()
    out["B1_top1_swd"] = boot_ci(hit, grp)
    out["B1_confusion"] = C.confusion(pred.to(truth.device), truth, 5)

    # ---- B2: patch_shuffle effect on dino_s ---------------------------
    n = 12
    zc = z_q[:, :n].reshape(-1, 4, 64, 64)
    src = torch.arange(5).repeat_interleave(n).to(zc.device)
    src_img = C.decode(zc).to(zc.device)
    sh_img = C.decode(perturb(zc, "patch_shuffle")).to(zc.device)
    grp_sub = torch.tensor([s * z_q.shape[1] + i for s in range(5) for i in range(n)])
    d_id = per_image_dino_s(src_img, pools)
    d_sh = per_image_dino_s(sh_img, pools)
    own_id = d_id.cpu().gather(1, src.cpu().view(-1, 1)).squeeze(1)
    own_sh = d_sh.cpu().gather(1, src.cpu().view(-1, 1)).squeeze(1)
    out["B2_shuffle_dino_s_after"] = boot_ci(own_sh, grp[grp_sub])
    out["B2_shuffle_dino_s_drop"] = boot_ci(own_id - own_sh, grp[grp_sub])

    # ---- B3: constrained latent ascent, per-image gain ----------------
    tgt = (src + 1) % 5
    z = zc.clone()
    for _ in range(15):
        zb = z.clone().requires_grad_(True)
        v = zb.new_zeros(())
        for s in range(5):
            m = tgt == s
            if m.any():
                v = v + loss("swd", zb[m], s) * int(m.sum()) / zc.shape[0]
        (gr,) = torch.autograd.grad(v, zb)
        d = gr.flatten(1)
        d = d / d.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
        z = rms_proj((z - 0.04 * d.reshape_as(z)).detach(), zc, 0.2)
    opt_img = C.decode(z).to(zc.device)
    d_opt = per_image_dino_s(opt_img, pools).cpu()
    d_base = per_image_dino_s(src_img, pools).cpu()
    tgt_c = tgt.cpu()
    gain = (d_opt.gather(1, tgt_c.view(-1, 1)) - d_base.gather(1, tgt_c.view(-1, 1))).squeeze(1)
    out["B3_dino_s_gain"] = boot_ci(gain, grp[grp_sub])
    others = (d_opt.sum(1) - d_opt.gather(1, tgt_c.view(-1, 1)).squeeze(1)) / 4
    base_others = (d_base.sum(1) - d_base.gather(1, tgt_c.view(-1, 1)).squeeze(1)) / 4
    sel = (d_opt.gather(1, tgt_c.view(-1, 1)).squeeze(1) - others) - \
          (d_base.gather(1, tgt_c.view(-1, 1)).squeeze(1) - base_others)
    out["B3_selectivity_change"] = boot_ci(sel, grp[grp_sub])

    C.save_json("bootstrap_ci.json", out)
    for k in ("B1_top1_swd", "B2_shuffle_dino_s_drop", "B3_dino_s_gain",
              "B3_selectivity_change"):
        print("[BOOT]", k, out[k])
    print("[BOOT] confusion", out["B1_confusion"])


if __name__ == "__main__":
    main()
