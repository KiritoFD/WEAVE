"""V36 A/B/C objectives, shared verbatim in meaning by audit and E0 control.

E1 changes B alone to inverse real pairs; A's historical target-Q shortcut is
retained in both arms to isolate the B intervention. No evaluation referee here.
"""
import numpy as np
import torch
from exp.v35_cleave_one.prepare_data import _encode_rgb_scaled, _load_sd_vae_encoder
from exp.v36_art_foundation.ecore_losses import ECorePerceptualCritic, latent_proxy_reconstruction_loss
from exp.v36_art_foundation.proxy import destylize_proxy_v36
from exp.v36_art_foundation.train_ecore import _ECoreSeenTrainPool
from exp.v37_editing import OUT, SEED
from src.utils import decode_s10, load_s10_decoder
from src.utils.runtime import bf16_autocast


class Objectives:
    def __init__(self, route="E0", device="cuda"):
        self.device, self.route = device, route
        self.pool = _ECoreSeenTrainPool()
        self.priors = {k: v.to(device).float() if torch.is_tensor(v) else v
                       for k, v in self.pool.priors.items()}
        self.dec = load_s10_decoder(device)
        self.enc = _load_sd_vae_encoder(device)
        self.critic = ECorePerceptualCritic(device=device)
        self.gen = torch.Generator(device=device).manual_seed(SEED + 444)
        self.rng = np.random.default_rng(SEED + 888)
        self.exposed = {k: np.zeros(self.pool.N, dtype=np.int64) for k in "ABC"}
        self.inverse = None
        if route == "E1":
            self.inverse = torch.load(OUT / "pairs/index.pt", weights_only=False)
            self.pz = np.load(OUT / "pairs/proxy_z.npy", mmap_mode="r")
            self.pq = np.load(OUT / "pairs/proxy_q.npy", mmap_mode="r")
            assert (OUT / "pairs/approved.json").exists(), "Fixed proxy examples must be inspected first"

    def tensor(self, a):
        return torch.from_numpy(np.array(a)).to(self.device).float()

    def record(self, task, idx):
        np.add.at(self.exposed[task], idx, 1)

    def task_losses(self, model, step):
        p, rng, dev = self.pool, self.rng, self.device
        ac, ar = p.sample_task_a(6, rng)
        bc, bra, brb = p.sample_task_b_triplets(4, rng)
        za, qa, zra, sta = (self.tensor(a) for a in
                             (p.z_gt[ac], p.q32[ac], p.z_gt[ar], p.stat_crop[ar]))
        la = torch.empty(6, device=dev).uniform_(.35, 1.05).clamp_max(1)
        zb, qb, zra_b, zrb_b, sra, srb = (self.tensor(a) for a in
            (p.z_gt[bc], p.q32[bc], p.z_gt[bra], p.z_gt[brb], p.stat_crop[bra], p.stat_crop[brb]))
        lb = torch.empty(4, device=dev).uniform_(.40, 1.05).clamp_max(1)
        with torch.no_grad():
            xx = decode_s10(self.dec, torch.cat([za, zra, zb, zra_b, zrb_b]), batch=24, output_cpu=False)
            xa, xra, xb, xbra, xbrb = xx[:6], xx[6:12], xx[12:16], xx[16:20], xx[20:24]
            proxy = destylize_proxy_v36(xa, gen=self.gen)
            with bf16_autocast(dev):
                zp = _encode_rgb_scaled(self.enc, proxy).float()
        with bf16_autocast(dev):
            pred = model(zp, zra, qa, sta, la, self.priors, noise=torch.randn_like(zp))
        latent = latent_proxy_reconstruction_loss(pred, za, la)
        pixel, logs = self.critic.forward_proxy_pixel(self.dec, pred, xa, xra, use_ckpt=False)
        self.record("A", ac)
        yield "A", .70 * latent + .35 * pixel, {"latent": float(latent.detach()), **logs}
        del pred, latent, pixel

        eps = torch.randn_like(zb)
        if self.route == "E0":
            with bf16_autocast(dev):
                oa = model(zb, zra_b, qb, sra, lb, self.priors, noise=eps)
                ob = model(zb, zrb_b, qb, srb, lb, self.priors, noise=eps)
            swap, logs = self.critic.forward_swap_pair(self.dec, oa, ob, xb, xbra, xbrb, lb, use_ckpt=False)
            self.record("B", bc)
            yield "B", .90 * swap, logs
            del oa, ob, swap
        else:
            # Deterministic shuffled order; repeating epochs are explicit, no judge filtering.
            n = len(self.inverse["target"])
            cursor = (step - 4501) * 4
            epoch = cursor // n
            order = np.random.default_rng(SEED + 300 + epoch).permutation(n)
            ii = order[np.arange(cursor % n, cursor % n + 4) % n]
            target = self.inverse["target"][ii]
            zp, qp, gt = self.tensor(self.pz[ii]), self.tensor(self.pq[ii]), self.tensor(p.z_gt[target])
            with torch.no_grad():
                xgt = decode_s10(self.dec, gt, batch=4, output_cpu=False)
            self.record("B", target)
            # Match B's four source works and two output branches; both have real GT.
            for branch in range(2):
                ri = self.inverse["target_refs"][ii, branch]
                with bf16_autocast(dev):
                    pred = model(zp, self.tensor(p.z_gt[ri]), qp, self.tensor(p.stat[ri]),
                                 lb, self.priors, noise=eps)
                latent = latent_proxy_reconstruction_loss(pred, gt, lb)
                pixel, logs = self.critic.forward_proxy_pixel(self.dec, pred, xgt, xgt, use_ckpt=False)
                yield "B" + str(branch), .45 * (.70 * latent + .35 * pixel), {
                    "latent": float(latent.detach()), **logs}
                del pred, latent, pixel

        ci = p.sample_indices(4, rng, balanced_frac=.20)
        zi, qi, si = (self.tensor(a) for a in (p.z_gt[ci], p.q32[ci], p.stat[ci]))
        li = torch.empty(4, device=dev).uniform_(.10, .70)
        with bf16_autocast(dev):
            oi = model(zi, zi, qi, si, li, self.priors, noise=torch.randn_like(zi))
        loss = .30 * latent_proxy_reconstruction_loss(oi, zi, li, is_identity=torch.ones(4, dtype=torch.bool, device=dev))
        self.record("C", ci)
        yield "C", loss, {"lambda_mean": float(li.mean())}

    def exposure_summary(self):
        return {k: {"source_exposures": int(v.sum()), "unique_works": int((v > 0).sum()),
                    "equivalent_passes": float(v.sum() / len(self.pool.valid_idx))}
                for k, v in self.exposed.items()}


def grad_vector(model):
    return torch.cat([p.grad.detach().float().flatten() if p.grad is not None else
                      torch.zeros_like(p).flatten() for p in model.parameters()])
