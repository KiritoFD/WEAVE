"""Reference-E A/B/C supervision with deterministic, input-aligned proxies.

Imported on CPU without constructing models. All heavy models are loaded only by
``EObjectives`` inside an explicitly requested training/audit process.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def deterministic_seed(seed: int, step: int, source: int, slot: int = 0) -> int:
    raw = f"stage1-proxy-v1:{seed}:{step}:{source}:{slot}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def slices(n: int, micro: int):
    for start in range(0, n, micro):
        yield slice(start, min(n, start + micro))


def proxy_q_metrics(target_q: torch.Tensor, proxy_q: torch.Tensor) -> dict[str, float]:
    a, b = target_q.float(), proxy_q.float()
    return {"q_rms": float((a - b).square().mean().sqrt()),
            "q_cosine": float(F.cosine_similarity(a.flatten(1), b.flatten(1)).mean()),
            "q_spatial_rmse": float((a - b).square().mean(1).sqrt().mean())}


class EObjectives:
    def __init__(self, config: dict, arm: str, root: Path, device: str = "cuda"):
        from exp.v36_art_foundation.ecore_losses import ECorePerceptualCritic
        from exp.v35_cleave_one.prepare_data import _load_sd_vae_encoder
        from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor
        from src.utils import load_s10_decoder

        self.device, self.config, self.arm = device, config, arm
        self.arm_config = config["arms"][arm]
        self.settings = config.get("training", {}).get("e", {})
        self.seed = int(self.arm_config.get("seed", config.get("seed", 20263800)))
        self.rng = np.random.default_rng(self.seed + 888)
        cache = self._path(config["paths"].get("cache_root", "runs/v36_art_foundation/cache"), root)
        self.meta = torch.load(cache / "seen_train_index.pt", map_location="cpu", weights_only=False)
        self.priors = torch.load(cache / "v36_seen_train_priors.pt", map_location="cpu", weights_only=False)
        self.priors = {k: v.to(device).float() if torch.is_tensor(v) else v for k, v in self.priors.items()}
        # Keep the 76K pool memory-mapped; materializing the entire z/Q pool is unnecessary.
        self.N = int(self.meta["n_seen"])
        self.arrays = {k: np.load(root / self.meta[k + "_rel"], mmap_mode="r")
                       for k in ("z_gt", "q32", "stat", "stat_crop")}
        self.valid_idx = np.asarray(self.meta["valid_idx"])
        probs = np.asarray(self.meta["sample_probs"])[self.valid_idx]
        self.probs = probs / probs.sum()
        self.artist_nbrs = np.asarray(self.meta["artist_nbrs"])
        self.has_nbr = self.artist_nbrs[:, 0] >= 0
        self.exposed = {k: np.zeros(self.N, dtype=np.int64) for k in "ABC"}
        self.dec, self.enc = load_s10_decoder(device), _load_sd_vae_encoder(device)
        self.cnx, self.critic = ConvNeXtTinyStage3Extractor(device), ECorePerceptualCritic(device)
        from exp.v36_art_foundation import proxy as proxy_module
        version = hashlib.sha256((cache / "seen_train_index.pt").read_bytes()
                                 + (cache / "v36_seen_train_priors.pt").read_bytes()
                                 + Path(proxy_module.__file__).read_bytes()
                                 + Path(__file__).read_bytes()).hexdigest()[:16]
        # Bump the version if proxy/source extraction changes; config hashes forbid accidental resume drift.
        self.proxy_cache = self._path(config["paths"].get("proxy_cache", "runs/stage1_recipe/proxy_cache"), root) / version
        self.proxy_cache.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _path(value, root):
        path = Path(value)
        return path if path.is_absolute() else Path(root) / path

    def sample_indices(self, n, *, balanced_frac):
        balanced = int(round(n * balanced_frac))
        a = self.rng.choice(self.valid_idx, n - balanced, replace=False)
        b = self.rng.choice(self.valid_idx, balanced, replace=True, p=self.probs)
        return np.concatenate([a, b])

    def sample(self):
        na, nb, nc = [int(self.settings.get(k, v)) for k, v in
                      (("batch_a", 6), ("batch_b", 4), ("batch_c", 4))]
        ac = self.sample_indices(na, balanced_frac=.25)
        ar = ac.copy()
        for j, ci in enumerate(ac):
            if self.has_nbr[ci] and self.rng.random() < .70:
                row = self.artist_nbrs[ci]
                row = row[row >= 0]
                ar[j] = int(self.rng.choice(row))
        bc = self.sample_indices(nb, balanced_frac=.30)
        bra = self.sample_indices(nb, balanced_frac=.50)
        brb = self.sample_indices(nb, balanced_frac=.50)
        # Match the historical sampling recipe, including its one-pass degeneracy repair.
        same = bra == brb
        if same.any():
            brb[same] = self.sample_indices(int(same.sum()), balanced_frac=.50)
        ci = self.sample_indices(nc, balanced_frac=.20)
        return {"A": (ac, ar), "B": (bc, bra, brb), "C": (ci,)}

    def tensor(self, key, ids):
        return torch.from_numpy(np.array(self.arrays[key][ids])).to(self.device).float()

    @torch.no_grad()
    def decode(self, z, micro):
        from src.utils import decode_s10
        return decode_s10(self.dec, z, batch=micro, output_cpu=False)

    @torch.no_grad()
    def proxy(self, rgb, ids, step, slot_offset=0):
        from exp.v35_cleave_one.prepare_data import _encode_rgb_scaled
        from exp.v32_nodino.cnx_layout import extract_cnx_q32
        from exp.v36_art_foundation.proxy import destylize_proxy_v36
        from src.utils.runtime import bf16_autocast

        zz, qq = [], []
        for slot, (image, source) in enumerate(zip(rgb, ids)):
            seed = deterministic_seed(self.seed, step, int(source), slot + slot_offset)
            path = self.proxy_cache / f"{seed:016x}.pt"
            if path.exists():
                cached = torch.load(path, map_location="cpu", weights_only=False)
                if (cached["source"], cached["seed"]) != (int(source), seed):
                    raise ValueError(f"Proxy provenance mismatch: {path}")
                z, q = cached["z"].to(self.device).float(), cached["q"].to(self.device).float()
            else:
                gen = torch.Generator(device=self.device).manual_seed(seed)
                px = destylize_proxy_v36(image[None], gen=gen)
                with bf16_autocast(self.device):
                    z = _encode_rgb_scaled(self.enc, px).float()
                    layout = self.cnx.extract_gray_layout(px)
                q = extract_cnx_q32(layout.float(), self.priors["W_stiefel_cnx"].float())
                tmp = path.with_suffix(".tmp")
                torch.save({"source": int(source), "seed": seed, "z": z.half().cpu(),
                            "q": q.half().cpu(), "input_contract": "same_proxy_RGB_for_z_Q32"}, tmp)
                tmp.replace(path)
                # All arms, including first producer, consume identical cached precision.
                z, q = z.half().float(), q.half().float()
            zz.append(z); qq.append(q)
        return torch.cat(zz), torch.cat(qq)

    def task_losses(self, model, step, micro=1, checkpointing=False, monitor=False):
        """Yield graph-bearing weighted micro losses; caller backprops each immediately.

        Samples, strengths and model noise are drawn for the full logical task
        before slicing. Weights are n_micro/n_task, not one mean per micro.
        """
        from exp.v36_art_foundation.ecore_losses import latent_proxy_reconstruction_loss
        from src.utils.runtime import bf16_autocast
        from torch.utils.checkpoint import checkpoint

        batch = self.sample()
        ac, ar = batch["A"]
        bc, bra, brb = batch["B"]
        ci, = batch["C"]
        for task, ids in (("A", ac), ("B", bc), ("C", ci)):
            np.add.at(self.exposed[task], ids, 1)
        strengths = {"A": torch.empty(len(ac), device=self.device).uniform_(.35, 1.05).clamp_max(1),
                     "B": torch.empty(len(bc), device=self.device).uniform_(.40, 1.05).clamp_max(1),
                     "C": torch.empty(len(ci), device=self.device).uniform_(.10, .70)}
        noises = {k: torch.randn((len(ids), 4, 64, 64), device=self.device)
                  for k, ids in (("A", ac), ("B", bc), ("C", ci))}

        def forward(z, zr, q, sr, lam, eps):
            def fn(a, b, c, d, e, f):
                with bf16_autocast(self.device):
                    return model(a, b, c, d, e, self.priors, noise=f)
            if checkpointing:
                return checkpoint(fn, z, zr, q, sr, lam, eps, use_reentrant=False)
            return fn(z, zr, q, sr, lam, eps)

        for task, ids in (("A", ac), ("B", bc), ("C", ci)):
            for sl in slices(len(ids), micro):
                ii = ids[sl]
                weight = len(ii) / len(ids)
                z, q = self.tensor("z_gt", ii), self.tensor("q32", ii)
                lam, eps = strengths[task][sl], noises[task][sl]
                # Fixed rays per logical task: micro choice does not change SWD projections.
                random_seed = deterministic_seed(self.seed + 31, step, ord(task))
                with torch.random.fork_rng(devices=[torch.device(self.device)]):
                    torch.manual_seed(random_seed)
                    if task == "A":
                        zr, sr = self.tensor("z_gt", ar[sl]), self.tensor("stat_crop", ar[sl])
                        xa, xr = self.decode(z, micro), self.decode(zr, micro)
                        zp, qp = self.proxy(xa, ii, step, sl.start)
                        qa = q if self.arm_config.get("a_q32", "target" if self.arm == "E0" else "proxy") == "target" else qp
                        pred = forward(zp, zr, qa, sr, lam, eps)
                        latent = latent_proxy_reconstruction_loss(pred, z, lam)
                        pixel, logs = self.critic.forward_proxy_pixel(self.dec, pred, xa, xr, use_ckpt=checkpointing)
                        loss = float(self.arm_config.get("a_weight", 1.0)) * (.70 * latent + .35 * pixel)
                        logs.update(proxy_q_metrics(q, qp))
                        logs.update(latent=float(latent.detach()), input_q="target_control" if qa is q else "proxy",
                                    c64_source="internally_derived_from_proxy_latent")
                        components = None
                    elif task == "B":
                        za, zb = self.tensor("z_gt", bra[sl]), self.tensor("z_gt", brb[sl])
                        sa, sb = self.tensor("stat_crop", bra[sl]), self.tensor("stat_crop", brb[sl])
                        x, xa, xb = self.decode(z, micro), self.decode(za, micro), self.decode(zb, micro)
                        oa, ob = forward(z, za, q, sa, lam, eps), forward(z, zb, q, sb, lam, eps)
                        loss, logs, components = self.critic.forward_swap_pair(
                            self.dec, oa, ob, x, xa, xb, lam, use_ckpt=checkpointing,
                            style_multiplier=float(self.arm_config.get("b_style_weight", 1.25 if self.arm == "E2" else 1)),
                            style_scale=strengths["B"].clamp(.2, 1).mean(), return_components=True)
                        loss = .90 * loss
                        components = {k: .90 * weight * v for k, v in components.items()}
                        with torch.no_grad():
                            residual = oa.float() - z
                            from src.utils.structure import dwt2_haar
                            bands = dwt2_haar(residual)
                            logs.update(residual_rms=float(residual.square().mean().sqrt()),
                                        residual_hf_rms=float(torch.stack([v.square().mean() for v in bands[1:]]).mean().sqrt()))
                    else:
                        sr = self.tensor("stat", ii)
                        oi = forward(z, z, q, sr, lam, eps)
                        loss = .30 * latent_proxy_reconstruction_loss(oi, z, lam,
                                      is_identity=torch.ones(len(ii), dtype=torch.bool, device=self.device))
                        logs, components = {"lambda_mean": float(lam.mean())}, None
                yield task, weight * loss, logs, components if monitor else None

    def state_dict(self):
        return {"numpy_generator": self.rng.bit_generator.state, "exposed": self.exposed}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["numpy_generator"]
        self.exposed = state["exposed"]

    def exposure_summary(self):
        return {k: {"source_exposures": int(v.sum()), "unique_works": int((v > 0).sum()),
                    "equivalent_passes": float(v.sum() / len(self.valid_idx))} for k, v in self.exposed.items()}
