"""WEAVE W-stage distillation losses (frozen Encoder / SD-VAE latent space).

Root cause being attacked (R2 table D): the pixel-wise latent MSE
    ||G(z_s,t) - z_oracle||^2
cannot teach a smooth 218K UNet to reproduce the per-image, phase-sensitive
high-frequency R1 gradient (R2 P1: ~48% of the edit energy sits in LH/HL/HH).
The network averages over samples -> half-step (RMS ratio 0.55), ~45deg direction
error, 53-60% relative fit error, and a 1.3-1.8 R1-margin gap.

This module replaces that loss with, all in the FROZEN R1 critic's space:
  1. multi-scale FEATURE matching at 64/32/16 (what the conv kernels can actually produce);
  2. KL alignment of the classification distribution (teacher logits as soft targets);
  3. an anti-attractor competition margin that repels the two dominant attractors
     (Impressionism / Minimalism) for the hard targets (ER / Rococo / Ukiyo-e);
  4. LL-only latent L2: keep the exact low-frequency coordinates, but STOP forcing
     per-pixel agreement on the unpredictable high-frequency subbands.

Also provides class-conditional teacher RMS budgets (experiment 3) so saturated
styles are throttled and under-fit styles get a stronger (but still bounded) push.

No CSD / DINO / image decode happens here; that is W_S10_ImgCalib's job.
Every function is pure-torch and CPU-testable with tiny tensors.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.legacy_root.wavelet import dwt2_haar

STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]
IMP_IDX = STYLES.index("Impressionism")
MIN_IDX = STYLES.index("Minimalism")
HARD_STYLES = (
    STYLES.index("Early_Renaissance"),
    STYLES.index("Rococo"),
    STYLES.index("Ukiyo_e"),
)

# Experiment 3: class-conditional teacher step budget. Saturated styles are
# throttled; hard/boundary styles get a larger effective push (paired with the
# anti-attractor margin so a bigger step does not just slide into Imp/Min).
CLASS_RMS = {
    "Early_Renaissance": 0.32,
    "Impressionism": 0.15,
    "Minimalism": 0.15,
    "Rococo": 0.32,
    "Ukiyo_e": 0.32,
}


def class_rms_tensor(target_id: torch.Tensor, device=None) -> torch.Tensor:
    """(B,) -> (B,1,1,1) per-sample step budget, looked up by class id."""
    table = torch.tensor(
        [CLASS_RMS[s] for s in STYLES], dtype=torch.float32, device=device
    )
    return table[target_id.long()].view(-1, 1, 1, 1)


# ---------------------------------------------------------------------------
# 1. Multi-scale R1 feature extraction
# ---------------------------------------------------------------------------
class R1FeatureExtractor:
    """Capture the frozen LatentStyleNet's 64/32/16 feature maps + logits.

    Hooks are registered once; the critic itself is never modified and stays
    frozen. Calling ``__call__(z)`` returns (logits, {scale: feature}).
    """

    SCALES = ("b64", "b32", "b16")

    def __init__(self, critic: nn.Module):
        self.critic = critic
        self._hooks = []
        self._cache: Dict[str, torch.Tensor] = {}
        for name in self.SCALES:
            layer = dict(critic.named_modules()).get(name)
            if layer is None:
                raise ValueError(f"critic has no layer '{name}'")
            self._hooks.append(
                layer.register_forward_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self._cache[name] = out

        return hook

    def __call__(self, z: torch.Tensor):
        self._cache.clear()
        logits = self.critic(z.float())
        feats = {k: self._cache[k] for k in self.SCALES}
        return logits, feats

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def feature_matching_loss(
    feats_student: Dict[str, torch.Tensor],
    feats_teacher: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
    ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MSE between L2-normalised student/teacher R1 features at each scale.

    Accepts pre-extracted feature dicts (so the caller forwards the frozen critic
    only once per side). Per-sample L2 normalisation stops the deep 16x16 scale
    from dominating the shallow 64x64 one by raw magnitude.
    """
    if weights is None:
        weights = {"b64": 1.0, "b32": 1.0, "b16": 1.0}
    total = ref.new_tensor(0.0) if ref is not None else None
    for name, w in weights.items():
        a = F.normalize(feats_student[name].float().flatten(1), dim=1)
        b = F.normalize(feats_teacher[name].float().flatten(1), dim=1)
        term = w * F.mse_loss(a, b)
        total = term if total is None else total + term
    return total


# ---------------------------------------------------------------------------
# 2. Classification-boundary KL alignment
# ---------------------------------------------------------------------------
def kl_boundary_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    tau: float = 2.0,
) -> torch.Tensor:
    """KL(p_teacher || p_student) on temperature-softened R1 logits.

    Teacher logits are produced under no_grad, so this is a one-way distillation
    of the classifier's decision boundary (not just its argmax).
    """
    log_p_s = F.log_softmax(logits_student.float() / tau, dim=1)
    p_t = F.softmax(logits_teacher.float().detach() / tau, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau**2)


# ---------------------------------------------------------------------------
# 3. Anti-attractor competition margin
# ---------------------------------------------------------------------------
def anti_attractor_margin_per_sample(
    logits: torch.Tensor,
    target_id: torch.Tensor,
    beta: float = 1.0,
    tau: float = 1.0,
    hard_boost: float = 2.0,
) -> torch.Tensor:
    """Per-sample margin to MAXIMISE: s_t - beta * logsumexp_{k!=t}(w_k*s_k/tau).

    For hard targets (ER / Rococo / Ukiyo-e) the competition weights of the two
    dominant attractors (Impressionism, Minimalism) are boosted, so the gradient
    contains a component normal to the Imp/Min decision boundaries -- the
    "repulsion" that R2's uniform-weight plain margin objective lacked.

    Returns (B,) so a teacher can do ``grad(margin.sum(), z)``, while
    ``anti_attractor_loss`` is simply ``-margin.mean()``.
    """
    q = logits.float()
    s_t = q.gather(1, target_id.long()[:, None]).squeeze(1)
    others = q.masked_fill(
        F.one_hot(target_id.long(), q.shape[1]).bool(), torch.finfo(q.dtype).min
    )
    w = torch.ones_like(q)
    hard = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
    for hs in HARD_STYLES:
        hard |= target_id.long() == hs
    if hard.any():
        w[hard, IMP_IDX] = hard_boost
        w[hard, MIN_IDX] = hard_boost
    competition = torch.logsumexp(others / tau * w, dim=1)
    return s_t - beta * competition


def anti_attractor_loss(
    logits: torch.Tensor,
    target_id: torch.Tensor,
    beta: float = 1.0,
    tau: float = 1.0,
    hard_boost: float = 2.0,
) -> torch.Tensor:
    """softplus(-margin / tau).mean() -- a BOUNDED version of -margin.

    A raw ``-margin.mean()`` is unbounded below (maximising margin forever), so the
    optimizer drives R1 logits to infinity -- the same lesson as the R1 ranking loss
    in P2, which used softplus for exactly this reason. softplus saturates once the
    margin is comfortably positive, while keeping the same gradient *direction*
    (target up, Imp/Min competitors down).
    """
    margin = anti_attractor_margin_per_sample(
        logits, target_id, beta, tau, hard_boost
    )
    return F.softplus(-margin / tau).mean()


# ---------------------------------------------------------------------------
# 4. LL-only latent L2
# ---------------------------------------------------------------------------
def ll_only_latent_l2(
    z_student: torch.Tensor, z_teacher: torch.Tensor
) -> torch.Tensor:
    """MSE restricted to the Haar LL (low-frequency) subband of the displacement.

    High-frequency LH/HL/HH coordinates are deliberately NOT penalised: those
    carry the per-image, phase-sensitive R1 gradient noise a small UNet cannot
    predict (R2 P1). This is the single most important change vs B3/B4/S1-S4.
    """
    delta = z_student.float() - z_teacher.float().detach()
    ll, _, _, _ = dwt2_haar(delta)
    return ll.pow(2).mean()


# ---------------------------------------------------------------------------
# Composite W-distillation loss
# ---------------------------------------------------------------------------
# Fixed per-term normalisers, measured from the W unit tests / smoke step1:
#   l_feat (normalised feature MSE) O(0.01), l_kl (tau^2*KL) O(1),
#   l_rep (softplus) O(1-2), l_ll (Haar-LL MSE) O(0.001-0.01).
# Without these, the LL term (which must keep the exact low-frequency coordinates)
# is ~1000x smaller than rep/kl and silently vanishes. Dividing each by its scale
# makes w_feat/w_kl/w_rep/w_ll genuine RELATIVE weights.
TERM_SCALE = {"feat": 0.01, "kl": 1.0, "rep": 1.0, "ll": 0.01}
# Bring the composite back to the historical pixel-MSE magnitude (~0.0625), so the
# existing lambda=20/50 calibration and the B4 anchor stay meaningful. Empirically
# the four normalised terms sum to ~8-15 on a real (student != teacher) step, so a
# scale of 0.006 lands the composite at ~0.05-0.10, matching the old MSE term.
W_LOSS_SCALE = 0.006


class WDistillLoss(nn.Module):
    """Composite loss replacing the pixel-wise latent MSE.

    L = W_LOSS_SCALE * ( w_feat*feat/s + w_kl*kl/s + w_rep*rep/s + w_ll*ll/s )
    where /s is the per-term normaliser above. Each term is O(1) after scaling, so
    no single term is drowned out by raw magnitude differences.
    """

    def __init__(
        self,
        critic: nn.Module,
        w_feat: float = 1.0,
        w_kl: float = 1.0,
        w_rep: float = 0.5,
        w_ll: float = 1.0,
        tau_kl: float = 2.0,
        rep_beta: float = 1.0,
        rep_hard_boost: float = 2.0,
        loss_scale: float = W_LOSS_SCALE,
    ):
        super().__init__()
        self.extractor = R1FeatureExtractor(critic)
        self.w_feat = w_feat
        self.w_kl = w_kl
        self.w_rep = w_rep
        self.w_ll = w_ll
        self.tau_kl = tau_kl
        self.rep_beta = rep_beta
        self.rep_hard_boost = rep_hard_boost
        self.loss_scale = loss_scale

    def forward(
        self, z_student: torch.Tensor, z_teacher: torch.Tensor, target_id: torch.Tensor
    ):
        # One forward per side; the frozen critic keeps student gradients flowing.
        with torch.no_grad():
            logits_t, feats_t = self.extractor(z_teacher)
        logits_s, feats_s = self.extractor(z_student)
        l_feat = feature_matching_loss(feats_s, feats_t, ref=z_student)
        l_kl = kl_boundary_loss(logits_s, logits_t, self.tau_kl)
        l_rep = anti_attractor_loss(
            logits_s, target_id, self.rep_beta, hard_boost=self.rep_hard_boost
        )
        l_ll = ll_only_latent_l2(z_student, z_teacher)
        # Normalise each raw term to O(1) before weighting, then scale the composite
        # back to the historical pixel-MSE magnitude.
        n_feat = l_feat / TERM_SCALE["feat"]
        n_kl = l_kl / TERM_SCALE["kl"]
        n_rep = l_rep / TERM_SCALE["rep"]
        n_ll = l_ll / TERM_SCALE["ll"]
        total = self.loss_scale * (
            self.w_feat * n_feat
            + self.w_kl * n_kl
            + self.w_rep * n_rep
            + self.w_ll * n_ll
        )
        parts = {
            "feat": float(l_feat.detach()),
            "kl": float(l_kl.detach()),
            "rep": float(l_rep.detach()),
            "ll": float(l_ll.detach()),
            "n_feat": float(n_feat.detach()),
            "n_kl": float(n_kl.detach()),
            "n_rep": float(n_rep.detach()),
            "n_ll": float(n_ll.detach()),
        }
        return total, parts

    def close(self):
        self.extractor.close()
