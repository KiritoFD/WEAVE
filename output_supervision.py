"""Route A: differentiable output-level style/content supervision.

Post-training stage objective:

    L = L_FM + lambda_s * L_style + lambda_c * L_content

where L_style / L_content are computed on the FINAL DECODED IMAGE obtained by
unrolling 2-4 Euler ODE steps through the live WEAVE velocity field and
decoding with the VAE — gradients flow along the real inference path into the
model. This is deliberately different from:

  * the historical Cycle-NCE / single-step SWD experiments (failed): those
    applied SWD to a random target latent at a single step with no ODE-path
    backprop and no decoded-image multi-layer features;
  * the FM target itself: FM regresses per-position velocities toward an
    unrelated painting's coefficients, while here style is matched at the
    distribution level (SWD over patch descriptors, no spatial correspondence).

Content side: DINO patch self-similarity + perceptual feature L1 + Sobel edge
structure loss. We never force output RGB to equal input RGB.

Memory controls: few unrolled steps (2-4), gradient checkpointing per step,
optional microbatching over the batch dim, AMP-friendly.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from features import FrozenFeatureExtractor, patch_self_similarity, sobel_edges
from swd import PatchSWDStyleLoss
from wavelet import dwt2_haar, idwt2_haar


# ---------------------------------------------------------------------------
# Pluggable latent->image decoders.
# ---------------------------------------------------------------------------

class LatentDecoder(nn.Module):
    """Interface: decode latent (B, 4, H, W) -> image (B, 3, H*s, W*s) in [0,1].

    Frozen w.r.t. its own parameters but differentiable w.r.t. its input.
    """

    scale_factor: float = 1.0  # latent_scale_factor used when *encoding*

    def decode_image(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_image(z)


class TinyConvDecoder(LatentDecoder):
    """Deterministic tiny conv decoder for CPU smoke tests (NOT a real VAE)."""

    def __init__(self, latent_channels: int = 4, width: int = 32, seed: int = 4321) -> None:
        super().__init__()
        self.scale_factor = 1.0
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, width, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width, 3, 3, padding=1), nn.Sigmoid(),
        )
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Conv2d):
                    w = torch.empty(m.weight.shape)
                    w.normal_(0.0, 0.05, generator=g)
                    m.weight.copy_(w)
                    if m.bias is not None:
                        m.bias.zero_()
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def decode_image(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DiffusersVADDecoder(LatentDecoder):
    """Wrap a diffusers AutoencoderKL / AutoencoderTiny decoder (server side)."""

    def __init__(self, vae: nn.Module, scaling_factor: float) -> None:
        super().__init__()
        self.vae = vae
        self.scale_factor = float(scaling_factor)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

    def decode_image(self, z: torch.Tensor) -> torch.Tensor:
        out = self.vae.decode(z / self.scale_factor)
        img = getattr(out, "sample", out[0] if isinstance(out, (tuple, list)) else out)
        return ((img.float() * 0.5 + 0.5).clamp(0.0, 1.0)).to(z.dtype)


def build_latent_decoder(name: str, *, device: torch.device | str = "cpu") -> LatentDecoder:
    """Factory: 'tiny' for smoke tests; 'taesd'/'sd15' load real VAEs (server)."""
    key = str(name).strip().lower()
    if key in ("tiny", "tiny_conv"):
        return TinyConvDecoder().to(device)
    if key in ("taesd", "sd15"):
        from diffusers import AutoencoderKL, AutoencoderTiny  # lazy

        if key == "taesd":
            from pathlib import Path

            vae = AutoencoderTiny.from_pretrained(
                str(Path(__file__).parent / "data" / "taesd"), local_files_only=True
            )
            return DiffusersVADDecoder(vae, scaling_factor=1.0).to(device)
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        return DiffusersVADDecoder(vae, scaling_factor=0.18215).to(device)
    raise ValueError(f"Unknown decoder {name!r}; expected tiny|taesd|sd15")


# ---------------------------------------------------------------------------
# Differentiable ODE unroll.
# ---------------------------------------------------------------------------

def _euler_step(model, h, t_val, dt, *, style_id, style_latent, style_text_tokens, reference_feats=None):
    v = model(
        h, t=t_val, style_id=style_id, style_latent=style_latent,
        style_text_tokens=style_text_tokens, reference_feats=reference_feats,
    )
    ll, lh, hl, hh = dwt2_haar(h)
    ll = ll + v["ll"] * dt
    lh = lh + v["lh"] * dt
    hl = hl + v["hl"] * dt
    if "hh" in v:
        hh = hh + v["hh"] * dt
    return idwt2_haar(ll, lh, hl, hh)


def unroll_euler_decode(
    model,
    decoder: LatentDecoder,
    content_latent: torch.Tensor,
    *,
    steps: int,
    style_id=None,
    style_latent: torch.Tensor | None = None,
    style_text_tokens: torch.Tensor | None = None,
    reference_feats: torch.Tensor | None = None,
    use_grad_checkpoint: bool = True,
    microbatch: int = 0,
    decoder_checkpoint: bool = False,
) -> torch.Tensor:
    """Unroll ``steps`` Euler ODE steps from content latent, then decode.

    Fully differentiable: gradients flow through VAE decode and every ODE step
    into the model parameters. ``microbatch`` > 0 splits the batch to cap peak
    memory; ``use_grad_checkpoint`` checkpoints each Euler step when the model
    is in training mode.
    """
    steps = max(1, int(steps))
    mb = int(microbatch)

    def _chunks(v, n_chunks):
        """Split per-sample conditioning like the latent when its batch matches."""
        if torch.is_tensor(v) and v.shape[0] == content_latent.shape[0] and n_chunks > 1:
            if v.numel() == v.shape[0]:  # e.g. per-sample scalar style_id
                return list(v.split(mb, dim=0))
            return list(v.split(mb, dim=0))
        return [v] * n_chunks

    if mb > 0 and content_latent.shape[0] > mb:
        chunks = list(content_latent.split(mb, dim=0))
        n = len(chunks)
        sid_c = _chunks(style_id, n) if not (torch.is_tensor(style_id) and style_id.numel() == 1) else [style_id] * n
        slat_c = _chunks(style_latent, n) if not (torch.is_tensor(style_latent) and style_latent.shape[0] == 1) else [style_latent] * n
        stok_c = _chunks(style_text_tokens, n)
        ref_c = _chunks(reference_feats, n)
        imgs = []
        for i, chunk in enumerate(chunks):
            imgs.append(unroll_euler_decode(
                model, decoder, chunk, steps=steps,
                style_id=sid_c[i], style_latent=slat_c[i], style_text_tokens=stok_c[i],
                reference_feats=ref_c[i],
                use_grad_checkpoint=use_grad_checkpoint, microbatch=0,
                decoder_checkpoint=decoder_checkpoint,
            ))
        return torch.cat(imgs, dim=0)

    h = content_latent
    dt = 1.0 / steps
    for i in range(steps):
        t_val = i * dt
        if use_grad_checkpoint and model.training and h.requires_grad:
            h = grad_checkpoint(
                _euler_step, model, h, t_val, dt,
                use_reentrant=False,
                **{
                    "style_id": style_id, "style_latent": style_latent,
                    "style_text_tokens": style_text_tokens, "reference_feats": reference_feats,
                },
            )
        else:
            h = _euler_step(
                model, h, t_val, dt,
                style_id=style_id, style_latent=style_latent,
                style_text_tokens=style_text_tokens, reference_feats=reference_feats,
            )
    if decoder_checkpoint and torch.is_grad_enabled() and h.requires_grad:
        return grad_checkpoint(decoder.decode_image, h, use_reentrant=False)
    return decoder.decode_image(h)


# ---------------------------------------------------------------------------
# Content structure loss: self-similarity + perceptual + edge (no RGB match).
# ---------------------------------------------------------------------------

class ContentStructureLoss(nn.Module):
    """Content preservation without forcing output RGB == input RGB.

    Components (all on frozen-extractor features of output vs content image):
      * perceptual: L1 between early-layer feature maps (upsampled to match);
      * self-similarity: L1 between patch self-similarity matrices at a deep
        layer (DINO patch self-similarity analogue);
      * edge: L1 between Sobel edge magnitude maps of the two images.
    """

    def __init__(
        self,
        extractor: FrozenFeatureExtractor,
        perceptual_layers: Iterable[str],
        selfsim_layer: str,
        *,
        w_perceptual: float = 1.0,
        w_selfsim: float = 1.0,
        w_edge: float = 0.5,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.perceptual_layers: List[str] = list(perceptual_layers)
        self.selfsim_layer = str(selfsim_layer)
        self.w_perceptual = float(w_perceptual)
        self.w_selfsim = float(w_selfsim)
        self.w_edge = float(w_edge)

    def forward(
        self,
        img_out: torch.Tensor,
        img_content: torch.Tensor,
        *,
        feats_out: Dict[str, torch.Tensor] | None = None,
        feats_content: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        needed = list(dict.fromkeys(self.perceptual_layers + [self.selfsim_layer]))
        if feats_content is None:
            with torch.no_grad():
                feats_content = self.extractor.forward_layers(img_content.detach(), needed)
        if feats_out is None:
            feats_out = self.extractor.forward_layers(img_out, needed)

        total = img_out.new_tensor(0.0)
        logs: Dict[str, torch.Tensor] = {}

        if self.w_perceptual > 0.0:
            lp = img_out.new_tensor(0.0)
            for layer in self.perceptual_layers:
                fo, fc = feats_out[layer], feats_content[layer]
                if fo.shape[-2:] != fc.shape[-2:]:
                    fc = F.interpolate(fc, size=fo.shape[-2:], mode="bilinear", align_corners=False)
                lp = lp + F.l1_loss(fo.float(), fc.float())
            logs["content_perceptual"] = lp.detach()
            total = total + self.w_perceptual * lp

        if self.w_selfsim > 0.0:
            so = patch_self_similarity(feats_out[self.selfsim_layer].float())
            sc = patch_self_similarity(feats_content[self.selfsim_layer].float())
            if so.shape[-1] != sc.shape[-1]:
                # Different spatial sizes: compare downsampled similarity structure.
                n = min(so.shape[-1], sc.shape[-1])
                so = F.adaptive_avg_pool1d(so.flatten(1), n).view(so.shape[0], -1)
                sc = F.adaptive_avg_pool1d(sc.flatten(1), n).view(sc.shape[0], -1)
            ls = F.l1_loss(so, sc)
            logs["content_selfsim"] = ls.detach()
            total = total + self.w_selfsim * ls

        if self.w_edge > 0.0:
            eo = sobel_edges(img_out.float())
            with torch.no_grad():
                ec = sobel_edges(img_content.detach().float())
            if eo.shape[-2:] != ec.shape[-2:]:
                ec = F.interpolate(ec, size=eo.shape[-2:], mode="bilinear", align_corners=False)
            le = F.l1_loss(eo, ec)
            logs["content_edge"] = le.detach()
            total = total + self.w_edge * le

        logs["loss"] = total
        return logs


# ---------------------------------------------------------------------------
# Orchestrator: one call -> decoded image, style SWD, content loss.
# ---------------------------------------------------------------------------

class OutputSupervision(nn.Module):
    """Route A post-training objective on the decoded image.

    Usage (trainer): given a batch (content latents, style image, content
    image), unroll the model a few Euler steps, decode, and compute
    distribution-level style SWD + content structure losses.
    """

    def __init__(
        self,
        extractor: FrozenFeatureExtractor,
        decoder: LatentDecoder,
        *,
        style_layers: Iterable[str],
        content_perceptual_layers: Iterable[str],
        content_selfsim_layer: str,
        ode_steps: int = 2,
        swd_projections: int = 64,
        patch_size: int = 8,
        patch_stride: int = 4,
        w_perceptual: float = 1.0,
        w_selfsim: float = 1.0,
        w_edge: float = 0.5,
        grad_checkpoint: bool = True,
        microbatch: int = 0,
        feature_res: int = 0,
        chunk_rows: int = 0,
        decoder_checkpoint: bool = False,
        extractor_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.ode_steps = int(ode_steps)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.microbatch = int(microbatch)
        self.feature_res = max(0, int(feature_res))
        self.chunk_rows = max(0, int(chunk_rows))
        self.decoder_checkpoint = bool(decoder_checkpoint)
        self.extractor_checkpoint = bool(extractor_checkpoint)
        self.style_loss = PatchSWDStyleLoss(
            extractor, style_layers,
            patch_size=patch_size, patch_stride=patch_stride,
            num_projections=swd_projections, chunk_rows=self.chunk_rows,
        )
        self.content_loss = ContentStructureLoss(
            extractor, content_perceptual_layers, content_selfsim_layer,
            w_perceptual=w_perceptual, w_selfsim=w_selfsim, w_edge=w_edge,
        )

    def forward(
        self,
        model,
        *,
        content_latent: torch.Tensor,
        style_image: torch.Tensor | None,
        content_image: torch.Tensor | None,
        style_id=None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        reference_feats: torch.Tensor | None = None,
        style_feats: Dict[str, torch.Tensor] | None = None,
        content_feats: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        img_out = unroll_euler_decode(
            model, self.decoder, content_latent,
            steps=self.ode_steps, style_id=style_id, style_latent=style_latent,
            style_text_tokens=style_text_tokens, reference_feats=reference_feats,
            use_grad_checkpoint=self.grad_checkpoint, microbatch=self.microbatch,
            decoder_checkpoint=self.decoder_checkpoint,
        )
        if self.feature_res and img_out.shape[-1] != self.feature_res:
            size = (self.feature_res, self.feature_res)
            img_out = F.interpolate(img_out, size=size, mode="bilinear", align_corners=False)
            if style_image is not None:
                style_image = F.interpolate(style_image, size=size, mode="bilinear", align_corners=False)
            if content_image is not None:
                content_image = F.interpolate(content_image, size=size, mode="bilinear", align_corners=False)
        # Single extractor pass on the output for all layers used by the two
        # losses (previously each loss ran its own forward, 5x total).
        needed = list(dict.fromkeys(
            self.style_loss.layers
            + self.content_loss.perceptual_layers
            + [self.content_loss.selfsim_layer]
        ))
        feats_style = style_feats
        feats_content = content_feats
        if self.extractor_checkpoint and torch.is_grad_enabled() and img_out.requires_grad:
            feats_out = grad_checkpoint(
                self.style_loss.extractor.forward_layers, img_out, needed, use_reentrant=False,
            )
        else:
            feats_out = self.style_loss.extractor.forward_layers(img_out, needed)
        if feats_style is None or feats_content is None:
            with torch.no_grad():
                if feats_style is None:
                    if style_image is None:
                        raise ValueError("style_image is required when style_feats are not cached")
                    feats_style = self.style_loss.extractor.forward_layers(
                        style_image.detach(), self.style_loss.layers,
                    )
                if feats_content is None:
                    if content_image is None:
                        raise ValueError("content_image is required when content_feats are not cached")
                    feats_content = self.content_loss.extractor.forward_layers(
                        content_image.detach(),
                        self.content_loss.perceptual_layers + [self.content_loss.selfsim_layer],
                    )
        if self.content_loss.w_edge > 0.0 and content_image is None:
            raise ValueError("content_image is required for the edge term (w_edge > 0)")
        style_logs = self.style_loss(
            img_out, style_image, feats_out=feats_out, feats_style=feats_style,
        )
        content_logs = self.content_loss(
            img_out, content_image, feats_out=feats_out, feats_content=feats_content,
        )
        logs: Dict[str, torch.Tensor] = {
            "loss_style_out": style_logs["loss"],
            "loss_content_out": content_logs["loss"],
        }
        for k, v in style_logs.items():
            if k != "loss":
                logs[f"style_{k}"] = v
        for k, v in content_logs.items():
            if k != "loss":
                logs[k] = v
        return logs

