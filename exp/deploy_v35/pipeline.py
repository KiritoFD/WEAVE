"""CLEAVE V35 Unified Pure-PyTorch Inference Pipeline (Reference / Optimized / Fast).

Enforces strict E-only / P-Core memory isolation and clean separation between:
  - profile="reference": verbatim V35 reference path (A0 BF16 + original UNets + SVD Stiefel + integrate_v33_trajectory)
  - profile="optimized": zero-engine pure-PyTorch infra optimizations (TF32, FastA0, FastConditioner + NS10 Stiefel,
    SDPA BottleneckAttention, HoistedPCoreStep trajectory factorization, FastS10B channels_last, host ODE grid,
    and optional fixed-shape torch.compile(reduce-overhead) with persistent FX graph disk cache).
  - profile="fast": optional solver-accelerated profile (Euler startup / fewer steps, gated separately from Optimized).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from exp.deploy_v35 import BUNDLE_DEFAULT_DIR, OPT_OUT_DIR, P8_LOCKED_CFG
from exp.deploy_v35.bundle import (
    load_a0_encoder_from_file,
    load_convnext_s3_from_file,
    load_ecore_from_file,
    load_pcore_from_file,
    load_s10b_from_file,
)
from exp.deploy_v35.torch_opt import (
    FastA0Encoder,
    FastConditioner,
    FastECore,
    FastS10BDecoder,
    HoistedPCoreStep,
    enable_persistent_inductor_cache,
)
from exp.v20.slf_flow import project_latent_y16_lcn
from exp.v21.stiefel_ref_cache import stiefel_polar_quotient
from exp.v32_nodino.cnx_layout import match_cnx_k8_submode
from exp.v33_semantic_control.models import integrate_v33_trajectory
from exp.v34_frontier.stage_a_startpoint import reconstruct_analytic_draft
from exp.v35_final.common import make_v35_schedule
from src.utils.runtime import REPO as ROOT, bf16_autocast
from src.utils.structure import rgb_to_structure_64


@dataclass
class CallStats:
    """Runtime hook counters verifying actual model invocations and condition-equivalent NFE."""

    a0_calls: int = 0
    cnx_calls: int = 0
    ecore_calls: int = 0
    pcore_batch_calls: int = 0
    pcore_cond_evals: int = 0
    s10_calls: int = 0

    def reset(self) -> None:
        self.a0_calls = 0
        self.cnx_calls = 0
        self.ecore_calls = 0
        self.pcore_batch_calls = 0
        self.pcore_cond_evals = 0
        self.s10_calls = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "a0_calls": self.a0_calls,
            "cnx_calls": self.cnx_calls,
            "ecore_calls": self.ecore_calls,
            "pcore_batch_calls": self.pcore_batch_calls,
            "pcore_cond_evals": self.pcore_cond_evals,
            "s10_calls": self.s10_calls,
            "total_flow_cond_evals": self.ecore_calls + self.pcore_cond_evals,
        }


class SourceFeatureCache:
    """In-memory per-source conditioning cache keyed by pixel sha256 + shape + dtype + device."""

    def __init__(self, version: str = "cleave_v35_1"):
        self.version = version
        self._store: dict[tuple[str, int, int, str, str, str], dict[str, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _pixel_key(rgb01: torch.Tensor) -> str:
        u8 = (rgb01.detach().clamp(0.0, 1.0) * 255.0).round().byte().cpu().numpy().tobytes()
        return hashlib.sha256(u8).hexdigest()

    def get_or_compute(self, rgb01: torch.Tensor, *, compute_fn, dtype_tag: str, device_tag: str) -> dict[str, torch.Tensor]:
        B, _, H, W = rgb01.shape
        key = (self._pixel_key(rgb01), B, H * 10000 + W, self.version, dtype_tag, device_tag)
        hit = self._store.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        val = compute_fn(rgb01)
        self._store[key] = val
        return val


def _precompute_dpm2m_schedule(
    steps: int,
    tau0: float,
    ws: float,
    wc: float,
    wsp: float,
    startup: str = "midpoint",
) -> list[dict[str, float]]:
    """Precompute host-side ODE grid constants so zero .item() GPU sync occurs inside the step loop."""
    grid = torch.linspace(0.0, 1.0, steps + 1).tolist()
    rows: list[dict[str, float]] = []
    dp_prev = 0.0
    for i in range(steps):
        p0, p1 = float(grid[i]), float(grid[i + 1])
        dp_i = p1 - p0
        dt_i = dp_i * (1.0 - float(tau0))
        t0_i = float(tau0) + p0 * (1.0 - float(tau0))

        def _cfg_weights(prog: float) -> tuple[float, float, float]:
            gl = min(1.0, max(0.0, (prog - 0.35) / 0.35))
            return ws - 0.6 * gl, wc, wsp * (0.4 + 0.6 * gl)

        ws0, wc0, wsp0 = _cfg_weights(p0)
        pm = 0.5 * (p0 + p1)
        tm = float(tau0) + pm * (1.0 - float(tau0))
        wsm, wcm, wspm = _cfg_weights(pm)
        rows.append({
            "step": i,
            "p0": p0,
            "p1": p1,
            "dp": dp_i,
            "dt": dt_i,
            "t0": t0_i,
            "ws0": ws0,
            "wc0": wc0,
            "wsp0": wsp0,
            "pm": pm,
            "tm": tm,
            "wsm": wsm,
            "wcm": wcm,
            "wspm": wspm,
            "need_midpoint": (i == 0 or startup != "euler") and (i == 0 and startup == "midpoint"),
            "ratio_prev": (dp_i / max(dp_prev, 1e-5)) if i > 0 else 0.0,
        })
        dp_prev = dp_i
    return rows


@dataclass
class CleaveDeployPipeline:
    """Unified pure-PyTorch CLEAVE V35 inference runtime."""

    bundle_dir: Path | None = None
    device: str = "cuda"
    load_pcore: bool = True
    profile: str = "reference"  # "reference" | "optimized" | "fast"

    # Precision & optimization knobs (used by optimized/fast; reference locks original baseline)
    a0_dtype: torch.dtype = torch.bfloat16
    pcore_dtype: torch.dtype = torch.bfloat16
    ecore_dtype: torch.dtype = torch.float32
    s10_dtype: torch.dtype = torch.float16
    use_host_grid: bool = False
    use_torch_opt: bool = False
    use_newton_schulz_stiefel: bool = True
    ns_steps: int = 10
    cnx_fp16: bool = True
    channels_last_s10: bool = False
    compile_mode: str = ""  # "" | "reduce-overhead" | "default"
    compile_batches: tuple[int, ...] = (1, 2, 4, 8, 12, 16)
    inductor_cache_dir: Path = field(default_factory=lambda: OPT_OUT_DIR / "inductor_cache")

    solver_startup: str = "midpoint"  # "midpoint" | "euler"
    pcore_steps: int = 6

    stats: CallStats = field(default_factory=CallStats)
    source_cache: SourceFeatureCache = field(default_factory=SourceFeatureCache)

    def __post_init__(self) -> None:
        self.sf = float(P8_LOCKED_CFG["scaling_factor"])
        if self.profile != "reference":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if self.compile_mode:
                enable_persistent_inductor_cache(self.inductor_cache_dir)

        bdir = Path(self.bundle_dir) if self.bundle_dir else None
        if bdir is not None and (bdir / "manifest.json").exists():
            p_pcore = bdir / "pcore.pt"
            p_ecore = bdir / "ecore.pt"
            p_a0 = bdir / "a0_encoder.pt"
            p_s10 = bdir / "s10b_decoder.pt"
            p_bank = bdir / "k8_bank.pt"
            p_a3 = bdir / "a3_priors.pt"
            p_tiny = bdir / "tiny_tensors.pt"
            p_cnx = bdir / "convnext_tiny.pth"
        else:
            p_pcore = ROOT / P8_LOCKED_CFG["pcore_ckpt"]
            p_ecore = ROOT / P8_LOCKED_CFG["ecore_ckpt"]
            p_a0 = ROOT / P8_LOCKED_CFG["a0_ckpt"]
            p_s10 = ROOT / P8_LOCKED_CFG["s10b_ckpt"]
            p_bank = ROOT / P8_LOCKED_CFG["k8_bank"]
            p_a3 = ROOT / P8_LOCKED_CFG["a3_priors"]
            p_tiny = ROOT / P8_LOCKED_CFG["tiny_tensors"]
            p_cnx = None

        self._pcore_path = p_pcore
        self._compiled: dict[tuple[str, int], Any] = {}

        # Load A0 Encoder
        a0_raw = load_a0_encoder_from_file(p_a0, device=self.device, dtype=self.a0_dtype)
        # Load S10B Decoder
        s10_raw = load_s10b_from_file(p_s10, device=self.device, dtype=self.s10_dtype)
        # Load ConvNeXt Stage-3
        if p_cnx is not None:
            cnx_raw = load_convnext_s3_from_file(p_cnx, device=self.device, dtype=torch.float32)
        else:
            from exp.v32_nodino.cnx_layout import ConvNeXtTinyStage3Extractor
            cnx_raw = ConvNeXtTinyStage3Extractor(device=self.device).eval()

        # Load E-Core
        ec_dtype = torch.float32 if (self.profile == "reference" or not self.use_torch_opt) else torch.float32
        ec_raw, self.raw_stats, self.priors_ec = load_ecore_from_file(p_ecore, device=self.device, dtype=ec_dtype)

        self.priors_a = {
            k: v.to(device=self.device, dtype=torch.float32)
            for k, v in torch.load(p_a3, map_location="cpu", weights_only=False).items()
        }
        bank = torch.load(p_bank, map_location="cpu", weights_only=False)
        self.sub_scene = bank["sub_scene"].to(device=self.device, dtype=torch.float32)
        self.sub_raw417 = bank["sub_raw417"].to(device=self.device, dtype=torch.float32)
        self.cnx_bank = {"sub_scene": self.sub_scene, "sub_raw417": self.sub_raw417}

        if p_tiny.exists():
            tiny = torch.load(p_tiny, map_location="cpu", weights_only=False)
        else:
            st = torch.load(ROOT / "runs/v32_nodino/cache/cnx_stiefel_cache.pt", map_location="cpu", weights_only=False)
            v18 = torch.load(ROOT / "runs/v18/cache/ccf_pool_cache.pt", map_location="cpu", weights_only=False)
            tiny = {"W_stiefel_cnx": st["W_stiefel_cnx"], "luma_proj": v18["luma_proj"]}
        self.w32_st = tiny["W_stiefel_cnx"].to(device=self.device, dtype=torch.float32)
        self.w_luma = torch.as_tensor(tiny["luma_proj"]["w"], device=self.device, dtype=torch.float32)
        self.b_luma = torch.as_tensor(tiny["luma_proj"]["b"], device=self.device, dtype=torch.float32)

        if self.use_torch_opt and self.profile != "reference":
            self.fast_a0 = FastA0Encoder(a0_raw, sf=self.sf).to(device=self.device, dtype=self.a0_dtype).eval().requires_grad_(False)
            self.fast_s10 = FastS10BDecoder(s10_raw, sf=self.sf, dtype=self.s10_dtype, channels_last=self.channels_last_s10).to(self.device).eval()
            self.fast_cond = FastConditioner(
                cnx_raw,
                self.w32_st,
                self.w_luma,
                self.b_luma,
                ns_steps=self.ns_steps,
                use_exact_svd=(not self.use_newton_schulz_stiefel),
                cnx_fp16=self.cnx_fp16,
            ).to(self.device).eval()
            self.fast_ecore = FastECore(ec_raw, self.raw_stats, self.priors_ec, dtype=self.ecore_dtype).to(self.device).eval()
            self.a0 = None
            self.s10 = None
            self.cnx_ext = None
            self.ecore = None
        else:
            self.a0 = a0_raw
            self.s10 = s10_raw
            self.cnx_ext = cnx_raw
            self.ecore = ec_raw
            self.fast_a0 = None
            self.fast_s10 = None
            self.fast_cond = None
            self.fast_ecore = None

        self.pcore = None
        self.fast_pcore = None
        if self.load_pcore:
            self.ensure_pcore_loaded()

    def ensure_pcore_loaded(self) -> None:
        if self.pcore is not None or self.fast_pcore is not None:
            return
        pc_raw, _ = load_pcore_from_file(self._pcore_path, device=self.device, dtype=torch.float32)
        if self.use_torch_opt and self.profile != "reference":
            self.fast_pcore = HoistedPCoreStep(pc_raw, dtype=self.pcore_dtype).to(self.device).eval()
        else:
            self.pcore = pc_raw

    def _get_callable(self, name: str, bs: int, mod: nn.Module) -> Any:
        if not self.compile_mode or bs not in self.compile_batches:
            return mod
        key = (name, bs)
        fn = self._compiled.get(key)
        if fn is None:
            raw_compiled = torch.compile(mod, mode=self.compile_mode, dynamic=False)
            if self.compile_mode == "reduce-overhead":
                def _safe_cg(*args, _c=raw_compiled):
                    torch.compiler.cudagraph_mark_step_begin()
                    out = _c(*args)
                    if isinstance(out, tuple):
                        return tuple(t.clone() for t in out)
                    return out.clone()
                fn = _safe_cg
            else:
                fn = raw_compiled
            self._compiled[key] = fn
        return fn

    def _encode_a0(self, rgb01: torch.Tensor) -> torch.Tensor:
        self.stats.a0_calls += 1
        B = rgb01.shape[0]
        if self.fast_a0 is not None:
            x = rgb01.to(device=self.device, dtype=self.a0_dtype)
            return self._get_callable("a0", B, self.fast_a0)(x)
        x = (rgb01.to(device=self.device, dtype=torch.float32) * 2.0 - 1.0).to(dtype=self.a0_dtype)
        if self.profile == "reference":
            return (self.a0.encode(x).latent_dist.mean * self.sf).float()
        h = self.a0.quant_conv(self.a0.encoder(x))
        mean, _ = torch.chunk(h, 2, dim=1)
        return (mean * self.sf).float()

    def _decode_s10(self, zo: torch.Tensor) -> torch.Tensor:
        self.stats.s10_calls += 1
        B = zo.shape[0]
        if self.fast_s10 is not None:
            return self._get_callable("s10", B, self.fast_s10)(zo)
        z_in = (zo / self.sf).to(dtype=self.s10_dtype)
        x = self.s10(z_in)
        return x.float().mul(0.5).add(0.5).clamp(0.0, 1.0)

    def _extract_conditioning(self, rgb01: torch.Tensor, *, need_y16: bool, z_s: torch.Tensor) -> dict[str, torch.Tensor]:
        self.stats.cnx_calls += 1
        B = rgb01.shape[0]
        if self.fast_cond is not None:
            c64, g16, q16, y16 = self._get_callable("cond", B, self.fast_cond)(rgb01.to(self.device), z_s)
            return {"z_s": z_s, "c64": c64, "g16": g16, "q16": q16, "y16": y16}
        c64 = rgb_to_structure_64(rgb01)
        g16 = self.cnx_ext.extract_gray_layout(rgb01)
        q16 = stiefel_polar_quotient(g16, self.w32_st)
        out = {"z_s": z_s, "c64": c64, "g16": g16, "q16": q16}
        if need_y16:
            out["y16"] = project_latent_y16_lcn(z_s, self.w_luma, self.b_luma)
        return out

    def prepare_source(self, rgb01: torch.Tensor, *, need_y16: bool = True) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            z_s = self._encode_a0(rgb01)
            return self._extract_conditioning(rgb01, need_y16=need_y16, z_s=z_s)

    def _run_ecore(self, z_s: torch.Tensor, c64: torch.Tensor, q16: torch.Tensor, tid: torch.Tensor, lam: float, eps: torch.Tensor) -> torch.Tensor:
        self.stats.ecore_calls += 1
        B = z_s.shape[0]
        lam_t = torch.full((B,), float(lam), device=self.device, dtype=torch.float32)
        if self.fast_ecore is not None:
            return self._get_callable("ecore", B, self.fast_ecore)(z_s, c64, q16, tid, lam_t, eps)
        if self.ecore_dtype == torch.float32:
            return self.ecore(z_s, c64, q16, tid, lam_t, self.raw_stats, self.priors_ec, noise=eps)
        with torch.autocast("cuda", dtype=self.ecore_dtype):
            return self.ecore(z_s, c64, q16, tid, lam_t, self.raw_stats, self.priors_ec, noise=eps).float()

    @torch.inference_mode()
    def run_e_only(
        self,
        rgb01: torch.Tensor,
        tid: torch.Tensor,
        lam: float,
        eps: torch.Tensor,
        *,
        use_a3: bool = False,
        cached_src: dict[str, torch.Tensor] | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Execute 1-NFE E-Core RGB->RGB pipeline without touching P-Core."""
        if lam <= 0.0:
            return (rgb01.clamp(0.0, 1.0), {}) if return_intermediates else rgb01.clamp(0.0, 1.0)
        src = cached_src if cached_src is not None else self.prepare_source(rgb01, need_y16=False)
        ze = self._run_ecore(src["z_s"], src["c64"], src["q16"], tid, lam, eps)
        z_out = reconstruct_analytic_draft(ze, tid, self.priors_a, mode="A3") if use_a3 else ze
        rgb_out = self._decode_s10(z_out)
        if not return_intermediates:
            return rgb_out
        return rgb_out, {**src, "ze": ze, "z_out": z_out}

    @torch.inference_mode()
    def run_e_to_p(
        self,
        rgb01: torch.Tensor,
        tid: torch.Tensor,
        knob_s: float,
        eps: torch.Tensor,
        *,
        cached_src: dict[str, torch.Tensor] | None = None,
        steps: int | None = None,
        startup: str | None = None,
        tau0_override: float | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Execute full E-Core + A3 + P-Core Dual-CFG + S10B RGB->RGB pipeline."""
        self.ensure_pcore_loaded()
        kw = make_v35_schedule(
            knob_s,
            ws_mul=P8_LOCKED_CFG["ws_mul"],
            wc_mul=P8_LOCKED_CFG["wc_mul"],
            dtau0=P8_LOCKED_CFG["dtau0"],
        )
        lam = float(kw.pop("draft_lambda", 0.75))
        n_steps = int(steps if steps is not None else self.pcore_steps)
        st_mode = str(startup if startup is not None else self.solver_startup)

        if knob_s <= 0.0:
            src = cached_src if cached_src is not None else self.prepare_source(rgb01, need_y16=True)
            rgb_out = self._decode_s10(src["z_s"])
            return (rgb_out, {**src, "zo": src["z_s"]}) if return_intermediates else rgb_out

        src = cached_src if cached_src is not None else self.prepare_source(rgb01, need_y16=True)
        u_st = match_cnx_k8_submode(src["g16"], tid, self.cnx_bank)

        tau0 = float(kw["tau0"] if tau0_override is None else tau0_override)
        if tau0 <= 0.0:
            ze, ze_s = None, None
            kw["tau0"] = 0.0
        else:
            ze = self._run_ecore(src["z_s"], src["c64"], src["q16"], tid, lam, eps)
            ze_s = reconstruct_analytic_draft(ze, tid, self.priors_a, mode="A3")

        if self.profile == "reference" and not self.use_host_grid and not self.use_torch_opt:
            parent = self

            class _HookModel(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                    self.use_gamma = getattr(m, "use_gamma", True)

                def forward(self, *args, **kwargs):
                    parent.stats.pcore_batch_calls += 1
                    parent.stats.pcore_cond_evals += 4
                    return self.m(*args, **kwargs)

            hooked = _HookModel(self.pcore)
            zo = integrate_v33_trajectory(
                hooked, ze_s, eps, src["y16"], src["q16"], tid, u_st,
                steps=n_steps, solver="dpm2m", **kw,
            )
        else:
            zo = self._integrate_zero_sync(
                ze_s, eps, src["y16"], src["q16"], tid, u_st,
                tau0=tau0, ws=float(kw["ws"]), wc=float(kw["wc"]), wsp=float(kw["wsp"]),
                rescale_phi=float(kw["rescale_phi"]), gamma=float(kw["gamma"]),
                lscale=float(kw["lscale"]), steps=n_steps, startup=st_mode,
            )

        rgb_out = self._decode_s10(zo)
        if not return_intermediates:
            return rgb_out
        return rgb_out, {**src, "u": u_st, "ze": ze, "ze_s": ze_s, "zo": zo}

    def _integrate_zero_sync(
        self,
        ze: torch.Tensor | None,
        eps: torch.Tensor,
        y16: torch.Tensor,
        q32: torch.Tensor,
        tid: torch.Tensor,
        u_style: torch.Tensor,
        *,
        tau0: float,
        ws: float,
        wc: float,
        wsp: float,
        rescale_phi: float,
        gamma: float,
        lscale: float,
        steps: int,
        startup: str,
    ) -> torch.Tensor:
        B = eps.shape[0]
        dev = self.device
        if ze is None or tau0 <= 0.0:
            t_eff = 0.0
            z = eps.clone()
        else:
            t_eff = float(tau0)
            z = (1.0 - t_eff) * eps + t_eff * ze

        f0 = torch.zeros(B, dtype=torch.bool, device=dev)
        f1 = torch.ones(B, dtype=torch.bool, device=dev)
        ls = torch.full((B,), float(lscale), device=dev)
        ls0 = torch.zeros(B, device=dev)
        ds4 = torch.cat([f0, f0, f1, f1], 0)
        dc4 = torch.cat([f0, f1, f0, f1], 0)
        lsc4 = torch.cat([ls, ls0, ls, ls0], 0)
        sched = _precompute_dpm2m_schedule(steps, t_eff, ws, wc, wsp, startup=startup)

        if self.fast_pcore is not None:
            gc1, bc1, gc2, bc2, cb4 = self.fast_pcore.prepare_traj(y16, q32, tid, u_style, ds4, dc4, lsc4, float(gamma))
            pcore_fn = self._get_callable("pcore", B, self.fast_pcore)

            def _eval_4b(zz: torch.Tensor, t_val: float):
                self.stats.pcore_batch_calls += 1
                self.stats.pcore_cond_evals += 4
                t_1b = torch.full((B,), t_val, device=dev, dtype=torch.float32)
                return pcore_fn(zz, t_1b, cb4, gc1, bc1, gc2, bc2).chunk(4, 0)
        else:
            g4 = torch.full((B * 4,), float(gamma), device=dev)
            y4 = y16.repeat(4, 1, 1, 1)
            q4 = q32.repeat(4, 1, 1, 1)
            s4 = tid.repeat(4)
            u4 = u_style.repeat(4, 1)

            def _eval_4b(zz: torch.Tensor, t_val: float):
                self.stats.pcore_batch_calls += 1
                self.stats.pcore_cond_evals += 4
                t4 = torch.full((B * 4,), t_val, device=dev, dtype=torch.float32)
                if self.pcore_dtype == torch.bfloat16:
                    with bf16_autocast(dev):
                        return self.pcore(zz.repeat(4, 1, 1, 1), y4, q4, s4, u4, t4, drop_style=ds4, drop_content=dc4, layout_scale=lsc4, gamma=g4).float().chunk(4, 0)
                if self.pcore_dtype == torch.float16:
                    with torch.autocast("cuda", dtype=torch.float16):
                        return self.pcore(zz.repeat(4, 1, 1, 1), y4, q4, s4, u4, t4, drop_style=ds4, drop_content=dc4, layout_scale=lsc4, gamma=g4).float().chunk(4, 0)
                return self.pcore(zz.repeat(4, 1, 1, 1), y4, q4, s4, u4, t4, drop_style=ds4, drop_content=dc4, layout_scale=lsc4, gamma=g4).float().chunk(4, 0)

        prev_v = None
        for row in sched:
            v_sc, v_s0, v_0c, v_00 = _eval_4b(z, row["t0"])
            v1 = v_00 + row["wc0"] * (v_0c - v_00) + row["ws0"] * (v_sc - v_0c) + row["wsp0"] * (v_s0 - v_00)
            if rescale_phi > 0.0:
                s_ref = v_sc.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                s_raw = v1.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                v1 = rescale_phi * v1 * (s_ref / s_raw) + (1.0 - rescale_phi) * v1

            dt = row["dt"]
            if not row["need_midpoint"] and prev_v is not None:
                z = z + dt * (v1 + 0.5 * row["ratio_prev"] * (v1 - prev_v))
            elif row["need_midpoint"]:
                z_mid = z + 0.5 * dt * v1
                vm_sc, vm_s0, vm_0c, vm_00 = _eval_4b(z_mid, row["tm"])
                v2 = vm_00 + row["wcm"] * (vm_0c - vm_00) + row["wsm"] * (vm_sc - vm_0c) + row["wspm"] * (vm_s0 - vm_00)
                if rescale_phi > 0.0:
                    s_ref = vm_sc.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                    s_raw = v2.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-4)
                    v2 = rescale_phi * v2 * (s_ref / s_raw) + (1.0 - rescale_phi) * v2
                z = z + dt * v2
            else:
                z = z + dt * v1
            prev_v = v1
        return z

    def warmup(self, batch_sizes: tuple[int, ...] = (1, 4)) -> None:
        """Warm fixed-shape compiled subgraphs once before latency measurement."""
        with torch.inference_mode():
            for bs in batch_sizes:
                rgb = torch.rand(bs, 3, 512, 512, device=self.device, dtype=torch.float32)
                tid = torch.arange(bs, device=self.device, dtype=torch.long) % 5
                eps = torch.randn(bs, 4, 64, 64, device=self.device, dtype=torch.float32)
                for _ in range(3):
                    if self.load_pcore:
                        _ = self.run_e_to_p(rgb, tid, 1.00, eps)
                    else:
                        _ = self.run_e_only(rgb, tid, 0.60, eps)
            torch.cuda.synchronize()


def make_optimized_pipeline(
    bundle_dir: Path | str = BUNDLE_DEFAULT_DIR,
    *,
    load_pcore: bool = True,
    compile_mode: str = "reduce-overhead",
    compile_batches: tuple[int, ...] = (1, 2, 4, 8, 12, 16),
    a0_dtype: torch.dtype = torch.float16,
    pcore_dtype: torch.dtype = torch.float16,
    ecore_dtype: torch.dtype = torch.float16,
    solver_startup: str = "midpoint",
    pcore_steps: int = 6,
    profile: str = "optimized",
) -> CleaveDeployPipeline:
    return CleaveDeployPipeline(
        bundle_dir=Path(bundle_dir),
        load_pcore=load_pcore,
        profile=profile,
        a0_dtype=a0_dtype,
        pcore_dtype=pcore_dtype,
        ecore_dtype=ecore_dtype,
        s10_dtype=torch.float16,
        use_host_grid=True,
        use_torch_opt=True,
        use_newton_schulz_stiefel=True,
        ns_steps=10,
        cnx_fp16=True,
        channels_last_s10=True,
        compile_mode=compile_mode,
        compile_batches=compile_batches,
        solver_startup=solver_startup,
        pcore_steps=pcore_steps,
    )
