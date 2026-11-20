r"""V26 P0-A: Source-Repulsive CFG (SR0–SR7) — 零训练源参考速度差分引导 + P0-D 少步扫描.

docs/1120/01_diagnostics_and_srcfg.md §P0-A / §P0-D 实现.

数学形式 (docs 预注册)
----------------------
基线 C2nt 速度 (2 分支, v_sc = drop_style 无风格 / v_0c = 目标参考条件):

    v_cfg(tau) = v_0c + wj * (v_sc - v_0c)

SR 替换式 CFG (**与基线同 NFE**: 分支 B 从"无风格"换成"源图自参考"全条件分支):

    v_sr(tau) = v_pos + w * P (v_pos - v_src)

- v_pos : drop_style=False, 参考 = 目标参考 (u_style_t / u_glob_t / d16_ref_t / v16_ref_t)
- v_src : drop_style=False, 参考 = 源图自身 (u_style_s = adapter(u_src,u_src),
          u_glob_s = fib_n[src], d16_ref_s = d16_src, v16_ref_s = v16_tr[src])
- P = I (SR1–SR4) 或 P = P_C = I4 - wy wy^T 色度子空间 (SR5/SR6)

SR7 (3 分支, 1.5×NFE 高算力对照): v = v_cfg + w * (v_pos - v_src).

关键恒等式 (tests/v26 契约 1): 当参考 = 源 (Same150 / 自参考) 时
    u_style_t == u_style_src, d16_ref_t == d16_ref_src, v16_ref_t == v16_ref_src,
    u_glob_t == u_glob_src  =>  v_pos - v_src == 0 (数值精度内)  =>  v_sr == v_pos.
    即 SR 引导对自参考**精确无操作** (不改变速度场), 由单测机械断言.

评测协议 (与 V25 C2nt t45 完全同口径): test750, eps seed 20261185, batch=25,
Same150 bypass (z_out = z_src), S10 解码, 四裁判隔离进程.

用法 (3060 / 4090 皆可):
  python -m exp.v26.sr_cfg --arms all            # SR0(复用) + SR1–SR7 + selfref 审计
  python -m exp.v26.sr_cfg --arms SR3_w10
  python -m exp.v26.sr_cfg --few-step            # P0-D: t45 下 steps {10,8,6,4} RK2 + 4 步 Euler
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import (
    C2NT_CKPT,
    OUT,
    RawWriterV26,
    V16_TR_CACHE,
    V21_CACHE,
    V25_MEASURED_C2NT_T45,
    V25_RAW_C2NT_T45,
    arm_done,
    autocast,
    config_hash,
    file_sha,
    read_json,
    run_measure_v26,
    sanitize_cuda_env,
    save_json,
    seed_all,
    utc,
    validate_raw_meta,
)
from exp.v18 import tasks_and_sources
from exp.v19.fiber_duality import StyleFiberEncoder, build_or_load_v19_foundation
from exp.v20.slf_flow import coarse_y16_lcn
from exp.v20.submode_patchswd import load_convnext_tiny_offline
from exp.v22.ref_adapter import RefAdapter
from exp.v24.p0_o1_o3_value import SLFUNetV24OneShot, _sg5_cnx_v16
from exp.v23.p2_attn_cross import _sg5_ref_d16
from src.utils import decode_s10, load_s10_decoder

EVAL_SEED_750 = 20261185
BATCH = 25
TAU0, WJ = 0.45, 2.2  # 与 V25 C2nt t45 读取数完全同口径

# 预注册 SR 臂矩阵 (docs/1120 §P0-A; 不得臂间私改)
SR_ARMS: dict[str, dict] = {
    "SR0_t45":     dict(mode="baseline", w=0.0,  proj="none"),   # 原始 C2nt 2 分支 CFG
    "SR1_w03":     dict(mode="sr",      w=0.3,  proj="none"),
    "SR2_w06":     dict(mode="sr",      w=0.6,  proj="none"),
    "SR3_w10":     dict(mode="sr",      w=1.0,  proj="none"),
    "SR4_w15":     dict(mode="sr",      w=1.5,  proj="none"),
    "SR5_w06_pc":  dict(mode="sr",      w=0.6,  proj="chroma"),
    "SR6_w10_pc":  dict(mode="sr",      w=1.0,  proj="chroma"),
    "SR7_3br_w06": dict(mode="sr3",     w=0.6,  proj="none"),   # 3 分支 1.5×NFE
}


# ------------------------------------------------------------ 投影 ----

def chroma_projector(w_luma: torch.Tensor) -> torch.Tensor:
    """P_C = I4 - wy wy^T, wy = w_luma / ||w_luma|| (正交投影到色度子空间).

    契约: P_C @ wy == 0, P_C == P_C^T, P_C @ P_C == P_C (幂等) — 单测断言.
    """
    wy = w_luma.flatten().float()
    wy = wy / wy.norm().clamp_min(1e-8)
    pc = torch.eye(4, device=wy.device, dtype=wy.dtype) - torch.outer(wy, wy)
    return pc


def apply_proj(diff: torch.Tensor, pc: torch.Tensor | None) -> torch.Tensor:
    """对 (B,4,H,W) 速度差逐像素做 4 通道投影 (None → I)."""
    if pc is None:
        return diff
    return torch.einsum("ij,bjhw->bihw", pc.to(diff.device, diff.dtype), diff)


# ------------------------------------------------- 速度场 (可单测) ----

def sr_velocity(fm, ad, u_style_t, u_style_s, u_glob_t, u_glob_s,
                d16, d16_r_t, v16_r_t, v16_r_s, d16_r_s, y16, zz, tv,
                *, mode: str, w: float, wj: float, pc: torch.Tensor | None = None,
                device: str = "cuda"):
    """V26 SR-CFG 速度场 (单批). mode ∈ {baseline, sr, sr3}.

    - baseline: 2 分支 [v_pos(目标参考, drop_style=F), v_sc(无风格, drop_style=T)]
    - sr:       2 分支 [v_pos(目标参考), v_src(源自参考)] — 同 NFE 替换式
    - sr3:      3 分支 [v_pos, v_src, v_sc] — v_cfg + w·(v_pos−v_src), 1.5×NFE
    返回 (velocity, n_branches).
    """
    B, dev = zz.shape[0], zz.device
    f0 = torch.zeros(B, dtype=torch.bool, device=dev)
    f1 = torch.ones(B, dtype=torch.bool, device=dev)
    l1 = torch.ones(B, device=dev)
    slot = torch.zeros(B, dtype=torch.long, device=dev)

    def _branch(ref_style, ref_glob, ref_d16, ref_v16, drop_st):
        k = len(ref_style)
        out = fm(torch.cat([zz] * k, 0), y16.repeat(k, 1, 1, 1), d16.repeat(k, 1, 1, 1),
                 slot.repeat(k), torch.cat(ref_style, 0), tv.repeat(k),
                 drop_style=torch.cat([f0 if not s else f1 for s in drop_st], 0),
                 drop_content=torch.zeros(k * B, dtype=torch.bool, device=dev),
                 layout_scale=l1.repeat(k),
                 d16_ref=torch.cat(ref_d16, 0), v16_ref=torch.cat(ref_v16, 0),
                 u_glob=torch.cat(ref_glob, 0)).float()
        return out.chunk(k, 0)

    if mode == "baseline":
        # 标准 2 分支: 分支 1 目标参考全条件 (drop_style=F), 分支 2 无风格 (drop_style=T)
        out = fm(torch.cat([zz, zz], 0), y16.repeat(2, 1, 1, 1), d16.repeat(2, 1, 1, 1),
                 slot.repeat(2), torch.cat([u_style_t, u_style_t], 0), tv.repeat(2),
                 drop_style=torch.cat([f0, f1], 0),
                 drop_content=torch.zeros(2 * B, dtype=torch.bool, device=dev),
                 layout_scale=l1.repeat(2),
                 d16_ref=d16_r_t.repeat(2, 1, 1, 1), v16_ref=v16_r_t.repeat(2, 1, 1, 1),
                 u_glob=u_glob_t.repeat(2, 1)).float()
        v_pos, v_sc = out.chunk(2, 0)
        return v_pos + wj * (v_sc - v_pos), 2

    if mode == "sr":
        v_pos, v_src = _branch([u_style_t, u_style_s], [u_glob_t, u_glob_s],
                               [d16_r_t, d16_r_s], [v16_r_t, v16_r_s], [False, False])
        return v_pos + w * apply_proj(v_pos - v_src, pc), 2

    if mode == "sr3":
        v_pos, v_src, v_sc = _branch([u_style_t, u_style_s, u_style_t],
                                     [u_glob_t, u_glob_s, u_glob_t],
                                     [d16_r_t, d16_r_s, d16_r_t],
                                     [v16_r_t, v16_r_s, v16_r_t], [False, False, True])
        v_cfg = v_pos + wj * (v_sc - v_pos)
        return v_cfg + w * apply_proj(v_pos - v_src, pc), 3

    raise ValueError(f"未知 SR mode: {mode}")


def v_0c_ret(v_pos: torch.Tensor, v_sc: torch.Tensor, wj: float) -> torch.Tensor:
    """基线 CFG 组合式 (供单测引用)."""
    return v_pos + wj * (v_sc - v_pos)


# ------------------------------------------------- RK2 / Euler 积分 ----

def integrate_sr(vel_fn, z_init: torch.Tensor, tau0: float, steps: int,
                 sampler: str = "rk2") -> torch.Tensor:
    """Heun(RK2) 或 Euler 积分; vel_fn(zz, prog) → velocity; prog ∈ [0,1]."""
    dt = (1.0 - tau0) / float(steps)
    z = z_init
    for i in range(steps):
        v1 = vel_fn(z, i / float(steps))
        if sampler == "euler":
            z = z + dt * v1
        else:
            z = z + dt * vel_fn(z + 0.5 * dt * v1, (i + 0.5) / float(steps))
    return z


# ------------------------------------------------- 环境装载 ----

def load_sr_env(pool: str, device: str = "cuda", ckpt_path: Path | None = None) -> dict:
    """装载 C2nt 推理环境 (bundle / c21 / v16_tr / SG-5 参考), 全部只读复用 V25 资产.

    ckpt_path 缺省用 V25 正式 C2nt (runs/v25/oneshot_val/C2nt_sw6_lc0.8.pt);
    V26 新训臂传入自己的 ckpt (runs/v26/oneshot_val/...).
    """
    bundle = build_or_load_v19_foundation(pool, device=device)
    c21 = torch.load(V21_CACHE, map_location="cpu", weights_only=False)
    cnx = load_convnext_tiny_offline(device).eval()
    for p in cnx.parameters():
        p.requires_grad_(False)

    ck = torch.load(ckpt_path if ckpt_path is not None else C2NT_CKPT,
                    map_location="cpu", weights_only=False)
    fm = SLFUNetV24OneShot(v_dim=192, use_conf_gate=True, use_global_fib=True).to(device).eval()
    fm.load_state_dict(ck["fm"], strict=True)
    ad = RefAdapter(dim=128).to(device).eval()
    ad.load_state_dict(ck["adapter"], strict=True)
    fib_mean, fib_std = ck["fib_mean"].to(device), ck["fib_std"].to(device)

    fiber_enc = StyleFiberEncoder(in_dim=bundle["fnd"]["tr_fib"].shape[1], dim=128).to(device).eval()
    fiber_enc.load_state_dict(bundle["fnd"]["fiber_state"])
    u_src_750 = fiber_enc.encode(bundle["fnd"]["test750_fib"].to(device).float()).cpu()
    fib_n_750 = (bundle["fnd"]["test750_fib"].to(device).float() - fib_mean) / fib_std
    del fiber_enc

    v16_tr = (torch.load(V16_TR_CACHE, map_location="cpu", weights_only=False)["v16_tr"]
              if V16_TR_CACHE.exists() else None)
    if v16_tr is None:
        from exp.v24.p0_o1_o3_value import build_or_load_train_cnx_v16
        from src.utils import load_s10_decoder as _lsd
        v16_tr = build_or_load_train_cnx_v16(bundle, _lsd(device), cnx, device=device)

    d16_sg5 = _sg5_ref_d16(bundle, c21, device=device)
    v16_sg5 = _sg5_cnx_v16(c21, cnx, device=device)
    u_sg5 = c21["sg5"]["u"].float()

    # SG-5 参考图的原始 795D 风格纤维 (协议口径与 r3.eval_o_tau_list 一致):
    # u_glob 必须取 SG-5 目标画派纤维 (5,795), 而非 test750 样本纤维!
    from PIL import Image

    import torchvision.transforms.functional as TF
    from exp.v19.fiber_duality import extract_color_fiber_27d, extract_dino_rgb_moments_768d
    from src.utils._local_eval_refs import load_dino_model

    dino_model = load_dino_model(device=device)
    clean_dir = ROOT / "datasets/distinct5_512/train_512clean"
    sg_fibs = []
    for st in c21["sg5"]["stems"]:
        p = clean_dir / st.split("__")[0] / f"{st}.jpg"
        if not p.exists():
            p = list(clean_dir.rglob(f"{st}.*"))[0]
        xr = TF.to_tensor(Image.open(p).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
        sg_fibs.append(torch.cat([extract_color_fiber_27d(xr),
                                  extract_dino_rgb_moments_768d(dino_model, xr, batch=1, device=device).to(device)], dim=-1))
    fib_sg5 = torch.cat(sg_fibs, 0)                                  # (5, 795)
    fib_n_sg5 = (fib_sg5 - fib_mean) / fib_std
    del dino_model
    torch.cuda.empty_cache()

    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float()
    return dict(bundle=bundle, c21=c21, cnx=cnx, fm=fm, ad=ad, fib_mean=fib_mean,
                fib_std=fib_std, u_src_750=u_src_750, fib_n_750=fib_n_750,
                fib_n_sg5=fib_n_sg5,
                v16_tr=v16_tr, d16_sg5=d16_sg5, v16_sg5=v16_sg5, u_sg5=u_sg5,
                w_l=w_l, b_l=b_l)


# ------------------------------------------------- SR 评测臂 ----

@torch.no_grad()
def eval_sr_arm(env: dict, arm: str, *, steps: int = 10, sampler: str = "rk2",
                device: str = "cuda", tau0: float = TAU0, wj: float = WJ) -> dict:
    """单 SR 臂全量 test750 评测 (幂等: 已有 summary_v26 则直接读)."""
    if arm_done(arm):
        s = read_json(OUT / "measured" / arm / "summary_v26.json")
        print(f"[V26 {arm}] Reusing measured summary", flush=True)
        return s
    spec = SR_ARMS[arm]
    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)
    fm, ad = env["fm"], env["ad"]
    pc = chroma_projector(env["w_l"]).to(device) if spec["proj"] == "chroma" else None

    w_l = env["w_l"].to(device)
    b_l = env["b_l"].to(device)
    y16_750 = coarse_y16_lcn(((z_src_750.float().to(device) * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED_750))
    d16_750 = env["bundle"]["test_struct"]["dino_16"]

    # 源自参考条件 (Same150 语义与 v_src 分支共用; d16_ref_s = 内容自身 d16)
    u_style_self = ad(env["u_src_750"].to(device), env["u_src_750"].to(device)).cpu()

    cfg_hash = config_hash({"arm": arm, "tau0": tau0, "wj": wj, "steps": steps,
                            "sampler": sampler, "spec": spec,
                            "ckpt": file_sha(C2NT_CKPT)[:12]})
    if (OUT / "raw" / arm / "meta.json").exists() and validate_raw_meta(
                read_json(OUT / "raw" / arm / "meta.json")):
        pass  # raw 已在, 只补 measure
    else:
        wr = RawWriterV26(arm, dec=dec, device=device)
        t0 = time.time()
        for st in range(0, 750, BATCH):
            sl = slice(st, min(st + BATCH, 750))
            grp = tasks[sl]
            B = len(grp)
            zs = z_src_750[sl].to(device).float()
            sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
            tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)

            u_style_t = ad(env["u_sg5"][tid_t.cpu()].to(device), env["u_src_750"][sl].to(device))
            u_style_s = u_style_self[sl].to(device)
            u_glob_t = env["fib_n_sg5"][tid_t.cpu()]  # SG-5 目标画派纤维 (协议口径)
            u_glob_s = env["fib_n_750"][sl]           # 源图自参考: 源样本自身纤维
            d16 = d16_750[sl].to(device).float()
            d16_r_t = env["d16_sg5"][tid_t.cpu()].to(device)
            v16_r_t = env["v16_sg5"][tid_t.cpu()].to(device)
            v16_r_s = env["v16_tr"][sl].to(device).float()
            y16 = y16_750[sl].to(device).float()
            z = (1.0 - tau0) * eps_750[sl].to(device).float() + tau0 * zs

            def vel_fn(zz, prog):
                tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
                v, _ = sr_velocity(fm, ad, u_style_t, u_style_s, u_glob_t, u_glob_s,
                                   d16, d16_r_t, v16_r_t, v16_r_s, d16, y16, zz, tv,
                                   mode=spec["mode"], w=spec["w"], wj=wj, pc=pc, device=device)
                return v

            z_gen = integrate_sr(vel_fn, z, tau0, steps, sampler)
            same = sid_p == tid_t
            z_out = torch.where(same.view(B, 1, 1, 1), zs, z_gen)
            wr.write_batch(z_out, grp)
        meta = wr.finalize(cfg_hash=cfg_hash, tau0=tau0, wj=wj, steps=steps, sampler=sampler,
                           mode=spec["mode"], w=spec["w"], proj=spec["proj"])
        assert validate_raw_meta(meta)
        print(f"[V26 {arm}] generated in {time.time() - t0:.0f}s", flush=True)
    s = run_measure_v26(arm, device=device)
    return s


@torch.no_grad()
def selfref_identity_audit(env: dict, *, n_tasks: int = 150, device: str = "cuda",
                           steps: int = 10, sampler: str = "rk2") -> dict:
    """SR3 (w=1.0) 无 Same150 bypass 的真实自参考审计: 150 条 same 任务全流程推理.

    预期: v_pos - v_src ≡ 0 (恒等式) => 输出 ≈ z_src (积分误差级); 报告
    latent MAE / 解码后 DINO-C(对源内容) / dino_s(对源风格) — 无需裁判子进程.
    """
    from src.utils import FeatureStack  # 延迟: 保持模块 CPU 可导入

    tasks, z_src_750, _ = tasks_and_sources()
    same_ids = [i for i, t in enumerate(tasks) if t["source_style_id"] == t["target_style_id"]][:n_tasks]
    sel = torch.tensor(same_ids, dtype=torch.long)
    dec = load_s10_decoder(device)
    fm, ad = env["fm"], env["ad"]
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)

    zs = z_src_750[sel].to(device).float()
    B = len(sel)
    y16 = coarse_y16_lcn(((zs * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()
    d16 = env["bundle"]["test_struct"]["dino_16"][sel].to(device).float()
    u_src = env["u_src_750"][sel].to(device)
    u_style_self = ad(u_src, u_src)
    u_glob_self = env["fib_n_750"][sel]
    v16_self = env["v16_tr"][sel].to(device).float()
    tau0, w = TAU0, SR_ARMS["SR3_w10"]["w"]

    eps = torch.randn(B, 4, 64, 64, generator=torch.Generator().manual_seed(EVAL_SEED_750 + 7)).to(device)
    z_init = (1.0 - tau0) * eps + tau0 * zs

    def vel_fn(zz, prog):
        tv = torch.full((B,), tau0 + prog * (1.0 - tau0), device=device)
        v, _ = sr_velocity(fm, ad, u_style_self, u_style_self, u_glob_self, u_glob_self,
                           d16, d16, v16_self, v16_self, d16, y16, zz, tv,
                           mode="sr", w=w, wj=WJ, pc=None, device=device)
        return v

    z_out = integrate_sr(vel_fn, z_init, tau0, steps, sampler)
    latent_mae = float((z_out - zs).abs().mean())
    latent_max = float((z_out - zs).abs().max())

    stack = FeatureStack(device)
    rgb_out = decode_s10(dec, z_out, batch=16).clamp(0.0, 1.0)
    rgb_src = decode_s10(dec, zs, batch=16).clamp(0.0, 1.0)

    def _dino(imgs):
        from PIL import Image as _Img
        pil = [_Img.fromarray((x.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype("uint8"))
               for x in imgs]
        return stack.extract(pil, batch=16)["dino"].cpu()

    f_out, f_src = _dino(rgb_out), _dino(rgb_src)
    dino_c = float(F.cosine_similarity(f_out, f_src, dim=-1).mean())
    res = {"n_tasks": B, "steps": steps, "sampler": sampler, "w": w,
           "latent_mae": round(latent_mae, 6), "latent_max_abs": round(latent_max, 6),
           "dino_c_vs_src": round(dino_c, 5), "identity_pass": bool(latent_mae < 0.05 and dino_c > 0.95),
           "updated_utc": utc()}
    save_json(OUT / "sr_selfref_identity_audit.json", res)
    print(f"[V26 SR-selfref] {res}", flush=True)
    return res


# ------------------------------------------------------- P0-D 少步 ----

FEW_STEP_ARMS: dict[str, dict] = {
    "FS_t45_rk2_10": dict(steps=10, sampler="rk2"),
    "FS_t45_rk2_08": dict(steps=8, sampler="rk2"),
    "FS_t45_rk2_06": dict(steps=6, sampler="rk2"),
    "FS_t45_rk2_04": dict(steps=4, sampler="rk2"),
    "FS_t45_eul_04": dict(steps=4, sampler="euler"),
}


@torch.no_grad()
def eval_few_step_arm(env: dict, arm: str, device: str = "cuda") -> dict:
    """P0-D 少步扫描臂: 基线 SR0 条件 (标准 2 分支 CFG), 仅改 steps/sampler."""
    if arm_done(arm):
        return read_json(OUT / "measured" / arm / "summary_v26.json")
    spec = FEW_STEP_ARMS[arm]
    s = eval_sr_arm(env, "SR0_t45", steps=spec["steps"], sampler=spec["sampler"], device=device) \
        if False else None  # placeholder: eval_sr_arm 按 SR_ARMS 取 spec, 故走专用路径
    # 直接内联 (与 eval_sr_arm 相同逻辑, baseline mode, 自定义 steps/sampler)
    return _eval_baseline_steps(env, arm, spec["steps"], spec["sampler"], device)


@torch.no_grad()
def _eval_baseline_steps(env: dict, arm: str, steps: int, sampler: str, device: str) -> dict:
    tasks, z_src_750, _ = tasks_and_sources()
    dec = load_s10_decoder(device)
    fm, ad = env["fm"], env["ad"]
    w_l, b_l = env["w_l"].to(device), env["b_l"].to(device)
    y16_750 = coarse_y16_lcn(((z_src_750.float().to(device) * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).half()
    eps_750 = torch.randn(z_src_750.shape, generator=torch.Generator().manual_seed(EVAL_SEED_750))
    d16_750 = env["bundle"]["test_struct"]["dino_16"]

    cfg_hash = config_hash({"arm": arm, "tau0": TAU0, "wj": WJ, "steps": steps,
                            "sampler": sampler, "ckpt": file_sha(C2NT_CKPT)[:12]})
    if (OUT / "raw" / arm / "meta.json").exists():
        s = run_measure_v26(arm, device=device)
        return s
    wr = RawWriterV26(arm, dec=dec, device=device)
    t0 = time.time()
    for st in range(0, 750, BATCH):
        sl = slice(st, min(st + BATCH, 750))
        grp = tasks[sl]
        B = len(grp)
        zs = z_src_750[sl].to(device).float()
        sid_p = torch.tensor([t["source_style_id"] for t in grp], device=device)
        tid_t = torch.tensor([t["target_style_id"] for t in grp], device=device)
        u_style_t = ad(env["u_sg5"][tid_t.cpu()].to(device), env["u_src_750"][sl].to(device))
        u_glob_t = env["fib_n_sg5"][tid_t.cpu()]  # SG-5 目标画派纤维 (协议口径)
        d16 = d16_750[sl].to(device).float()
        d16_r_t = env["d16_sg5"][tid_t.cpu()].to(device)
        v16_r_t = env["v16_sg5"][tid_t.cpu()].to(device)
        y16 = y16_750[sl].to(device).float()
        z = (1.0 - TAU0) * eps_750[sl].to(device).float() + TAU0 * zs

        def vel_fn(zz, prog):
            tv = torch.full((B,), TAU0 + prog * (1.0 - TAU0), device=device)
            v, _ = sr_velocity(fm, ad, u_style_t, None, u_glob_t, None,
                               d16, d16_r_t, v16_r_t, None, None, y16, zz, tv,
                               mode="baseline", w=0.0, wj=WJ, pc=None, device=device)
            return v

        z_gen = integrate_sr(vel_fn, z, TAU0, steps, sampler)
        same = sid_p == tid_t
        z_out = torch.where(same.view(B, 1, 1, 1), zs, z_gen)
        wr.write_batch(z_out, grp)
    meta = wr.finalize(cfg_hash=cfg_hash, tau0=TAU0, wj=WJ, steps=steps, sampler=sampler)
    assert validate_raw_meta(meta)
    print(f"[V26 {arm}] generated in {time.time() - t0:.0f}s", flush=True)
    return run_measure_v26(arm, device=device)


# ----------------------------------------------------------- main ----

def main():
    from . import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="all", help="逗号分隔 SR 臂名 或 all / none")
    ap.add_argument("--few-step", action="store_true", help="追加 P0-D 少步扫描")
    ap.add_argument("--skip-selfref", action="store_true")
    args = ap.parse_args()

    sanitize_cuda_env()
    seed_all(20261185)
    env = load_sr_env(args.pool, device=args.device)

    results = {}
    if args.arms != "none":
        names = list(SR_ARMS) if args.arms == "all" else [a.strip() for a in args.arms.split(",")]
        # SR0 复用 V25 raw (同 seed 同协议): 拷贝 raw 目录 → v26, 重跑 measure
        if "SR0_t45" in names and not arm_done("SR0_t45") and \
                V25_RAW_C2NT_T45.exists() and not (OUT / "raw" / "SR0_t45" / "meta.json").exists():
            shutil.copytree(V25_RAW_C2NT_T45, OUT / "raw" / "SR0_t45")
            if V25_MEASURED_C2NT_T45.exists():
                (OUT / "measured" / "SR0_t45").mkdir(parents=True, exist_ok=True)
                shutil.copy(V25_MEASURED_C2NT_T45, OUT / "measured" / "SR0_t45" / "summary_v24.json")
                shutil.copy(V25_MEASURED_C2NT_T45, OUT / "measured" / "SR0_t45" / "summary_v26.json")
            print("[V26 SR0] Reused V25 C2nt t45 raw (identical protocol/seed)", flush=True)
        for arm in names:
            results[arm] = {k: v for k, v in eval_sr_arm(env, arm, device=args.device).items()
                            if k in ("cross600", "same150")}
        if not args.skip_selfref:
            results["selfref_audit"] = selfref_identity_audit(env, device=args.device)

    if args.few_step:
        for arm in FEW_STEP_ARMS:
            s = eval_few_step_arm(env, arm, device=args.device)
            results[arm] = {k: v for k, v in s.items() if k in ("cross600", "same150")}

    save_json(OUT / "sr_cfg_results.json", {"updated_utc": utc(), "results": results})
    print(f"\n=== V26 SR-CFG done → {OUT / 'sr_cfg_results.json'} ===", flush=True)


if __name__ == "__main__":
    main()
