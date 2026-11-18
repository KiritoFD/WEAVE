"""Fix 1: T4 锚点缓存的明确生成函数与四种缓存情形校验（存在 / 不存在 / 不完整 / 版本变化）。"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from . import OUT, ROOT, RawWriterV22, config_hash, read_json, validate_raw_meta

T4_CONFIG = {
    "arm": "V19_T4_sweet_vgg_nosid",
    "mode": "s10_vgg",
    "alpha_c": 0.18,
    "m_steps": 2,
    "rho_y_edge": 0.065,
    "rho_y_low": 0.12,
    "rho_c": 0.22,
    "lcn_budget": 0.024,
    "etas": [0.055, 0.105],
    "is_wrong": False,
    "batch": 25,
}
T4_HASH = config_hash(T4_CONFIG)
T4_V19_RAW = ROOT / "runs/v19/raw/V19_T4_sweet_vgg_nosid"
T4_V22_CACHE = OUT / "cache/t4_latents_750.pt"


def _load_raw_latents(raw_dir: Path) -> torch.Tensor | None:
    try:
        meta = read_json(raw_dir / "meta.json")
        if not validate_raw_meta(meta):
            return None
        if meta.get("t4_hash") not in (None, T4_HASH):
            return None
        zs = []
        for b in meta["batches"]:
            d = torch.load(raw_dir / b["file"], map_location="cpu", weights_only=False)
            zs.append(d["z_final"].float())
        return torch.cat(zs, dim=0)
    except Exception:
        return None


def _regenerate_t4(bundle: dict, device: str = "cuda") -> torch.Tensor:
    """按 V19 T4 原始配置重生成 750 锚点（教师路径与 run_sprint1_teachers 逐位一致）。"""
    import time

    from exp.v18 import tasks_and_sources
    from exp.v19.aniso_teacher import AnisotropicS10Teacher
    from src.utils import decode_s10

    cfg = T4_CONFIG
    teacher = AnisotropicS10Teacher(bundle, device=device)
    tasks, z_src_750, n2c_cache = tasks_and_sources()
    z0_750 = n2c_cache["z0"].float()
    zn2_750 = n2c_cache["z_n2"].float()
    fib_750 = bundle["fnd"]["test750_fib"].float()

    zs_out: list[torch.Tensor] = []
    t0 = time.time()
    for st in range(0, 750, cfg["batch"]):
        sl = slice(st, min(st + cfg["batch"], 750))
        grp = tasks[sl]
        sid_proto = torch.tensor([t["source_style_id"] for t in grp], device=device)
        tid_true = torch.tensor([t["target_style_id"] for t in grp], device=device)
        same = sid_proto == tid_true
        tid_used = tid_true  # T4 恒为 correct-target

        z0_b = z0_750[sl].to(device)
        zn2_b = zn2_750[sl].to(device)
        z_out = teacher.solve_batch(
            zn2_b, z0_b, z_src_750[sl], fib_750[sl], tid_used,
            mode=cfg["mode"], alpha_chroma=cfg["alpha_c"], m_steps=cfg["m_steps"],
            rho_y_edge=cfg["rho_y_edge"], rho_y_low=cfg["rho_y_low"], rho_c=cfg["rho_c"],
            lcn_budget=cfg["lcn_budget"], etas=tuple(cfg["etas"]), same_mask=same,
        )
        zs_out.append(z_out.detach().cpu().float())
    del teacher
    torch.cuda.empty_cache()
    z_t4 = torch.cat(zs_out, dim=0)
    print(f"[V22 T4] Regenerated 750 T4 anchors in {time.time()-t0:.0f}s (hash={T4_HASH})", flush=True)
    return z_t4


def build_or_load_t4_latents_750(bundle: dict, device: str = "cuda") -> tuple[torch.Tensor, str]:
    """返回 (z_t4 (750,4,64,64) float32 CPU, source_tag)。

    情形 1 缓存存在:   v22 cache 文件且内嵌 hash 一致 -> 直接用。
    情形 2 原始存在:   runs/v19/raw/V19_T4_sweet_vgg_nosid 完整(750 唯一) -> 优先采用原始产物。
    情形 3 不完整:     raw meta 缺失/不完整 -> 重新生成。
    情形 4 版本变化:   raw meta 携带的 t4_hash 与当前配置 hash 不一致 -> 重新生成并另存。
    """
    OUT.mkdir(parents=True, exist_ok=True)
    T4_V22_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # 情形 1: v22 cache 存在且版本一致
    if T4_V22_CACHE.exists():
        try:
            ck = torch.load(T4_V22_CACHE, map_location="cpu", weights_only=False)
            if ck.get("t4_hash") == T4_HASH and ck["z_t4"].shape[0] == 750:
                print(f"[V22 T4] Reusing v22 cache (hash={T4_HASH})", flush=True)
                return ck["z_t4"].float(), "v22_cache"
        except Exception:
            pass

    # 情形 2/4: v19 原始产物（完整性 + 版本双重校验）
    z_raw = _load_raw_latents(T4_V19_RAW)
    if z_raw is not None:
        torch.save({"z_t4": z_raw.half(), "t4_hash": T4_HASH, "source": "v19_raw"}, T4_V22_CACHE)
        print(f"[V22 T4] Loaded original v19 raw T4 (hash={T4_HASH})", flush=True)
        return z_raw, "v19_raw"

    # 情形 3: 不存在 / 不完整 -> 重新生成
    z_t4 = _regenerate_t4(bundle, device=device)
    torch.save({"z_t4": z_t4.half(), "t4_hash": T4_HASH, "source": "regenerated"}, T4_V22_CACHE)
    return z_t4, "regenerated"


def ensure_t4_latents_750(bundle: dict, device: str = "cuda") -> torch.Tensor:
    """兼容旧 exp.v20.run_v20 的调用签名。"""
    z, _tag = build_or_load_t4_latents_750(bundle, device=device)
    return z
