r"""P0 流 B 探针: 78k 开放域"风格污染→异块复原" Direct Bridge (用户 2026-11-22 方案).

RefTeacher 熔断 FAIL (36.67%, ER/Uki 坍缩) → 按两步止损转纯流 B. 探针在 v18 train
latent 缓存 (4400 张) 上验证机制, 不依赖任何教师/泄漏锚:

  污染增广: 原画 z_x → 伪源 z̃_s = P_Y 锁定 + P_C 矩对齐到随机异图 (异图色度交换)
            + Haar 高频随机缩放 (0.5–1.5×) — 布局不变、风格被剥离, 杜绝抄袭偷懒;
  参考:     同画无重叠象限块 (或全图) latent 的去几何统计 → E_ref MLP (11→128) = u_vivid;
  Bridge:   z0⊥ = P_Y(z̃_s) + P_C(z̃_s) 矩对齐 E_ref 色度矩 → 直流流向 z_x
            (rectified, 无噪声项), L = MSE(v, z_x − z0⊥);
  手术:     SLFUNet cond_mlp 的 onehot(5) 切片置零 (免标签), 解冻 cond_mlp + 全部
            style blocks (AdaGN), 20% Style Dropout 激活推理期 CFG.
  探针验收: 1500 upd → 96 任务严格 1-shot (E_ref = SG-5 exemplar 统计, onehot 置零)
            + dual-gate 落盘 (红线 3).

用法:  python -m exp.v26.gpu4090.p0_stream_b --pool ../pseudo_gt_24h/orig_mean_fp16
       --updates 1500 --smoke 0
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from exp.v26 import OUT, ROOT, save_json, utc
from exp.v26.gpu4090.p0_common import (STYLES, chroma_align_anchor, dual_gate_record)

EVAL_SEED = 20261185


# ------------------------------------------------- 增广与统计 ----

def _luma(z: torch.Tensor, w_l: torch.Tensor, b_l: torch.Tensor) -> torch.Tensor:
    return (z * w_l).sum(1, keepdim=True) + b_l


def haar_energy(y: torch.Tensor) -> torch.Tensor:
    """亮度 3 子带能量 (B, 3) — LH/HL/HH RMS."""
    a = y[:, :, 0::2, 0::2]
    b = y[:, :, 0::2, 1::2]
    c = y[:, :, 1::2, 0::2]
    d = y[:, :, 1::2, 1::2]
    lh = ((a - b) / 2).flatten(1).square().mean(1)
    hl = ((a - c) / 2).flatten(1).square().mean(1)
    hh = ((a - d) / 2).flatten(1).square().mean(1)
    return torch.stack([lh, hl, hh], 1).clamp_min(1e-6).sqrt()


def ref_stats(z_ref: torch.Tensor, w_l: torch.Tensor, b_l: torch.Tensor) -> torch.Tensor:
    """去几何参考统计 (11,): P_C 通道矩 8 + 亮度 Haar 能量 3."""
    P_Y = None  # 通道矩直接取 latent 通道 (VAE 4 通道近正交分解)
    mu = z_ref.mean(dim=(2, 3))          # (B, 4)
    sd = z_ref.std(dim=(2, 3)).clamp_min(1e-4)
    y = _luma(z_ref, w_l, b_l).clamp(0.0, 1.0)
    return torch.cat([mu, sd, haar_energy(y)], dim=1)  # (B, 11)


def pollute(z_x: torch.Tensor, z_alien: torch.Tensor, w_l: torch.Tensor, b_l: torch.Tensor,
            g: torch.Generator, priors: dict | None = None) -> torch.Tensor:
    """流 B' 强污染: 异图完整色度场搬运 (P_C 场直取异图, 非矩匹配) + 亮度结构锁源."""
    if priors is None:
        B = z_x.shape[0]
        z_p = chroma_align_anchor(z_x, z_alien)
        hf = torch.empty(B, 1, 1, 1, device=z_x.device).uniform_(0.5, 1.5)
        y = _luma(z_p, w_l, b_l)
        y_src = _luma(z_x, w_l, b_l)
        detail = y - F.avg_pool2d(y, 3, stride=1, padding=1)
        y_new = y_src + detail * hf
        z_p = z_p.clone()
        z_p[:, 0:1] = z_p[:, 0:1] + (y_new - y)
        return z_p
    P_Y = priors["P_Y"].to(z_x.device)
    P_C = priors["P_C"].to(z_x.device)
    B = z_x.shape[0]
    zy = torch.einsum("ij,bjhw->bihw", P_Y, z_x.float())
    zc_alien = torch.einsum("ij,bjhw->bihw", P_C, z_alien.float())
    # 亮度笔触破坏: 每样本随机 [模糊 pool 2/3 | 高频衰减 0.3-0.6]
    y = zy
    r = torch.rand(B, 1, 1, 1, device=z_x.device)
    k = 2 + (torch.rand(B, 1, 1, 1, device=z_x.device) > 0.5).long()
    y_blur = F.interpolate(F.avg_pool2d(y, 2, 1, 1) if False else F.avg_pool2d(y, 4, 1, 1),
                           size=y.shape[-2:], mode="bilinear", align_corners=False)
    detail = y - F.avg_pool2d(y, 3, stride=1, padding=1)
    y_att = y - detail * (0.3 + 0.4 * torch.rand_like(r))
    y_deg = torch.where(r > 0.5, y_blur, y_att)
    # 色度场 = 异图完整 P_C 场 (笔触统计整体替换)
    return y_deg + zc_alien


def quadrant_stats(z: torch.Tensor, w_l: torch.Tensor, b_l: torch.Tensor) -> torch.Tensor:
    """同画四象限统计 (B, 4, 11) — 训练时随机取非本象限块作参考."""
    B, C, H, W = z.shape
    qs = []
    for i in (0, 1):
        for j in (0, 1):
            q = z[:, :, i * H // 2:(i + 1) * H // 2, j * W // 2:(j + 1) * W // 2]
            qs.append(ref_stats(q, w_l, b_l))
    return torch.stack(qs, 1)


# ------------------------------------------------- E_ref 编码器 ----

class ERefMLP(torch.nn.Module):
    def __init__(self, in_dim: int = 11, dim: int = 128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class BridgeSLFUNet(torch.nn.Module):
    """SLFUNet 包装: onehot(5) 切片置零 (免标签手术), 其余接口不变."""

    def __init__(self, slfunet, n_style: int = 5):
        super().__init__()
        self.slf = slfunet
        self.n_style = n_style

    def forward(self, z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, **kw):
        return self.slf(z_tau, y16_lcn, dino_16, style_id, u_vivid, tau, **kw)

    def zero_onehot(self):
        """手术: cond_mlp 首层 weight/bias 的 onehot 列置零 (列号 64..68)."""
        first = self.slf.cond_mlp[0]
        with torch.no_grad():
            first.weight[:, 64:64 + self.n_style].zero_()
            if first.bias is not None:
                pass  # bias 无列对应, 保持


# ------------------------------------------------- 训练 ----

def train_stream_b(pool: str, updates: int = 1500, device: str = "cuda",
                   seed: int = 20266600, smoke: bool = False) -> dict:
    import copy

    import exp.v25 as _v25
    from exp.v19.fiber_duality import build_or_load_v19_foundation
    from exp.v20.slf_flow import coarse_y16_lcn
    from exp.v22.core import PrismCore
    from exp.v20.submode_patchswd import build_or_load_core_knn_bank

    _v25.seed_all(seed)
    bundle = build_or_load_v19_foundation(pool, device=device)
    bank = build_or_load_core_knn_bank(bundle, device=device)
    tr = bundle["v18_cache"]["train"]
    z_all = tr["z"].float()
    d16_all = tr["dino_16"]
    sid_tr = tr["sid"].long()
    N = len(z_all)
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)
    priors = {k: v.to(device) if torch.is_tensor(v) else v for k, v in bundle["fnd"]["priors"].items()}
    y16_all = coarse_y16_lcn(((z_all.to(device) * w_l).sum(1, keepdim=True) + b_l)
                             .clamp(0.0, 1.0)).half()

    host_cfg = dict(ckpt=ROOT / "runs/v23/joint_c1ab/slf_Joint_Pk_C1AB.pt")
    core = PrismCore(bundle=bundle, core_bank=bank, ckpt_path=host_cfg["ckpt"], device=device)
    slf = core.model
    for p in slf.parameters():
        p.requires_grad_(False)
    # 解冻: cond_mlp + 全部 style blocks (enc64/enc32/dec32/dec64/mid*/up/down convs 保留冻结)
    unfreeze = ("cond_mlp", "enc64", "enc32", "dec32", "dec64", "mid1", "mid2", "up32", "up64")
    new_p = []
    for n, p in slf.named_parameters():
        if n.startswith(unfreeze):
            p.requires_grad_(True)
            new_p.append(p)
    bridge = BridgeSLFUNet(slf).to(device)
    bridge.zero_onehot()  # 免标签手术: onehot(5) 通道永久置零
    eref = ERefMLP().to(device)
    opt = torch.optim.AdamW([
        {"params": eref.parameters(), "lr": 3e-4},
        {"params": new_p, "lr": 3e-5}], weight_decay=1e-3)
    ema_slf = copy.deepcopy(slf).eval()
    ema_eref = copy.deepcopy(eref).eval()
    for p in list(ema_slf.parameters()) + list(ema_eref.parameters()):
        p.requires_grad_(False)

    micro, accum = 8, 4
    t0 = time.time()
    for step in range(1, updates + 1):
        scale = 0.2 + 0.4 * (1.0 + math.cos(math.pi * step / max(1, updates)))
        for pg, lr in zip(opt.param_groups, (3e-4, 3e-5)):
            pg["lr"] = lr * scale
        opt.zero_grad(set_to_none=True)
        for ai in range(accum):
            g = torch.Generator().manual_seed(seed * 7919 + step * 97 + ai)
            idx = torch.randint(0, N, (micro,), generator=g).tolist()
            # 异图 (供色度交换)
            alien = torch.randint(0, N, (micro,), generator=g).tolist()
            zx = z_all[idx].to(device).float()
            za = z_all[alien].to(device).float()
            zt = pollute(zx, za, w_l, b_l, g, priors=priors)
            # 参考: 同画非本象限块 (latent) → 去几何统计 + Bridge 色度锚
            H = zx.shape[-1]
            qi_i, qi_j = int(torch.randint(0, 2, (1,), generator=g)), int(torch.randint(0, 2, (1,), generator=g))
            zq = zx[:, :, qi_i * H // 2:(qi_i + 1) * H // 2, qi_j * H // 2:(qi_j + 1) * H // 2]
            u_ref = eref(ref_stats(zq, w_l, b_l))
            # Bridge 起点: 源亮度(降级)+异图色度场; 参考 = 同画象限; 目标 = 原画 z_x
            # 直连目标 = 原画 z_x; v_target = z_x − z0⊥
            tau = torch.rand(micro, device=device)
            z_tau = (1.0 - tau.view(-1, 1, 1, 1)) * z0p + tau.view(-1, 1, 1, 1) * zx
            d16 = d16_all[idx].to(device).float()
            y16 = y16_all[idx].to(device).float()
            slot = torch.zeros(micro, dtype=torch.long, device=device)
            drop_s = torch.rand(micro, device=device) < 0.20  # 20% Style Dropout
            drop_c = torch.zeros(micro, dtype=torch.bool, device=device)
            with torch.autocast("cuda", torch.bfloat16):
                v = bridge(z_tau, y16, d16, slot, u_ref, tau,
                           drop_style=drop_s, drop_content=drop_c,
                           layout_scale=torch.ones(micro, device=device)).float()
            v_target = zx - z0p
            loss = F.mse_loss(v, v_target) / accum
            loss.backward()
        all_p = [p for pg_ in opt.param_groups for p in pg_["params"]]
        grads = [p_.grad for p_ in all_p if p_.grad is not None]
        if grads and not all(bool(torch.isfinite(g_.all())) for g_ in grads):
            opt.zero_grad(set_to_none=True)
            print(f"[StreamB] step {step}: non-finite grads skipped", flush=True)
            continue
        torch.nn.utils.clip_grad_norm_(all_p, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pw in zip(ema_slf.parameters(), slf.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
            for pe, pw in zip(ema_eref.parameters(), eref.parameters()):
                pe.mul_(0.996).add_(pw, alpha=0.004)
        if step == 1 or step % 100 == 0 or step == updates:
            print(f"[StreamB] step {step} loss={float(loss) * accum:.4f} "
                  f"({step / max(time.time() - t0, 1e-6):.2f} ups)", flush=True)
        if smoke and step >= 100:
            break
    return {"bridge": bridge, "ema_slf": ema_slf, "ema_eref": ema_eref,
            "bundle": bundle, "bank": bank, "updates_done": step}


import time  # noqa: E402


# ------------------------------------------------- 探针评测 ----

@torch.no_grad()
def probe_eval(trained: dict, pool: str, device: str = "cuda", n_tasks: int = 96) -> dict:
    from exp.v18 import tasks_and_sources
    from exp.v20.slf_flow import coarse_y16_lcn
    from src.utils import FeatureStack, decode_s10, load_s10_decoder
    from exp.v26.gpu4090.p0_common import _pil

    bundle = trained["bundle"]
    tasks, z_src_750, _ = tasks_and_sources()
    cross = [i for i, t in enumerate(tasks) if t["source_style_id"] != t["target_style_id"]][:n_tasks]
    sel = torch.tensor(cross)
    tid_true = torch.tensor([tasks[i]["target_style_id"] for i in cross])
    c21 = torch.load(ROOT / "runs/v21/cache/v21_stiefel_ref_cache.pt",
                     map_location="cpu", weights_only=False)
    z_ref5 = c21["sg5"]["z"].float().to(device)  # exemplar latents (参考)
    w_l = bundle["v18_cache"]["luma_proj"]["w"].float().view(1, 4, 1, 1).to(device)
    b_l = bundle["v18_cache"]["luma_proj"]["b"].float().to(device)
    z_src = z_src_750[sel].to(device).float()
    d16 = bundle["test_struct"]["dino_16"][sel].to(device).float()
    y16 = coarse_y16_lcn(((z_src * w_l).sum(1, keepdim=True) + b_l).clamp(0.0, 1.0)).float()
    eref = trained["ema_eref"]
    slf = trained["ema_slf"]

    u_ref = eref(ref_stats(z_ref5[tid_true], w_l, b_l))  # 严格 1-shot: 参考统计 → E_ref
    z0p = chroma_align_anchor(z_src, z_ref5[tid_true])  # 免标签锚: P_Y 锁源, P_C 对齐参考
    eps = torch.randn(z_src.shape, generator=torch.Generator().manual_seed(EVAL_SEED)).to(device)
    z = z0p.clone()
    tau0 = 0.0
    steps = 8
    slot = torch.zeros(len(sel), dtype=torch.long, device=device)
    outs = []
    for i in range(steps):
        tau = torch.full((len(sel),), tau0 + (i + 0.5) / steps, device=device)
        v = slf(z, y16, d16, slot, u_ref, tau,
                drop_style=torch.zeros(len(sel), dtype=torch.bool, device=device),
                drop_content=torch.zeros(len(sel), dtype=torch.bool, device=device),
                layout_scale=torch.ones(len(sel), device=device)).float()
        z = z + (1.0 - tau0) / steps * v
    z_gen = z

    dec = load_s10_decoder(device)
    stack = FeatureStack(device)
    src_dino = torch.load(ROOT / "runs/v6_r2/cache/tasks750_source_features.pt",
                          map_location="cpu", weights_only=False)
    pred_all, dc_all = [], []
    for st in range(0, len(sel), 24):
        rgb = decode_s10(dec, z_gen[st:st + 24].to(device).float(), batch=24).clamp(0.0, 1.0)
        feats = stack.extract([_pil(x) for x in rgb], batch=24)
        cs = feats["csd"] @ stack.csd_prototypes.T
        pred_all += cs.argmax(1).tolist()
        stems = [tasks[i]["source_stem"] for i in cross][st:st + 24]
        f_src = torch.stack([torch.as_tensor(src_dino["raw_dino"][s]) for s in stems])
        dc_all += torch.diagonal(feats["dino"] @ f_src.to(feats["dino"].device).T).tolist()
    want = tid_true.tolist()
    top1 = round(sum(int(p == w) for p, w in zip(pred_all, want)) / len(pred_all), 4)
    five = {STYLES[t]: round(sum(1 for p, w in zip(pred_all, want) if w == t and p == t)
                             / max(1, sum(1 for w in want if w == t)), 4) for t in range(5)}
    dino_c = round(sum(dc_all) / len(dc_all), 4)
    rec = dual_gate_record("P0_StreamB_probe",
                           {"csd_top1": top1, "dino_c": dino_c,
                            "top1_per_target_style": five}, None,
                           out_name="p0_dualgate_streamb_probe.json")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    from exp.v26 import sanitize_cuda_env
    sanitize_cuda_env()
    trained = train_stream_b(args.pool, updates=args.updates, device=args.device,
                             smoke=bool(args.smoke))
    rec = probe_eval(trained, pool=args.pool, device=args.device)
    save_json(OUT / "gpu4090" / "p0_stream_b_probe.json",
              {"updated_utc": utc(), "updates": args.updates, "probe": rec})
    print(f"[StreamB] probe → {rec['cross600']['top1']} min_style={rec['cross600']['top1_min_style']}",
          flush=True)


if __name__ == "__main__":
    main()
