"""LD0-LD4 latent edit runner. No VAE/metrics inside — latents out only.

Arms:
  LD0  identity passthrough (z_N2) — local re-scoring baseline
  LD1  extra R1 steps only (dirs={'r1'})
  LD2  LD1 + region mask (dirs={'r1'}, mask on)
  LD3  local style gradient only, full-domain (dirs={'local'})
  LD4  mask + {r1, local, mix} shortlist, eta backtrack
  LD5  (optional, separate trainer) — not here
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from exp.v8.guidance import SPDHead, normalize, target_margin
from src.infer import load_critic
from .local_style import LocalStyleEnergy
from .direction import DirectionComputer
from .solver import solve_batch

REPO = Path(__file__).resolve().parents[2]
TASKS = json.loads((REPO / "eval_protocol/tasks_750.json").read_text(encoding="utf-8"))["tasks"]
CACHE = REPO / "runs/v9_local_4070/cache/n2c_z.pt"
BANK = REPO / "runs/v9_local_4070/bank/style_bank.pt"
N2C_CKPT = REPO / "runs/v8/train/N2C/step_04000.pt"
OUT = REPO / "runs/v9_local_4070"


def task_meta(i):
    t = TASKS[i]
    return t["task_id"], t["source_style_id"], t["target_style_id"]


def _stem_hash(stem: str) -> int:
    import hashlib
    return int(hashlib.md5(stem.encode()).hexdigest()[:8], 16)


def split_indices(name: str):
    """Deterministic source-disjoint split: dev = stems with md5%5==0 (~20%)."""
    out = []
    for i, t in enumerate(TASKS):
        is_dev = _stem_hash(t["source_stem"]) % 5 == 0
        if is_dev == (name == "dev"):
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["LD0", "LD1", "LD2", "LD3", "LD4"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--mask-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = a.device

    idx_all = split_indices(a.split)
    if a.limit:
        idx_all = idx_all[:a.limit]
    out_dir = OUT / a.split / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() and a.resume else set()
    todo = [i for i in idx_all if TASKS[i]["task_id"] not in done]
    print(f"[{a.arm}/{a.split}] {len(todo)}/{len(idx_all)} tasks, batch={a.batch}")
    if not todo:
        return

    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    tid_of = {t["task_id"]: i for i, t in enumerate(TASKS)}

    critic = None
    head = lse = dcomp = None
    if a.arm != "LD0":
        critic = load_critic(dev)
        ck = torch.load(N2C_CKPT, map_location="cpu", weights_only=False)
        head = SPDHead(rank=int(ck.get("rank", 0))).to(dev).eval()
        head.load_state_dict(ck["head"])
        for p in head.parameters():
            p.requires_grad_(False)
        lse = LocalStyleEnergy(str(BANK), dev)
        dcomp = DirectionComputer(critic, head, lse, dev)

    t0 = time.time()
    bi = 0
    while bi < len(todo):
        group = todo[bi:bi + a.batch]
        rows = [tid_of[TASKS[i]["task_id"]] for i in group]
        # NOTE: TASKS index == cache row index (cache built in task order)
        rows = [i for i in group]
        z_n2 = cache["z_n2"][rows].to(dev)
        z0 = cache["z0"][rows].to(dev)
        z_src = cache["z_src"][rows].to(dev)
        sid = torch.tensor([TASKS[i]["source_style_id"] for i in rows], device=dev)
        tid = torch.tensor([TASKS[i]["target_style_id"] for i in rows], device=dev)

        if a.arm == "LD0":
            z_out = z_n2
            infos = [{"short": bool(s == t)} for s, t in zip(sid.tolist(), tid.tolist())]
        else:
            mode = {"LD1": "r1", "LD2": "r1", "LD3": "local", "LD4": "mix"}[a.arm]
            with torch.enable_grad():
                dirs = dcomp.directions(z_n2, z0, tid, mode)
            if a.arm == "LD4":
                dirs_list = [{"r1": dirs["r1"], "local": dirs["local"],
                              "mix": dirs["mix"]}]
            else:
                dirs_list = [{k: v for k, v in dirs.items() if k != "mix"}]
            mask = None
            if a.arm in ("LD2", "LD4"):
                mask = dcomp.region_mask(z_n2, z0, tid, frac=a.mask_frac)
            # identity tasks: force z_n2 afterwards
            z_out, infos = solve_batch(
                z_n2, z0, z_src, sid, tid, dirs_list, mask,
                critic, lse, eps=a.eps, k_steps=a.k)
            same = sid == tid
            if same.any():
                z_out[same] = z_n2[same]
                for j in same.nonzero(as_tuple=True)[0].tolist():
                    infos[j]["short"] = True

        payload = {"ids": [TASKS[i]["task_id"] for i in rows],
                   "z_src": z_src.cpu(), "z0": z0.cpu(),
                   "z_final": z_out.cpu(),
                   "infos": infos}
        bp = out_dir / f"batch_{bi:04d}.pt"
        torch.save(payload, bp.with_suffix(".tmp"))
        bp.with_suffix(".tmp").replace(bp)
        done.update(payload["ids"])
        done_path.write_text(json.dumps(sorted(done)))
        el = time.time() - t0
        print(f"  {len(done)}/{len(idx_all)} {el:.0f}s "
              f"({el / max(len(done),1):.2f}s/img)", flush=True)
        bi += a.batch
    print(f"[{a.arm}/{a.split}] DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
