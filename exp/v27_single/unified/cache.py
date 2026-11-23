"""Materialize the full paired teacher set. Cached gate examples are reused exactly."""

import argparse

import torch

from exp.v20.submode_patchswd import build_or_load_core_knn_bank
from exp.v21.stiefel_ref_cache import build_or_load_v21_cache
from exp.v27_mainline import OUT, ROOT
from exp.v27_single.unified.teacher_gate import DEST, SEED, load_teachers, make_endpoints, task_plan
from src.utils.runtime import read_json, save_json

RUN = OUT / "prism1_unified"
PAIR_PATH = RUN / "pairs.pt"


def atomic_save(payload, path):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", type=int, default=24, choices=(8, 16, 20, 24))
    args = parser.parse_args()
    if PAIR_PATH.exists():
        return
    RUN.mkdir(parents=True, exist_ok=True)
    pack = build_or_load_v21_cache(str(ROOT.parent / "pseudo_gt_24h/orig_mean_fp16"))
    bundle, c21 = pack["bundle"], pack["c21"]
    bank = build_or_load_core_knn_bank(bundle, k_sub=8, core_per_style=180, device="cuda")
    train = bundle["v18_cache"]["train"]
    selected, _, _ = task_plan(train)
    src = selected.repeat_interleave(5)
    tid = torch.arange(5).repeat(len(selected))
    same = train["sid"][src] == tid
    gate = read_json(DEST / "results.json")
    pilot = torch.load(DEST / "pilot_endpoints.pt", map_location="cpu", weights_only=False)
    keys = {int(s)*5+int(t): i for i, (s, t) in enumerate(zip(src, tid))}
    old = torch.tensor([keys[int(s)*5+int(t)] for s, t in zip(pilot["src"], pilot["tid"])])
    needs = ~same
    needs[old] = False
    remaining = needs.nonzero().flatten()
    chunks = []
    teachers = None
    base_batch = args.micro
    print(f"PAIR PLAN {len(src)} total, {int(same.sum())} identity, {len(old)} reused, {len(remaining)} new. "
          f"Teacher gate passed={gate['passed']}, continuing by user instruction.", flush=True)
    for i, ids in enumerate(remaining.split(256)):
        path = RUN / f"cache/teacher_{i:03d}.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if teachers is None:
                teachers = load_teachers(bundle)
            gen = torch.Generator().manual_seed(SEED + 100 + i)
            eps = torch.randn((len(ids), 4, 64, 64), generator=gen).half()
            while True:
                try:
                    payload = make_endpoints(bundle, c21, bank, src[ids], tid[ids],
                                             teachers=teachers, eps=eps, batch=base_batch)
                    break
                except torch.cuda.OutOfMemoryError:
                    if base_batch == 8:
                        raise
                    base_batch = {24: 20, 20: 16, 16: 8}[base_batch]
                torch.cuda.empty_cache()
                print(f"Teacher micro-batch lowered to {base_batch}", flush=True)
            atomic_save(payload, path)
        chunks.append((ids, path))
        print(f"CACHE {min((i+1)*256,len(remaining))}/{len(remaining)} new cross (micro={base_batch})", flush=True)
    fields = {name: train["z"][src].half().clone() for name in ("hiC", "hiS")}
    for name in fields:
        fields[name][old] = pilot[name]
    for ids, path in chunks:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.equal(payload["src"], src[ids]) or not torch.equal(payload["tid"], tid[ids]):
            raise ValueError(f"Teacher cache order differs at {path}")
        for name in fields:
            fields[name][ids] = payload[name]
    atomic_save({"src": src, "tid": tid, "same": same, **fields}, PAIR_PATH)
    save_json(RUN / "pairs_meta.json", {"pairs": len(src), "cross": int((~same).sum()),
                                       "identity": int(same.sum()), "teacher_gate_passed": gate["passed"],
                                       "pilot_reused": len(old), "seed": SEED})
    print(f"CACHE COMPLETE {PAIR_PATH}", flush=True)


if __name__ == "__main__":
    main()
