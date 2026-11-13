"""Smoke + correctness tests for the five-style trainer on synthetic caches (CPU).

Checks
  * every objective runs, produces finite loss, and saves a loadable checkpoint
  * the ten-pair schedule covers all pairs within each cycle of ten updates
  * the permutation sampler is balanced and never repeats within one epoch
  * shared global standardization is used (no per-style whitening)
  * eps=0 CFM path (exp 11) and no-EDM path (exp 06) run
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from tools.sbflow.data5 import STYLE_NAMES, load_caches, pair_schedule, PermutationSampler


def make_cache(path: Path, style: str, n: int, seed: int, shift: float):
    g = torch.Generator().manual_seed(seed)
    lat = (torch.randn(n, 32, 16, 16, generator=g) * 0.8 + shift).half()
    thumb = (torch.rand(n, 3, 32, 32, generator=g) * 255).byte()
    torch.save({"latent": lat, "thumb": thumb, "ids": [f"{style}_{i}.jpg" for i in range(n)],
                "ae": "synthetic", "style": style, "scaling_factor": 0.4552,
                "image_size": 512}, path)


def test_schedule_covers_pairs():
    pairs = [(a, b) for a in range(5) for b in range(a + 1, 5)]
    sched = pair_schedule(pairs, 40, torch.Generator().manual_seed(0))
    for cycle in range(4):
        block = sched[cycle * 10:(cycle + 1) * 10]
        assert sorted(block) == sorted(pairs), block
    print("PASS test_schedule_covers_pairs")


def test_perm_sampler_balanced():
    s = PermutationSampler(64, torch.Generator().manual_seed(1))
    d = s.draw(64)
    assert len(set(d.tolist())) == 64, "duplicate index inside one epoch"
    d2 = s.draw(100)
    assert len(d2) == 100
    print("PASS test_perm_sampler_balanced")


def test_shared_global_normalization(tmp: Path):
    paths = []
    for i, s in enumerate(STYLE_NAMES):
        p = tmp / f"{s}.pt"
        make_cache(p, s, 48, 100 + i, shift=0.5 * i)
        paths.append(p)
    data = load_caches(paths, "cpu")
    # global mean of the concatenated pool must be ~0 after standardization
    allz = torch.cat([z.mean((0, 2, 3)) for z in data.z], dim=0)
    assert abs(float(allz.mean())) < 0.35, float(allz.mean())
    # per-style means must still differ (no per-style whitening)
    means = torch.stack([z.mean((0, 2, 3)) for z in data.z])
    spread = float(means.mean(dim=1).std())
    assert spread > 0.05, f"per-style differences were erased: spread={spread}"
    print(f"PASS test_shared_global_normalization (per-style mean spread={spread:.3f})")
    return paths


def run_train(paths, tmp: Path, name: str, extra: list) -> dict:
    out = tmp / f"run_{name}"
    cmd = [sys.executable, "-m", "sbflow.train5",
           "--experiment-id", name, "--caches", *[str(p) for p in paths],
           "--output", str(out), "--steps", "6", "--batch", "16",
           "--log-interval", "3", "--save-interval", "6",
           "--width", "32", "--depth", "1", "--heads", "2",
           "--warmup", "2", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       cwd=str(Path(__file__).resolve().parents[3]))
    if r.returncode != 0:
        raise RuntimeError(f"{name} failed:\nSTDOUT\n{r.stdout}\nSTDERR\n{r.stderr}")
    summary = json.loads((out / "train_summary.json").read_text())
    losses = [h["loss"] for h in summary["loss_history"]]
    assert all(abs(x) < 1e6 for x in losses), losses
    ck = torch.load(out / "model_step6.pt", map_location="cpu", weights_only=False)
    assert "model" in ck and "style_stats" in ck and "styles" in ck
    assert ck["model_kwargs"]["num_styles"] == 5
    print(f"PASS train {name:22s} losses={['%.3f' % x for x in losses]} "
          f"params={summary['n_params']}")
    return summary


def main():
    torch.set_num_threads(2)
    test_schedule_covers_pairs()
    test_perm_sampler_balanced()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = test_shared_global_normalization(tmp)
        run_train(paths, tmp, "bidir", ["--objective", "bidir"])
        run_train(paths, tmp, "bidir_joint", ["--objective", "bidir_joint"])
        run_train(paths, tmp, "bidir_heads2", ["--objective", "bidir", "--heads-out", "2"])
        run_train(paths, tmp, "pair_emb", ["--objective", "bidir", "--style-cond", "pair"])
        run_train(paths, tmp, "single_vel", ["--objective", "single_velocity"])
        run_train(paths, tmp, "single_vel_noedm", ["--objective", "single_velocity", "--edm", "0"])
        run_train(paths, tmp, "score_current", ["--objective", "score_current"])
        run_train(paths, tmp, "endpoint", ["--objective", "bidir", "--target", "endpoint"])
        run_train(paths, tmp, "stratified", ["--objective", "bidir", "--time-sampling", "stratified"])
        run_train(paths, tmp, "eps0_cfm", ["--objective", "single_velocity", "--eps", "0.0"])
        run_train(paths, tmp, "ot", ["--objective", "bidir", "--ot-ratio", "0.5"])
        run_train(paths, tmp, "replacement", ["--objective", "bidir", "--sampler", "replacement"])
        run_train(paths, tmp, "localconv_lora", ["--objective", "bidir", "--local-conv",
                                                 "--pair-rank", "4"])
    print("ALL TRAIN5 SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
