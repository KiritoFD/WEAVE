"""End-to-end smoke test on the GPU box: synthetic 5-style caches -> train5 -> gen5 -> CPU score.

Keeps the model tiny so it finishes in well under a minute and only proves plumbing:
every artifact is produced, loadable, and has the documented fields.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/mnt/i/Latent_Style/WEAVE")
STYLES = ["Early_Renaissance", "Impressionism", "Minimalism", "Rococo", "Ukiyo_e"]


def make_caches(tmp: Path):
    import torch
    paths = []
    for i, s in enumerate(STYLES):
        g = torch.Generator().manual_seed(200 + i)
        p = tmp / f"{s}.pt"
        torch.save({
            "latent": (torch.randn(40, 32, 16, 16, generator=g) * 0.8 + 0.4 * i).half(),
            "thumb": (torch.rand(40, 3, 32, 32, generator=g) * 255).byte(),
            "ids": [f"{s}_{k}.jpg" for k in range(40)],
            "ae": "synthetic", "style": s, "scaling_factor": 0.4552, "image_size": 512,
        }, p)
        paths.append(p)
    return paths


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), **kw)
    if r.returncode != 0:
        print("STDOUT\n", r.stdout[-3000:])
        print("STDERR\n", r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd[:6]))} ...")
    return r


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        caches = make_caches(tmp)

        train_out = tmp / "train"
        print("[e2e] train5 ...", flush=True)
        run([sys.executable, "-m", "sbflow.train5",
             "--experiment-id", "e2e", "--caches", *[str(p) for p in caches],
             "--output", str(train_out), "--steps", "12", "--batch", "16",
             "--width", "32", "--depth", "1", "--heads", "2",
             "--log-interval", "6", "--save-interval", "12", "--warmup", "2"])
        summary = json.loads((train_out / "train_summary.json").read_text())
        ck_path = train_out / "model_step12.pt"
        assert ck_path.exists(), "no checkpoint"
        print(f"[e2e] trained: updates={summary['updates']} pairs={len(summary['unordered_pairs'])} "
              f"exposures={summary['endpoint_exposures']} params={summary['n_params']}", flush=True)

        # gen5 needs a real AE for decode; use --limit 0 with synthetic AE is impossible,
        # so verify the task/manifest logic with a dry structural check instead.
        sys.path.insert(0, str(REPO))
        from sbflow.gen5 import load_tasks
        for subset, expect in (("full600", 600), ("quick200", 200), ("same150", 150)):
            tasks = load_tasks(REPO, subset)
            assert len(tasks) == expect, (subset, len(tasks))
            dirs = {(t["source_style"], t["target_style"]) for t in tasks}
            if subset != "same150":
                assert len(dirs) == 20, (subset, len(dirs))
            print(f"[e2e] subset {subset:9s} -> {len(tasks)} tasks, {len(dirs)} directed pairs", flush=True)

        # identity must not call the network and must return the input unchanged
        import torch
        from sbflow.infer5 import canonical, load_model5, solve_transfer
        _, model, stats = load_model5(str(ck_path), "cpu", "raw")
        z = torch.randn(2, 32, 16, 16)
        out = solve_transfer(model, z, 2, 2, stats, 0.04, steps=4)
        assert torch.allclose(out, z), "identity transfer must be exact"
        print("[e2e] identity transfer exact", flush=True)

        for (i, j) in [(0, 4), (4, 0), (1, 3), (3, 1)]:
            o = solve_transfer(model, z, i, j, stats, 0.04, steps=4, method="heun")
            assert torch.isfinite(o).all(), (i, j)
            a, b, rev = canonical(i, j)
            assert a < b and rev == (i != a)
        print("[e2e] all four sampled directions produce finite latents", flush=True)

    print("E2E SMOKE PASSED")


if __name__ == "__main__":
    main()
