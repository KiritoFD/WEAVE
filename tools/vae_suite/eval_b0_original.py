"""Evaluate original SD-VAE (B0) baseline with the SWD metric on the 2048 val set."""
import sys
sys.path.insert(0, "/root/Workspace/WEAVE_VAE")
import json
from pathlib import Path
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tools.vae_suite.dataset import RawMemmapWikiArtDataset
from tools.vae_suite.swd_metric import LatentSWDDescriptor
from tools.vae_suite.audit_multistep import evaluate_checkpoint

calib = json.loads(Path("/root/Workspace/WEAVE_VAE/swd_calibration.json").read_text())
descriptor = LatentSWDDescriptor()
descriptor.set_calibration(
    torch.tensor(calib["chan_mean"]),
    torch.tensor(calib["chan_std"]),
    calib["scale_swd1"],
    calib["scale_swd2"],
    calib["scale_moment"],
    calib["median_inter_dist"],
)

ds = RawMemmapWikiArtDataset(
    Path("/root/Workspace/data/wikiart_512_raw/wikiart_512_raw_manifest.json"),
    is_training=False,
)
loader = DataLoader(ds, batch_size=16, shuffle=False)

vae = AutoencoderKL.from_pretrained(
    "/root/Workspace/xy/DiT/data/pretrained/sd-vae-ft-ema"
).eval()

res = evaluate_checkpoint(vae, descriptor, loader, num_batches=16)

print("===== B0 ORIGINAL SD-VAE BASELINE =====")
print("Overall AUROC      :", res["overall_auroc"])
print("CrossArtist AUROC  :", res["cross_artist_auroc"])
print("kNN-5 Top1         :", res["knn5_top1"])
print("p50 sigma          :", res["p50_sigma"])
print("Mean KL            :", res["mean_kl"])
print("Effective Rank     :", res["effective_rank"])

out = Path("/root/Workspace/WEAVE_VAE/runs_vae/V2/b0_original_baseline.json")
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("[Saved]", out)
