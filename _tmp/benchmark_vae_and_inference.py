"""Accurate Micro-benchmark on RTX 3060 (12GB):
Measures inference latency breakdown between Model Forward (D1, ScoreDiff) and VAE Decode (TensorRT b8 fp16 vs PyTorch fp16).
"""
import time
import sys
from pathlib import Path
import torch
import numpy as np

REPO_ROOT = Path("/mnt/i/Latent_Style/WEAVE")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools" / "12h"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "trt"))

from tools.v5.route_c import load_vae, load_trt, decode_trt
from tools.v6.models_v6 import C3UNet, ConditionalScoreNet
from tools.v6.canonicalize import ContentCanonicalizer
from tools.v6.score_diff import edit_with_score_difference
from tools.v6.train_eval_v6 import generate_v6

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running latency micro-benchmark on: {torch.cuda.get_device_name(0)}")

# 1. Load VAE and TensorRT engine
vae = load_vae(device)
trt = load_trt(vae, device)

# Warmup CUDA
torch.cuda.synchronize()
_ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
torch.cuda.synchronize()

# -------------------------------------------------------------
# Test VAE Decode: PyTorch Native vs TensorRT
# -------------------------------------------------------------
z1 = torch.randn(1, 4, 64, 64, device=device, dtype=torch.half)
z8 = torch.randn(8, 4, 64, 64, device=device, dtype=torch.half)

# Warmup TRT
for _ in range(10):
    _ = decode_trt(trt, vae, z8, batch=8)
torch.cuda.synchronize()

# Benchmark TRT b8 (batch=8)
n_runs = 50
t0 = time.perf_counter()
for _ in range(n_runs):
    _ = decode_trt(trt, vae, z8, batch=8)
torch.cuda.synchronize()
trt_b8_ms_per_batch = ((time.perf_counter() - t0) / n_runs) * 1000.0
trt_b8_ms_per_img = trt_b8_ms_per_batch / 8.0

# Benchmark PyTorch VAE decode (batch=1)
for _ in range(5):
    with torch.inference_mode():
        _ = vae.decode(z1 / vae.config.scaling_factor).sample
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(n_runs):
    with torch.inference_mode():
        _ = vae.decode(z1 / vae.config.scaling_factor).sample
torch.cuda.synchronize()
pyt_b1_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

# Benchmark PyTorch VAE decode (batch=8)
t0 = time.perf_counter()
for _ in range(n_runs):
    with torch.inference_mode():
        _ = vae.decode(z8 / vae.config.scaling_factor).sample
torch.cuda.synchronize()
pyt_b8_ms_per_batch = ((time.perf_counter() - t0) / n_runs) * 1000.0
pyt_b8_ms_per_img = pyt_b8_ms_per_batch / 8.0

# -------------------------------------------------------------
# Test D1 Model Forward (Single-step Direct Operator)
# -------------------------------------------------------------
global_stats = torch.load(REPO_ROOT / "runs" / "v6" / "cache" / "canonicalizer_stats.pt", map_location="cpu")
canon = ContentCanonicalizer(global_stats).to(device)

d1_model = C3UNet(base_dim=48, num_styles=5, mode="direct").to(device).eval()
d1_ck = torch.load(REPO_ROOT / "runs" / "v6" / "D1_T1_strong" / "D1_T1_strong.pt", map_location="cpu")
d1_model.load_state_dict(d1_ck["model"])

# Warmup D1
t_ids1 = torch.tensor([1], device=device, dtype=torch.long)
t_ids8 = torch.tensor([1]*8, device=device, dtype=torch.long)
for _ in range(10):
    with torch.inference_mode():
        _ = generate_v6(d1_model, canon, z1.float(), t_ids1, canon_mode="c1_gentle")
torch.cuda.synchronize()

# Benchmark D1 batch=1
t0 = time.perf_counter()
for _ in range(n_runs):
    with torch.inference_mode():
        _ = generate_v6(d1_model, canon, z1.float(), t_ids1, canon_mode="c1_gentle")
torch.cuda.synchronize()
d1_b1_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

# Benchmark D1 batch=8
t0 = time.perf_counter()
for _ in range(n_runs):
    with torch.inference_mode():
        _ = generate_v6(d1_model, canon, z8.float(), t_ids8, canon_mode="c1_gentle")
torch.cuda.synchronize()
d1_b8_ms_per_img = (((time.perf_counter() - t0) / n_runs) * 1000.0) / 8.0

# -------------------------------------------------------------
# Test ScoreDiff Model (12 steps and 16 steps)
# -------------------------------------------------------------
sd_model = ConditionalScoreNet(base_dim=64, num_styles=5).to(device).eval()
sd_ck = torch.load(REPO_ROOT / "runs" / "v6" / "score_diff" / "score_diff_net.pt", map_location="cpu")
sd_model.load_state_dict(sd_ck["model"])

s_ids8 = torch.tensor([0]*8, device=device, dtype=torch.long)
task_ids8 = [f"Early_Renaissance/test_{i}/Impressionism" for i in range(8)]

# Warmup ScoreDiff
for _ in range(2):
    _ = edit_with_score_difference(
        sd_model, z8.float(), s_ids8, t_ids8,
        num_steps=12, step_size=0.10, struct_anchor_weight=0.08,
        task_ids=task_ids8,
    )
torch.cuda.synchronize()

# Benchmark ScoreDiff 12 steps (batch=8)
n_runs_sd = 10
t0 = time.perf_counter()
for _ in range(n_runs_sd):
    _ = edit_with_score_difference(
        sd_model, z8.float(), s_ids8, t_ids8,
        num_steps=12, step_size=0.10, struct_anchor_weight=0.08,
        task_ids=task_ids8,
    )
torch.cuda.synchronize()
sd12_b8_ms_per_img = (((time.perf_counter() - t0) / n_runs_sd) * 1000.0) / 8.0

# Benchmark ScoreDiff 16 steps (batch=8)
t0 = time.perf_counter()
for _ in range(n_runs_sd):
    _ = edit_with_score_difference(
        sd_model, z8.float(), s_ids8, t_ids8,
        num_steps=16, step_size=0.12, struct_anchor_weight=0.08,
        task_ids=task_ids8,
    )
torch.cuda.synchronize()
sd16_b8_ms_per_img = (((time.perf_counter() - t0) / n_runs_sd) * 1000.0) / 8.0

print("\n=== LATENCY BENCHMARK RESULTS (Per Image, ms) ===")
print(f"1. VAE Decode (TensorRT b8 fp16, amortized per img) : {trt_b8_ms_per_img:6.2f} ms")
print(f"   VAE Decode (PyTorch fp16, batch=1)              : {pyt_b1_ms:6.2f} ms")
print(f"   VAE Decode (PyTorch fp16, batch=8 amortized)    : {pyt_b8_ms_per_img:6.2f} ms")
print(f"2. D1 Model Forward (Direct Operator, batch=1)     : {d1_b1_ms:6.2f} ms")
print(f"   D1 Model Forward (Direct Operator, batch=8)     : {d1_b8_ms_per_img:6.2f} ms")
print(f"3. ScoreDiff 12-Step (Iterative edit, batch=8)     : {sd12_b8_ms_per_img:6.2f} ms")
print(f"   ScoreDiff 16-Step (Iterative edit, batch=8)     : {sd16_b8_ms_per_img:6.2f} ms")

print("\n=== END-TO-END INFERENCE BREAKDOWN & VAE TIME RATIO ===")

# Case A: D1 (Batch=8, TRT)
e2e_d1_trt = d1_b8_ms_per_img + trt_b8_ms_per_img
pct_d1_trt = (trt_b8_ms_per_img / e2e_d1_trt) * 100.0
print(f"[D1 + TensorRT b8] Total: {e2e_d1_trt:.2f} ms | VAE Decode: {trt_b8_ms_per_img:.2f} ms ({pct_d1_trt:.1f}%) | Model: {d1_b8_ms_per_img:.2f} ms ({100-pct_d1_trt:.1f}%)")

# Case B: D1 (Batch=1, PyTorch Native VAE)
e2e_d1_pyt = d1_b1_ms + pyt_b1_ms
pct_d1_pyt = (pyt_b1_ms / e2e_d1_pyt) * 100.0
print(f"[D1 + PyTorch Native b1] Total: {e2e_d1_pyt:.2f} ms | VAE Decode: {pyt_b1_ms:.2f} ms ({pct_d1_pyt:.1f}%) | Model: {d1_b1_ms:.2f} ms ({100-pct_d1_pyt:.1f}%)")

# Case C: D1 (Batch=8, PyTorch Native VAE)
e2e_d1_pyt8 = d1_b8_ms_per_img + pyt_b8_ms_per_img
pct_d1_pyt8 = (pyt_b8_ms_per_img / e2e_d1_pyt8) * 100.0
print(f"[D1 + PyTorch Native b8] Total: {e2e_d1_pyt8:.2f} ms | VAE Decode: {pyt_b8_ms_per_img:.2f} ms ({pct_d1_pyt8:.1f}%) | Model: {d1_b8_ms_per_img:.2f} ms ({100-pct_d1_pyt8:.1f}%)")

# Case D: ScoreDiff 12-Step (Batch=8, TRT)
e2e_sd12_trt = sd12_b8_ms_per_img + trt_b8_ms_per_img
pct_sd12_trt = (trt_b8_ms_per_img / e2e_sd12_trt) * 100.0
print(f"[ScoreDiff 12-step + TRT b8] Total: {e2e_sd12_trt:.2f} ms | VAE Decode: {trt_b8_ms_per_img:.2f} ms ({pct_sd12_trt:.1f}%) | Model: {sd12_b8_ms_per_img:.2f} ms ({100-pct_sd12_trt:.1f}%)")

# Case E: ScoreDiff 16-Step (Batch=8, TRT)
e2e_sd16_trt = sd16_b8_ms_per_img + trt_b8_ms_per_img
pct_sd16_trt = (trt_b8_ms_per_img / e2e_sd16_trt) * 100.0
print(f"[ScoreDiff 16-step + TRT b8] Total: {e2e_sd16_trt:.2f} ms | VAE Decode: {trt_b8_ms_per_img:.2f} ms ({pct_sd16_trt:.1f}%) | Model: {sd16_b8_ms_per_img:.2f} ms ({100-pct_sd16_trt:.1f}%)")
