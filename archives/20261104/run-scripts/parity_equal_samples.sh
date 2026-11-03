#!/usr/bin/env bash
# Equal-sample parity: b1 fp32 x 2800 steps vs the existing b14 bf16 x 200 steps.
# Both see 2800 samples with the same config/seed/lr; compare loss trajectories.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/batch_parity
BASE=/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json
tag=eq_b1_fp32_2800
vlm=$(python3 -c "print(2800*1/5000)")
cfg="$R/config_$tag.json"
python3 - "$cfg" "$tag" "$vlm" <<'PY'
import json, sys
cfg, tag, vlm = sys.argv[1], sys.argv[2], float(sys.argv[3])
d = {"_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/batch_parity/ckpt_%s" % tag},
     "training": {"batch_size": 1, "num_epochs": 11, "seed": 0, "log_interval": 50,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "use_amp": False, "amp_dtype": "fp32",
                  "internal_probe_enabled": False, "internal_early_stop_enabled": False,
                  "numeric_debug": True, "numeric_debug_interval": 1, "numeric_debug_dump_limit": 5000},
     "bridge": {"output_supervision_microbatch": 1},
     "data": {"virtual_length_multiplier": vlm}}
json.dump(d, open(cfg, "w"), indent=2)
PY

echo "== $tag b=1 mb=1 steps=2800 vlm=$vlm ==" | tee -a "$R/parity_equal.log"
timeout 3000 python3 run.py --config "$cfg" > "$R/$tag.log" 2>&1
echo "[rc=$?] $tag" | tee -a "$R/parity_equal.log"

python3 - "$R" <<'PY'
import json
from pathlib import Path
R = Path("/mnt/i/Latent_Style/runs/batch_parity")

def losses(tag):
    p = R / ("ckpt_%s" % tag) / "numeric_debug.jsonl"
    out = []
    if not p.exists():
        return out
    for line in p.read_text(errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("stage") == "forward":
            m = rec.get("metrics", {})
            v = m.get("total", m.get("loss"))
            if v is not None:
                out.append(float(v))
    return out

b1 = losses("eq_b1_fp32_2800")
b14 = losses("p_b14_bf16")
print("b1_records", len(b1), "b14_records", len(b14))
print("b1  first/last/mean", (b1[0], b1[-1], sum(b1) / len(b1)) if b1 else None)
print("b14 first/last/mean", (b14[0], b14[-1], sum(b14) / len(b14)) if b14 else None)
if b1 and b14:
    # b14 step ~8 == 112 samples, step ~72 == 1008, step 200 == 2800
    for b14_idx, b1_idx in ((7, 99), (71, 999), (199, 2799)):
        if b14_idx < len(b14) and b1_idx < len(b1):
            print("match samples=%d b1=%.4f b14=%.4f diff=%+.4f"
                  % ((b14_idx + 1) * 14, b1[b1_idx], b14[b14_idx], b1[b1_idx] - b14[b14_idx]))
    mean_diff = abs(sum(b1) / len(b1) - sum(b14) / len(b14))
    print("mean_abs_loss_diff %.4f (noise band ~0.05)" % mean_diff)
PY
echo PARITY_EQUAL_DONE | tee -a "$R/parity_equal.log"
