#!/usr/bin/env bash
# Batch/AMP parity check: 200-step loss trajectory at batch=1 fp32 vs batch=14 bf16-AMP.
# Also repeats batch=14 with seed 1 to bound run-to-run noise.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/mnt/i/Latent_Style/runs/batch_parity
mkdir -p "$R"
BASE=/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json
: > "$R/parity.jsonl"

run_one () {
  tag="$1"; b="$2"; mb="$3"; seed="$4"; amp="$5"; dtype="$6"
  vlm=$(python3 -c "print(200*$b/5000)")
  cfg="$R/config_$tag.json"
  python3 - "$cfg" "$tag" "$b" "$mb" "$seed" "$amp" "$dtype" "$vlm" <<'PY'
import json, sys
cfg, tag, b, mb, seed, amp, dtype, vlm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7], float(sys.argv[8])
d = {"_base": "/mnt/i/Latent_Style/runs/nextgen_routeA_post/config_routeA.json",
     "checkpoint": {"save_dir": "/mnt/i/Latent_Style/runs/batch_parity/ckpt_%s" % tag},
     "training": {"batch_size": b, "num_epochs": 11, "seed": seed, "log_interval": 1,
                  "use_tqdm": False, "resume_optimizer": False,
                  "learning_rate": 5e-5, "min_learning_rate": 5e-5,
                  "use_amp": amp == "1", "amp_dtype": dtype,
                  "numeric_debug": True, "numeric_debug_interval": 1, "numeric_debug_dump_limit": 400},
     "bridge": {"output_supervision_microbatch": mb,
                "output_supervision_extractor": "dinov2",
                "output_supervision_extractor_local_dir": "/mnt/i/Latent_Style/weights/dinov2-base",
                "output_supervision_feature_res": 384,
                "output_supervision_chunk_rows": 4,
                "output_supervision_patch_size": 3,
                "output_supervision_patch_stride": 3,
                "output_supervision_style_layers": "block4,block8,block11",
                "output_supervision_content_layers": "block2,block4",
                "output_supervision_selfsim_layer": "block8"},
     "data": {"virtual_length_multiplier": vlm}}
json.dump(d, open(cfg, "w"), indent=2)
PY
  echo "== $tag b=$b mb=$mb seed=$seed amp=$amp $dtype vlm=$vlm ==" | tee -a "$R/parity.log"
  timeout 1800 python3 run.py --config "$cfg" > "$R/$tag.log" 2>&1
  echo "[rc=$?] $tag" | tee -a "$R/parity.log"
}

run_one p_b1_fp32   1  1  0 0 fp32
run_one p_b14_bf16 14 14  0 1 bf16
run_one p_b14_s1   14 14  1 1 bf16
echo PARITY_DONE | tee -a "$R/parity.log"

python3 - "$R" <<'PY'
import json, re
from pathlib import Path
R = Path("/mnt/i/Latent_Style/runs/batch_parity")
for tag in ("p_b1_fp32", "p_b14_bf16", "p_b14_s1"):
    n = 0; fwd = []
    p = R / f"ckpt_{tag}" / "numeric_debug.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try: rec = json.loads(line)
            except Exception: continue
            if rec.get("stage") == "forward":
                m = rec.get("metrics", {})
                fwd.append(m.get("total") or m.get("loss"))
    log = (R / f"{tag}.log").read_text(errors="ignore")
    lr = set(re.findall(r"lr=([0-9.e-]+)", log))
    print(tag, "fwd_records", len(fwd), "first", fwd[0] if fwd else None,
          "last", fwd[-1] if fwd else None, "lr", sorted(lr)[:2])
PY
