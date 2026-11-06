#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
for arm in S1_positive_b256 S2_reverse_b256; do
  echo "========== $arm"
  echo "-- epochs:"; ls -1 runs/pgt/$arm | grep -c epoch_ || true
  echo "-- step ckpts:"; ls -1 runs/pgt/$arm | grep step_ || true
  echo "-- config (training/bridge/data/ckpt keys):"
  python3 - <<PY
import json,pathlib
p=pathlib.Path("runs/pgt/$arm/config.json")
c=json.loads(p.read_text())
def pick(sec,keys):
    d=c.get(sec,{})
    print(sec, {k:d.get(k) for k in keys})
pick("training",["batch_size","accumulation_steps","learning_rate","min_learning_rate","scheduler","num_epochs","stop_after_global_steps","save_step_milestones","seed","use_amp","num_workers","resume_checkpoint","resume_optimizer","resume_training_state"])
pick("bridge",["paired_latent_passthrough","t_sampling_mode","training_objective_mode","spectral_w_ll","spectral_w_lh","spectral_w_hl","objective_mode","structure_aligned_target","ll_partial_style_enabled","ll_partial_alpha"])
pick("data",["paired_latent_cache_dir","latent_cache_dir","train_sample_limit","content_style_sampling_weights","target_style_sampling_weights"])
pick("model",["inference_adain","endpoint_adain_scale","target_latent_hf_subband_fusion_enabled"])
print("ablation",c.get("ablation"))
PY
  echo "-- train.log head:"; head -n 12 runs/pgt/$arm/train.log
  echo "-- train.log tail:"; tail -n 14 runs/pgt/$arm/train.log
done
echo "========== PROGRESS/gt"
cat runs/gt/PROGRESS.md 2>/dev/null | tail -n 40
