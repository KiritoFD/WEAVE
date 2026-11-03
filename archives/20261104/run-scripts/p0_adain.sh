#!/usr/bin/env bash
# P0.1 Inference AdaIN ablation: same checkpoint, same 750-image protocol,
#   arm A "adain_on"  -> config.json (endpoint_adain_scale = 1.0, per-step stats matching)
#   arm B "adain_off" -> endpoint_adain_scale = 0.0 (pure network output)
# The delta is how much instance-level stylisation the network itself learned.
set -u
cd /mnt/i/Latent_Style/WEAVE || exit 1
R=/mnt/i/Latent_Style/runs/p0_diag
mkdir -p "$R"
python3 - "$R/config_adain0.json" <<'PY'
import json, sys
d = json.load(open('/mnt/i/Latent_Style/WEAVE/config.json'))
d.setdefault('model', {})['endpoint_adain_scale'] = 0.0
d.setdefault('model', {})['style_extrap_alpha'] = 0.0
json.dump(d, open(sys.argv[1], 'w'), indent=2)
print('[p0] adain-off config ->', sys.argv[1])
PY
CKPT=exp/dino_s_break/brk_a_ll03_10ep/epoch_0010.pt
bash /mnt/i/Latent_Style/runs/eval_one.sh adain_on  "$CKPT" "$R" /mnt/i/Latent_Style/WEAVE/config.json
bash /mnt/i/Latent_Style/runs/eval_one.sh adain_off "$CKPT" "$R" "$R/config_adain0.json"
echo P0_ADAIN_DONE
