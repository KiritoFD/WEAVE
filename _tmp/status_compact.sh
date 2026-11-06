#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "== NOW $(date -u '+%F %T') UTC =="
echo "-- tmux --"
tmux ls 2>/dev/null
echo "-- py procs --"
ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E "python" | grep -v grep | head -6 | cut -c1-160
echo "-- gpu --"
/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
echo "-- PROGRESS tail --"
tail -n 12 runs/pgt/PROGRESS.md 2>/dev/null
echo "-- b256 dynamics --"
for c in S0_base_b256 S1_positive_b256 S2_reverse_b256; do
  f=runs/pgt/$c/internal_dynamics_summary.json
  if [ -f "$f" ]; then echo "[$c]"; python - "$f" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(" armed=%s fired=%s ep=%s step=%s reason=%s"%(d.get("armed"),d.get("fired"),d.get("fired_epoch"),d.get("global_step"),d.get("stop_reason")))
r=d.get("rho_series") or []; g=d.get("gate_mean_series") or []
if r: print(" rho_last=%s gate_last=%s n_probe=%d"%(r[-1],g[-1] if g else None,len(r)))
PY
  else echo "[$c] no summary"; fi
done
echo "-- b256 logs (last line, truncated) --"
for c in S0_base_b256 S1_positive_b256 S2_reverse_b256; do
  f=runs/pgt/$c/train.log
  if [ -f "$f" ]; then echo "[$c] $(grep -o 'Epoch [0-9]*/[0-9]*\|global_step=[0-9]*\|step [0-9]*/[0-9]*' "$f" | tail -1)"; fi
done
echo "-- b256 ckpts --"
ls -l runs/pgt/S0_base_b256/epoch_*.pt runs/pgt/S1_positive_b256/step_*.pt runs/pgt/S1_positive_b256/epoch_*.pt runs/pgt/S2_reverse_b256/step_*.pt runs/pgt/S2_reverse_b256/epoch_*.pt 2>/dev/null | tail -12
echo "-- markers --"
ls -l runs/pgt/B256_DONE runs/p2/pgt_q0_controls.json runs/p2/pgt_contact_sheet.png runs/p2/pgt_e_direction.json 2>/dev/null
echo "-- p2 metrics json --"
ls -l runs/p2/pgt_*.json 2>/dev/null | tail -12
echo "-- disk --"
df -h /mnt/i | tail -1
