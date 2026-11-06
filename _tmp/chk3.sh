#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE
echo "== progress tail =="
tail -8 runs/gt/PROGRESS.md
echo "== early stop lines =="
grep -E "Internal early stop|early_stop|rho|gate" runs/v4/S1_A_P5/train.log | tail -20
echo "== eval timing =="
grep -E "wall|elapsed" runs/v4/S1_A_P5/eval750.log | tail -3
python3 - <<'PY'
import json
rows=json.load(open('/mnt/i/Latent_Style/pseudo_gt_24h/eval_render/S1_A_P5_final/v4_eval750_rows.json'))
print("rows",len(rows),"distinct_src",len({r['src'] for r in rows}),
      "distinct_src_style",len({r['src_style'] for r in rows}),
      "cross",sum(r['cross'] for r in rows))
print("nan dino_s",sum(1 for r in rows if r['dino_s']!=r['dino_s']),
      "nan swd",sum(1 for r in rows if r['swd']!=r['swd']),
      "nan lpips",sum(1 for r in rows if r['lpips']!=r['lpips']),
      "nan clip",sum(1 for r in rows if r['clip_s']!=r['clip_s']))
PY
echo "== ckpt =="
ls -l runs/v4/S1_A_P5/*.pt
