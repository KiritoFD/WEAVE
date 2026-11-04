#!/usr/bin/env bash
# (1) network vs post-processing attribution: 3 ckpts x 3 AdaIN modes.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
base=archive/runs_submission/canonical_oriented_epoch4/epoch_0004.pt
a3=runs/12h/A3/step_002000.pt
a4=runs/12h/A4/step_002000.pt
mkdir -p runs/12h/attrib
for spec in "base:$base" "A3:$a3" "A4:$a4"; do
  name="${spec%%:*}"; ck="${spec#*:}"
  for mode in net_only last_step every_step; do
    case "$mode" in
      net_only)  ov=configs/12h/infer_net_only.json ;;
      last_step) ov=configs/12h/infer_last_step.json ;;
      every_step) ov=inference.json ;;
    esac
    out="runs/12h/attrib/${name}_${mode}"
    [ -f "$out/dino_summary.json" ] && { echo "SKIP $out"; continue; }
    bash tools/12h/eval_path_attr.sh "attr_${name}_${mode}_b8" "$ck" "$out" 8 "$ov"
    echo "DONE $out"
  done
done
date

