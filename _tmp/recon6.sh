#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "=== S2 INTERNAL SUMMARY ==="; cat runs/v4/S2_E7/internal_dynamics_summary.json
echo
echo "=== S2 GATE TAIL ==="; grep -iE 'gate|early|stop|rho|shared' runs/v4/S2_E7/train.log | tail -n 30 | cut -c1-400
echo "=== S2 LAST EPOCH LINE ==="; grep -E 'Epoch|epoch' runs/v4/S2_E7/train.log | tail -n 6 | cut -c1-300
echo "=== S1 INTERNAL SUMMARY ==="; cat runs/v4/S1_A_P5/internal_dynamics_summary.json
echo
echo "=== BUILD LOG TIMING ==="; ls -la runs/pgt/build_reverse_e6.log 2>/dev/null; tail -n 5 runs/pgt/build_reverse_e6.log 2>/dev/null | cut -c1-200
echo "=== RENDER LOG S2 ==="; cat runs/v4/S2_E7/render.log
echo "=== EVAL LOG S2 ==="; tail -n 20 runs/v4/S2_E7/eval750.log
