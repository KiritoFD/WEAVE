#!/usr/bin/env bash
# Run the full V10 test battery on the GPU box.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

echo "########## 1. V10 math tests ##########"
python3 -u sbflow/tests/test_v10_math.py

echo "########## 2. legacy sbflow math tests ##########"
python3 -u sbflow/tests/test_core.py
python3 -u sbflow/tests/test_infer.py

echo "########## 3. train5 smoke (all objectives/ablations) ##########"
python3 -u sbflow/tests/test_train5_smoke.py

echo "########## 4. end-to-end train -> infer ##########"
python3 -u sbflow/tests/test_e2e_v10.py

echo "########## ALL V10 TESTS DONE ##########"
