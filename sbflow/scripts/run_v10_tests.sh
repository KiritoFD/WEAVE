#!/usr/bin/env bash
# Full V10 battery on the GPU box, including the D/E mechanism tests.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

echo "########## 1. V10 math ##########"
python3 -u sbflow/tests/test_v10_math.py
echo "########## 2. legacy sbflow math ##########"
python3 -u sbflow/tests/test_core.py
python3 -u sbflow/tests/test_infer.py
echo "########## 3. D/E mechanism tests ##########"
python3 -u sbflow/tests/test_v10_de.py
echo "########## 4. train5 smoke (12 arms) ##########"
python3 -u sbflow/tests/test_train5_smoke.py
echo "########## 5. end-to-end ##########"
python3 -u sbflow/tests/test_e2e_v10.py
echo "########## ALL V10 TESTS DONE ##########"
