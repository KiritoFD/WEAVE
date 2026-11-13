#!/usr/bin/env bash
# Run the complete V10 engineering test battery on the 3060 before launching suite40.
set -euo pipefail
cd /mnt/i/Latent_Style/WEAVE
export PYTHONPATH=/mnt/i/Latent_Style/WEAVE:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

echo "########## config dry test ##########"
python3 -u sbflow/tests/test_suite40_dry.py
echo "########## compat tests ##########"
python3 -u sbflow/tests/test_suite40_compat.py
echo "########## mechanism activation ##########"
python3 -u sbflow/tests/test_suite40_mechanisms.py
echo "########## V10 math ##########"
python3 -u sbflow/tests/test_v10_math.py
echo "########## legacy math ##########"
python3 -u sbflow/tests/test_core.py
python3 -u sbflow/tests/test_infer.py
echo "########## D/E math ##########"
python3 -u sbflow/tests/test_v10_de.py
echo "########## ALL SUITE40 TESTS DONE ##########"
