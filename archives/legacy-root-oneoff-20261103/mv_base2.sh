set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
find results/baselines -type d -name images -print -exec rm -rf {} +
echo "=== baselines remaining structure ==="
find results/baselines -maxdepth 3 -type d | sed 's#.*/##' | sort | tr '\n' ' '
echo
echo -n "baselines png after cleanup: "; find results/baselines -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l
echo -n "baselines metric files kept: "; find results/baselines -type f \( -name '*.csv' -o -name '*.json' -o -name '*.log' -o -name '*.txt' \) | wc -l
