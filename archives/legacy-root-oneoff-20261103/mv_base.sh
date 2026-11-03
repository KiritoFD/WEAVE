set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== all 'images' dirs under baselines ==="
find results/baselines -type d -name images
echo "=== total baselines images png (before) ==="
find results/baselines -type d -name images -exec sh -c 'echo -n "$1: "; find "$1" -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l' _ {} \;
echo "=== baselines top-level metric files (keep) sample ==="
find results/baselines -maxdepth 3 -type f \( -name '*.csv' -o -name '*.json' -o -name '*.log' -o -name '*.txt' \) | sed 's#.*/##' | sort -u | head -20
