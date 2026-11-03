set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== ours/d5_750_endpoint_only/images structure ==="
find results/ours/d5_750_endpoint_only/images -maxdepth 1 | sed 's#.*/##' | sort | tr '\n' ' '
echo
echo "=== png count ==="
find results/ours/d5_750_endpoint_only/images -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l
# move into D5-512 as weave_endpoint_only
mkdir -p results/D5-512/weave_endpoint_only
mv results/ours/d5_750_endpoint_only/images/* results/D5-512/weave_endpoint_only/
rm -rf results/ours/d5_750_endpoint_only
echo "moved. D5-512 now: $(ls results/D5-512 | tr '\n' ' ')"
