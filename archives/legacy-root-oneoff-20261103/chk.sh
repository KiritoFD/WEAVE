set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== D5-512/weave_endpoint_only top ==="
find results/D5-512/weave_endpoint_only -maxdepth 1 | sed 's#.*/##' | sort | tr '\n' ' '
echo
echo -n "nested images/ png: "; find results/D5-512/weave_endpoint_only/images -type f -name '*.png' 2>/dev/null | wc -l
echo -n "top-level png: "; find results/D5-512/weave_endpoint_only -maxdepth 1 -type f -name '*.png' | wc -l
