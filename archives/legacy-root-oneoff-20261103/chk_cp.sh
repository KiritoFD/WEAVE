cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== real cross_protocol top ==="
ls -la results/cross_protocol/
echo "=== all_summary.json exists? ==="
ls -la results/cross_protocol/all_summary.json 2>/dev/null
echo "=== any dirs under cross_protocol ==="
find results/cross_protocol -maxdepth 1 -type d | sed 's#.*/##'
