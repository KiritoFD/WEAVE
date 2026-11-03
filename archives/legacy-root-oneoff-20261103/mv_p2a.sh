set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "P256 method dirs: $(ls results/P256 | tr '\n' ' ')"
mv results/P256 results/P2A-256
echo "renamed -> P2A-256: $(ls results/P2A-256 | tr '\n' ' ')"
