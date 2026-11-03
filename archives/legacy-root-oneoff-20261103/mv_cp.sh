cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== cross_protocol top (find) ==="
find results/cross_protocol -maxdepth 1 | sed 's#.*/##' | sort
echo "=== cross_protocol/D5-512 files ==="
find results/cross_protocol/D5-512 -type f 2>/dev/null | head
echo "=== cross_protocol/P2A-256 files ==="
find results/cross_protocol/P2A-256 -type f 2>/dev/null | head
echo "=== main D5-512 csv count ==="
find results/D5-512 -maxdepth 1 -name '*.csv' | wc -l
echo "=== main P2A-256 csv count ==="
find results/P2A-256 -maxdepth 1 -name '*.csv' | wc -l
