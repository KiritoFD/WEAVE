cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== cross_protocol/cross_protocol ==="
find results/cross_protocol/cross_protocol -maxdepth 2 | sed 's#.*/##' | sort | head -30
echo "=== compare a cross_protocol csv with main to see if dup ==="
echo "--- cross_protocol/D5-512/stylealigned_metrics.csv head ---"
head -3 results/cross_protocol/D5-512/stylealigned_metrics.csv
echo "--- main D5-512 csv list ---"
ls results/D5-512/*.csv
echo "--- does main D5-512 have stylealigned_metrics.csv? ---"
find results/D5-512 -maxdepth 1 -name 'stylealigned_metrics.csv' | wc -l
