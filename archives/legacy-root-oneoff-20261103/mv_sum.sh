cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== files matching _*.json/_*.csv at results top (count) ==="
ls results/_*.json results/_*.csv 2>/dev/null | wc -l
echo "=== tables dir ==="
find results/tables -maxdepth 2 | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo "=== aaai2027_v4_tables dir ==="
find results/aaai2027_v4_tables -maxdepth 2 | sed 's#.*/##' | sort | tr '\n' ' '; echo
