cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== results top ==="
ls results
echo
for d in D5-512 P2A-256 R5; do
  n=$(find results/$d -maxdepth 1 -mindepth 1 -type d | wc -l)
  png=$(find results/$d -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)
  echo "$d: $n method-dirs, $png images"
done
echo
echo "=== old dirs should NOT exist ==="
for o in results/R5-512 results/R5/images results/R5-WikiArt results/P256 results/ours/R5 results/_teaser_top100 results/_teaser_p2a_top100 results/tables results/aaai2027_v4_tables; do
  [ -e "$o" ] && echo "STILL EXISTS: $o" || echo "gone: $o"
done
echo
echo "total images under results: $(find results -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)"
