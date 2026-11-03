set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
mkdir -p results/D5-512/teaser_top100 results/P2A-256/teaser_top100
mv results/_teaser_top100/* results/D5-512/teaser_top100/
mv results/_teaser_p2a_top100/* results/P2A-256/teaser_top100/
rmdir results/_teaser_top100 results/_teaser_p2a_top100
echo "D5-512/teaser_top100: $(ls results/D5-512/teaser_top100 | wc -l) items"
echo "P2A-256/teaser_top100: $(ls results/P2A-256/teaser_top100 | wc -l) items"
echo "=== results top now ==="
ls results
