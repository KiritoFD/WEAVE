set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "PWD=$(pwd)"
mkdir -p results/R5
echo "moving R5/images/* -> results/R5/"
mv results/R5/images/* results/R5/
rmdir results/R5/images
echo "R5 now has: $(ls results/R5 | tr '\n' ' ')"
echo "removing ours/R5 (exact dup of R5/images)"
rm -rf results/ours/R5
echo "done"
