cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== _teaser_top100 ==="
find results/_teaser_top100 -maxdepth 2 | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo -n "png: "; find results/_teaser_top100 -type f -name '*.png' | wc -l
echo "=== _teaser_p2a_top100 ==="
find results/_teaser_p2a_top100 -maxdepth 2 | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo -n "png: "; find results/_teaser_p2a_top100 -type f -name '*.png' | wc -l
