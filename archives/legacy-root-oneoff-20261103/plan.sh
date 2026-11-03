cd /mnt/g/GitHub/Latent_Style/WEAVE
echo "=== D5-512 method dirs ==="
find results/D5-512 -maxdepth 1 -type d | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo "=== R5-512 method dirs ==="
find results/R5-512 -maxdepth 1 -type d | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo "=== R5/images method dirs ==="
find results/R5/images -maxdepth 1 -type d | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo "=== R5-WikiArt method dirs ==="
find results/R5-WikiArt -maxdepth 1 -type d | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo "=== P256 method dirs ==="
find results/P256 -maxdepth 1 -type d | sed 's#.*/##' | sort | tr '\n' ' '; echo
echo
echo "=== R5-512 methods NOT in R5/images ==="
comm -23 <(find results/R5-512 -maxdepth 1 -type d | sed 's#.*/##' | sort) <(find results/R5/images -maxdepth 1 -type d | sed 's#.*/##' | sort)
echo "=== R5/images methods NOT in R5-512 ==="
comm -13 <(find results/R5-512 -maxdepth 1 -type d | sed 's#.*/##' | sort) <(find results/R5/images -maxdepth 1 -type d | sed 's#.*/##' | sort)
echo "=== R5-WikiArt methods NOT in R5/images ==="
comm -23 <(find results/R5-WikiArt -maxdepth 1 -type d | sed 's#.*/##' | sort) <(find results/R5/images -maxdepth 1 -type d | sed 's#.*/##' | sort)
