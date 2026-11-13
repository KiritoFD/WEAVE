#!/usr/bin/env bash
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
STYLES="Early_Renaissance Impressionism Minimalism Rococo Ukiyo_e"

count() {  # $1 = dir
  if [ -d "$1" ]; then
    find "$1" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' -o -name '*.webp' \) | wc -l
  else
    echo "MISSING"
  fi
}

echo "############ 1) datasets/distinct5_512 (the pool V9 caches were built from) ############"
for split in train val test; do
  TOT=0
  LINE="  distinct5_512/$split:"
  for S in $STYLES; do
    N=$(count "datasets/distinct5_512/$split/$S")
    LINE="$LINE $S=$N"
    if [ "$N" != "MISSING" ]; then TOT=$((TOT+N)); fi
  done
  echo "$LINE  | total=$TOT"
done

echo
echo "############ 2) data/test (frozen Test600 sources) ############"
TOT=0; LINE="  data/test:"
for S in $STYLES; do
  N=$(count "data/test/$S"); LINE="$LINE $S=$N"
  if [ "$N" != "MISSING" ]; then TOT=$((TOT+N)); fi
done
echo "$LINE  | total=$TOT"

echo
echo "############ 3) other 5-style pools present ############"
for D in datasets/wikiart_new_512 datasets/wikiart_random20_512 datasets/distinct5_512; do
  if [ -d "$D" ]; then
    echo "  $D/ :"
    for SUB in "$D"/*/; do
      [ -d "$SUB" ] || continue
      case "$(basename "$SUB")" in
        train|val|test)
          for S in $STYLES; do
            N=$(count "$SUB$S")
            [ "$N" = "0" ] && continue
            echo "    $(basename "$SUB")/$S = $N"
          done
          ;;
        Early_Renaissance|Impressionism|Minimalism|Rococo|Ukiyo_e)
          echo "    $(basename "$SUB") = $(count "$SUB")"
          ;;
      esac
    done
  fi
done

echo
echo "############ 4) D5 latent caches actually used by V8/V9 tooling ############"
find /mnt/i/Latent_Style -maxdepth 4 -name "*.pt" -path "*d5*" 2>/dev/null | head -20
