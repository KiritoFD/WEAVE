set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE

# --- R5-512 ---
# unique method sdturbo_r5 -> move
if [ -d results/R5-512/sdturbo_r5 ]; then mv results/R5-512/sdturbo_r5 results/R5/; fi
# weave_eval variant (R5-512/weave is structured differently, keep as weave_eval)
if [ -d results/R5-512/weave ]; then mv results/R5-512/weave results/R5/weave_eval; fi
# remove duplicate methods present in R5 (verified identical)
for m in adain adain_stytr2 aespa idt latent_wct sdturbo stylealigned styleid styleshot stytr2 tgt wct wct_stytr2 zstar; do
  if [ -d results/R5-512/$m ]; then rm -rf results/R5-512/$m; fi
done
# remove leftover non-method files
rm -rf results/R5-512
echo "R5-512 handled"

# --- R5-WikiArt ---
# unique method seedream -> move
if [ -d results/R5-WikiArt/seedream ]; then mv results/R5-WikiArt/seedream results/R5/; fi
# weave_wikiart variant
if [ -d results/R5-WikiArt/weave ]; then mv results/R5-WikiArt/weave results/R5/weave_wikiart; fi
# duplicate methods
for m in aespa cut samam samst sdturbo stylealigned styleid styleshot stytr2 tgt zstar; do
  if [ -d results/R5-WikiArt/$m ]; then rm -rf results/R5-WikiArt/$m; fi
done
rm -rf results/R5-WikiArt
echo "R5-WikiArt handled"

echo "R5 final methods: $(ls results/R5 | tr '\n' ' ')"
