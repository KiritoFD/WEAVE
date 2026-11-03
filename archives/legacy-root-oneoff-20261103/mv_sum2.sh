set -e
cd /mnt/g/GitHub/Latent_Style/WEAVE
mkdir -p results/summary_tables
# move loose _*.json/_*.csv (excluding dirs)
for f in results/_*.json results/_*.csv; do
  [ -f "$f" ] && mv "$f" results/summary_tables/
done
# move other loose summary csv/jsons (ablation/task2/musiq/phase/tgt/semantic/cfsd/cut/remote)
for f in results/ablation_summary_capacity.csv results/ablation_summary_capacity.json \
         results/ablation_summary_loss.csv results/ablation_summary_loss.json \
         results/task2_1_attention.csv results/task2_1_attention.json \
         results/task2_2_stylefilm.csv results/task2_2_stylefilm.json \
         results/task2_3_endpoint.csv results/task2_3_endpoint.json \
         results/task2_4_gate_init.csv results/task2_4_gate_init.json \
         results/musiq_results_r1r2r3.json results/musiq_results_r4.json results/musiq_verified_clean.json \
         results/r5_musiq_baseline.json results/r5_musiq_sem_r8_ht008.json \
         results/semantic_swd_metrics_summary.json results/semantic_swd_musiq_results.json \
         results/phase32_loss_new_summary.csv results/phase32_loss_new_summary.json \
         results/phase3_minimal.csv results/phase3_minimal.json \
         results/tgt_artfid_random_stability.csv results/tgt_artfid_random_stability.json \
         results/_cfsd_summary.json results/_cut_p256_musiq.json results/_remote_map.json \
         results/_verify_hh.json results/_verify_ladder.json \
         results/artfid_details.json results/artfid_summary.json; do
  [ -f "$f" ] && mv "$f" results/summary_tables/
done
# move table dirs
[ -d results/tables ] && mv results/tables results/summary_tables/tables
[ -d results/aaai2027_v4_tables ] && mv results/aaai2027_v4_tables results/summary_tables/aaai2027_v4_tables
echo "=== summary_tables now ==="
ls results/summary_tables
echo "=== results top now ==="
ls results
