echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== trainer ==="
pgrep -af "A1_hf_b256" | head -5
pgrep -af "eval_fast" | head -5
echo "=== ckpts ==="
ls -lt /mnt/i/Latent_Style/WEAVE/runs/1m/A1_hf_b256 2>/dev/null | head -8
echo "=== disk I ==="
df -h /mnt/i | tail -1
