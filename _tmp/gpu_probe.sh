echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "=== compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "=== procs ==="
ps aux | grep -E "trainer|A1_hf|preencode" | grep -v grep | head -20
echo "=== tmux ==="
tmux ls 2>&1 | head -20
echo "=== home ==="
pwd
ls
