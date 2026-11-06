echo "=== tools ==="
command -v tmux; command -v screen; command -v nohup; command -v conda
echo "=== python ==="
python3 -c "import sys; print('py', sys.executable, sys.version)" 2>&1 | head -3
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))" 2>&1 | head -3
echo "=== conda envs ==="
ls /root/miniconda3/envs 2>/dev/null; ls /opt/conda/envs 2>/dev/null
echo "=== project ==="
ls /mnt/i/Latent_Style/WEAVE | head -30
echo "=== eval proc ==="
ps -o pid,etime,rss,cmd -p 1526906 2>/dev/null | cut -c1-140
echo "=== root ==="
ls /root | head -20
