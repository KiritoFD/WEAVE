#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
mkdir -p runs/p2
# 旧原型是用 2 张源图排除集建的，全量前必须删掉按 300 源图重建
rm -f /mnt/i/Latent_Style/pseudo_gt_24h/csd_proto.pt
python3 -u tools/pseudo_gt/q0_quality.py --out runs/p2/pgt_q0.json 2>&1 | tee runs/p2/pgt_q0.log
echo EXIT=$?
