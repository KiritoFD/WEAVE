#!/bin/bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
nohup bash _tmp/v4_chain2.sh > _tmp/v4_chain2.log 2>&1 &
echo "pid=$!"
sleep 4
tail -n 5 _tmp/v4_chain2.log
