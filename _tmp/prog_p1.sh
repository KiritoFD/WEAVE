#!/usr/bin/env bash
cd /mnt/i/Latent_Style/WEAVE || exit 1
echo "[$(date -u '+%F %T')] v4_p1_report done: pool1 300图/8040记录/3118s; 8-Pattern共同可行 173/300(57.7%); per-pattern feasible R64 300 G64 217 C64 260 S64 300 F64 300 R128 299 G128 199 C128 269 A_P5 207; supp 内 coverage_actual 恒等于名义值(不可行布局被丢弃)，真实覆盖率偏差只在训练缓存路径(P5 0.2340) -> runs/p2/v4_pattern_report.{md,json}" >> runs/gt/PROGRESS.md
tail -n 3 runs/gt/PROGRESS.md
