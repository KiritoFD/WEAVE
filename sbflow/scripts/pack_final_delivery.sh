#!/usr/bin/env bash
# Build the final delivery bundle and emit base64.
#   table/    the consolidated results table
#   docs/     the phase reports
#   compact/  aggregate metrics only
#   full/     original JSONs including per-task rows
set -uo pipefail
cd /mnt/i/Latent_Style/WEAVE
OUT=runs/v6_next/deliver_pack
rm -rf "$OUT"; mkdir -p "$OUT/compact" "$OUT/full" "$OUT/docs" "$OUT/table"

python3 - <<'PY'
import json
from pathlib import Path
OUT = Path("runs/v6_next/deliver_pack")

def strip_rows(d):
    if isinstance(d, dict):
        return {k: strip_rows(v) for k, v in d.items() if k != "rows"}
    if isinstance(d, list):
        return [strip_rows(x) for x in d]
    return d

srcs = []
srcs += list(Path("runs/v6_next/amortize").glob("*.json"))
srcs += list(Path("runs/v6_next").glob("eval_750_*/eval.json"))
srcs += list(Path("runs/v6_next").glob("eval_heldout400_*/eval.json"))
srcs += list(Path("runs/v6_next").glob("eval_same150_*/eval.json"))
srcs += list(Path("runs/v6_next/probes").glob("*.json"))
for a in ("Q2a_embed_style", "P2_G0_bf16", "P2_G1a_bf16", "P2_G1b_bf16",
          "P1_B0_ctrl218", "P1_B1_distill218", "P1_B2c_ctrl845",
          "P1_B2_distill845", "P1_B3_distill20", "P1_B4_distill50"):
    p = Path(f"runs/v6/{a}/report.json")
    if p.exists():
        srcs.append(p)

def out_name(p: Path) -> str:
    if p.name in ("eval.json", "report.json"):
        return f"{p.parent.name}.json"
    return f"{p.stem}.json"

seen = {}
for p in srcs:
    if not p.exists():
        continue
    name = out_name(p)
    if name in seen:
        print(f"  COLLISION {name}")
        continue
    seen[name] = p
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    (OUT / "compact" / name).write_text(
        json.dumps(strip_rows(d), ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "full" / name).write_bytes(p.read_bytes())
print(f"json files: {len(seen)}")
PY

cp -f deliver/V6_RESULTS_TABLE.md "$OUT/table/" 2>/dev/null || true
for d in V6_NEXT_P0_AMORTIZE V6_NEXT_P1_DISTILL V6_NEXT_FINAL_REPORT \
         V6_NEXT_P2_RESULTS V6_NEXT_Q1_REPLICATION V6_NEXT_Q2_RESULTS \
         V6_NEXT_RESULTS V6_NEXT_DECISIONS V6_NEXT_STYLE_PLAN \
         V6_NEXT_P0_CORRECTIONS; do
  f="docs/v6/$d.md"
  [ -f "$f" ] && cp -f "$f" "$OUT/docs/"
done
cp -f Negative_Results/L3_implementation/autograd_probe_zeroes_gradient.md "$OUT/docs/" 2>/dev/null || true
cp -f Negative_Results/L3_implementation/fp16_underflow_kills_small_loss.md "$OUT/docs/" 2>/dev/null || true
cp -f Negative_Results/L3_implementation/pred_class_int_vs_str.md "$OUT/docs/" 2>/dev/null || true

cat > "$OUT/README.md" <<'EOF'
# V6 下一阶段 · 交付包

## 目录

| 目录 | 内容 |
|---|---|
| `table/V6_RESULTS_TABLE.md` | **统一结果表**（含每行来源标注） |
| `docs/` | 阶段报告 + 3 篇静默失败记录 |
| `compact/` | 聚合指标（无逐任务行），约 100 KB，用于核对表格 |
| `full/` | 原始 JSON（含逐任务行），约 5 MB，用于复算与再分析 |

## 关键说明

1. **Cross600 指标一律 cross-only**：`eval_tiers` 的行里不含 `is_cross`，
   按 `tgt_style != src_style` 分离。用默认值会把 750 全算成 cross，Top-1 虚高 3.4 倍
   （0.256 vs 0.075）。
2. **Minimalism 误判率**已从错误的 0.000 更正为实测值（Direct ≈ 0.26，Oracle ≈ 0.31，
   ScoreDiff ≈ 0.36）。原 0.000 是 `pred_class` 整数索引与字符串比较的 bug。
3. **Oracle 不是独立模型**：`Q2a_embed_style.pt` + `r1_r1.pt` 两个 checkpoint 缺一不可。
4. 未在本轮测量的行（D1_T1、历史 WEAVE 原版、L11 的 Cross600）在表格末尾单独列出，
   **不与实测行混排**。

## 复现

```bash
bash sbflow/scripts/run_p0_amortize.sh      # P0 五变体
bash sbflow/scripts/run_p1_distill.sh       # P1 B0/B1/B2c/B2
bash sbflow/scripts/run_p1b_lambda.sh       # P1 B3/B4 (λ=20/50)
bash sbflow/scripts/run_p1_750.sh           # 750 全协议
bash sbflow/scripts/run_750_scorediff12.sh  # ScoreDiff N=12
bash sbflow/scripts/gen_delivery_table.sh   # 生成本表
```
EOF

tar czf - -C "$OUT" . | base64 -w0
