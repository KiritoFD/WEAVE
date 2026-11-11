"""WEAVE W P1 — report generator for score-prior guided editing.

Reads:
  runs/v6_w/score_prior/train_report.json
  runs/v6_w/eval_guided_r1/eval.json
  runs/v6_w/eval_guided_score/eval.json
  runs/v6_w/eval_guided_both/eval.json
  runs/v6_w/w_paired_verdict.json
  runs/v6_r2/eval_B3_distill20/eval.json (anchor)

Writes:
  docs/v6/W_P1_REPORT.md
  runs/v6_w/w_p1_verdict.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "runs/v6_w"
OUT = ROOT / "docs/v6/W_P1_REPORT.md"


def load_eval(rel: str):
    p = W / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"


# Backward-compatible alias used throughout the report templates.
def f(x, n=4):
    return fmt(x, n)


def main():
    train_p = W / "score_prior/train_report.json"
    train = json.loads(train_p.read_text(encoding="utf-8")) if train_p.exists() else None

    evals = {
        "R1 only": load_eval("eval_guided_r1/eval.json"),
        "Score only": load_eval("eval_guided_score/eval.json"),
        "R1 + Score": load_eval("eval_guided_both/eval.json"),
    }
    anchors = {
        "Q2a Direct": load_eval("../v6_r2/eval_Q2a_direct/eval.json"),
        "B3 λ=20": load_eval("../v6_r2/eval_B3_distill20/eval.json"),
        "B4 λ=50": load_eval("../v6_r2/eval_B4_distill50/eval.json"),
        "Q2a + R1 Oracle": load_eval("../v6_r2/eval_Q2a_oracle/eval.json"),
    }

    paired_path = W / "w_paired_verdict.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8")) if paired_path.exists() else {}
    gates = paired.get("gates_vs_B3", {})
    paired_details = paired.get("paired_details_vs_B3", {})

    L = []
    L.append("# WEAVE W P1 报告 — 真实 latent 密度先验 + 引导编辑")
    L.append("")
    L.append("> 协议：`tasks_750.json`（600 cross + 150 same），`paper_rgb_v1`，"
             "原始 RGB 零点，统一 TRT fp16 SD-VAE 解码。")
    L.append("> 本阶段不重训 Oracle、不改评测口径；只测试在冻结生成器上加入"
             "**真实 latent 分布 score 先验**是否能改善 R1 引导。")
    L.append("")
    L.append("## 1. Score prior 训练")
    L.append("")
    if train:
        L.append("| 项目 | 值 |")
        L.append("|---|---:|")
        L.append(f"| 参数量 | {train['n_params']/1e6:.2f}M |")
        L.append(f"| Steps | {train['steps']} |")
        L.append(f"| 训练 latent 数 | {train['n_train']} |")
        L.append(f"| 验证 latent 数 | {train['n_val']} |")
        L.append(f"| 最终训练 loss | {f(train['final_train_loss'])} |")
        L.append(f"| 验证 denoise MSE | {json.dumps(train['val_denoise_mse_by_sigma'])} |")
        L.append(f"| 训练墙钟 | {train['train_wall_sec']}s |")
        L.append(f"| 峰值显存 | {train['peak_vram_mib']}MiB |")
        L.append(f"| Latent pool | `{train['latent_dir']}` |")
    else:
        L.append("- score prior 训练结果缺失。")
    L.append("")

    L.append("## 2. 引导编辑 750 任务结果（Cross600）")
    L.append("")
    L.append("| 方法 | Top-1 | Margin | CSD ΔT | DINO-S | DINO-C | LPIPS | Min 误判 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, rel in [("B3 λ=20", "../v6_r2/eval_B3_distill20/eval.json"),
                      ("B4 λ=50", "../v6_r2/eval_B4_distill50/eval.json"),
                      ("Q2a+Oracle", "../v6_r2/eval_Q2a_oracle/eval.json")]:
        d = load_eval(rel)
        if d:
            s = d["cross600"]
            L.append(f"| {name} | {f(s['top1']*100,1)}% | {f(s['margin'])} | {f(s['dT'])} | "
                     f"{f(s['dino_s'])} | {f(s['dino_c'])} | {f(s['lpips'])} | "
                     f"{f(s['minimalism_offtarget']*100,1)}% |")
    for name, d in evals.items():
        if d:
            s = d["cross600"]
            L.append(f"| {name} | {f(s['top1']*100,1)}% | {f(s['margin'])} | {f(s['dT'])} | "
                     f"{f(s['dino_s'])} | {f(s['dino_c'])} | {f(s['lpips'])} | "
                     f"{f(s['minimalism_offtarget']*100,1)}% |")
        else:
            L.append(f"| {name} | — | — | — | — | — | — | — |")
    L.append("")

    L.append("## 3. 相对 B3 的配对聚类 bootstrap")
    L.append("")
    L.append("| 方法 | ΔTop-1 [95% CI] | ΔDINO-S [95% CI] | ΔDINO-C | ΔLPIPS | P4.3 门槛 |")
    L.append("|---|---:|---:|---:|---:|---|")
    for name in ["R1 only", "Score only", "R1 + Score"]:
        det = paired_details.get(name)
        gate = gates.get(name)
        d = evals[name]
        s = d["cross600"] if d else {}
        if isinstance(det, dict):
            t = det["top1"]; ds = det["dino_s"]; dc = det["dino_c"]; lp = det["lpips"]
            L.append(
                f"| {name} | {f(t['diff']*100,1)}pp [{f(t['ci'][0]*100,1)}, {f(t['ci'][1]*100,1)}] | "
                f"{f(ds['diff'])} [{f(ds['ci'][0])}, {f(ds['ci'][1])}] | {f(dc['diff'])} | "
                f"{f(lp['diff'])} | {gates.get(name,'—')} |")
        else:
            # Older paired-comparison runs stored only the gate string for some arms.
            L.append(f"| {name} | — | — | — | — | {det or '—'} |")
    L.append("")
    L.append("> 详细数值来自 `runs/v6_w/w_paired_verdict.json`；"
             "P4.3 门槛要求 ΔDINO-S ≥ +0.020 且 CI 下界 > 0，同时 DINO-C/LPIPS/Top-1 不明显退化。")
    L.append("")
    L.append("## 4. 判定")
    L.append("")
    any_pass = False
    for name in ["R1 only", "Score only", "R1 + Score"]:
        d = evals[name]
        if d:
            s = d["cross600"]
            ok = s["top1"] >= 0.20 and s["dino_s"] >= 0.4214 + 0.020 and s["dino_c"] >= 0.80
            any_pass = any_pass or ok
    if any_pass:
        L.append("- 有引导编辑配置达到预设的 Top-1/DINO-S/DINO-C 联合门槛，进入下一阶段。")
    else:
        L.append("- **无引导编辑配置达到预设联合门槛**。若 score prior 的 DINO-C 有提升，"
                 "仍可作为机制证据继续；若风格与内容均未改善，应停止该路线。")
    L.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[W-P1] wrote {OUT}")

    verdict = {
        "guided_evals_present": {k: bool(v) for k, v in evals.items()},
        "score_prior_present": train is not None,
        "any_guided_meets_preset_joint_gate": any_pass,
    }
    (W / "w_p1_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"[W-P1] wrote {W/'w_p1_verdict.json'}")


if __name__ == "__main__":
    main()
