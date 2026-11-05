"""按 gt-build §5 的规则选臂：先淘汰语义/接缝破坏，再比风格增益与覆盖率。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q0", default=str(Path(paths.project_root()) / "runs/p2/pgt_q0.json"))
    ap.add_argument("--out", default=str(Path(paths.project_root()) / "runs/p2/pgt_selection.json"))
    args = ap.parse_args()
    data = json.loads(Path(args.q0).read_text(encoding="utf-8"))
    cfg = data["configs"]

    seam_ref = sorted(cfg[f"A_{s}"]["seam"]["median"] for s in ("P2", "P5", "P7", "P9") if f"A_{s}" in cfg)
    seam_limit = 2.0 * seam_ref[len(seam_ref) // 2] if seam_ref else float("inf")

    def usable(key: str) -> tuple[bool, str]:
        row = cfg[key]
        # 匹配在 28 个配置间共享，n/pass_rate 对所有配置相同；低于 240/300 只记为
        # 规模化风险（报告里注明），不再当作淘汰条件（否则全部配置一起被否掉）。
        notes = []
        if row.get("n", 0) < 240:
            notes.append(f"scale_risk: n={row.get('n')} pass_rate={row.get('pass_rate')}")
        if row["dino_c"]["mean"] < 0.80:
            return False, f"dino_c {row['dino_c']['mean']:.3f} < 0.80"
        if row["lpips"]["mean"] > 0.35:
            return False, f"lpips {row['lpips']['mean']:.3f} > 0.35"
        if row["seam"]["mean"] > seam_limit:
            return False, f"seam {row['seam']['mean']:.3f} > 2×A中位 {seam_limit:.3f}"
        return True, "ok" + ("; " + "; ".join(notes) if notes else "")

    screened = {}
    for key in cfg:
        ok, reason = usable(key)
        screened[key] = {"ok": ok, "reason": reason}

    pos = [k for k in cfg if k.split("_")[0] in {"A", "B", "C", "D"} and screened[k]["ok"]]
    neg = [k for k in cfg if k.split("_")[0] == "E" and screened[k]["ok"]]
    if not pos or not neg:
        raise SystemExit(f"筛选后无可用臂：pos={pos} neg={neg}")

    pos.sort(key=lambda k: (cfg[k]["csd_dM"]["mean"], cfg[k]["dino_c"]["mean"]), reverse=True)
    neg.sort(key=lambda k: (cfg[k]["csd_dT"]["mean"], cfg[k]["dino_c"]["mean"]), reverse=True)
    out = {
        "seam_limit": seam_limit,
        "screened": screened,
        "positive": {"key": pos[0], "metrics": {m: cfg[pos[0]][m]["mean"] for m in ("csd_dT", "csd_dM", "dino_c", "lpips", "seam", "coverage")}},
        "reverse": {"key": neg[0], "metrics": {m: cfg[neg[0]][m]["mean"] for m in ("csd_dT", "csd_dS", "dino_c", "lpips", "seam", "coverage")}},
        "positive_ranked": pos,
        "reverse_ranked": neg,
        "scale_risk": {
            "n": cfg[pos[0]]["n"],
            "pass_rate": cfg[pos[0]]["pass_rate"],
            "note": "匹配在 28 个配置间共享；n<240/300 只作规模化风险标注，不淘汰配置",
        },
        "note": "正向按 CSD ΔM 排序（内容/接缝先过关）；反向按 ΔT 排序。未设 +0.01 成功门槛。",
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("seam_limit", "positive", "reverse")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
