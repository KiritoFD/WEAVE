# Tasks: infra 补丁包

- [ ] T1 新增 `tools/apply_infra_patch.py`（old→new 精确替换，支持 check/apply/emit-diff）
- [ ] T2 新增 `frozen_feature_cache.py`（memmap 运行时查找，兼容性自检 + 降级）
- [ ] T3 新增 `tools/build_dino_feature_cache.py`（离线构建 P1 缓存）
- [ ] T4 新增 `runs/eval_one_fast.sh`（P8，不改 `eval_one.sh`）
- [ ] T5 在临时副本执行 `--check` / `--apply` / `py_compile`
- [ ] T6 生成 `docs/113/patches/infra_20261104.patch` 并验证可反向应用
- [ ] T7 更新 `docs/113/INFRA_AUDIT_PATCH.md`：应用顺序、收益/风险、GPU 验证清单、b96 判断
