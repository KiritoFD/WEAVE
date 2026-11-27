# B07：Minimal PRISM CPU 测试套件整合 —— 覆盖矩阵报告

- 任务：B07（docs/19_pull_together/agent2.md §六），Slot B，2026-11-20
- 交付：`tests/v25/test_b07_regression.py`（8 条缺口测试）＋本报告
- 边界遵守：`tests/v25/__init__.py` 已存在（空文件），未动；未修改任何既有测试文件与被测代码；无 git 命令、无 CUDA、无绝对路径；`OMP_NUM_THREADS=2`。
- 原则：**219→232 条既有测试是资产，只补缺口＋统一入口，不重写**。矩阵中"已覆盖"项一律引用既有测试（文件::测试名），不复制断言。

---

## 1. 覆盖矩阵：agent2.md §六 B07 清单 11 项 × 现有测试文件

| # | B07 清单项 | 状态 | 覆盖证据（文件::测试） |
|---|---|---|---|
| 1 | 非劣性统计边界 | **已覆盖** | `tests/v25/test_stats.py`：`test_verdict_pass_at_exact_band_edges`（边界含端点 PASS）、`test_verdict_fail_when_complex_significantly_better`（FAIL，含单指标触发）、`test_verdict_inconclusive_when_ci_overlaps_band_edge`（边缘重叠 INCONCLUSIVE）、`test_lpips_direction_lower_is_better`（LPIPS 方向）、`test_verdict_rejects_metric_set_mismatch`、`test_compare_arms_guard_downgrades_pass_to_inconclusive`＋`test_same_blind_guard_uses_non_cross_rows`（worst-class/same-blind 守卫降级）、`test_compare_arms_pipeline_and_foundation_equivalence`（与 `bootstrap_source_ci`/`paired_cluster_ci` 同种子逐位等价，共享重采样索引） |
| 2 | γ=0 回退 | **已覆盖** | `tests/v25/test_mechanism.py`：`test_gamma_zero_exact_fallback`（随机权重 `torch.equal` 逐元素＋per-sample γ 向量）、`test_gatenet_zero_init_equals_global_blend`（γ=0 全局混合）；`tests/v25/test_r12.py`：`test_gamma_zero_exact_identity_and_sweep_residual_zero`（扫描 residual 精确 0＋bypass 后 same==z_src） |
| 3 | 空间门控均值与有效编辑能量 | **已覆盖** | `tests/v25/test_mechanism.py`：`test_gate_norm_mean_one_after_clamp`、`test_gate_norm_order_matters`（clamp→mean-norm 次序铁律＋cap 分母下界）、`test_gate_latent_grid_mean_one`（16→64 上采样二次归一析取不变量）、`test_edit_energy_definition_and_monotonicity`（E=γ·\|\|M⊙Δz\|\|/\|\|Δz\|\| 精确定义）、`test_energy_matched_gamma_restores_target`（能量匹配 γ*）、`test_heuristic_gate_finite_and_normalized`；`tests/v25/test_r12.py`：`test_run_mechanism_a_reeval_cpu_mock`（A0 能量恒等于 γ） |
| 4 | 双分支条件保持 | **已覆盖** | `tests/v25/test_mechanism_c.py`：`test_swap_loss_lambda_default_and_linearity`（λ_c 线性）、`test_swap_loss_content_drift_penalty_direction`＋`test_swap_loss_style_direction`（两输出方向数值梯度）、`test_swap_loss_gradient_flows_to_both_outputs`、`test_swap_loss_images_mock_feature_fn`；四条件参考评测 S1/S2 反转触发：`test_eval_ref_conditions_smoke_s1_triggers_on_correct_ref_inversion` / `_s2_triggers_on_own_ref_inversion`；C3 强度混合：`test_mix_teacher_strengths_formula`、`test_build_strength_mix_pack_alignment_formula_and_reproducible` |
| 5 | One-Shot 参考变化 | **部分→已补** | 原有：`test_mechanism_c.py`：`test_ref_feature_cache_hit_and_key_stability`（参考内容 sha 变→新键 miss）、`test_oneshot_integrate_cpu_smoke`（单参考积分烟测）、`test_legacy_style_slot_is_inert_in_reused_model`。缺口（模型输出对参考替换的实际响应）→ **新** `test_b07_regression.py::test_oneshot_output_responds_to_reference_change` |
| 6 | teacher/config hash | **已覆盖** | `tests/v25/test_a02_lineage.py`：`test_solver_hash_deterministic_and_order_independent`、`test_solver_hash_distinguishes_solvers`、`test_lineage_record_identity_key_shape`（model_sha→anchor_sha→solver_hash）、`test_inconsistency_detection_triggers`；`tests/v25/test_registry.py`：`test_param_hash_deterministic_and_order_independent`＋`test_param_hash_matches_a02_solver_hash_convention`/`_v22_config_hash_convention`（单一 hash 约定防两套）；`tests/v25/test_prism_v25.py`：`test_default_core_config_hash_deterministic_and_pinned`（冻结 pin 表）、`test_from_descriptor_restores_all_historical_arms_with_pinned_hash`；`tests/v25/test_mechanism.py`：`test_teacher_spec_mismatch_asserts`、`test_teacher_cache_contract`（缓存 teacher_spec 篡改即 AssertionError） |
| 7 | 损坏缓存 | **部分→已补** | 原有：`test_registry.py::test_corrupt_line_tolerated_on_read`（JSONL 坏行跳过）、`test_result_writer.py`：`test_read_missing_and_corrupt_returns_none`/`test_corrupt_status_json_falls_back_to_log_replay`/`test_torn_jsonl_tail_still_recovers_state`/`test_torn_tmp_file_does_not_affect_original`（JSON/JSONL 容错读）、`test_a02_lineage.py::test_probe_pt_non_torch_file_is_unknown_not_raise`（非 torch .pt 容错探测）。缺口→ **新** `test_b07_regression.py`：`test_ref_feature_cache_corrupt_entry_fails_loud`（特征缓存损坏 .pt 条目响亮失败，绝不静默重编码/删改）、`test_eval_latent_v25_cache_state_branches`（空缓存→再生分支 sentinel 拦截＋H1 半完成窗口响亮 FileNotFoundError） |
| 8 | 重复任务 | **已覆盖（引用）** | `tests/v25/test_stats.py::test_pairing_duplicate_task_id_raises`（a01 配对层 duplicate cross task_id raise）、`test_pairing_mismatched_task_sets_raise`；`tests/v25/test_result_writer.py::test_validate_raw_meta_dict`＋`test_done_refuses_duplicate_ids`（750 唯一 id 闸堵"失败记 done"）；`raw_io.RawWriter.finish` 重复 id raise 由 `b06_recovery_report.md` §1.5 审计记录（评测协议层已有防线，不加测） |
| 9 | 输出 schema | **缺口→已补** | **新** `test_b07_regression.py::test_eval_latent_v25_output_schema_from_cached_summary`：`eval_latent_v25` 返回 dict 的权威 7 键集合（arm/dino_c/dino_s/lpips/style_affinity/csd_top1/same_top1）＋类型＋5 位舍入值逐项核对；CPU 走"raw meta 完整→读 measured summary"分支（合成 summary＋fake OUT，不触 GPU） |
| 10 | Windows 路径 | **部分→已补** | 原有：`test_result_writer.py::test_windows_path_compat`（atomic 写：嵌套/空格/中文/大小写重开/覆盖）、`test_mechanism.py::test_teacher_spec_separator_normalization`（TeacherSpec `\\`↔`/` 判等）、`test_prism_v25.py`（`_norm_separators`）。缺口（血缘注册表跨平台键匹配）→ **新** `test_b07_regression.py::test_registry_lookup_windows_path_key_equivalence`：posix 写（4090/Linux）↔ 反斜杠/大小写查（3060/Windows）命中同一记录（`_path_key` resolve+normcase 契约） |
| 11 | CPU-only 运行 | **已覆盖＋已补全链** | 原有：`test_mechanism.py`/`test_r12.py::test_no_cuda_in_tests`（CVD=''＋device_count）、`test_mechanism_c.py::test_no_cuda_in_tests`（**B02 NVML 陷阱判据：`is_initialized()` 为权威**；并记录 torch.optim.Adam CPU 上也初始化 CUDA context 的教训）、`test_prism_v25.py::test_lazy_import_no_exp_modules_and_device_count_zero`（子进程 lazy import）。缺口→ **新** `test_b07_regression.py::test_cpu_only_full_suite_import_smoke`（CVD='' 子进程按文件路径导入全部 10 个测试模块＋B02 判据断言）＋本文件级 `test_no_cuda_in_tests` 护栏 |

统计：11 项中 **7 项已覆盖**（引用既有测试）、**4 项部分/缺口**（One-Shot 参考变化、损坏缓存、输出 schema、Windows 血缘路径），全部由 8 条新测试补齐。

---

## 2. 新增测试清单（tests/v25/test_b07_regression.py，8 条，全部 CPU）

1. `test_eval_latent_v25_output_schema_from_cached_summary` — 输出 schema 键集/类型/舍入口径（清单 9）。
2. `test_eval_latent_v25_cache_state_branches` — 空缓存→再生分支（哨兵拦截，证明不静默编造）＋H1 窗口（meta 有效、summary 缺失→响亮 FileNotFoundError，与 B06 `scan_status` 的 `summary_missing` 分诊语义一致）（清单 7/9）。
3. `test_ref_feature_cache_corrupt_entry_fails_loud` — `ReferenceFeatureCache` 损坏 .pt 条目：响亮抛错、encode_fn 零调用、磁盘条目不删不改（清单 7）。空缓存 miss→encode 路径已在 `test_mechanism_c` 覆盖，仅引用。
4. `test_registry_lookup_windows_path_key_equivalence` — 血缘注册表 posix/反斜杠/大小写键等价（normcase 为 Windows 语义，Linux 下自动跳过大小写断言，双平台安全）（清单 10）。
5. `test_oneshot_output_responds_to_reference_change` — 参考替换→输出改变、同参考→逐位确定；刻意打破 `SLFUNet.out_conv` 与 `RefAttnV24.out_proj` 两处零初始化热启动头（`test_mechanism` 的 randomize_out 同款模式），否则输出恒 0、参考分支恒 0，无法考察响应性（清单 5）。
6. `test_forged_mean_extrapolated_rows_pass_structural_gate_but_fail_ci_guard` — **A01 验收第 5 条防线**：均值外推常数伪造行（600 cross/150 簇完整协议规模）能通过 a01 全部结构闸（`pair_cross_rows`/`compare_arms` 不抛）并给出"完美 PASS"，但三指标 CI 必然零宽度退化——本文件提供的上游包装 `assert_paired_ci_not_degenerate(res)` 在该形态上抛 ValueError；对照组（含任务级噪声的真实形态行）CI 非退化、闸放行。**只测不改 a01 本体**；包装函数可直接被 B08 提升进 `compare_arms` 调用方（verdict 消费前）。残差已记录：逐簇均值插值形态（非全常数）无自动检测签名，防线仍是 `run_round1` 的 UNDECIDED 口头约定。
7. `test_cpu_only_full_suite_import_smoke` — CVD='' 子进程导入全部 10 个测试模块；`is_initialized()==False` 权威＋`device_count()==0` 辅证（本机全新子进程实测两者成立，与 B02 NVML 判据对齐）（清单 11）。
8. `test_no_cuda_in_tests` — 文件级护栏（对齐姊妹文件约定）。

---

## 3. 入口统一与全量数字

```
python -m pytest tests/v25 -q
  B07 前：232 passed in 22.40s
  B07 后：240 passed in 31.16s   （+8，零失败零跳过，无回归）
python -m pytest tests/v25/test_b07_regression.py -q
  → 8 passed in 13.71s（含 2 个子进程冒烟）
护栏：python -m pytest tests/test_no_absolute_paths.py tests/test_markdown_links.py -q
  → 4 passed（新文件无绝对路径、报告无坏链）
```

一条命令 `python -m pytest tests/v25 -q` 跑通全部 240 条；耗时 31.16s（增量 ~8.8s 主要来自两个子进程冒烟与 2 次 compare_arms 全协议规模 bootstrap）。

---

## 4. 遗留缺口（需要 GPU / 真实产物，列给 B08 与总控）

1. **`eval_latent_v25` 再生成分支的实际执行**（`RawWriterV25` 写批→`run_measure_v25` 子进程→`export_raw_pngs`）：需要 S10 decoder＋GPU；CPU 侧只验证了分支选择逻辑与哨兵拦截。B08 做跨平台运行器时应在 3060/4090 队列加臂级冒烟（跑 1 个真臂后 `scan_status` 全绿）。
2. **measure 全链**（`src.eval.runner.run_measure_subprocess`：DINOv2/CSD/CLIP/LPIPS 裁判加载与推理）：无 CPU 单测可行路径；下游统计已由 `test_stats.py::test_real_round1_primary_comparison_smoke`（有真实 eval.json 的机器自动启用）覆盖。
3. **teacher/config hash 与真实 checkpoint 的绑定**：`test_a02_lineage.py` 已用 skip-if-absent 形态覆盖（B4 ckpt 存在才验），sha 一致性只能在持有 ckpt 的机器（4090）上完成——4090 队列下次跑评测时顺带执行即可。
4. **γ=0 回退 / 双分支条件保持的端到端训练语义**（真实 750 上 loss 下降、bypass 伪迹消失）：属 A08 独立审查范围，非 CPU 单测目标。
5. **`ReferenceFeatureCache` 损坏条目"自动重编"策略**：当前响亮失败（本测试钉死）；若总控决定改为惰性重编需改 `exp/v25/reference_teacher_v2.py`（所有权归总控），改后同步更新本测试语义。

---

## 5. RISKS / 发现

- **(a) `site-packages` 的 `tests` 包遮蔽（新发现）**：本机 `Lib/site-packages/tests` 是常规包，`import tests` 时胜过仓库 namespace package（cwd 只贡献 namespace portion）——`import tests.v25` 字面不可用。pytest 下模块名实为 `v25.test_*`（`tests/` 无 `__init__.py`，pytest 取 `tests/v25` 为包根）。冒烟测试已按文件路径加载，不受影响；但任何新代码禁止 `import tests.v25`（会静默拿到 site-packages 的无关包）。
- **(b) `device_count()==0` 断言的机器相关性**：torch 2.13 NVML 路径下 `device_count()` 在 CVD='' 时可能失真（B02 已知）。本机全新子进程实测为 0（与 `test_prism_v25` lazy-import 判据一致）；若其它机器误报，按断言消息放宽该条，以 `is_initialized()==False` 为权威。
- **(c) 两个 schema/结构 pin 是有意报警器**：`EVAL_LATENT_V25_KEYS` 变化或 `out_conv`/`out_proj` 改名会 fail 并提示同步更新——属预期防护，不是脆弱测试。
- **(d) 增时**：全量套件 22.4s→31.2s，仍在时间盒内；如需提速可把 `test_cpu_only_full_suite_import_smoke` 标 `@pytest.mark.slow`（本次未加，保持"一条命令跑通全部"的验收口径）。

## 6. NEXT_TASK_SUGGESTION（给 B08）

1. 把 `assert_paired_ci_not_degenerate`（本测试文件内，纯函数）提升进 a01 的 `compare_arms` 调用方或 `run_round1`（总控集成时一行 import＋一次调用）。
2. 提案 A（`measure.py:109` eval.json 原子化）＋提案 C（ckpt tmp+replace）＋R4 merge-on-start（B06 报告 §6 遗留）随跨平台运行器一起落地。
3. 3060 队列加臂级 `scan_status` 全绿冒烟（补 §4.1 的 GPU 缺口）。
