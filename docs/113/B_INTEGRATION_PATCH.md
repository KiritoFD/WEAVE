# Route B 集成补丁草稿（默认关闭，未合并）

状态：**草稿，不落盘到 model/trainer/flow/config_schema**。另一条训练线正在改同一批文件，
本文件只给出精确插入位置与 diff，等训练线空窗期再由维护者合并。

安全保证：
- 新增模块 `b1_style_operator.py` / `b2_destyle.py` / `b_pseudo_targets.py` /
  `b_training_diagnostics.py` 只被显式导入，导入时不改变任何现有前向；
- 下列补丁的每个入口默认 `enabled=False`，默认路径与现状逐位一致；
- 路由 B 训练时**不做教师前向**：DINO/VGG 特征只从
  `OfflineStyleFeatureCache`（`b_pseudo_targets.py`）读取。

---

## 1. `config_schema.py`

### 1.1 新增 `RouteBConfig`（插入到 `BridgeConfig` 之后、`TrainingConfig` 之前，约第 941 行）

```diff
@@ after class BridgeConfig.to_dict(...)
+@dataclass
+class RouteBConfig:
+    """Route B: content-anchored operator T_phi + pseudo targets. Default OFF."""
+
+    enabled: bool = False
+    stage: int = 1                     # 1=train T_phi, 2=freeze+endpoint, 3=reference conditioning
+    # --- T_phi 容量（默认配置 10.82M 参数） ---
+    widths: tuple[int, int, int] = (96, 192, 288)
+    encoder_blocks: tuple[int, int, int] = (2, 2, 2)
+    attention_heads: tuple[int, int, int] = (4, 8, 8)
+    attention_layers: tuple[int, int, int] = (2, 2, 2)
+    attention_dropout: float = 0.0
+    token_side: int = 16
+    decoder_hidden: int = 96
+    num_directions: int = 8
+    num_brush_scales: int = 3
+    direction_kernel: int = 5
+    block_kernel: int = 5
+    style_gate_init: float = 0.2
+    ll_shift_max: float = 2.0
+    hf_shift_max: float = 0.8
+    style_dim: int = 32
+    patch_size: int = 4
+    patch_stride: int = 4
+    # --- 阶段 1 优化 / 闸门 ---
+    lr: float = 3e-3
+    unpaired_weight: float = 0.5
+    grad_clip: float = 1.0
+    min_style_gain: float = 0.05
+    min_content_keep: float = 0.60
+    max_reference_copy: float = 0.45
+    # --- 曲线 / 诊断 ---
+    milestones: tuple[int, ...] = (2000, 10000, 30000)
+    curve_every: int = 250             # 非里程碑也记录（JSONL 全量）
+    holdout_fraction: float = 0.05
+    overfit_samples: int = 32
+    overfit_steps: int = 200
+    adain_off_min_gain: float = 0.05
+    # --- 数据范围：先 5k fast-validate，再 18k（18k 需先编码 latent cache） ---
+    data_scope: str = "5k"             # "5k" | "18k"
+    extra: dict[str, Any] = field(default_factory=dict)
+
+    @classmethod
+    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RouteBConfig":
+        known, extra = _split_known_fields(cls, payload)
+        cfg = cls(**known)
+        cfg.extra = extra
+        _rehydrate_extra_attributes(cfg)
+        return cfg
+
+    def to_dict(self) -> dict[str, Any]:
+        _materialize_missing_dataclass_fields(self)
+        payload = asdict(self)
+        extra = payload.pop("extra", {})
+        payload.update(extra)
+        return payload
```

### 1.2 `ExperimentConfig` 增加字段与解析（约第 1171–1196 行）

```diff
 class ExperimentConfig:
     model: ModelConfig = field(default_factory=ModelConfig)
     bridge: BridgeConfig = field(default_factory=BridgeConfig)
+    route_b: RouteBConfig = field(default_factory=RouteBConfig)
     training: TrainingConfig = field(default_factory=TrainingConfig)
@@ from_mapping
-        known_sections = {"model", "bridge", "training", "data", "checkpoint", "inference", "full_eval", "ablation"}
+        known_sections = {"model", "bridge", "route_b", "training", "data", "checkpoint", "inference", "full_eval", "ablation"}
         cfg = cls(
             model=ModelConfig.from_mapping(data.get("model")),
             bridge=BridgeConfig.from_mapping(data.get("bridge")),
+            route_b=RouteBConfig.from_mapping(data.get("route_b")),
             training=TrainingConfig.from_mapping(data.get("training")),
```

对应 YAML（新增，默认不写即关闭）：

```yaml
route_b:
  enabled: true
  stage: 1
  data_scope: 5k
  widths: [96, 192, 288]
```

---

## 2. `model.py`

T_phi **不进入 WEAVE 主干**（避免影响默认前向与参数量统计）。仅在阶段 3 需要把
参考图条件接进主模型时，提供一个可选槽位。

插入位置：`WEAVE.__init__` 第 506 行 `self.texture_refiner = None` 之后。

```diff
         else:
             self.texture_refiner = None
+        # === Route B: content-anchored operator slot (default None) ===
+        # Set by SBTrainer only when config.route_b.enabled. Not created here:
+        # the operator is trained in its own stage and never changes the default
+        # forward when left as None.
+        self.b1_operator = None
+        # === /Route B ===
```

并在类内追加：

```diff
+    def set_b1_operator(self, operator: torch.nn.Module | None) -> None:
+        """Attach/detach the Route B operator (stage 2/3 only)."""
+        self.b1_operator = operator
+
+    def forward_with_b1_endpoint(self, content_latent: torch.Tensor) -> torch.Tensor | None:
+        """Return z* for FM endpoint when the operator is attached; else None."""
+        if self.b1_operator is None:
+            return None
+        ref = getattr(self, "last_reference_latent", None)
+        if ref is None:
+            return None
+        return self.b1_operator(content_latent, ref)["z_star"]
```

> 说明：真正训练阶段 1 时不需要动 `model.py`；算子由 `trainer.py` 持有。这个槽位只是
> 阶段 3 的接线点，默认 `None` 时 `forward` 完全不变。

---

## 3. `flow.py`

插入位置：`_apply_pseudo_endpoints`（第 287–320 行）开头，cache 分支之前。

```diff
     def _apply_pseudo_endpoints(
         self, target_style: torch.Tensor, conditioning: Dict
     ) -> torch.Tensor:
         if self.endpoint_mode == "target_hf":
             return target_style
+        # Route B stage 2: direct z1 from the frozen operator (no cache hop).
+        # Only used when the trainer passes conditioning["route_b_z1"].
+        direct = conditioning.get("route_b_z1")
+        if torch.is_tensor(direct):
+            if direct.shape != target_style.shape:
+                raise ValueError(
+                    f"route_b_z1 shape {tuple(direct.shape)} != target_style {tuple(target_style.shape)}"
+                )
+            out = target_style.clone()
+            rows = conditioning.get("route_b_rows")
+            if rows is None:
+                return direct.to(device=target_style.device, dtype=target_style.dtype)
+            idx = torch.as_tensor(list(rows), dtype=torch.long, device=target_style.device)
+            out[idx] = direct.to(device=target_style.device, dtype=target_style.dtype)
+            return out
         pseudo = conditioning.get("pseudo_target_latent")
```

到阶段 2 默认仍走 `pseudo_target_cache_dir`（离线导出 `z1`），`route_b_z1` 是给
"算子常驻显存、在线算端点"的变体用；两条路径都只替换端点，不改 FM 其余项。

---

## 4. `trainer.py`

### 4.1 初始化（第 287 行 `endpoint_mode='pseudo' without ...` 之后）

```diff
             elif _endpoint_mode == "pseudo":
                 logger.warning("endpoint_mode='pseudo' without pseudo_target_cache_dir — all rows fall back to target_hf.")
+        # === Route B: operator + stage-1 training ===
+        self._route_b = None
+        self._route_b_provider = None
+        self._route_b_curve = None
+        if bool(getattr(getattr(config, "route_b", None), "enabled", False)):
+            from b1_style_operator import B1OperatorConfig, B1StyleOperator
+            from b_pseudo_targets import BPseudoTargetProvider, OfflineStyleFeatureCache, ProviderConfig
+            from b_training_diagnostics import CurveRecorder
+
+            _rb = config.route_b
+            _op_cfg = B1OperatorConfig(
+                latent_channels=int(getattr(config.model, "latent_channels", 4)),
+                widths=tuple(_rb.widths),
+                encoder_blocks=tuple(_rb.encoder_blocks),
+                attention_heads=tuple(_rb.attention_heads),
+                attention_layers=tuple(_rb.attention_layers),
+                attention_dropout=float(_rb.attention_dropout),
+                token_side=int(_rb.token_side),
+                decoder_hidden=int(_rb.decoder_hidden),
+                num_directions=int(_rb.num_directions),
+                num_brush_scales=int(_rb.num_brush_scales),
+                direction_kernel=int(_rb.direction_kernel),
+                block_kernel=int(_rb.block_kernel),
+                style_gate_init=float(_rb.style_gate_init),
+                ll_shift_max=float(_rb.ll_shift_max),
+                hf_shift_max=float(_rb.hf_shift_max),
+                style_dim=int(_rb.style_dim),
+                patch_size=int(_rb.patch_size),
+                patch_stride=int(_rb.patch_stride),
+            )
+            self._route_b = B1StyleOperator(_op_cfg).to(self.device)
+            _style_cache = None
+            _cache_dir = str(getattr(_rb, "style_feature_cache_dir", "") or "").strip()
+            if _cache_dir:
+                _style_cache = OfflineStyleFeatureCache(_cache_dir)
+            self._route_b_provider = BPseudoTargetProvider(
+                self._route_b,
+                style_feature_cache=_style_cache,
+                cfg=ProviderConfig(unpaired_weight=float(_rb.unpaired_weight)),
+            )
+            self._route_b_curve = CurveRecorder(milestones=tuple(_rb.milestones))
+            self._route_b_opt = torch.optim.AdamW(
+                self._route_b.parameters(), lr=float(_rb.lr), weight_decay=0.0
+            )
+            logger.info(
+                "Route B enabled | stage=%d | params=%.3fM | data_scope=%s",
+                int(_rb.stage),
+                sum(p.numel() for p in self._route_b.parameters() if p.requires_grad) / 1e6,
+                str(getattr(_rb, "data_scope", "5k")),
+            )
+        # === /Route B ===
```

### 4.2 阶段 1：主训练循环里替换 loss 分支（第 1571–1581 行）

```diff
+                if self._route_b_provider is not None and self.global_step < _ROUTE_B_STAGE1_STEPS:
+                    # Stage 1: T_phi only. Main WEAVE forward is skipped on purpose;
+                    # we train the operator on B2 pairs + unpaired content and log to the curve.
+                    _rb_logs = self._route_b_provider.stage1_step(
+                        {
+                            "content_latent": content,
+                            "reference_latent": target_style,
+                            "target_latent": target_style,
+                        },
+                        {
+                            "content_latent": content,
+                            "reference_latent": target_style,
+                        },
+                        optimizer=self._route_b_opt,
+                        grad_clip=1.0,
+                        unpaired_weight=float(getattr(config.route_b, "unpaired_weight", 0.5)),
+                    )
+                    if self.global_step % int(getattr(config.route_b, "curve_every", 250)) == 0:
+                        self._route_b_curve.record(
+                            int(self.global_step), "train", {"loss": float(_rb_logs["loss"].detach())}
+                        )
+                    if self._route_b_curve.should_record(int(self.global_step)):
+                        from b_training_diagnostics import StyleGainDiagnostics, evaluate_holdout
+
+                        _diag = StyleGainDiagnostics(self._route_b)
+                        _holdout = [
+                            {
+                                "content_latent": content[i : i + 1],
+                                "reference_latent": target_style[i : i + 1],
+                            }
+                            for i in range(min(8, content.shape[0]))
+                        ]
+                        self._route_b_curve.record(
+                            int(self.global_step), "holdout", evaluate_holdout(self._route_b_provider, _holdout)
+                        )
+                        _cmp = _diag.compare(content[:1], target_style[:1])
+                        logger.info(
+                            "Route B @%d | adain_off_gain=%.4f adain_on_gain=%.4f network_ok=%s",
+                            self.global_step,
+                            _cmp["adain_off"]["style_gain"],
+                            _cmp["adain_on"]["style_gain"],
+                            _cmp["network_style_ok"],
+                        )
+                    loss_dict = {"loss": _rb_logs["loss"]}
+                    loss = _rb_logs["loss"]
+                    continue
                 if self.distill_enabled and self.teacher_model is not None:
```

> 注意：`continue` 的位置需与训练循环的实际结构对齐（若循环尾部有 scheduler/日志，
> 应把 stage-1 分支放进 `loss_dict` 赋值前的 if/else，而不是裸 `continue`）。
> patch 合并时以实际循环为准，这里给出的是逻辑骨架。

### 4.3 阶段切换（提交门禁）

```python
def _route_b_stage_gate(self) -> bool:
    """阶段 1 -> 阶段 2 的门禁：曲线 + AdaIN-off + 小样本过拟合三项同时通过。"""
    if self._route_b_provider is None or self._route_b_curve is None:
        return False
    from b_training_diagnostics import OverfitProbe, StyleGainDiagnostics
    summ = self._route_b_curve.summary()
    last = summ["milestones"].get(max(summ["milestones"]), {})
    hold = last.get("holdout", {})
    if hold.get("style_gain", 0.0) < float(self.config.route_b.min_style_gain):
        logger.warning("Route B stage gate: holdout style_gain too low (%s)", hold)
        return False
    probe = OverfitProbe(
        self._route_b_provider,
        OverfitConfig(steps=int(self.config.route_b.overfit_steps), n_samples=32),
    )
    if not probe.run(self._route_b_samples).get("overfit_ok", False):
        logger.warning("Route B stage gate: 32-sample overfit probe failed")
        return False
    self._route_b_provider.mark_screened({"passed": True})
    self._route_b_provider.freeze_operator()
    return True
```

`mark_screened` 的正式版本应使用 `evaluate_and_mark(...)` 的真实报告，而不是
只写 `{"passed": True}`；这里保持与 `b_pseudo_targets.py` 的 API 一致。

### 4.4 阶段 2/3

- 阶段 2 端点：优先用离线 cache（`BPseudoTargetProvider.export_endpoints` 产出
  `PseudoTargetCache`），或把 `route_b_z1` 塞进 `conditioning`；
- 阶段 3：`provider.stage3_checks(content, ref_a, ref_b)` 必须同时通过
  swap（`output_delta_rel >= 1e-4`）与 permutation（`output_delta_max <= 1e-4`）。

---

## 5. 数据范围

| 范围 | 现状 | 用法 |
|---|---|---|
| **5k 缓存** | 已有 latent cache | fast-validate：先跑阶段 1 到 2k/10k/30k，看曲线与诊断 |
| **18k** | 只有原图，无 latent cache | **后续工程项**：先离线编码成 latent cache，再替换 `data_scope: 18k`；不阻塞当前开发 |

18k 编码脚本建议（后续）：

```bash
python tools/encode_latents.py --manifest data/paintings_18k.jsonl --vae <frozen-vae> \
    --out data/latents_18k --shard-size 4096
```

（该脚本尚不存在；列入后续工程项。）

---

## 6. 诊断启动命令

```bash
# 1) 阶段 1 曲线（训练循环内自动记录）
#    -> <exp_dir>/b_curve.jsonl, <exp_dir>/b_curve_summary.md

# 2) 小样本过拟合（32 张，独立脚本/Notebook 均可）
python - <<'PY'
from b_training_diagnostics import OverfitProbe, OverfitConfig
...
PY

# 3) AdaIN-off 风格增益
python - <<'PY'
from b_training_diagnostics import StyleGainDiagnostics
...
PY
```

判定规则：
- 32 张过拟合 `ratio > 0.5` → 判定架构/目标问题，**不继续堆复杂度**，先回报；
- AdaIN-off `style_gain < 0.05` → 风格来自推理后处理而不是网络，淘汰。

---

## 7. 容量与资源（`tools/bench_route_b.py` 实测/估算）

```
params                        10.820M
param+optimizer (fp32 Adam)   0.16 GB
forward FLOPs / sample        22.21 GFLOPs  (latent 64x64)
train FLOPs / sample           ≈66.6 GFLOPs (fwd+bwd≈3x)
activation est batch=8        1.79 GB
activation est batch=16       3.58 GB   (含参数 0.16 GB，10 GB 预算内)
CPU forward (1x64x64)          60–95 ms（仅 sanity check）
```

3060 12G 估算（有效算力是假设值，非实测）：

| 精度 | 有效 TFLOPS | batch=8 步时 | 30k 步墙钟 |
|---|---|---|---|
| fp32 | 8.0 | ≈67 ms | ≈0.56 h |
| bf16 | 14.0 | ≈38 ms | ≈0.32 h |

阶段 1 每步有 paired + unpaired 两次前向，且带 SWD patch 描述子，实测量级
应在上述数字的 **1.5–3×**，即 30k 步约 **1–2 h**（5k 缓存、batch 8）。
真实数字等训练线释放 GPU 后用同脚本在 CUDA 上复测。

---

## 8. 合并顺序建议

1. `config_schema.py`（只加字段，默认 `enabled=False`）；
2. `trainer.py` 初始化 + stage-1 分支（默认分支不变）；
3. `flow.py` 的 `route_b_z1` 直连路径；
4. `model.py` 的槽位（阶段 3 才需要）；
5. 每步合并后跑 `python -m pytest tests/test_route_b_style_operator.py -q`
   与一次默认配置的 `python tools/smoke_route_b.py --steps 40`。
