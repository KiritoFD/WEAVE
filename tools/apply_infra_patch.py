"""Apply / verify the infra patch of docs/113/patches/infra_20261104.patch.

The patch is expressed as exact ``old -> new`` replacements so it can be applied
to the current working tree (which already carries uncommitted edits to
``trainer.py`` / ``config_schema.py`` / ``run.py``) without relying on ``HEAD``.

Modes:
    --check       verify every anchor matches exactly once (no writes)
    --apply       write the patched files
    --emit-diff   print a unified diff (git-apply compatible) to stdout

Every switch keeps its pre-patch default, so applying the patch alone does not
change any numerical behaviour.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Edit blocks: file -> [(old, new), ...]
# ---------------------------------------------------------------------------

CONFIG_SCHEMA_FIELDS_OLD = """    output_supervision_chunk_rows: int = 0         # descriptor unfold chunking (0=off)
"""

CONFIG_SCHEMA_FIELDS_NEW = """    output_supervision_chunk_rows: int = 0         # descriptor unfold chunking (0=off)
    # === infra patch (docs/113/patches/infra_20261104.patch): all default OFF ===
    output_supervision_max_samples: int = 0        # 0=whole batch; >0=random K aux samples per step
    output_supervision_split_backward: bool = False  # FM backward first, aux backward second
    output_supervision_decoder_checkpoint: bool = False  # grad-checkpoint the latent decoder
    output_supervision_extractor_checkpoint: bool = False  # grad-checkpoint the img_out teacher pass
    output_supervision_frozen_feats_dir: str = ""  # offline dinov2 feature cache root (P1)
    output_supervision_frozen_feats_dtype: str = "float16"
"""

CONFIG_SCHEMA_PROBE_OLD = """    log_interval: int = 20
    use_tqdm: bool = True
"""

CONFIG_SCHEMA_PROBE_NEW = """    log_interval: int = 20
    probe_log_interval: int = 1   # 1=probe scalars every step (current); N=sample every N steps
    use_tqdm: bool = True
"""

DATASET_ITEM_OLD = """            "content_index": c_idx,
        }
"""

DATASET_ITEM_NEW = """            "content_index": c_idx,
        }
        # infra patch P1: frozen-feature cache coordinates (additive keys only).
        _flip_t = bool(self._cache_flip_target[index]) if self._cache_flip_target is not None else False
        _flip_c = bool(self._cache_flip_content[index]) if self._cache_flip_content is not None else False
        item["target_cache_index"] = t_idx
        item["content_cache_index"] = c_idx
        item["target_cache_flip"] = torch.tensor(float(_flip_t), dtype=torch.float32)
        item["content_cache_flip"] = torch.tensor(float(_flip_c), dtype=torch.float32)
        item["target_cache_valid"] = torch.tensor(
            float(self.pairing_cache_dual_target_mix <= 0.0), dtype=torch.float32
        )
"""

UNROLL_SIG_OLD = """    use_grad_checkpoint: bool = True,
    microbatch: int = 0,
) -> torch.Tensor:
"""

UNROLL_SIG_NEW = """    use_grad_checkpoint: bool = True,
    microbatch: int = 0,
    decoder_checkpoint: bool = False,
) -> torch.Tensor:
"""

UNROLL_RECURSE_OLD = """                use_grad_checkpoint=use_grad_checkpoint, microbatch=0,
            ))
"""

UNROLL_RECURSE_NEW = """                use_grad_checkpoint=use_grad_checkpoint, microbatch=0,
                decoder_checkpoint=decoder_checkpoint,
            ))
"""

UNROLL_RETURN_OLD = """        else:
            h = _euler_step(
                model, h, t_val, dt,
                style_id=style_id, style_latent=style_latent,
                style_text_tokens=style_text_tokens, reference_feats=reference_feats,
            )
    return decoder.decode_image(h)
"""

UNROLL_RETURN_NEW = """        else:
            h = _euler_step(
                model, h, t_val, dt,
                style_id=style_id, style_latent=style_latent,
                style_text_tokens=style_text_tokens, reference_feats=reference_feats,
            )
    if decoder_checkpoint and torch.is_grad_enabled() and h.requires_grad:
        return grad_checkpoint(decoder.decode_image, h, use_reentrant=False)
    return decoder.decode_image(h)
"""

OS_INIT_SIG_OLD = """        feature_res: int = 0,
        chunk_rows: int = 0,
    ) -> None:
"""

OS_INIT_SIG_NEW = """        feature_res: int = 0,
        chunk_rows: int = 0,
        decoder_checkpoint: bool = False,
        extractor_checkpoint: bool = False,
    ) -> None:
"""

OS_INIT_BODY_OLD = """        self.feature_res = max(0, int(feature_res))
        self.chunk_rows = max(0, int(chunk_rows))
"""

OS_INIT_BODY_NEW = """        self.feature_res = max(0, int(feature_res))
        self.chunk_rows = max(0, int(chunk_rows))
        self.decoder_checkpoint = bool(decoder_checkpoint)
        self.extractor_checkpoint = bool(extractor_checkpoint)
"""

OS_FORWARD_SIG_OLD = """        content_latent: torch.Tensor,
        style_image: torch.Tensor,
        content_image: torch.Tensor,
        style_id=None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        reference_feats: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
"""

OS_FORWARD_SIG_NEW = """        content_latent: torch.Tensor,
        style_image: torch.Tensor | None,
        content_image: torch.Tensor | None,
        style_id=None,
        style_latent: torch.Tensor | None = None,
        style_text_tokens: torch.Tensor | None = None,
        reference_feats: torch.Tensor | None = None,
        style_feats: Dict[str, torch.Tensor] | None = None,
        content_feats: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
"""

OS_UNROLL_CALL_OLD = """            use_grad_checkpoint=self.grad_checkpoint, microbatch=self.microbatch,
        )
"""

OS_UNROLL_CALL_NEW = """            use_grad_checkpoint=self.grad_checkpoint, microbatch=self.microbatch,
            decoder_checkpoint=self.decoder_checkpoint,
        )
"""

OS_RESIZE_OLD = """        if self.feature_res and img_out.shape[-1] != self.feature_res:
            size = (self.feature_res, self.feature_res)
            img_out = F.interpolate(img_out, size=size, mode="bilinear", align_corners=False)
            style_image = F.interpolate(style_image, size=size, mode="bilinear", align_corners=False)
            content_image = F.interpolate(content_image, size=size, mode="bilinear", align_corners=False)
"""

OS_RESIZE_NEW = """        if self.feature_res and img_out.shape[-1] != self.feature_res:
            size = (self.feature_res, self.feature_res)
            img_out = F.interpolate(img_out, size=size, mode="bilinear", align_corners=False)
            if style_image is not None:
                style_image = F.interpolate(style_image, size=size, mode="bilinear", align_corners=False)
            if content_image is not None:
                content_image = F.interpolate(content_image, size=size, mode="bilinear", align_corners=False)
"""

OS_FEATS_OLD = """        feats_out = self.style_loss.extractor.forward_layers(img_out, needed)
        with torch.no_grad():
            feats_style = self.style_loss.extractor.forward_layers(
                style_image.detach(), self.style_loss.layers,
            )
            feats_content = self.content_loss.extractor.forward_layers(
                content_image.detach(),
                self.content_loss.perceptual_layers + [self.content_loss.selfsim_layer],
            )
"""

OS_FEATS_NEW = """        feats_style = style_feats
        feats_content = content_feats
        if self.extractor_checkpoint and torch.is_grad_enabled() and img_out.requires_grad:
            feats_out = grad_checkpoint(
                self.style_loss.extractor.forward_layers, img_out, needed, use_reentrant=False,
            )
        else:
            feats_out = self.style_loss.extractor.forward_layers(img_out, needed)
        if feats_style is None or feats_content is None:
            with torch.no_grad():
                if feats_style is None:
                    if style_image is None:
                        raise ValueError("style_image is required when style_feats are not cached")
                    feats_style = self.style_loss.extractor.forward_layers(
                        style_image.detach(), self.style_loss.layers,
                    )
                if feats_content is None:
                    if content_image is None:
                        raise ValueError("content_image is required when content_feats are not cached")
                    feats_content = self.content_loss.extractor.forward_layers(
                        content_image.detach(),
                        self.content_loss.perceptual_layers + [self.content_loss.selfsim_layer],
                    )
        if self.content_loss.w_edge > 0.0 and content_image is None:
            raise ValueError("content_image is required for the edge term (w_edge > 0)")
"""

TRAINER_INIT_OLD = """                getattr(config.bridge, "output_supervision_feature_res", 0),
                getattr(config.bridge, "output_supervision_chunk_rows", 0),
            )
        # Route C: InfoNCE weight for reference encoder (encoder lives in the model)
"""

TRAINER_INIT_NEW = """                getattr(config.bridge, "output_supervision_feature_res", 0),
                getattr(config.bridge, "output_supervision_chunk_rows", 0),
            )
        # === infra patch (docs/113/patches/infra_20261104.patch): all default OFF ===
        self._output_sup_max_samples = max(0, int(getattr(config.bridge, "output_supervision_max_samples", 0)))
        self._output_sup_split_backward = bool(
            getattr(config.bridge, "output_supervision_split_backward", False)
        )
        self._output_sup_cache = None
        self._sup_cache_fallback_warned = False
        _feats_dir = str(getattr(config.bridge, "output_supervision_frozen_feats_dir", "") or "").strip()
        if self._output_supervision is not None and _feats_dir:
            from frozen_feature_cache import FrozenFeatureCache

            _sup = self._output_supervision
            self._output_sup_cache = FrozenFeatureCache(
                _feats_dir,
                feature_res=int(getattr(config.bridge, "output_supervision_feature_res", 0)),
                layers=list(dict.fromkeys(
                    _sup.style_loss.layers
                    + _sup.content_loss.perceptual_layers
                    + [_sup.content_loss.selfsim_layer]
                )),
                dtype=str(getattr(config.bridge, "output_supervision_frozen_feats_dtype", "float16")),
            )
        # Route C: InfoNCE weight for reference encoder (encoder lives in the model)
"""

TRAINER_CACHE_HELPER_OLD = """    def step_scheduler(self) -> None:
"""

TRAINER_CACHE_HELPER_NEW = '''    def _warn_sup_cache_fallback(self, reason: str) -> None:
        if not self._sup_cache_fallback_warned:
            logger.warning(
                "Frozen feature cache disabled for this run: %s; using the online extractor.", reason
            )
            self._sup_cache_fallback_warned = True

    def _sup_cache_rows(self, batch: Dict[str, torch.Tensor], side: str, sel: torch.Tensor | None):
        """Global cache rows for ``side`` ('target'|'content'); ``None`` = online path."""
        cache = self._output_sup_cache
        if cache is None or not cache.enabled:
            return None
        idx_key = f"{side}_cache_index"
        sid_key = "target_style_id" if side == "target" else "source_style_id"
        if idx_key not in batch or sid_key not in batch:
            self._warn_sup_cache_fallback(f"batch has no '{idx_key}'/'{sid_key}' keys")
            return None
        if side == "target" and "target_cache_valid" in batch:
            if not bool(torch.all(batch["target_cache_valid"] > 0).item()):
                self._warn_sup_cache_fallback("dual-target blend rows are not cacheable")
                return None
        _sid = batch[sid_key]
        _idx = batch[idx_key]
        _flip = batch.get(f"{side}_cache_flip")
        if sel is not None:
            if torch.is_tensor(_sid):
                _sid = _sid.index_select(0, sel)
            if torch.is_tensor(_idx):
                _idx = _idx.index_select(0, sel)
            if torch.is_tensor(_flip):
                _flip = _flip.index_select(0, sel)
        rows = cache.rows(_sid, _idx, _flip)
        if rows is None:
            self._warn_sup_cache_fallback("cache row out of range or unknown style id")
        return rows

    def _output_sup_step(self, batch, content, target_style, target_style_id):
        """Route A auxiliary loss (P1 cache + P2 sub-sampling aware)."""
        if content.device.type == "cuda":
            _autocast = torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype)
        else:
            _autocast = torch.autocast("cpu", enabled=False)
        with _autocast:
            # --- infra P2: optional aux sub-sampling (K random samples) ---
            _c_aux, _t_aux, _sid_aux = content, target_style, target_style_id
            _tok_aux = batch.get("target_style_text_tokens")
            _ref_aux = batch.get("reference_feats")
            _sel = None
            _aux_k = self._output_sup_max_samples
            if 0 < _aux_k < content.shape[0]:
                _sel = torch.randperm(content.shape[0], device=content.device)[:_aux_k]
                _c_aux = content.index_select(0, _sel)
                _t_aux = target_style.index_select(0, _sel)
                if torch.is_tensor(target_style_id):
                    _sid_aux = target_style_id.index_select(0, _sel)
                if torch.is_tensor(_tok_aux):
                    _tok_aux = _tok_aux.index_select(0, _sel)
                if torch.is_tensor(_ref_aux):
                    _ref_aux = _ref_aux.index_select(0, _sel)
            # --- infra P1: frozen feature cache lookup (None => online path) ---
            _feats_style = _feats_content = None
            _rows_t = self._sup_cache_rows(batch, "target", _sel)
            _rows_c = self._sup_cache_rows(batch, "content", _sel)
            if _rows_t is not None and _rows_c is not None:
                _feats_style = self._output_sup_cache.lookup(
                    _rows_t, self._output_supervision.style_loss.layers, device=content.device,
                )
                _feats_content = self._output_sup_cache.lookup(
                    _rows_c,
                    self._output_supervision.content_loss.perceptual_layers
                    + [self._output_supervision.content_loss.selfsim_layer],
                    device=content.device,
                )
            with torch.no_grad():
                _style_img = (
                    None if _feats_style is not None
                    else self._output_supervision.decoder.decode_image(_t_aux)
                )
                _content_img = self._output_supervision.decoder.decode_image(_c_aux)
            _sup_logs = self._output_supervision(
                self.model,
                content_latent=_c_aux,
                style_image=_style_img,
                content_image=_content_img,
                style_id=_sid_aux,
                style_latent=batch.get("target_style_latent") if torch.is_tensor(batch.get("target_style_latent")) else _t_aux,
                style_text_tokens=_tok_aux,
                reference_feats=_ref_aux,
                style_feats=_feats_style,
                content_feats=_feats_content,
            )
            _sup_total = (
                self._output_sup_lambda_style * _sup_logs["loss_style_out"]
                + self._output_sup_lambda_content * _sup_logs["loss_content_out"]
            )
        return _sup_total, _sup_logs

    def step_scheduler(self) -> None:
'''

TRAINER_ROUTEA_OLD = """                if self._output_supervision is not None and (step_idx % self._output_sup_apply_every == 0):
                    with torch.no_grad():
                        _style_img = self._output_supervision.decoder.decode_image(target_style)
                        _content_img = self._output_supervision.decoder.decode_image(content)
                    _sup_logs = self._output_supervision(
                        self.model,
                        content_latent=content,
                        style_image=_style_img,
                        content_image=_content_img,
                        style_id=target_style_id,
                        style_latent=batch.get("target_style_latent") if torch.is_tensor(batch.get("target_style_latent")) else target_style,
                        style_text_tokens=batch.get("target_style_text_tokens"),
                        reference_feats=batch.get("reference_feats"),
                    )
                    _sup_total = (
                        self._output_sup_lambda_style * _sup_logs["loss_style_out"]
                        + self._output_sup_lambda_content * _sup_logs["loss_content_out"]
                    )
                    loss = loss + _sup_total
                    loss_dict["loss"] = loss
                    loss_dict["loss_output_sup"] = _sup_total
                    for _k, _v in _sup_logs.items():
                        loss_dict[f"out_{_k}"] = _v.detach() if torch.is_tensor(_v) else _v
"""

TRAINER_ROUTEA_NEW = """                if (
                    self._output_supervision is not None
                    and (step_idx % self._output_sup_apply_every == 0)
                    and not self._output_sup_split_backward
                ):
                    _sup_total, _sup_logs = self._output_sup_step(
                        batch, content, target_style, target_style_id,
                    )
                    loss = loss + _sup_total
                    loss_dict["loss"] = loss
                    loss_dict["loss_output_sup"] = _sup_total
                    for _k, _v in _sup_logs.items():
                        loss_dict[f"out_{_k}"] = _v.detach() if torch.is_tensor(_v) else _v
"""

TRAINER_SPLIT_BACKWARD_OLD = """            backward_time_total += max(0.0, time.perf_counter() - t0)
            grad_report = None
"""

TRAINER_SPLIT_BACKWARD_NEW = """            backward_time_total += max(0.0, time.perf_counter() - t0)
            if (
                self._output_sup_split_backward
                and self._output_supervision is not None
                and (step_idx % self._output_sup_apply_every == 0)
            ):
                # --- infra P3: the FM graph is already freed, build the aux graph now ---
                _t0 = time.perf_counter()
                _sup_total, _sup_logs = self._output_sup_step(
                    batch, content, target_style, target_style_id,
                )
                _aux_term = _sup_total / self.accumulation_steps
                if self.scaler is not None:
                    self.scaler.scale(_aux_term).backward()
                else:
                    _aux_term.backward()
                backward_time_total += max(0.0, time.perf_counter() - _t0)
                loss_dict["loss_output_sup"] = _sup_total
                loss_dict["loss"] = loss.detach() + _sup_total.detach()
                for _k, _v in _sup_logs.items():
                    loss_dict[f"out_{_k}"] = _v.detach() if torch.is_tensor(_v) else _v
            grad_report = None
"""

TRAINER_PROBE_INIT_OLD = """        self.log_interval = max(0, int(train_cfg.get("log_interval", 20)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
"""

TRAINER_PROBE_INIT_NEW = """        self.log_interval = max(0, int(train_cfg.get("log_interval", 20)))
        self.probe_log_interval = max(1, int(train_cfg.get("probe_log_interval", 1)))
        self.use_tqdm = bool(train_cfg.get("use_tqdm", True))
"""

TRAINER_PROBE_DUE_OLD = """            probe_due = (
"""

TRAINER_PROBE_DUE_NEW = """            probe_interval = self.probe_log_interval
            probe_due = (
"""

RUN_STARTUP_OLD = """    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)
"""

RUN_STARTUP_NEW = """    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Seed: %d", seed)
    if device.type == "cuda":
        # infra P7: read-only startup sanity checks (shared 3060 => catch interlopers).
        _free_bytes, _total_bytes = torch.cuda.mem_get_info()
        logger.info(
            "GPU memory before init: free=%.2fG total=%.2fG used=%.2fG",
            _free_bytes / 1024 ** 3, _total_bytes / 1024 ** 3,
            (_total_bytes - _free_bytes) / 1024 ** 3,
        )
        if _free_bytes < 9.5 * 1024 ** 3:
            logger.warning(
                "Only %.2fG free on %s; another run may share this GPU, peak VRAM numbers will be polluted.",
                _free_bytes / 1024 ** 3, torch.cuda.get_device_name(0),
            )
        if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
            logger.warning(
                "PYTORCH_CUDA_ALLOC_CONF is unset; expandable_segments will not be enabled. "
                "Export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for long runs."
            )
"""

EDIT_BLOCKS: dict[str, list[tuple[str, str]]] = {
    "config_schema.py": [
        (CONFIG_SCHEMA_FIELDS_OLD, CONFIG_SCHEMA_FIELDS_NEW),
        (CONFIG_SCHEMA_PROBE_OLD, CONFIG_SCHEMA_PROBE_NEW),
    ],
    "utils/dataset.py": [
        (DATASET_ITEM_OLD, DATASET_ITEM_NEW),
    ],
    "output_supervision.py": [
        (UNROLL_SIG_OLD, UNROLL_SIG_NEW),
        (UNROLL_RECURSE_OLD, UNROLL_RECURSE_NEW),
        (UNROLL_RETURN_OLD, UNROLL_RETURN_NEW),
        (OS_INIT_SIG_OLD, OS_INIT_SIG_NEW),
        (OS_INIT_BODY_OLD, OS_INIT_BODY_NEW),
        (OS_FORWARD_SIG_OLD, OS_FORWARD_SIG_NEW),
        (OS_UNROLL_CALL_OLD, OS_UNROLL_CALL_NEW),
        (OS_RESIZE_OLD, OS_RESIZE_NEW),
        (OS_FEATS_OLD, OS_FEATS_NEW),
    ],
    "trainer.py": [
        (TRAINER_INIT_OLD, TRAINER_INIT_NEW),
        (TRAINER_CACHE_HELPER_OLD, TRAINER_CACHE_HELPER_NEW),
        (TRAINER_ROUTEA_OLD, TRAINER_ROUTEA_NEW),
        (TRAINER_SPLIT_BACKWARD_OLD, TRAINER_SPLIT_BACKWARD_NEW),
        (TRAINER_PROBE_INIT_OLD, TRAINER_PROBE_INIT_NEW),
        (TRAINER_PROBE_DUE_OLD, TRAINER_PROBE_DUE_NEW),
    ],
    "run.py": [
        (RUN_STARTUP_OLD, RUN_STARTUP_NEW),
    ],
}

NEW_FILES = {
    "frozen_feature_cache.py": "frozen_feature_cache.py",
    "tools/build_dino_feature_cache.py": "tools/build_dino_feature_cache.py",
    "runs/eval_one_fast.sh": "runs/eval_one_fast.sh",
}


def _read(path: Path) -> str:
    """Read bytes verbatim (no newline translation) so CRLF files keep their endings."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _patched_text(text: str, edits: list[tuple[str, str]], name: str, errors: list[str]) -> str:
    # trainer.py / config_schema.py / output_supervision.py / dataset.py are CRLF in
    # this working tree, run.py is LF: adapt every anchor to the file's own endings.
    newline = "\r\n" if "\r\n" in text else "\n"
    for old, new in edits:
        if newline != "\n":
            old = old.replace("\n", newline)
            new = new.replace("\n", newline)
        count = text.count(old)
        if count != 1:
            errors.append(f"{name}: anchor found {count} times (expected 1): {old.splitlines()[0]!r}")
            continue
        text = text.replace(old, new, 1)
    return text


def cmd_check(root: Path) -> int:
    errors: list[str] = []
    for rel, edits in EDIT_BLOCKS.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"{rel}: missing file")
            continue
        text = _read(path)
        _patched_text(text, edits, rel, errors)
        if "infra patch" in text or "infra P" in text:
            errors.append(f"{rel}: patch markers already present (applied twice?)")
    if errors:
        for err in errors:
            print(f"FAIL {err}")
        return 1
    total = sum(len(v) for v in EDIT_BLOCKS.values())
    print(f"OK   {total} anchors across {len(EDIT_BLOCKS)} files match exactly once")
    return 0


def cmd_apply(root: Path) -> int:
    errors: list[str] = []
    staged: list[tuple[Path, str, int]] = []
    for rel, edits in EDIT_BLOCKS.items():
        path = root / rel
        text = _read(path)
        staged.append((path, _patched_text(text, edits, rel, errors), len(edits)))
    if errors:
        for err in errors:
            print(f"FAIL {err}")
        return 1
    for path, patched, n_edits in staged:
        _write(path, patched)
        print(f"APPLIED {path.relative_to(root)} ({n_edits} edits)")
    return 0


def _unified(rel: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{rel}",
        tofile=f"b/{rel}",
        n=4,
    )
    return f"diff --git a/{rel} b/{rel}\n" + "".join(diff)


def cmd_emit_diff(root: Path) -> int:
    chunks: list[str] = []
    for rel, edits in EDIT_BLOCKS.items():
        before = _read(root / rel)
        after = _patched_text(before, edits, rel, [])
        chunks.append(_unified(rel, before, after))
    for rel, src in NEW_FILES.items():
        content = _read(root / src)
        body = content.splitlines(keepends=True)
        if body and not body[-1].endswith("\n"):
            body[-1] += "\n"
        diff = difflib.unified_diff(
            [], body, fromfile="/dev/null", tofile=f"b/{rel}", n=0,
        )
        chunks.append(f"diff --git a/{rel} b/{rel}\nnew file mode 100644\n" + "".join(diff))
    sys.stdout.write("\n".join(chunks))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(REPO_ROOT), help="Repository root to operate on.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Verify anchors only (no writes).")
    group.add_argument("--apply", action="store_true", help="Write the patched files.")
    group.add_argument("--emit-diff", action="store_true", help="Print a unified diff to stdout.")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.check:
        return cmd_check(root)
    if args.apply:
        return cmd_apply(root)
    return cmd_emit_diff(root)


if __name__ == "__main__":
    raise SystemExit(main())
