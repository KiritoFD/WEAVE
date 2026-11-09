import pathlib

p = pathlib.Path("sbflow/style_loss.py")
s = p.read_text(encoding="utf-8")
start = s.index("# --------------------------------------------------------------------------- #\n# SWD")
end = s.index("def supervised_contrastive(")
# keep everything before SWD header and after swd2_local block
new_block = '''# --------------------------------------------------------------------------- #
# SWD
# --------------------------------------------------------------------------- #
# V11 fix (docs/v11/V11_THEORY.md section 7, defect 4): the V10 implementation
# sliced the generated cloud inside `torch.no_grad()`, so `xg` was detached and the
# distribution loss carried NO gradient to the generator. That is why exp24 reported
# the same SWD as the unsupervised baseline. The estimator below is defined ONCE
# (rank-matched SWD) and implemented twice -- full batch and chunked -- so the
# chunked version can be verified to have exactly the full-batch gradient.
def _projections(dim: int, n_proj: int, device, dtype, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    p = torch.randn(dim, n_proj, generator=g).to(device=device, dtype=dtype)
    return F.normalize(p, dim=0)


def _reference_quantiles(py: torch.Tensor, n: int) -> torch.Tensor:
    """Sorted reference projections resampled to `n` ranks. Detached by construction."""
    m = py.shape[0]
    py_sorted = py.sort(dim=0).values
    if m == n:
        return py_sorted
    pos = torch.linspace(0, m - 1, n, device=py.device, dtype=py.dtype)
    lo = pos.floor().long().clamp(0, m - 1)
    hi = (lo + 1).clamp(0, m - 1)
    w = (pos - lo.to(pos.dtype)).unsqueeze(1)
    return py_sorted[lo] * (1 - w) + py_sorted[hi] * w


def _swd2_ranked(x: torch.Tensor, y: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Canonical estimator: sort both clouds per projection, match by rank.

    Gradient flows into `x` only (through the scatter), which is what we want: the
    reference cloud is data, never a parameter.
    """
    px = x @ proj
    n = px.shape[0]
    with torch.no_grad():
        perm = px.argsort(dim=0)                      # sorted position of each sample
        yq = _reference_quantiles(y @ proj, n)        # (n, P), constant
    px_sorted = torch.zeros_like(px).scatter(0, perm, px)
    return (px_sorted - yq).square().mean()


def swd2(x: torch.Tensor, y: torch.Tensor, n_proj: int = 64, seed: int = 0,
         proj: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Squared sliced Wasserstein distance between two point clouds (N, D), (M, D)."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("swd2 expects (N, D) and (M, D)")
    p = proj if proj is not None else _projections(x.shape[1], n_proj, x.device, x.dtype, seed)
    return _swd2_ranked(x, y, p)


def swd2_chunked(x: torch.Tensor, y: torch.Tensor, n_proj: int = 64, seed: int = 0,
                 chunk: int = 64, proj: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Same estimator as `swd2`, with the forward recomputed in chunks.

    Pass 1 (no_grad) fixes the global sort permutation and the reference quantiles.
    Pass 2 recomputes `x_chunk @ proj` WITH grad and scatters each sample to its
    global sorted position, so the value and the gradient both equal the full-batch
    computation. Verified numerically against `swd2` in tests/test_swd_grad.py.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("swd2_chunked expects (N, D) and (M, D)")
    p = proj if proj is not None else _projections(x.shape[1], n_proj, x.device, x.dtype, seed)
    n = x.shape[0]
    with torch.no_grad():
        perm = (x @ p).argsort(dim=0)
        yq = _reference_quantiles(y @ p, n)
    px_sorted = None
    for s in range(0, n, chunk):
        pxc = x[s:s + chunk] @ p
        if px_sorted is None:
            px_sorted = torch.zeros(n, p.shape[1], dtype=pxc.dtype, device=pxc.device)
        px_sorted = px_sorted.scatter(0, perm[s:s + chunk], pxc)
    return (px_sorted - yq).square().mean()


def swd2_local(x_patch: torch.Tensor, y_patch: torch.Tensor, n_proj: int = 32,
               seed: int = 0, chunk: int = 64,
               proj: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Local (patch) SWD: x_patch/y_patch are (B, P, D); compares pooled patch clouds."""
    if x_patch.ndim != 3 or y_patch.ndim != 3:
        raise ValueError("swd2_local expects (B, P, D)")
    xf = x_patch.reshape(-1, x_patch.shape[-1])
    yf = y_patch.reshape(-1, y_patch.shape[-1])
    return swd2_chunked(xf, yf, n_proj, seed, chunk, proj)


'''
p.write_text(s[:start] + new_block + s[end:], encoding="utf-8")
print("rewrote SWD block")
