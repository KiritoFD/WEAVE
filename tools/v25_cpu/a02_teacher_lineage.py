"""A02: teacher & anchor lineage registry and consistency audit (V25 Slot A).

Builds the lineage chain required by ``docs/19_pull_together/agent2.md`` A02:

    model_sha -> anchor_sha -> solver_hash -> output_hash

for every teacher-target generation path (train side) and every evaluation
path (eval side) of the D6 / PRISM-1 / Round-2 lineages, then compares the
two sides and reports (a) fully consistent pairs, (b) pairs claimed
equivalent but pending the 4090 B4-vs-T4 check, (c) definitely inconsistent
pairs.  Any inconsistency is surfaced programmatically (``find_inconsistencies``)
so the orchestrator can fail fast.

Design constraints (agent2.md section 11 + task card):
- CPU-only here: numpy + stdlib at module level; ``torch`` is imported lazily
  and only by ``probe_pt`` / ``check_b4_vs_t4`` (the latter is meant to run on
  the 4090 queue, never on the local laptop).
- ``solver_hash`` mirrors the repo convention ``exp.v22.config_hash``
  (``json.dumps(cfg, sort_keys=True, default=str)`` -> sha256), so solver
  dicts registered here can be cross-checked against the live experiment
  modules (drift test in ``tests/v25/test_a02_lineage.py``).
- Records whose checkpoint/tensor files are not present locally carry
  ``"unknown"`` hashes and never raise (legacy-probe semantics; the V19 raw
  T4 meta predates the ``t4_hash`` binding and is accepted as ``None`` by
  ``exp/v22/t4.py:34`` -- unknown-by-construction, not an error).
- No git, no ssh, no CUDA.  The 4090 validation command is *printed*, not run.

Anchor families (``anchor_kind``):
    t4_opt          AnisotropicS10Teacher optimisation output on test750,
                    version-bound by T4_HASH (exp/v22/t4.py).
    b4_canon        feedforward P1_B4_distill50 on canonicalized clean latent,
                    ``b4(canon(z, c1_gentle), 0.0, canon, tid)`` recipe
                    (exp/v19/routed_student.py:407-408 origin).
    z_src_clean     the clean source latent itself (Round-2 AB bridge; also
                    the P1-3 ``Anc_Source`` arm judged negative at eval time).
    p1_chroma       chroma-shift + PRISM-1 single-step output used AS an
                    anchor (PRISM-1 lambda=1.0 => alpha=0.20).
    avg_pool_pseudo avg_pool2d(3x3, replicate pad) of z_src pretending to be a
                    clean latent (V24 M3/M4/M5 historical negative).
    zp1_as_d6_anchor V24 M6 chain: avg_pool -> PRISM-1 -> its output fed to
                    the D6 flow as anchor (historical negative).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

UNKNOWN = "unknown"

# --------------------------------------------------------------------------- #
# hashes
# --------------------------------------------------------------------------- #
def _sorted_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def solver_hash(solver: Mapping, n: int | None = None) -> str:
    """Order-independent sha256 of a solver/config dict.

    Same serialization convention as ``exp.v22.config_hash`` (sorted JSON,
    ``default=str``); returns the full 64-hex digest, or its first ``n``
    characters when ``n`` is given.
    """
    digest = hashlib.sha256(_sorted_json(dict(solver)).encode("utf-8")).hexdigest()
    return digest if n is None else digest[:n]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Chunked file sha256; returns ``UNKNOWN`` for missing files (no raise)."""
    p = Path(path)
    if not p.is_file():
        return UNKNOWN
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tensor_bytes(t: Any) -> str:
    """sha256 over a tensor's raw bytes (contiguous, little-endian view)."""
    import torch

    tt = t.detach().cpu().contiguous()
    return hashlib.sha256(tt.numpy().tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# lineage record
# --------------------------------------------------------------------------- #
VERDICT_IDENTICAL = "IDENTICAL"
VERDICT_PENDING_4090 = "EQUIVALENT_PENDING_4090"
VERDICT_INCONSISTENT = "INCONSISTENT"
VERDICT_DISTINCT = "DISTINCT_LINEAGE"

#: anchor-family pairs that some document claims are interchangeable, together
#: with the evidence still required before the claim may be used for
#: attribution.  Keys are frozensets so order does not matter.
ANCHOR_EQUIVALENCE_CLAIMS = {
    frozenset(("t4_opt", "b4_canon")): (
        "'T4-equivalent' claim (exp/v25/precompute_train_latents.py:5); "
        "requires the 4090 B4-vs-T4 per-task L2 + per-channel quantile check "
        "(check_b4_vs_t4) to pass"
    ),
    frozenset(("t4_opt", "p1_chroma")): (
        "'PRISM-1 anchor ~ T4 anchor' point estimate (04_round1_results.md "
        "P1-3, tolerance edges grazed, tau arms misaligned); requires the "
        "tau-aligned paired bootstrap CI (A01 comparison #3) to pass"
    ),
}


@dataclass
class LineageRecord:
    """One teacher-target generation path or one evaluation path.

    ``*_sha`` fields hold hex digests, ``UNKNOWN`` when the artifact is not
    locally verifiable.  ``identity_key`` is the attribution key: two records
    may only be mutually attributed when it matches (modulo declared-equivalent
    anchors pending the 4090 check).
    """

    name: str
    side: str                      # "eval" | "train" | "legacy"
    stage: str                     # e.g. "V25-R1", "V25-R2", "V24-P0"
    model_path: str
    model_sha: str = UNKNOWN
    anchor_kind: str = UNKNOWN
    anchor_def: str = ""
    anchor_sha: str = UNKNOWN
    solver: Optional[dict] = None
    solver_h: str = UNKNOWN        # kept in sync with solver by __post_init__
    branches: str = UNKNOWN
    gate: str = UNKNOWN
    dataset: str = UNKNOWN         # "test750(cross600)" | "train4400" | ...
    noise_seed: str = UNKNOWN
    code_ref: str = ""             # "file.py:line" provenance
    notes: str = ""
    output_ref: str = ""           # measured eval.json / cache path
    output_sha: str = UNKNOWN

    def __post_init__(self) -> None:
        if self.solver is not None:
            self.solver_h = solver_hash(self.solver)
        elif not self.solver_h:
            self.solver_h = UNKNOWN

    def identity_key(self) -> tuple:
        return (self.model_sha, self.anchor_sha, self.solver_h, self.branches, self.gate)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping) -> "LineageRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def compare(a: LineageRecord, b: LineageRecord) -> dict:
    """Compare two records; never raises on unknown hashes.

    Verdicts:
      IDENTICAL             identity keys equal -> mutual attribution allowed.
                            UNKNOWN components are tolerated ONLY when the
                            underlying artifacts share the same path (sha fill
                            pending from the 4090 manifest), otherwise they
                            keep the pair pending.
      EQUIVALENT_PENDING_4090
                            same model/solver/branches/gate; anchors differ but
                            their family pair carries a written equivalence
                            claim (ANCHOR_EQUIVALENCE_CLAIMS) that is not yet
                            evidenced -> conditional, gated on the claim's
                            required check.
      INCONSISTENT          anchor families are semantically different without a
                            claim (or same family with different known
                            anchor_sha) -> attribution forbidden, reported
                            immediately.
      DISTINCT_LINEAGE      model/solver/branches/gate really differ -> both
                            records are valid but only water-level-interpolated
                            comparisons (A01) are meaningful.
    """
    components = (
        ("model_sha", a.model_sha, b.model_sha),
        ("solver_hash", a.solver_h, b.solver_h),
        ("branches", a.branches, b.branches),
        ("gate", a.gate, b.gate),
    )
    real_diffs: list[str] = []
    pending_components: list[str] = []
    model_sha_both_unknown = False
    for name, x, y in components:
        if x == y:
            if name == "model_sha" and x == UNKNOWN:
                model_sha_both_unknown = True  # cannot verify locally
            continue
        if UNKNOWN in (x, y):
            pending_components.append(name)
        else:
            real_diffs.append(name)

    reasons: list[str] = []
    for name in real_diffs:
        x, y = dict((n, (u, v)) for n, u, v in components)[name]
        if name == "model_sha":
            reasons.append(f"model_sha differs: {str(x)[:8]} vs {str(y)[:8]}")
        elif name == "solver_hash":
            reasons.append(f"solver_hash differs: {str(x)[:8]} vs {str(y)[:8]}")
        else:
            reasons.append(f"{name} differs: {x} vs {y}")

    anchor_equal = a.anchor_kind == b.anchor_kind and a.anchor_sha == b.anchor_sha
    claim = ANCHOR_EQUIVALENCE_CLAIMS.get(frozenset((a.anchor_kind, b.anchor_kind)))
    pipeline_pair = {a.side, b.side} <= {"eval", "train"} and a.model_path != b.model_path

    if not anchor_equal and claim is None:
        # anchors genuinely differ without a written equivalence claim
        if a.anchor_kind == b.anchor_kind and UNKNOWN not in (a.anchor_sha, b.anchor_sha):
            reasons.append(
                f"same anchor family ({a.anchor_kind}) but different anchor_sha "
                f"({str(a.anchor_sha)[:8]} vs {str(b.anchor_sha)[:8]}) -> version drift"
            )
        else:
            reasons.append(
                f"anchor families are semantically different: {a.anchor_kind} "
                f"({a.anchor_def}) vs {b.anchor_kind} ({b.anchor_def})"
            )
        return _verdict(VERDICT_INCONSISTENT, reasons, a, b)

    if real_diffs:
        # different weights / solver / branches regardless of anchor claims
        if claim is not None:
            reasons.append(f"(anchor pair also carries an unverified claim: {claim})")
        if pending_components:
            reasons.append("UNKNOWN identity components: " + ", ".join(sorted(pending_components)))
        return _verdict(VERDICT_DISTINCT, reasons, a, b)

    if claim is not None:
        reasons.append(
            f"anchor kinds differ ({a.anchor_kind} vs {b.anchor_kind}); "
            f"claim on file but evidence pending: {claim}"
        )
        if pipeline_pair:
            reasons.append(
                "model artifacts differ by path -> different weights entirely; "
                "the anchor claim cannot rescue attribution"
            )
            return _verdict(VERDICT_DISTINCT, reasons, a, b)
        if pending_components:
            reasons.append("UNKNOWN identity components: " + ", ".join(sorted(pending_components)))
        return _verdict(VERDICT_PENDING_4090, reasons, a, b)

    if pending_components:
        # identity equal except for UNKNOWN components
        same_artifact = (
            a.model_path == b.model_path and {a.side, b.side} <= {"eval", "train", "anchor"}
        )
        reasons.append(
            "identity equal up to UNKNOWN components: "
            + ", ".join(sorted(pending_components))
        )
        if same_artifact:
            reasons.append("(same artifact path: consistency by construction, sha fill pending)")
            return _verdict(VERDICT_IDENTICAL, reasons, a, b)
        return _verdict(VERDICT_PENDING_4090, reasons, a, b)

    if model_sha_both_unknown:
        if a.model_path == b.model_path and {a.side, b.side} <= {"eval", "train", "anchor"}:
            reasons.append(
                "(model sha unknown on this machine: same artifact path + "
                "config, sha fill pending from the 4090 manifest)"
            )
            return _verdict(VERDICT_IDENTICAL, reasons, a, b)
        reasons.append(
            "model sha unknown for BOTH records and artifact paths differ: "
            "identity unverifiable until the 4090 manifest fills the shas"
        )
        return _verdict(VERDICT_PENDING_4090, reasons, a, b)

    return _verdict(VERDICT_IDENTICAL, reasons, a, b)


def _verdict(verdict: str, reasons: list[str], a: LineageRecord, b: LineageRecord) -> dict:
    attribution = {
        VERDICT_IDENTICAL: "yes (same lineage)",
        VERDICT_PENDING_4090: "conditional - only after the 4090 check passes",
        VERDICT_INCONSISTENT: "NO - forbidden in either direction",
        VERDICT_DISTINCT: "no - A01 water-level interpolation only",
    }[verdict]
    return {
        "a": a.name,
        "b": b.name,
        "verdict": verdict,
        "reasons": reasons,
        "attribution_allowed": attribution,
        "a_identity": list(a.identity_key()),
        "b_identity": list(b.identity_key()),
    }


def find_inconsistencies(comparisons: list[dict]) -> list[dict]:
    """All comparisons whose verdict is INCONSISTENT (A02: report immediately)."""
    return [c for c in comparisons if c["verdict"] == VERDICT_INCONSISTENT]


# --------------------------------------------------------------------------- #
# fixed literals copied from the experiment code (code_ref proves provenance)
# --------------------------------------------------------------------------- #
# exp/v22/core.py:23-32
D5_SOLVER_LIT = dict(
    tau_min=0.28, tau_hi=0.66, steps=10,
    ws_fig=4.2, wsp_fig_late=2.4, wsp_abs=5.2,
    wc_fig=1.85, wc_abs=0.80, lscale_abs=0.34, rescale_phi=0.48,
)
D6_SOLVER_LIT = dict(
    tau_min=0.40, tau_hi=0.74, steps=8,
    ws_fig=4.4, wsp_fig_late=1.8, wsp_abs=4.8,
    wc_fig=1.95, wc_abs=1.00, lscale_abs=0.44, rescale_phi=0.48,
)
# exp/v25/r1_inference_ablation.py:31  D6H_SOLVER = dict(D6_SOLVER, tau_min=0.48, tau_hi=0.78)
D6H_SOLVER_LIT = dict(D6_SOLVER_LIT, tau_min=0.48, tau_hi=0.78)
# exp/v25/r1_inference_ablation.py:151 (DualSWD arm solver)
DSWD_SOLVER_LIT = dict(D6_SOLVER_LIT, tau_min=0.58, tau_hi=0.78, ws_fig=5.2)
# exp/v22/t4.py:11-23 (T4 teacher config; T4_HASH = config_hash of this)
T4_CONFIG_LIT = {
    "arm": "V19_T4_sweet_vgg_nosid",
    "mode": "s10_vgg",
    "alpha_c": 0.18,
    "m_steps": 2,
    "rho_y_edge": 0.065,
    "rho_y_low": 0.12,
    "rho_c": 0.22,
    "lcn_budget": 0.024,
    "etas": [0.055, 0.105],
    "is_wrong": False,
    "batch": 25,
}
T4_HASH_LOCAL = solver_hash(T4_CONFIG_LIT, n=12)

P1_CKPT = "runs/v19/students/s2_prism1_v2.pt"          # exp/v24/p0_m0_m2.py:26
V20_CKPT = "runs/v20/slf_flow/slf_core_v2.pt"          # exp/v24/p0_m0_m2.py:27
B4_CKPT = "runs/v6/P1_B4_distill50/P1_B4_distill50.pt" # exp/v19/routed_student.py:361
SLF_FINAL_CKPT = "runs/v20/slf_flow/slf_final.pt"      # exp/v22/tmatrix.py:34
DUAL_SWD_CKPT = "runs/v23/dual_swd/slf_DualSWD_pure.pt"  # exp/v25/r1_inference_ablation.py:29

EVAL_SEED_PROTOCOL = "20261185"   # src/utils/protocol.py:12
R2_EVAL_SEED = "20261119"         # exp/v25/r2_train_ablation.py:181
TEACHER_EPS_SEED = "20263502"     # exp/v25/precompute_train_latents.py:115


def _checkpoint_shas() -> dict[str, str]:
    return {
        P1_CKPT: sha256_file(REPO_ROOT / P1_CKPT),
        V20_CKPT: sha256_file(REPO_ROOT / V20_CKPT),
        B4_CKPT: sha256_file(REPO_ROOT / B4_CKPT),
        SLF_FINAL_CKPT: sha256_file(REPO_ROOT / SLF_FINAL_CKPT),
        DUAL_SWD_CKPT: sha256_file(REPO_ROOT / DUAL_SWD_CKPT),
    }


def _output_sha(rel: str) -> str:
    return sha256_file(REPO_ROOT / rel)


# --------------------------------------------------------------------------- #
# registry (every entry cites its code_ref)
# --------------------------------------------------------------------------- #
def build_registry() -> list[LineageRecord]:
    """All teacher-target generation paths and evaluation paths relevant to the
    V25 mechanism A/B/C readouts.  Checkpoint shas are filled when the file is
    locally present (4090-only artifacts stay UNKNOWN, by design)."""
    ck = _checkpoint_shas()

    # standalone anchor-provenance records (side="anchor": no own model)
    anchor_records = [
        LineageRecord(
            name="ANCHOR_T4_TEST750", side="anchor", stage="V19->V25",
            model_path="(none: AnisotropicS10Teacher optimisation, no ckpt file)",
            model_sha=UNKNOWN,
            anchor_kind="t4_opt",
            anchor_def=(
                "AnisotropicS10Teacher.solve_batch(mode='s10_vgg', 2 steps) on "
                "test750; cache runs/v22/cache/t4_latents_750.pt bound to "
                "T4_HASH (exp/v22/t4.py:86-117; config exp/v22/t4.py:11-23); "
                "V19 raw meta predates t4_hash and is accepted as None -> "
                "version-unknown legacy probe (exp/v22/t4.py:34)"
            ),
            anchor_sha=T4_HASH_LOCAL,
            solver=None, branches="anchor_cache", gate="none",
            dataset="test750", noise_seed=EVAL_SEED_PROTOCOL,
            code_ref="exp/v22/t4.py:45-83",
            notes="tensor sha256 pending from the 4090 manifest",
        ),
        LineageRecord(
            name="ANCHOR_B4_TRAIN4400", side="anchor", stage="V25-precompute",
            model_path="(none: feedforward anchor, ckpt = " + B4_CKPT + ")",
            model_sha=ck[B4_CKPT],
            anchor_kind="b4_canon",
            anchor_def=(
                "b4(canon(z, c1_gentle), 0.0, canon, tid) on train4400; recipe "
                "origin exp/v19/routed_student.py:407-408 (there: wrong-label "
                "counterfactual anchors); declared 'T4-equivalent' by "
                "exp/v25/precompute_train_latents.py:5"
            ),
            anchor_sha=ck[B4_CKPT],
            solver=None, branches="anchor_cache", gate="none",
            dataset="train4400", noise_seed="deterministic_feedforward",
            code_ref="exp/v25/precompute_train_latents.py:75-84",
            notes="B4 = 218K-param C3UNet distill (V6/P1 era), NOT a "
                  "bit-equivalent of the 2-step optimised T4: equivalence is "
                  "a written claim, pending the 4090 tensor check",
        ),
    ]

    t4_anchor = dict(
        anchor_kind="t4_opt",
        anchor_def=(
            "T4 anchor (see ANCHOR_T4_TEST750); cache bound to T4_HASH"
        ),
        anchor_sha=T4_HASH_LOCAL,
    )
    b4_anchor = dict(
        anchor_kind="b4_canon",
        anchor_def=(
            "b4(canon(z, c1_gentle), 0.0, canon, tid) feedforward anchor, "
            "recipe origin exp/v19/routed_student.py:407-408; declared "
            "'T4-equivalent' by exp/v25/precompute_train_latents.py:5"
        ),
        anchor_sha=ck[B4_CKPT],
    )
    r2_ab_anchor = dict(
        anchor_kind="z_src_clean",
        anchor_def=(
            "Round-2 Anchor-Bridge: z_init=(1-tau0)*noise+tau0*z1 with "
            "tau0~U(0.40,0.78), z1 = clean train latent (za = z1, "
            "exp/v25/r2_train_ablation.py:123-137)"
        ),
        anchor_sha=UNKNOWN,  # derived from v18 train cache tensor; sha on 4090
    )
    p1_anchor = dict(
        anchor_kind="p1_chroma",
        anchor_def=(
            "PRISM-1(lambda=1.0) output as anchor: chroma shift alpha=0.20 + "
            "s2_prism1_v2 single step (exp/v24/p0_m0_m2.py:31-63, cached in "
            "runs/v24/cache/p0_base_latents_750.pt z_p1_100)"
        ),
        anchor_sha=UNKNOWN,  # V24 cache tensor lives on the 4090
    )
    zsrc_anchor = dict(
        anchor_kind="z_src_clean",
        anchor_def="raw source latent z_src_750 as anchor (exp/v25/r1_inference_ablation.py:178-179)",
        anchor_sha=UNKNOWN,
    )

    recs: list[LineageRecord] = anchor_records

    # ---------------- eval side: D6 family (T4 anchor) ----------------------
    recs.append(LineageRecord(
        name="EVAL_V20_D6_ORIG", side="eval", stage="V20-stage2",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6_SOLVER_LIT), branches="full", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v20/run_v20.py:389-391,363-364",
        anchor_kind=t4_anchor["anchor_kind"], anchor_def=(
            "T4 via ensure_t4_latents_750 (exp/v20/run_v20.py:121-139 -> exp/v22/t4.py)"),
        anchor_sha=T4_HASH_LOCAL,
        notes="arm V20_D6_core_slf_c69; solver dict == V22 D6_SOLVER literal",
        output_ref="runs/v20/measured/V20_D6_core_slf_c69/eval.json",
        output_sha=_output_sha("runs/v20/measured/V20_D6_core_slf_c69/eval.json"),
    ))
    recs.append(LineageRecord(
        name="EVAL_V24_M0_V20_D6", side="eval", stage="V24-P0",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6_SOLVER_LIT), branches="full", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v24/p0_m0_m2.py:67-85,120-134",
        **t4_anchor,
        notes="re-eval reused by V25-R1 as M0_v20_d6 baseline",
        output_ref="runs/v24/measured/M0_v20_d6/eval.json",
        output_sha=_output_sha("runs/v24/measured/M0_v20_d6/eval.json"),
    ))
    recs.append(LineageRecord(
        name="EVAL_V25_P02_D6H_CSJ", side="eval", stage="V25-R1",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6H_SOLVER_LIT), branches="cs_j", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v25/r1_inference_ablation.py:165-172",
        **t4_anchor,
        notes="reference arm of P0-2/P1-3/P1-4; cs_j adopted as main config",
        output_ref="runs/v25/measured/P02_d6h_csj/eval.json",
        output_sha=_output_sha("runs/v25/measured/P02_d6h_csj/eval.json"),
    ))
    recs.append(LineageRecord(
        name="EVAL_V25_P02_D5_FULL", side="eval", stage="V25-R1",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D5_SOLVER_LIT), branches="full", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v25/r1_inference_ablation.py:160-172",
        **t4_anchor,
        notes="D5 water level; ws5.5/ws4.2/cs_j variants share this lineage",
        output_ref="runs/v25/measured/P02_d5_full/eval.json",
        output_sha=_output_sha("runs/v25/measured/P02_d5_full/eval.json"),
    ))
    recs.append(LineageRecord(
        name="EVAL_V25_CURVE_B_D6", side="eval", stage="V25-R1",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6_SOLVER_LIT, tau_min=0.66, tau_hi=0.78),
        branches="full", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v25/r1_inference_ablation.py:145-149",
        **t4_anchor,
        notes="tau_min shift arms t050/t057/t066; strongest single-model D6",
        output_ref="runs/v25/measured/B_d6_t066/eval.json",
        output_sha=_output_sha("runs/v25/measured/B_d6_t066/eval.json"),
    ))
    recs.append(LineageRecord(
        name="EVAL_V25_DSWD_T058", side="eval", stage="V25-R1",
        model_path=DUAL_SWD_CKPT, model_sha=ck[DUAL_SWD_CKPT],
        solver=dict(DSWD_SOLVER_LIT), branches="cs_j", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v25/r1_inference_ablation.py:150-153",
        **t4_anchor,
        notes="separate weights + ws_fig 5.2: own lineage, second candidate",
        output_ref="runs/v25/measured/B_dswd_t058/eval.json",
        output_sha=_output_sha("runs/v25/measured/B_dswd_t058/eval.json"),
    ))

    # ---------------- eval side: anchor swaps (P1-3) ------------------------
    recs.append(LineageRecord(
        name="EVAL_V25_P13_ANCP1", side="eval", stage="V25-R1",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6H_SOLVER_LIT, tau_min=0.40), branches="cs_j", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v25/r1_inference_ablation.py:175-184",
        **p1_anchor,
        notes="PRISM-1-anchor fast path candidate (tau not aligned with P02 ref)",
        output_ref="runs/v25/measured/P13_ancp1_t040/eval.json",
        output_sha=_output_sha("runs/v25/measured/P13_ancp1_t040/eval.json"),
    ))
    recs.append(LineageRecord(
        name="EVAL_V25_P13_ANCSRC", side="eval", stage="V25-R1",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6H_SOLVER_LIT, tau_min=0.40), branches="cs_j", gate="original",
        dataset="test750(cross600)", noise_seed=EVAL_SEED_PROTOCOL,
        code_ref="exp/v25/r1_inference_ablation.py:178-184",
        **zsrc_anchor,
        notes="JUDGED NEGATIVE (Top-1 7.17%, output ~= source): kept as the "
              "canonical counter-example of anchor semantics",
        output_ref="runs/v25/measured/P13_ancsrc_t040/eval.json",
        output_sha=_output_sha("runs/v25/measured/P13_ancsrc_t040/eval.json"),
    ))

    # ---------------- eval side: PRISM-1 single-step ------------------------
    recs.append(LineageRecord(
        name="EVAL_PRISM1_L100", side="eval", stage="V19/V24/V25",
        model_path=P1_CKPT, model_sha=ck[P1_CKPT],
        solver=None, branches="single_step", gate="none",
        dataset="test750(cross600)", noise_seed="deterministic_feedforward",
        code_ref="exp/v24/p0_m0_m2.py:31-63 (V24 M0_prism1_l100); V19 sprint2",
        anchor_kind="z0_chroma",
        anchor_def=(
            "apply_latent_chroma_shift(z0_750, tid, alpha=min(0.45,0.20*lam)) "
            "as flow-start basis; student output = z_anc + lam*delta"
        ),
        anchor_sha=UNKNOWN,
        notes="lambda in {0.75,1.0,1.3,1.6}; same_mask bypass on same150",
        output_ref="runs/v24/measured/M0_prism1_l100/eval.json",
        output_sha=_output_sha("runs/v24/measured/M0_prism1_l100/eval.json"),
    ))

    # ---------------- eval side: Round 2 trained arms -----------------------
    recs.append(LineageRecord(
        name="EVAL_V25_R2_TRAINED", side="eval", stage="V25-R2",
        model_path="runs/v25/r2_tmatrix/slf_Tr_T{0..3}.pt", model_sha=UNKNOWN,
        solver=dict(D6H_SOLVER_LIT), branches="cs_j", gate="original",
        dataset="test750(cross600)", noise_seed=R2_EVAL_SEED,
        code_ref="exp/v25/r2_train_ablation.py:186-195",
        **t4_anchor,
        notes="eps seed 20261119 differs from R1/V24/V20 (20261185): "
              "cross-ROUND paired CI carries a noise-draw confound",
        output_ref="runs/v25/measured/V25_Tr_T0_d6hcsj/eval.json",
        output_sha=_output_sha("runs/v25/measured/V25_Tr_T0_d6hcsj/eval.json"),
    ))

    # ---------------- train side: V25 teacher cache -------------------------
    recs.append(LineageRecord(
        name="TRAIN_V25_TEACHER_CACHE", side="train", stage="V25-precompute",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6H_SOLVER_LIT), branches="cs_j", gate="original",
        dataset="train4400", noise_seed=TEACHER_EPS_SEED,
        code_ref="exp/v25/precompute_train_latents.py:35-43,112-122",
        **b4_anchor,
        notes="teacher triple == EVAL_V25_P02_D6H_CSJ except anchor family "
              "(b4_canon vs t4_opt) and dataset; z_d6_tr is the mechanism A/B "
              "teacher target; also stores z_anc_tr/z_p1_tr(lambda=1.0)",
        output_ref="runs/v25/cache/train_latents_4400.pt",
        output_sha=_output_sha("runs/v25/cache/train_latents_4400.pt"),
    ))
    recs.append(LineageRecord(
        name="TRAIN_V25_PRISM1_TARGETS", side="train", stage="V25-precompute",
        model_path=P1_CKPT, model_sha=ck[P1_CKPT],
        solver=None, branches="single_step", gate="none",
        dataset="train4400", noise_seed="deterministic_feedforward",
        code_ref="exp/v25/precompute_train_latents.py:86-110",
        anchor_kind="z0_chroma",
        anchor_def="apply_latent_chroma_shift(z0_tr, tid, alpha=0.20) == lambda=1.0 recipe",
        anchor_sha=UNKNOWN,
        notes="recipe-identical to EVAL_PRISM1_L100 but on the train domain",
        output_ref="runs/v25/cache/train_latents_4400.pt",
        output_sha=_output_sha("runs/v25/cache/train_latents_4400.pt"),
    ))

    # ---------------- train side: Round 2 self-paired FM --------------------
    recs.append(LineageRecord(
        name="TRAIN_V25_R2_SELFPAIRED", side="train", stage="V25-R2",
        model_path=SLF_FINAL_CKPT, model_sha=ck[SLF_FINAL_CKPT],
        solver=None, branches="flow_matching_target", gate="none",
        dataset="train4400", noise_seed="20263500(seed_all shared)",
        code_ref="exp/v25/r2_train_ablation.py:54-165",
        **r2_ab_anchor,
        notes="target z1 = clean train latent (NO D6 teacher target); AB arm "
              "T2/T3 only; training-time anchor is z_src-clean, which P1-3 "
              "shows collapses style at inference (Top-1 7.17%)",
        output_ref="runs/v25/r2_tmatrix/slf_Tr_T3.pt",
        output_sha=_output_sha("runs/v25/r2_tmatrix/slf_Tr_T3.pt"),
    ))

    # ---------------- legacy negative paths (banned) ------------------------
    recs.append(LineageRecord(
        name="LEGACY_V24_M3_M4_POOL", side="legacy", stage="V24-P1",
        model_path=P1_CKPT, model_sha=ck[P1_CKPT],
        solver=None, branches="residual_training", gate="none",
        dataset="train4400", noise_seed=UNKNOWN,
        code_ref="exp/v24/p1_m3_m4_residual.py:179,213",
        anchor_kind="avg_pool_pseudo",
        anchor_def="z0_c = avg_pool2d(pad_replicate(z_src),3,1) pretending to be a clean latent",
        anchor_sha="pseudo_avgpool3x3",
        notes="M3/M5 failed-arm lineage; precompute_train_latents.py:4 explicitly "
              "replaced it with real z0_tr - BANNED from any attribution",
    ))
    recs.append(LineageRecord(
        name="LEGACY_V24_M6_ZP1_ANCHOR", side="legacy", stage="V24-P2",
        model_path=V20_CKPT, model_sha=ck[V20_CKPT],
        solver=dict(D6_SOLVER_LIT), branches="cs_j", gate="original",
        dataset="train4400", noise_seed=UNKNOWN,
        code_ref="exp/v24/p2_m6_flow_bridge.py:82-89",
        anchor_kind="zp1_as_d6_anchor",
        anchor_def="avg_pool z0 -> PRISM-1 -> its output fed to the D6 flow as anchor",
        anchor_sha="pseudo_p1_chain",
        notes="double-stylized fake anchor chain - BANNED from any attribution",
    ))
    return recs


def registry_by_name(registry: list[LineageRecord] | None = None) -> dict[str, LineageRecord]:
    return {r.name: r for r in (registry if registry is not None else build_registry())}


# --------------------------------------------------------------------------- #
# pre-registered pair plan -> a/b/c tables (+ informational distinct table)
# --------------------------------------------------------------------------- #
#: pairs (a, b, table) whose consistency matters for cross-attribution of
#: teacher readouts.  table: a=identical, b=pending-4090, c=inconsistent,
#: d=distinct-by-design (informational).
PAIR_PLAN = [
    ("EVAL_V20_D6_ORIG", "EVAL_V24_M0_V20_D6", "a"),
    ("TRAIN_V25_PRISM1_TARGETS", "EVAL_PRISM1_L100", "a"),
    ("ANCHOR_T4_TEST750", "ANCHOR_B4_TRAIN4400", "b"),
    ("TRAIN_V25_TEACHER_CACHE", "EVAL_V25_P02_D6H_CSJ", "b"),
    ("TRAIN_V25_R2_SELFPAIRED", "EVAL_V25_R2_TRAINED", "c"),
    ("EVAL_V25_P13_ANCSRC", "EVAL_V25_P02_D6H_CSJ", "c"),
    ("LEGACY_V24_M3_M4_POOL", "TRAIN_V25_TEACHER_CACHE", "c"),
    ("LEGACY_V24_M6_ZP1_ANCHOR", "EVAL_V20_D6_ORIG", "c"),
    ("EVAL_V24_M0_V20_D6", "EVAL_V25_P02_D6H_CSJ", "d"),
    ("EVAL_V25_CURVE_B_D6", "EVAL_V25_P02_D6H_CSJ", "d"),
    ("EVAL_V25_DSWD_T058", "EVAL_V25_P02_D6H_CSJ", "d"),
    ("EVAL_V25_P13_ANCP1", "EVAL_V25_P02_D6H_CSJ", "d"),
    ("TRAIN_V25_TEACHER_CACHE", "EVAL_V25_R2_TRAINED", "d"),
]


def run_pair_plan(registry: list[LineageRecord] | None = None) -> list[dict]:
    reg = registry_by_name(registry)
    out = []
    for a, b, table in PAIR_PLAN:
        comp = compare(reg[a], reg[b])
        comp["table"] = table
        out.append(comp)
    return out


# --------------------------------------------------------------------------- #
# hard usage constraints for the G queue and Slot B
# --------------------------------------------------------------------------- #
CONSTRAINT_RULES = [
    "C1 (teacher identity): mechanism A/B readouts may only be attributed to a "
    "teacher whose full identity key (model_sha, anchor_sha, solver_hash, "
    "branches, gate) matches the record; the single registered D6 teacher "
    "target is TRAIN_V25_TEACHER_CACHE (z_d6_tr).",
    "C2 (B4-T4 gate): any comparison between train-side teacher numbers "
    "(B4-canon anchor) and eval-side T4-anchor numbers is BLOCKED until the "
    "4090 check_b4_vs_t4 command passes its equivalence threshold; P1-3 "
    "already showed anchor semantics can move Top-1 by ~48pp, so this is a "
    "hard gate, not a formality.",
    "C3 (Round-2 AB): TRAIN_V25_R2_SELFPAIRED trains from the clean source "
    "latent, not from a T4/B4 anchor.  Its checkpoints may be *evaluated* on "
    "T4 anchors (that is what r2 does), but must never be described as "
    "teacher-anchor-consistent training; Tr_T2/T3 vs P02*/M0* cross-"
    "attribution is forbidden.",
    "C4 (noise seed): V20/V24/R1 evals share EVAL_SEED=20261185; Round-2 "
    "evals use 20261119.  Paired bootstrap between an R1 arm and an R2 arm "
    "mixes two noise draws -> rerun one side with the other's seed (or "
    "report as seed-unmatched) before any CI-based decision.",
    "C5 (legacy ban): V24 M3/M4/M5 avg_pool pseudo anchors and the M6 "
    "z_p1-as-anchor chain are closed negative lineages; their latents "
    "(ref_teacher_pairs_2000.pt et al.) must not seed any V25+ teacher or "
    "anchor cache.",
    "C6 (DualSWD): B_dswd_t058 uses different weights and ws_fig=5.2 -> its "
    "numbers form a separate lineage; compare only via A01 water-level "
    "interpolation, never as same-teacher pairs.",
    "C7 (test750): all quality decisions read cross600 on the full test750 "
    "(user decision 2026-11-14); dev previews never enter lineage tables.",
    "C8 (cache reuse): every new anchor/teacher cache must store its "
    "LineageRecord.to_dict() (identity key + shas) alongside the tensors; "
    "loaders must compare the stored key against the requested one and "
    "regenerate on mismatch (pattern of exp/v22/t4.py).",
]


# --------------------------------------------------------------------------- #
# cache probing (lazy torch; large files are never fully materialized)
# --------------------------------------------------------------------------- #
def probe_pt(path: str | Path) -> dict:
    """Read only metadata of a .pt (keys / teacher_spec / shapes).

    Files > 100 MB are memory-mapped so tensors stay on disk; only small
    python objects (teacher_spec, hashes) are materialized.  Missing or
    unreadable files yield ``{"status": "unknown", ...}`` -- never raises
    (legacy-probe contract).
    """
    p = Path(path)
    info: dict[str, Any] = {"path": str(p.as_posix()), "exists": p.is_file()}
    if not p.is_file():
        info["status"] = "missing"
        return info
    try:
        info["size_mb"] = round(p.stat().st_size / 1e6, 2)
    except OSError:
        info["status"] = "unknown"
        return info
    try:
        import torch

        if info["size_mb"] > 100:
            ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        else:
            ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as exc:  # unreadable / not a torch file / torch missing
        info["status"] = "unknown"
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    if not isinstance(ck, dict):
        info["status"] = "unknown"
        info["error"] = f"top-level object is {type(ck).__name__}, not dict"
        return info
    info["status"] = "ok"
    meta: dict[str, Any] = {}
    shapes: dict[str, list] = {}
    for k, v in ck.items():
        if isinstance(v, torch.Tensor):
            shapes[k] = list(v.shape)
        else:
            meta[k] = v
    info["tensor_shapes"] = shapes
    info["meta"] = meta
    info["hashes"] = {"file_sha256": sha256_file(p)}
    return info


# --------------------------------------------------------------------------- #
# B4 vs T4 per-task check (intended for the 4090 queue; pure math is testable)
# --------------------------------------------------------------------------- #
def b4_t4_stats(z_b4: Any, z_t4: Any, z_ref: Any | None = None) -> dict:
    """Per-task L2 + per-channel quantile comparison of two anchor sets.

    ``z_*`` are (N,4,64,64) tensors (torch) or arrays (numpy).  When ``z_ref``
    (e.g. the raw source latent z0) is given, the mean task L2 is normalized
    by the T4-vs-ref distance scale, giving a scale-free divergence ratio.
    Pure math: works on CPU numpy; torch tensors are converted transparently.
    """
    def np_(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    a, b = np_(z_b4).astype(np.float64), np_(z_t4).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    flat_a, flat_b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    l2 = np.linalg.norm(flat_a - flat_b, axis=1)
    out: dict[str, Any] = {
        "n": int(len(a)),
        "task_l2_mean": float(l2.mean()),
        "task_l2_p50": float(np.median(l2)),
        "task_l2_p95": float(np.percentile(l2, 95)),
        "task_l2_max": float(l2.max()),
    }
    ch_a = a.reshape(len(a), a.shape[1], -1).mean(axis=2)   # (N, C) per-channel means
    ch_b = b.reshape(len(b), b.shape[1], -1).mean(axis=2)
    qs = (0.01, 0.25, 0.50, 0.75, 0.99)
    q_a = np.quantile(ch_a, qs, axis=0)                     # (Q, C)
    q_b = np.quantile(ch_b, qs, axis=0)
    out["channel_quantile_absdiff_max"] = float(np.abs(q_a - q_b).max())
    out["channel_mean_absdiff_mean"] = float(np.abs(ch_a.mean(0) - ch_b.mean(0)).mean())
    if z_ref is not None:
        r = np_(z_ref).astype(np.float64).reshape(len(a), -1)
        l2_ref = np.linalg.norm(flat_b - r, axis=1)
        scale = float(l2_ref.mean())
        out["ref_scale_t4_vs_src_mean_l2"] = scale
        out["divergence_ratio"] = float(l2.mean() / scale) if scale > 0 else None
        ratio = out["divergence_ratio"]
        if ratio is None:
            out["reading"] = "inconclusive (zero reference scale)"
        elif ratio < 0.15:
            out["reading"] = "equivalent-scale (ratio < 0.15)"
        elif ratio > 0.60:
            out["reading"] = "distinct-scale (ratio > 0.60) -> treat anchors as different lineages"
        else:
            out["reading"] = "inconclusive (0.15 <= ratio <= 0.60) -> decide with measured evals"
    return out


def check_b4_vs_t4(
    t4_cache: str | Path = "runs/v22/cache/t4_latents_750.pt",
    b4_ckpt: str | Path = B4_CKPT,
    z0_cache: str | Path = "runs/v9_local_4070/cache/n2c_z.pt",
    tasks_json: str | Path = "eval_protocol/tasks_750.json",
    device: str = "cuda",
) -> dict:
    """Build B4+canonicalizer anchors for test750 and compare with the T4 cache.

    Meant to be executed ON THE 4090 (``python -m tools.v25_cpu.a02_teacher_lineage
    --check-b4-t4``); locally it exits with status BLOCKED because the caches
    and GPU are absent.  No ssh is ever performed here.
    """
    root = REPO_ROOT
    missing = [str(p) for p in (root / t4_cache, root / b4_ckpt, root / z0_cache) if not (root / p).is_file()]
    if missing:
        return {"status": "BLOCKED", "missing": missing,
                "reason": "T4/B4 artifacts not present on this machine; run the printed command on the 4090"}
    import torch

    from src.infer import canonicalizer, load_c3unet

    t4_ck = torch.load(root / t4_cache, map_location="cpu", weights_only=False)
    z_t4 = t4_ck.get("z_t4", t4_ck) if isinstance(t4_ck, dict) else t4_ck
    z_t4 = z_t4.float()
    if isinstance(t4_ck, dict) and t4_ck.get("t4_hash") not in (None, T4_HASH_LOCAL):
        return {"status": "BLOCKED",
                "reason": f"t4 cache hash {t4_ck.get('t4_hash')} != T4_HASH {T4_HASH_LOCAL}: regenerate first"}

    tasks = json.loads((root / tasks_json).read_text(encoding="utf-8"))["tasks"]
    z0 = torch.load(root / z0_cache, map_location="cpu", weights_only=False)["z0"].float()
    assert len(tasks) == len(z0) == len(z_t4) == 750, "anchor task count mismatch"

    b4, _ = load_c3unet(root / b4_ckpt, device)
    canon = canonicalizer(device)
    z_b4 = torch.zeros_like(z0)
    tid = torch.tensor([t["target_style_id"] for t in tasks])
    with torch.no_grad():
        for st in range(0, 750, 50):
            sl = slice(st, st + 50)
            zb = z0[sl].to(device)
            c_can = canon(zb, mode="c1_gentle", augment=False)
            z_b4[sl] = b4(c_can, 0.0, c_can, tid[sl].to(device)).cpu().float()

    stats = b4_t4_stats(z_b4, z_t4, z_ref=z0)
    return {
        "status": "OK",
        "stats": stats,
        "t4_cache_sha256": sha256_file(root / t4_cache),
        "t4_hash_bound": t4_ck.get("t4_hash") if isinstance(t4_ck, dict) else None,
        "b4_ckpt_sha256": sha256_file(root / b4_ckpt),
        "z_t4_sha256": sha256_tensor_bytes(z_t4),
        "z_b4_sha256": sha256_tensor_bytes(z_b4),
    }


def b4_t4_command() -> str:
    """The one G-queue line to run on the 4090 (printed, never executed here)."""
    return (
        'ssh 4090 "cd /root/Workspace/WEAVE && OMP_NUM_THREADS=2 '
        "python -m tools.v25_cpu.a02_teacher_lineage --check-b4-t4 --device cuda "
        '> runs/v25/cache/a02_b4_vs_t4.json"'
    )


# --------------------------------------------------------------------------- #
# report rendering
# --------------------------------------------------------------------------- #
def render_registry(registry: list[LineageRecord]) -> str:
    lines = [
        "| record | side | stage | model_path (sha8) | anchor_kind (sha8) | solver8 | branches/gate | dataset | seed | code_ref |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in registry:
        lines.append(
            f"| `{r.name}` | {r.side} | {r.stage} "
            f"| `{r.model_path}` ({r.model_sha[:8]}) "
            f"| {r.anchor_kind} ({r.anchor_sha[:8]}) "
            f"| {r.solver_h[:8] if r.solver_h != UNKNOWN else UNKNOWN} "
            f"| {r.branches}/{r.gate} | {r.dataset} | {r.noise_seed} | `{r.code_ref}` |"
        )
    return "\n".join(lines)


def render_table(comparisons: list[dict], table: str, title: str) -> str:
    rows = [c for c in comparisons if c["table"] == table]
    lines = [f"### {title}", "",
             "| a | b | verdict | attribution | reason |",
             "|---|---|---|---|---|"]
    for c in rows:
        reason = "; ".join(c["reasons"]) or "-"
        lines.append(
            f"| `{c['a']}` | `{c['b']}` | **{c['verdict']}** | {c['attribution_allowed']} | {reason} |"
        )
    return "\n".join(lines)


def render_constraints() -> str:
    return "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(CONSTRAINT_RULES))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="write the full registry+comparison JSON here")
    ap.add_argument("--probe", default=None, help="probe a .pt file (keys/teacher_spec only)")
    ap.add_argument("--check-b4-t4", action="store_true",
                    help="run the B4-vs-T4 per-task check (4090 only; prints BLOCKED elsewhere)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    if args.probe:
        print(json.dumps(probe_pt(args.probe), indent=2, default=str))
        return 0
    if args.check_b4_t4:
        print(json.dumps(check_b4_vs_t4(device=args.device), indent=2, default=str))
        return 0

    reg = build_registry()
    comps = run_pair_plan(reg)
    bad = find_inconsistencies(comps)
    print("== lineage registry ==")
    print(render_registry(reg))
    print()
    print(render_table(comps, "a", "(a) fully consistent pairs"))
    print()
    print(render_table(comps, "b", "(b) equivalent pending the 4090 B4-vs-T4 / sha check"))
    print()
    print(render_table(comps, "c", "(c) definitely inconsistent pairs"))
    print()
    print(render_table(comps, "d", "(d) distinct lineages (informational: A01 interpolation only)"))
    print()
    if bad:
        print(f"!! {len(bad)} INCONSISTENT pair(s): " + ", ".join(f"{c['a']} vs {c['b']}" for c in bad))
    print()
    print("== 4090 pending command ==")
    print(b4_t4_command())
    print()
    print("== usage constraints (G queue / Slot B) ==")
    print(render_constraints())
    if args.json:
        payload = {
            "registry": [r.to_dict() for r in reg],
            "comparisons": comps,
            "inconsistencies": bad,
            "constraints": CONSTRAINT_RULES,
            "b4_t4_command": b4_t4_command(),
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
