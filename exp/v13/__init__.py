"""V13: LocalH2-GCL — local stroke attention + differentiable GCL on H2.

Fixes the eight V11b issues (plan §1) in NEW modules; exp/v11b stays frozen as
the historical baseline. Old checkpoints keep their original (÷√2) sigmas; new
training uses the corrected per-band estimates.
"""
