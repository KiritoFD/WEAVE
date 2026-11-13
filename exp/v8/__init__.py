"""V8: generator-constrained target-style optimisation.

Three routes, runnable independently (24h plan, 2026-11-13):

  A  controls.py  Generator-Coordinate Editing: low-dimensional continuous
                  control of the trained C3UNet (StyleAdaLN / Palette / local).
                  u=0 is bit-identical to the checkpoint forward.
  B  guidance.py  Learned Gradient Geometry: positive-definite, content-aware
                  update operators trained against the FINAL generated output
                  (never fitted to a teacher endpoint).
  C  energy.py    Real-Supported Discriminative Energy: frozen R1 margin plus a
                  weak multi-prototype support term from real paintings.

Supporting: p0_contract.py (must pass before any arm), arms.py (fixed matrix).

Retired and archived under archive/v8_ot_geometry/: the patch-bank OT /
geometric-matching route, because a fixed-plan squared distance reduces to
weighted-mean regression and cannot represent multi-modal stroke statistics.
"""
from .controls import ControlSpec, ControlledGenerator, Controls, wrap
from .guidance import (ControlSPDHead, LatentSPDEditor, SPDHead, UnrollConfig,
                       UnsupervisedUnroll, generate_control_edit, generate_edit,
                       project_rms, target_margin)
from .energy import RealSupport, SupportedEnergy, build_real_support

__all__ = ["ControlSpec", "ControlledGenerator", "Controls", "wrap",
           "LatentSPDEditor", "SPDHead", "ControlSPDHead", "UnrollConfig",
           "UnsupervisedUnroll", "generate_edit", "generate_control_edit",
           "project_rms", "target_margin",
           "RealSupport", "SupportedEnergy", "build_real_support"]
