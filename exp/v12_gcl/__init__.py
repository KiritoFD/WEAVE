"""V12-GCL: Content-Query Geometry / Global Content Loss (pre-registered).

Plan: docs/v12/WEAVE_V12_GCL_PLAN.md
Goal: expand the style-content Pareto frontier at EQUAL content preservation —
new content-geometry constraints (Query GCL) on top of the frozen LD4 machinery,
NOT a new style gradient.

Arms (plan §5): G0=LD4(reuse merged-750), G0S=LD4-strong, G1=Haar Query gate,
G2O=old V5 self-sim gate, G2=multi-scale relation gate, G2S=G2+strong budget.
G3/G5 gated on P0 audits; G4/G6 are separate high-compute extensions.
"""
