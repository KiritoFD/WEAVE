# Archived: V8 OT / geometry-matching route (superseded 2026-11-13)

These files implemented the "content-conditioned style Patch Bank with
geometric matching / entropic OT" route:

  exp/v8/energies.py   PatchOTEnergy, SlicedWassersteinEnergy, geometry_prior
  exp/v8/bank.py       real-painting patch bank (20480 prototypes built)
  exp/v8/features.py   per-patch geometry / HF / amplitude / LL descriptors
  exp/v8/arms.py       H0-H5 matrix (96 arms) built on those energies
  tools/v7r_next/*     the earlier CPT transport + constrained editor

Retired because the new 24h plan moves the style direction INTO the trained
generator (A: continuous StyleAdaLN/Palette control coordinates), learns a
positive-definite update operator (B), and adds real-painting support to R1 (C).
The patch-bank OT objective also reduces to weighted-mean regression under a
fixed plan, which cannot represent multi-modal stroke statistics.

Still valid and reused:
  * CentroidEnergy is kept as an explicit NEGATIVE baseline (mean regression).
  * MixtureSupportEnergy soft-min logic informs the C route's real support term,
    but only as a weak support energy over frozen R1 features, not as the main
    objective.
  * The multi-modal check still stands and is why OT was dropped:
        on-mode  mixture 0.347   centroid 18.000
        midpoint mixture 9.000   centroid  9.000
