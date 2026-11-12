"""WEAVE source tree.

  src.model   network definitions (the only place a network class may live)
  src.infer   checkpoint -> raw latents / images; no metrics, no PNG
  src.eval    raw -> metrics and PNG; no generation
  src.utils   deployment plumbing and frozen measurement referees
"""
