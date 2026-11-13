"""V9-LD: latent-domain local style descent on top of cached N2C outputs.

All arms start from the SAME frozen z_N2 (N2C.e gen cache). Only R1 critic +
N2C SPDHead (rank-0 diagonal) + Haar patch style bank live on GPU.
"""
