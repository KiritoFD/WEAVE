"""WEAVE models: the single home for every network definition.

Policy for this package: model code lives HERE. `src.infer` must not define or
patch networks, and no experiment may keep a private copy of a network — that
is how variants silently drift apart.

Networks are re-exported from their proven implementations so the numbers stay
bit-identical to the results already published in this repository. New work
adds classes here; it never forks an existing one.
"""
from .unet import C3UNet, ConditionalScoreNet
from .critic import LatentStyleNet
from .canon import ContentCanonicalizer, canonicalizer
from .transport import SingleStepTransportNet

__all__ = [
    "C3UNet",
    "ConditionalScoreNet",
    "LatentStyleNet",
    "ContentCanonicalizer",
    "canonicalizer",
    "SingleStepTransportNet",
]
