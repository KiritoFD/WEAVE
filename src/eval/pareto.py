"""Provenance predicates for comparing WEAVE against public research."""

PUBLIC_LITERATURE_METHODS = frozenset({
    "AdaIN", "AesPA", "CUT", "Hist2Style", "SaMAM", "SaMST",
    "StyleGallery", "StyleID", "StyleShot", "StyTr2", "WCT",
})


def is_public_literature(method: str) -> bool:
    """Only reviewed external research entries are eligible for the literature front."""
    return method in PUBLIC_LITERATURE_METHODS


def frontier_indices(points, *, minimize_x=False):
    """Return non-dominated two-dimensional points in increasing x order."""
    coords = [((-x if minimize_x else x), y) for x, y in points]
    ordered = sorted(range(len(coords)), key=lambda i: coords[i], reverse=True)
    front = []
    best_y = float("-inf")
    for i in ordered:
        y = coords[i][1]
        if y > best_y:
            front.append(i)
            best_y = y
    return sorted(front, key=lambda i: points[i][0])


def composite_dino_lse(
    dino_c: float,
    dino_s: float,
    *,
    alpha: float = 10.0,
    beta: float = 15.0,
) -> float:
    """Smooth negative-log-sum-exp composite of DINO-C and DINO-S.

    Linear combinations (a*C + b*S) maximize at extreme endpoints (e.g. unedited
    C~0.96 or content-collapsed S~0.58) rather than balanced regimes.
    -log(exp(-10*C) + exp(-15*S)) acts as a smooth concave minimum that aligns
    typical DINO-C (~0.70) and DINO-S (~0.47) scales (2:3 ratio) with zero
    hard thresholds.
    """
    import math

    u_c = -alpha * float(dino_c)
    u_s = -beta * float(dino_s)
    m = max(u_c, u_s)
    return -(m + math.log(math.exp(u_c - m) + math.exp(u_s - m)))
