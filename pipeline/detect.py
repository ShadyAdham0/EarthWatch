"""Baseline flood detection. Owner: AI2.

Contract: docs/CONTRACT.md § 3.

Suggested baseline: log-ratio change (after_db - before_db) on HH, Otsu threshold
on valid pixels, keep pixels that got darker AND are dark in the after image
(new open water). Test first on small synthetic arrays, e.g.:

    before = np.full((100, 100), -8.0); after = before.copy(); after[40:60, 40:60] = -22.0
"""

import numpy as np

from .preprocess import Prepared


def baseline_flood_mask(before: Prepared, after: Prepared) -> np.ndarray:
    """Return a bool mask (True = new open water) with the same shape as the inputs."""
    raise NotImplementedError("AI2: implement log-ratio + threshold baseline")
