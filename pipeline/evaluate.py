"""Area statistics and accuracy vs. a reference map. Owner: AI2.

Contract: docs/CONTRACT.md § 3.
"""

import numpy as np


def summarize(mask: np.ndarray, valid: np.ndarray, pixel_area_m2: float,
              reference: np.ndarray | None = None) -> dict:
    """Return flooded_area_km2, valid_area_km2, iou, f1 (iou/f1 None without reference)."""
    raise NotImplementedError("AI2: implement area stats and IoU / F1")
