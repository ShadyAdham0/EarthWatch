"""Area statistics and accuracy vs. a reference map. Owner: AI2.

Contract: docs/CONTRACT.md § 3.
"""

import numpy as np


def summarize(mask: np.ndarray, valid: np.ndarray, pixel_area_m2: float,
              reference: np.ndarray | None = None) -> dict:
    """Return flooded_area_km2, valid_area_km2, iou, f1.

    Parameters
    ----------
    mask : np.ndarray (bool)
        Flood detection mask (True = detected flood).
    valid : np.ndarray (bool)
        Valid-data mask (True = usable pixel).
    pixel_area_m2 : float
        Area of one pixel in square metres. For 20 m NISAR GCOV: 400.0.
    reference : np.ndarray (bool) or None
        Reference flood mask for accuracy assessment. If None, metrics
        are returned as None.

    Returns
    -------
    dict
        Keys: flooded_area_km2, valid_area_km2, flooded_pct,
              n_flooded_pixels, n_valid_pixels, iou, f1, precision, recall.
    """
    # Restrict to valid pixels only
    mask_v = mask & valid
    n_flooded = int(mask_v.sum())
    n_valid = int(valid.sum())

    flooded_area_km2 = round(n_flooded * pixel_area_m2 / 1e6, 2)
    valid_area_km2 = round(n_valid * pixel_area_m2 / 1e6, 2)
    flooded_pct = round(100.0 * n_flooded / max(n_valid, 1), 4)

    result = {
        "flooded_area_km2": flooded_area_km2,
        "valid_area_km2": valid_area_km2,
        "flooded_pct": flooded_pct,
        "n_flooded_pixels": n_flooded,
        "n_valid_pixels": n_valid,
        "pixel_area_m2": pixel_area_m2,
        "iou": None,
        "f1": None,
        "precision": None,
        "recall": None,
    }

    if reference is not None:
        ref_v = reference & valid
        # True positives, false positives, false negatives
        tp = int((mask_v & ref_v).sum())
        fp = int((mask_v & ~ref_v).sum())
        fn = int((~mask_v & ref_v).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        result["iou"] = round(iou, 4)
        result["f1"] = round(f1, 4)
        result["precision"] = round(precision, 4)
        result["recall"] = round(recall, 4)
        result["tp"] = tp
        result["fp"] = fp
        result["fn"] = fn

    return result
