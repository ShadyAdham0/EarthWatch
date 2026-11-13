"""Baseline flood detection using log-ratio and Otsu thresholding. Owner: AI2.

Contract: docs/CONTRACT.md § 3.

Method:
  1. Compute HH log-ratio: after_db - before_db (in dB = log ratio in linear)
  2. Apply Otsu threshold on the negative tail (pixels that got darker)
  3. Require the after-image backscatter to also be below a darkness threshold
     (new open water should be both darker-than-before AND absolutely dark)
  4. Remove isolated small clusters (morphological opening)

Settings (documented for reproducibility):
  DARKNESS_THRESHOLD_DB = -15.0   # After-image HH must be darker than this
  MIN_PATCH_PIXELS = 5            # Minimum connected patch size to keep
  CROSS_POL_BOOST = True          # Also check HV ratio as supporting evidence
  CROSS_POL_WEIGHT = 0.3          # Weight for HV in combined score

References:
  - Twele, A. et al. (2016). "Sentinel-1-based flood mapping." Remote Sensing.
  - Otsu, N. (1979). "A threshold selection method from gray-level histograms."
    IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.

IMPORTANT: The difference image is NOT a final flood map. A change in
backscatter is not automatically a flood.  This baseline provides candidates
for further validation.
"""

import numpy as np
from scipy.ndimage import label as ndlabel, binary_opening, generate_binary_structure

from .preprocess import Prepared


# ============================================================================
# Detection parameters — documented for reproducibility
# ============================================================================

# Darkness threshold: after-image HH must be below this to be "water-like"
# Typical L-band HH over calm water: -18 to -25 dB
# Rough terrain / vegetation: -5 to -15 dB
DARKNESS_THRESHOLD_DB = -15.0

# Minimum connected component size (in pixels) to retain
# At 20 m spacing, 5 pixels ≈ 2000 m² ≈ 0.002 km²
MIN_PATCH_PIXELS = 5

# Whether to use cross-pol (HV) as additional evidence
CROSS_POL_BOOST = True
CROSS_POL_WEIGHT = 0.3


def _otsu_threshold(data: np.ndarray) -> float:
    """Compute Otsu's threshold on a 1-D array of finite values.

    Parameters
    ----------
    data : np.ndarray
        1-D array of values to threshold.

    Returns
    -------
    float
        Optimal threshold that minimizes intra-class variance.
    """
    if len(data) == 0:
        return -3.0
    if np.isclose(data.max(), data.min()):
        return float(data.max())

    # Histogram with 256 bins over the data range
    hist, bin_edges = np.histogram(data, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return np.median(data)

    # Cumulative sums
    weight_bg = np.cumsum(hist).astype(np.float64)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * bin_centers) / np.maximum(weight_bg, 1)
    mean_fg = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / np.maximum(weight_fg, 1)

    # Between-class variance
    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

    # Find the threshold that maximizes between-class variance
    idx = np.argmax(between_var)
    return float(bin_centers[idx])


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components smaller than min_size pixels.

    Parameters
    ----------
    mask : np.ndarray (bool)
        Binary mask.
    min_size : int
        Minimum number of connected pixels to retain.

    Returns
    -------
    np.ndarray (bool)
        Cleaned mask.
    """
    if min_size <= 1:
        return mask

    labeled, n_features = ndlabel(mask)
    if n_features == 0:
        return mask

    # Count pixels per label
    component_sizes = np.bincount(labeled.ravel())
    # component_sizes[0] is background, skip it
    too_small = component_sizes < min_size
    too_small[0] = False  # don't touch background
    mask_cleaned = mask.copy()
    mask_cleaned[too_small[labeled]] = False
    return mask_cleaned


def baseline_flood_mask(before: Prepared, after: Prepared,
                        darkness_threshold: float = DARKNESS_THRESHOLD_DB,
                        min_patch: int = MIN_PATCH_PIXELS) -> np.ndarray:
    """Return a bool mask (True = new open water) with the same shape as inputs.

    Algorithm:
      1. Log-ratio (after HH - before HH in dB) on valid pixels
      2. Otsu threshold on the ratio to find the "significantly darker" cutoff
      3. Require after-image HH below the absolute darkness threshold
      4. Optionally boost with HV evidence
      5. Morphological opening + small component removal

    Parameters
    ----------
    before, after : Prepared
        Aligned, filtered, dB-converted scenes from prepare_pair().
    darkness_threshold : float
        After-image HH must be darker than this (dB) to qualify as water.
    min_patch : int
        Minimum connected component size (pixels) to retain.

    Returns
    -------
    np.ndarray (bool)
        Flood candidate mask. True = new open water candidate.
        CAUTION: This is a CANDIDATE mask, not a confirmed flood map.
    """
    valid = before.valid & after.valid
    shape = before.hh_db.shape

    # --- Step 1: Log-ratio (dB difference = log ratio) ---
    # Negative values mean the surface got darker (potential new water)
    ratio_hh = np.full(shape, np.nan, dtype=np.float32)
    ratio_hh[valid] = after.hh_db[valid] - before.hh_db[valid]

    # --- Step 2: Otsu threshold on negative tail ---
    # Only consider pixels that got darker
    darker = ratio_hh[valid & (ratio_hh < 0)]
    if len(darker) < 100:
        # Not enough dark pixels to compute a meaningful threshold
        return np.zeros(shape, dtype=bool)

    otsu_thresh = _otsu_threshold(darker)
    # Ensure the threshold is negative (we want pixels that got darker)
    if otsu_thresh >= 0:
        otsu_thresh = -3.0  # fallback: 3 dB decrease

    # --- Step 3: Apply combined criteria ---
    # Pixel must satisfy ALL of:
    #   a) Got darker by more than the Otsu threshold
    #   b) After-image is absolutely dark (below darkness_threshold)
    #   c) Is in the valid region
    flood = (
        valid
        & (ratio_hh <= otsu_thresh + 1e-4)
        & (after.hh_db <= darkness_threshold)
    )

    # --- Step 4: Optional cross-pol (HV) boost ---
    if CROSS_POL_BOOST:
        hv_valid = np.isfinite(before.hv_db) & np.isfinite(after.hv_db) & valid
        ratio_hv = np.full(shape, 0.0, dtype=np.float32)
        ratio_hv[hv_valid] = after.hv_db[hv_valid] - before.hv_db[hv_valid]

        # HV should also decrease for water (cross-pol drops over smooth surfaces)
        hv_darker = ratio_hv < (otsu_thresh * CROSS_POL_WEIGHT)
        # Only ADD supporting pixels that are also dark and decreased in HV
        # Don't remove HH-only detections (HV is noisier)
        flood = flood & (hv_darker | ~hv_valid)

    # --- Step 5: Morphological cleaning ---
    # Opening with a 3×3 structuring element to remove isolated noise pixels
    struct = generate_binary_structure(2, 1)  # 4-connected cross
    flood = binary_opening(flood, structure=struct)

    # Remove small connected components
    flood = _remove_small_components(flood, min_patch)

    return flood
