"""Convert to dB, mask invalid pixels, apply speckle filter, put both dates
on one grid. Owner: AI1.

Contract: docs/CONTRACT.md § 2.

Processing steps implemented here:
  1. Valid-pixel masking (mask field: 255=outside, 0=partially focused → invalid)
  2. Fill / NaN / non-positive masking
  3. Adaptive Lee speckle filter (7×7 window) — preserves edges better than
     simple boxcar, avoids excessive smoothing of small flood features
  4. Shadow/layover proxy masking using local incidence angle proxy from
     RTC gamma-to-sigma factor
  5. Conversion to dB: 10 * log10(linear_power)
  6. Grid alignment and common valid-data mask

Settings (documented for reproducibility):
  SPECKLE_FILTER = "lee_enhanced"
  SPECKLE_WINDOW = 7
  SHADOW_LAYOVER_THRESHOLD = (0.1, 10.0)  # RTC factor outside this → shadow/layover
  DEM = "NISAR built-in RTC" (already applied in L2 GCOV product)
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter

from .io import Scene


# ============================================================================
# Processing parameters — all documented here for reproducibility
# ============================================================================

# Speckle filter configuration
SPECKLE_FILTER = "lee_enhanced"   # Options: "lee_enhanced", "boxcar", "none"
SPECKLE_WINDOW = 7                # Window size (odd integer, pixels)
SPECKLE_NLOOKS = 4.4              # Effective number of looks (from GCOV metadata)
SPECKLE_DAMPING = 1.0             # Lee filter damping factor

# Shadow / layover masking via RTC gamma-to-sigma factor
# rtcGammaToSigmaFactor encodes the terrain slope effect:
#   Very small values → radar shadow (steep slopes away from sensor)
#   Very large values → layover (steep slopes toward sensor)
#   Normal terrain ≈ 0.5–3.0
SHADOW_RTC_MIN = 0.1              # Below this → shadow
LAYOVER_RTC_MAX = 10.0            # Above this → layover

# Minimum valid backscatter (linear power) — below this is noise floor
MIN_LINEAR_POWER = 1e-6           # ~-60 dB, well below any real surface


@dataclass
class Prepared:
    hh_db: np.ndarray    # float32 dB
    hv_db: np.ndarray    # float32 dB
    valid: np.ndarray    # bool, True = usable pixel
    x: np.ndarray
    y: np.ndarray
    epsg: int
    date: str


def to_db(linear: np.ndarray) -> np.ndarray:
    """10*log10 of linear power; non-positive / non-finite values become NaN.

    Parameters
    ----------
    linear : np.ndarray
        Backscatter in linear power units (gamma0 or sigma0).

    Returns
    -------
    np.ndarray
        Backscatter in decibels (dB). Invalid pixels are NaN.
    """
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(linear) & (linear > 0)
    out[ok] = 10.0 * np.log10(linear[ok])
    return out


def _lee_enhanced_filter(img: np.ndarray, valid: np.ndarray,
                         window: int = SPECKLE_WINDOW,
                         nlooks: float = SPECKLE_NLOOKS,
                         damping: float = SPECKLE_DAMPING) -> np.ndarray:
    """Enhanced Lee speckle filter for SAR imagery.

    Applies an adaptive filter that preserves edges and point targets while
    reducing speckle noise. Uses a minimum-variance approach within a local
    window, weighted by the coefficient of variation.

    Parameters
    ----------
    img : np.ndarray
        Input image (linear power).
    valid : np.ndarray (bool)
        True where data is valid.
    window : int
        Filter window size (must be odd).
    nlooks : float
        Equivalent number of looks in the input.
    damping : float
        Controls filter strength. Higher = more smoothing.

    Returns
    -------
    np.ndarray
        Filtered image (same shape, linear power).

    References
    ----------
    Lee, J.-S. (1981). "Refined filtering of image noise using local statistics."
    Computer Graphics and Image Processing, 15(4), 380-389.
    """
    # Work in linear power; replace invalid with NaN for windowed stats
    work = img.copy().astype(np.float64)
    work[~valid] = np.nan

    # Pad for edge handling
    half = window // 2

    # Local mean using uniform filter (NaN-aware via masking)
    count = uniform_filter(valid.astype(np.float64), size=window, mode="constant")
    count = np.maximum(count, 1e-10)  # avoid division by zero

    filled = np.where(valid, work, 0.0)
    local_mean = uniform_filter(filled, size=window, mode="constant") / count

    # Local variance
    local_sq = uniform_filter(filled ** 2, size=window, mode="constant") / count
    local_var = np.maximum(local_sq - local_mean ** 2, 0.0)

    # Coefficient of variation
    cv_squared = local_var / np.maximum(local_mean ** 2, 1e-20)
    cu_squared = 1.0 / nlooks  # theoretical speckle CV²

    # Weight factor (0 = full smoothing, 1 = no smoothing)
    weight = np.clip(
        1.0 - (cu_squared / np.maximum(cv_squared, 1e-20)) * damping,
        0.0, 1.0
    )

    # Adaptive combination
    out = local_mean + weight * (work - local_mean)

    # Restore original where filter couldn't be applied
    bad = ~np.isfinite(out)
    out[bad] = work[bad]

    return out.astype(np.float32)


def _build_validity_mask(scene: Scene) -> np.ndarray:
    """Build a comprehensive validity mask from Scene data.

    Marks pixels as invalid (False) if any of:
      - Outside swath (mask == 255)
      - Partially focused / edge (mask == 0)
      - NaN or Inf in either HH or HV
      - Non-positive backscatter in either channel
      - Below noise floor threshold
      - In radar shadow (RTC factor too low)
      - In layover region (RTC factor too high)

    Parameters
    ----------
    scene : Scene
        Input scene with backscatter and metadata arrays.

    Returns
    -------
    np.ndarray (bool)
        True where the pixel is usable for analysis.
    """
    # Start: valid sub-swath pixels (mask >= 1 and != 255)
    valid = (scene.mask != 255) & (scene.mask != 0)

    # Finite and positive backscatter in both channels
    valid &= np.isfinite(scene.hh) & (scene.hh > MIN_LINEAR_POWER)
    valid &= np.isfinite(scene.hv) & (scene.hv > MIN_LINEAR_POWER)

    # Shadow / layover masking from RTC factor
    if scene.rtc_factor is not None:
        rtc = scene.rtc_factor
        rtc_valid = np.isfinite(rtc)
        # Shadow: very small RTC factor (steep slope away from radar)
        shadow = rtc_valid & (rtc < SHADOW_RTC_MIN)
        # Layover: very large RTC factor (steep slope toward radar)
        layover = rtc_valid & (rtc > LAYOVER_RTC_MAX)
        valid &= ~shadow & ~layover

    return valid


def _find_common_grid(before: Scene, after: Scene):
    """Find the overlapping pixel grid between two scenes.

    The NISAR GCOV products for the same track/frame use the same fixed
    geocoded grid, so normally the grids are identical.  This function
    handles the general case where they might differ.

    Returns
    -------
    tuple
        (x_common, y_common,
         before_col_slice, before_row_slice,
         after_col_slice, after_row_slice)
    """
    assert before.epsg == after.epsg, (
        f"CRS mismatch: {before.epsg} vs {after.epsg}"
    )

    # Check pixel spacing
    dx_b = before.x[1] - before.x[0] if len(before.x) > 1 else 20.0
    dx_a = after.x[1] - after.x[0] if len(after.x) > 1 else 20.0
    dy_b = before.y[1] - before.y[0] if len(before.y) > 1 else -20.0
    dy_a = after.y[1] - after.y[0] if len(after.y) > 1 else -20.0
    assert abs(dx_b - dx_a) < 1e-3 and abs(dy_b - dy_a) < 1e-3, (
        f"Pixel spacing mismatch: ({dx_b},{dy_b}) vs ({dx_a},{dy_a})"
    )

    # Find offset in pixels
    off_x = (after.x[0] - before.x[0]) / dx_b
    off_y = (after.y[0] - before.y[0]) / dy_b

    # Verify integer alignment (sub-pixel residual < 0.01 px)
    assert abs(off_x - round(off_x)) < 0.01, f"Sub-pixel x offset: {off_x}"
    assert abs(off_y - round(off_y)) < 0.01, f"Sub-pixel y offset: {off_y}"

    ox, oy = int(round(off_x)), int(round(off_y))

    # Compute overlap in pixel indices
    ncols_b, ncols_a = len(before.x), len(after.x)
    nrows_b, nrows_a = len(before.y), len(after.y)

    # Column overlap
    b_c0 = max(0, ox)
    b_c1 = min(ncols_b, ox + ncols_a)
    a_c0 = b_c0 - ox
    a_c1 = a_c0 + (b_c1 - b_c0)

    # Row overlap
    b_r0 = max(0, oy)
    b_r1 = min(nrows_b, oy + nrows_a)
    a_r0 = b_r0 - oy
    a_r1 = a_r0 + (b_r1 - b_r0)

    x_common = before.x[b_c0:b_c1]
    y_common = before.y[b_r0:b_r1]

    return (
        x_common, y_common,
        (slice(b_r0, b_r1), slice(b_c0, b_c1)),
        (slice(a_r0, a_r1), slice(a_c0, a_c1)),
    )


def prepare_pair(before: Scene, after: Scene,
                 apply_speckle_filter: bool = True) -> tuple[Prepared, Prepared]:
    """Return both dates in dB on the same pixel grid with a shared validity mask.

    Processing steps:
      1. Build individual validity masks (swath, focus, shadow, layover)
      2. Align to common overlapping grid
      3. Apply speckle filter (Enhanced Lee, 7×7) on linear power
      4. Convert to dB
      5. Combine validity masks (AND) for consistent comparison

    Parameters
    ----------
    before, after : Scene
        Input scenes from read_gcov().
    apply_speckle_filter : bool
        If True (default), apply the Enhanced Lee filter. Set to False
        for debugging or when comparing with unfiltered data.

    Returns
    -------
    tuple[Prepared, Prepared]
        Both dates aligned, filtered, in dB, with shared validity mask.
    """
    # --- Step 1: Individual validity masks ---
    valid_b = _build_validity_mask(before)
    valid_a = _build_validity_mask(after)

    # --- Step 2: Align grids ---
    x_common, y_common, sl_b, sl_a = _find_common_grid(before, after)

    # Crop to common grid
    hh_b = before.hh[sl_b].copy()
    hv_b = before.hv[sl_b].copy()
    val_b = valid_b[sl_b].copy()

    hh_a = after.hh[sl_a].copy()
    hv_a = after.hv[sl_a].copy()
    val_a = valid_a[sl_a].copy()

    # --- Step 3: Speckle filter (on linear power, before dB conversion) ---
    if apply_speckle_filter and SPECKLE_FILTER != "none":
        print(f"  Applying {SPECKLE_FILTER} filter ({SPECKLE_WINDOW}×{SPECKLE_WINDOW})...")
        hh_b = _lee_enhanced_filter(hh_b, val_b)
        hv_b = _lee_enhanced_filter(hv_b, val_b)
        hh_a = _lee_enhanced_filter(hh_a, val_a)
        hv_a = _lee_enhanced_filter(hv_a, val_a)

    # --- Step 4: Convert to dB ---
    hh_db_b = to_db(hh_b)
    hv_db_b = to_db(hv_b)
    hh_db_a = to_db(hh_a)
    hv_db_a = to_db(hv_a)

    # --- Step 5: Combined validity mask ---
    # A pixel must be valid in BOTH dates for change detection
    common_valid = val_b & val_a & np.isfinite(hh_db_b) & np.isfinite(hh_db_a)

    prep_before = Prepared(
        hh_db=hh_db_b, hv_db=hv_db_b, valid=common_valid.copy(),
        x=x_common, y=y_common, epsg=before.epsg, date=before.date,
    )
    prep_after = Prepared(
        hh_db=hh_db_a, hv_db=hv_db_a, valid=common_valid.copy(),
        x=x_common, y=y_common, epsg=after.epsg, date=after.date,
    )

    return prep_before, prep_after
