"""ML flood detection, compared against the baseline. Owner: AI3.

Contract: docs/CONTRACT.md § 3.

Method: Unsupervised clustering using Gaussian Mixture Model (GMM) on a
multi-feature stack: [before_HH, after_HH, ratio_HH, before_HV, after_HV].
The cluster with the lowest mean after-image HH backscatter AND the most
negative ratio is labelled as the "water" class.

This is compared with the baseline (log-ratio + Otsu) to assess whether
the ML approach finds different or additional flood candidates.

Settings (documented for reproducibility):
  N_CLUSTERS = 4          # Number of GMM components
  FEATURES = ["before_hh", "after_hh", "ratio_hh", "before_hv", "after_hv"]
  WATER_SELECTION = "min_after_hh"  # How to pick the water cluster
"""

import numpy as np
from scipy.ndimage import binary_opening, generate_binary_structure

from .preprocess import Prepared


# ============================================================================
# ML parameters — documented for reproducibility
# ============================================================================
N_CLUSTERS = 4               # Number of Gaussian mixture components
RANDOM_STATE = 42             # For reproducibility
MAX_TRAINING_PIXELS = 100_000  # Subsample if more valid pixels than this
MIN_PATCH_PIXELS = 5           # Minimum connected component size


def ml_flood_mask(before: Prepared, after: Prepared) -> np.ndarray:
    """Return a bool mask (True = new open water) using GMM clustering.

    Features used:
      - Before-event HH (dB)
      - After-event HH (dB)
      - HH log-ratio (after - before, dB)
      - Before-event HV (dB)
      - After-event HV (dB)

    The cluster whose centroid has the lowest after-HH value AND a
    negative ratio is selected as the water class.

    Parameters
    ----------
    before, after : Prepared
        Aligned, filtered, dB-converted scenes.

    Returns
    -------
    np.ndarray (bool)
        Flood candidate mask. Same shape as input arrays.
        CAUTION: This is a candidate mask, not a confirmed flood map.
    """
    from sklearn.mixture import GaussianMixture

    valid = before.valid & after.valid
    shape = before.hh_db.shape

    # Build feature matrix (only on valid pixels)
    ratio_hh = after.hh_db - before.hh_db

    # Check we have enough HV data
    hv_ok = np.isfinite(before.hv_db) & np.isfinite(after.hv_db) & valid

    if hv_ok.sum() > valid.sum() * 0.5:
        # Use all 5 features
        feature_names = ["before_hh", "after_hh", "ratio_hh", "before_hv", "after_hv"]
        features = np.column_stack([
            before.hh_db[valid],
            after.hh_db[valid],
            ratio_hh[valid],
            np.where(hv_ok[valid], before.hv_db[valid], 0),
            np.where(hv_ok[valid], after.hv_db[valid], 0),
        ])
    else:
        # HV unreliable; use only HH features
        feature_names = ["before_hh", "after_hh", "ratio_hh"]
        features = np.column_stack([
            before.hh_db[valid],
            after.hh_db[valid],
            ratio_hh[valid],
        ])

    # Replace any remaining NaN/Inf and ensure float64 precision
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=-50.0).astype(np.float64)

    n_valid = len(features)
    if n_valid < N_CLUSTERS * 10:
        print(f"  Warning: too few valid pixels ({n_valid}) for GMM clustering")
        return np.zeros(shape, dtype=bool)

    # Subsample for efficiency
    if n_valid > MAX_TRAINING_PIXELS:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(n_valid, MAX_TRAINING_PIXELS, replace=False)
        train_features = features[idx]
    else:
        train_features = features

    # Standardize features
    feat_mean = train_features.mean(axis=0)
    feat_std = train_features.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    train_normed = (train_features - feat_mean) / feat_std
    all_normed = (features - feat_mean) / feat_std

    # Fit GMM with regularization on covariance
    print(f"  Fitting GMM with {N_CLUSTERS} components on {len(train_features)} pixels...")
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        reg_covar=1e-2,
        random_state=RANDOM_STATE,
        max_iter=100,
        n_init=1,
    )
    gmm.fit(train_normed)

    # Predict all valid pixels
    labels = gmm.predict(all_normed)

    # Find the water cluster: lowest mean after_hh AND negative mean ratio
    # after_hh is feature index 1, ratio_hh is feature index 2
    cluster_stats = []
    for k in range(N_CLUSTERS):
        in_k = labels == k
        if in_k.sum() == 0:
            cluster_stats.append({"after_hh_mean": 0, "ratio_mean": 0, "count": 0})
            continue
        cluster_stats.append({
            "after_hh_mean": float(features[in_k, 1].mean()),
            "ratio_mean": float(features[in_k, 2].mean()),
            "count": int(in_k.sum()),
        })
        print(f"    Cluster {k}: n={in_k.sum()}, "
              f"after_HH={cluster_stats[-1]['after_hh_mean']:.1f} dB, "
              f"ratio={cluster_stats[-1]['ratio_mean']:.1f} dB")

    # Select water cluster: must have negative ratio AND low after_hh
    water_candidates = [
        (k, s) for k, s in enumerate(cluster_stats)
        if s["ratio_mean"] < -1.0 and s["after_hh_mean"] < -12.0 and s["count"] > 10
    ]

    if not water_candidates:
        print("  Warning: No cluster meets water criteria (ratio < -1 dB, after_HH < -12 dB)")
        return np.zeros(shape, dtype=bool)

    # Pick the one with the lowest after_hh_mean
    water_k = min(water_candidates, key=lambda x: x[1]["after_hh_mean"])[0]
    print(f"  Selected cluster {water_k} as water class")

    # Build output mask
    flood = np.zeros(shape, dtype=bool)
    valid_indices = np.where(valid)
    water_in_valid = labels == water_k
    flood[valid_indices[0][water_in_valid], valid_indices[1][water_in_valid]] = True

    # Morphological cleaning
    struct = generate_binary_structure(2, 1)
    flood = binary_opening(flood, structure=struct)

    # Remove small components
    from scipy.ndimage import label as ndlabel
    labeled, n_features = ndlabel(flood)
    if n_features > 0:
        sizes = np.bincount(labeled.ravel())
        too_small = sizes < MIN_PATCH_PIXELS
        too_small[0] = False
        flood[too_small[labeled]] = False

    return flood
