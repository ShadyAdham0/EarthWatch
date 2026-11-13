"""ML flood detection, compared against the baseline. Owner: AI3.

Contract: docs/CONTRACT.md § 3.

Options: unsupervised clustering (k-means / Gaussian mixture) on features such as
before_db, after_db, ratio, HV; or a random forest trained on reference-map labels.
"""

import numpy as np

from .preprocess import Prepared


def ml_flood_mask(before: Prepared, after: Prepared) -> np.ndarray:
    """Return a bool mask (True = new open water) with the same shape as the inputs."""
    raise NotImplementedError("AI3: implement ML step")
