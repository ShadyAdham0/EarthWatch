"""Convert to dB, mask invalid pixels, put both dates on one grid. Owner: AI1.

Contract: docs/CONTRACT.md § 2.
"""

from dataclasses import dataclass

import numpy as np

from .io import Scene


@dataclass
class Prepared:
    hh_db: np.ndarray   # float32 dB
    hv_db: np.ndarray   # float32 dB
    valid: np.ndarray   # bool, True = usable pixel
    x: np.ndarray
    y: np.ndarray
    epsg: int
    date: str


def to_db(linear: np.ndarray) -> np.ndarray:
    """10*log10 of linear power; non-positive / non-finite values become NaN."""
    out = np.full(linear.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(linear) & (linear > 0)
    out[ok] = 10.0 * np.log10(linear[ok])
    return out


def prepare_pair(before: Scene, after: Scene) -> tuple[Prepared, Prepared]:
    """Return both dates in dB on the same pixel grid with a shared validity mask."""
    raise NotImplementedError("AI1: check grid alignment, crop/shift to the common grid, build masks")
