"""Read NISAR L2 GCOV HDF5 files. Owner: AI1.

Contract: docs/CONTRACT.md § 1.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Scene:
    hh: np.ndarray      # HHHH, linear power, float32 (2-D)
    hv: np.ndarray      # HVHV, linear power, float32 (2-D)
    x: np.ndarray       # pixel-centre x coordinates (metres, product CRS)
    y: np.ndarray       # pixel-centre y coordinates (metres, product CRS)
    epsg: int           # e.g. 32642
    date: str           # "YYYY-MM-DD" (UTC)
    granule: str        # source file name


def read_gcov(path: str | Path, frequency: str = "A", step: int = 1) -> Scene:
    """Open a GCOV .h5 file read-only and return HHHH, HVHV and the grid.

    step > 1 reads every n-th pixel (for quick previews).
    """
    raise NotImplementedError("AI1: implement with h5py (open with mode='r')")
