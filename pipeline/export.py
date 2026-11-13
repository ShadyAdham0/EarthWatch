"""Write pipeline results in the dashboard format. Owner: AI3.

Contract: docs/CONTRACT.md § 4. Must reproduce the structure of sample/example-event/.
"""

from pathlib import Path

import numpy as np

from .preprocess import Prepared


def write_outputs(out_dir: str | Path, event: dict, before: Prepared, after: Prepared,
                  change_db: np.ndarray, flood_mask: np.ndarray, stats: dict) -> Path:
    """Reproject layers to lat/lon, write the four PNGs and meta.json, return the folder."""
    raise NotImplementedError("AI3: implement PNG + meta.json export")
