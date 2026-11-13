"""Read NISAR L2 GCOV HDF5 files. Owner: AI1.

Contract: docs/CONTRACT.md § 1.

NISAR GCOV data is L-band dual-pol (DH): HHHH and HVHV covariance terms
stored as gamma0 linear power.  There is no VV channel; HH is the co-pol
primary and HV is the cross-pol secondary.

Processing already applied by the GCOV producer:
  - Radiometric terrain correction (gamma0)
  - Geocoding to UTM (EPSG 32642, 20 m pixels)

NOT applied (must handle ourselves or document):
  - Thermal noise removal  → HV near noise floor, use HH as primary
  - Shadow / layover masking → dark shadow mimics water
  - Speckle filtering → multi-looking already done (numberOfLooks > 1)
  - Valid-sample masking → mask=0 and mask=255 must be excluded
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# HDF5 paths inside NISAR GCOV products
_GA = "/science/LSAR/GCOV/grids/frequencyA"
_ID = "/science/LSAR/identification"


@dataclass
class Scene:
    hh: np.ndarray       # HHHH, linear power, float32 (2-D)
    hv: np.ndarray       # HVHV, linear power, float32 (2-D)
    x: np.ndarray        # pixel-centre x coordinates (metres, product CRS)
    y: np.ndarray        # pixel-centre y coordinates (metres, product CRS)
    epsg: int            # e.g. 32642
    date: str            # "YYYY-MM-DD" (UTC)
    granule: str         # source file name
    mask: np.ndarray     # uint8: 255=outside swath, 0=partially focused, 1..n=valid
    num_looks: np.ndarray | None = None  # numberOfLooks, for speckle context
    rtc_factor: np.ndarray | None = None  # gamma0-to-sigma0 conversion factor


def read_gcov(path: str | Path, frequency: str = "A", step: int = 1) -> Scene:
    """Open a GCOV .h5 file read-only and return HHHH, HVHV and the grid.

    Parameters
    ----------
    path : str or Path
        Path to the NISAR L2 GCOV HDF5 file.
    frequency : str
        Frequency band: "A" (20 m, default) or "B" (80 m).
    step : int
        Sub-sampling factor. step > 1 reads every n-th pixel (fast previews).

    Returns
    -------
    Scene
        Dataclass with backscatter arrays, coordinates, and metadata.

    Notes
    -----
    - Files are always opened read-only (mode='r').
    - The data is identified by filename, not folder location (files may be
      swapped in the delivered zip — see phase1_report.md § 6).
    """
    path = Path(path)
    ga = f"/science/LSAR/GCOV/grids/frequency{frequency.upper()}"

    with h5py.File(str(path), "r") as f:
        # --- Identification ---
        granule_id = f[f"{_ID}/granuleId"][()].decode(errors="replace")
        date_str = f[f"{_ID}/zeroDopplerStartTime"][()].decode()[:10]  # YYYY-MM-DD

        # --- Coordinate grid ---
        x = f[f"{ga}/xCoordinates"][::step].astype(np.float64)
        y = f[f"{ga}/yCoordinates"][::step].astype(np.float64)
        epsg = int(f[f"{ga}/projection"][()])

        # --- Backscatter (gamma0, linear power) ---
        hh = f[f"{ga}/HHHH"][::step, ::step].astype(np.float32)
        hv = f[f"{ga}/HVHV"][::step, ::step].astype(np.float32)

        # --- Quality mask ---
        mask = f[f"{ga}/mask"][::step, ::step]  # uint8

        # --- Optional: number of looks and RTC factor ---
        num_looks = None
        if f"{ga}/numberOfLooks" in f:
            num_looks = f[f"{ga}/numberOfLooks"][::step, ::step].astype(np.float32)

        rtc_factor = None
        if f"{ga}/rtcGammaToSigmaFactor" in f:
            rtc_factor = f[f"{ga}/rtcGammaToSigmaFactor"][::step, ::step].astype(np.float32)

    return Scene(
        hh=hh, hv=hv, x=x, y=y,
        epsg=epsg, date=date_str, granule=granule_id,
        mask=mask, num_looks=num_looks, rtc_factor=rtc_factor,
    )
