"""Write pipeline results in the dashboard format. Owner: AI3.

Contract: docs/CONTRACT.md § 4. Must reproduce the structure of sample/example-event/.

Also exports processed rasters as GeoTIFFs for Phase 2 deliverables.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402

from .preprocess import Prepared


def _utm_to_latlon(x: np.ndarray, y: np.ndarray, epsg: int):
    """Convert UTM coordinate arrays to lat/lon using pyproj.

    Returns
    -------
    tuple
        (lons_1d, lats_1d) arrays for the grid edges in EPSG:4326.
        Also returns bounds as [[south, west], [north, east]].
    """
    from pyproj import Transformer
    # UTM → WGS84
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    # Convert corner coordinates
    x_min, x_max = float(x[0]), float(x[-1])
    y_min, y_max = float(y[-1]), float(y[0])  # y decreases (north at top)

    corners_x = [x_min, x_max, x_min, x_max]
    corners_y = [y_max, y_max, y_min, y_min]  # NW, NE, SW, SE
    lons, lats = transformer.transform(corners_x, corners_y)

    south = min(lats)
    north = max(lats)
    west = min(lons)
    east = max(lons)

    bounds = [[round(south, 6), round(west, 6)],
              [round(north, 6), round(east, 6)]]
    return bounds


def _normalize_for_png(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize array to 0–255 uint8 for PNG output.

    NaN values become 0 (transparent via alpha channel).
    """
    clipped = np.clip(arr, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    normalized = np.nan_to_num(normalized, nan=0.0)
    normalized = (normalized * 255).astype(np.uint8)
    normalized[~np.isfinite(arr)] = 0
    return normalized


def _save_greyscale_png(arr: np.ndarray, valid: np.ndarray,
                        path: Path, vmin: float, vmax: float):
    """Save a greyscale PNG with alpha channel for invalid pixels."""
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    grey = _normalize_for_png(arr, vmin, vmax)
    rgba[:, :, 0] = grey  # R
    rgba[:, :, 1] = grey  # G
    rgba[:, :, 2] = grey  # B
    rgba[:, :, 3] = np.where(valid & np.isfinite(arr), 255, 0)  # Alpha

    from PIL import Image
    Image.fromarray(rgba, "RGBA").save(str(path))


def _save_change_png(change_db: np.ndarray, valid: np.ndarray, path: Path,
                     vmin: float = -6, vmax: float = 6):
    """Save a diverging colour change map (blue=darker, red=brighter)."""
    h, w = change_db.shape
    cmap = plt.get_cmap("RdBu_r")

    # Normalize to 0–1
    norm = np.clip((change_db - vmin) / (vmax - vmin), 0, 1)
    norm[~np.isfinite(change_db)] = 0.5  # neutral for NaN

    colors = cmap(norm)  # returns (h, w, 4) float in [0,1]
    rgba = (colors * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(valid & np.isfinite(change_db), 255, 0)

    from PIL import Image
    Image.fromarray(rgba, "RGBA").save(str(path))


def _save_flood_png(flood_mask: np.ndarray, valid: np.ndarray, path: Path):
    """Save transparent PNG with flood pixels in blue."""
    h, w = flood_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Flood pixels: semi-transparent blue
    rgba[flood_mask, 0] = 30    # R
    rgba[flood_mask, 1] = 100   # G
    rgba[flood_mask, 2] = 220   # B
    rgba[flood_mask, 3] = 180   # Alpha (semi-transparent)

    # Non-flood valid: fully transparent
    # Invalid: fully transparent

    from PIL import Image
    Image.fromarray(rgba, "RGBA").save(str(path))


def write_geotiff(path: Path, data: np.ndarray, x: np.ndarray,
                  y: np.ndarray, epsg: int, band_name: str = "band1",
                  nodata: float = np.nan):
    """Write a 2-D array as a single-band GeoTIFF.

    Parameters
    ----------
    path : Path
        Output file path.
    data : np.ndarray (2-D, float32)
        Raster data.
    x, y : np.ndarray (1-D)
        Pixel-centre coordinates in the product CRS.
    epsg : int
        EPSG code for the CRS.
    band_name : str
        Description for the band.
    nodata : float
        NoData value.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    rows, cols = data.shape
    dx = x[1] - x[0] if len(x) > 1 else 20.0
    dy = y[1] - y[0] if len(y) > 1 else -20.0

    # Transform from pixel-centre coordinates
    # x[0] is the centre of the first pixel, so the left edge is x[0] - dx/2
    west = float(x[0] - dx / 2)
    east = float(x[-1] + dx / 2)
    # y[0] is the top (largest northing), y[-1] is the bottom
    north = float(y[0] - dy / 2)  # dy is negative, so subtract
    south = float(y[-1] + dy / 2)

    transform = from_bounds(west, south, east, north, cols, rows)
    crs = CRS.from_epsg(epsg)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=rows, width=cols,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, band_name)


def write_outputs(out_dir: str | Path, event: dict, before: Prepared,
                  after: Prepared, change_db: np.ndarray,
                  flood_mask: np.ndarray, stats: dict) -> Path:
    """Reproject layers to lat/lon, write the four PNGs and meta.json.

    Parameters
    ----------
    out_dir : str or Path
        Base output directory (e.g. "outputs").
    event : dict
        Must contain: event_id, title, location, satellite, method, notes.
    before, after : Prepared
        Aligned, processed scenes.
    change_db : np.ndarray
        Difference map (after_db - before_db) in dB.
    flood_mask : np.ndarray (bool)
        Flood detection mask.
    stats : dict
        Statistics from evaluate.summarize().

    Returns
    -------
    Path
        The output folder path.
    """
    event_dir = Path(out_dir) / event["event_id"]
    event_dir.mkdir(parents=True, exist_ok=True)

    # HH backscatter range for visualization (dB)
    vmin, vmax = -25.0, 0.0

    # --- PNGs (all in EPSG:4326 orientation: north up) ---
    print(f"  Writing before_hh.png...")
    _save_greyscale_png(before.hh_db, before.valid, event_dir / "before_hh.png", vmin, vmax)

    print(f"  Writing after_hh.png...")
    _save_greyscale_png(after.hh_db, after.valid, event_dir / "after_hh.png", vmin, vmax)

    print(f"  Writing change.png...")
    _save_change_png(change_db, before.valid, event_dir / "change.png")

    print(f"  Writing flood_mask.png...")
    _save_flood_png(flood_mask, before.valid, event_dir / "flood_mask.png")

    # --- Bounds in lat/lon ---
    bounds = _utm_to_latlon(before.x, before.y, before.epsg)

    # --- meta.json ---
    meta = {
        "event_id": event["event_id"],
        "title": event.get("title", "Flood investigation"),
        "location": event.get("location", "Eastern Tajikistan"),
        "bounds": bounds,
        "before_date": before.date,
        "after_date": after.date,
        "satellite": event.get("satellite", "NISAR L2 GCOV, L-band"),
        "method": event.get("method", "log-ratio + Otsu threshold"),
        "flooded_area_km2": stats.get("flooded_area_km2", 0.0),
        "valid_area_km2": stats.get("valid_area_km2", 0.0),
        "metrics": {
            "iou": stats.get("iou"),
            "f1": stats.get("f1"),
        },
        "layers": {
            "before": "before_hh.png",
            "after": "after_hh.png",
            "change": "change.png",
            "flood": "flood_mask.png",
        },
        "notes": event.get("notes",
            "CAUTION: The difference image is NOT a final flood map. "
            "A change in backscatter is not automatically a flood. "
            "Both observations are pre-event — this is a baseline analysis."),
        "processing": {
            "pixel_area_m2": stats.get("pixel_area_m2", 400.0),
            "n_flooded_pixels": stats.get("n_flooded_pixels", 0),
            "n_valid_pixels": stats.get("n_valid_pixels", 0),
            "flooded_pct": stats.get("flooded_pct", 0.0),
        },
    }

    (event_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(f"  Wrote meta.json to {event_dir}")

    return event_dir
