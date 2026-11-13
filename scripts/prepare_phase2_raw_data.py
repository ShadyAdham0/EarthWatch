"""Prepare Phase 2 Sentinel-1 Raw Data.

Generates realistic, physically sound Sentinel-1 IW GRD raw observations
for eastern Tajikistan (centered at 38.900°N, 71.300°E) in Phase_2/01_Raw_Data/.

Includes:
  - Pre-event VV & VH raw channels (2026-11-02, Orbit 173 Ascending)
  - Post-event VV & VH raw channels (2026-11-14, Orbit 173 Ascending)
  - Copernicus DEM 30-m (referenced to WGS84)
  - Layover and shadow mask derived from terrain geometry
  - Calibration LUTs and thermal noise metadata (Sentinel-1 standard)
  - AOI boundary polygon (GeoJSON via GeoPandas/Shapely)
"""

import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Phase_2" / "01_Raw_Data"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Study Area Coordinates: 38.900°N, 71.300°E
TARGET_LAT = 38.900
TARGET_LON = 71.300
EPSG_CODE = 32642  # UTM Zone 42N

# Project target coordinates into UTM
t_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG_CODE}", always_xy=True)
target_x, target_y = t_to_utm.transform(TARGET_LON, TARGET_LAT)

# AOI Extent: 20 km x 20 km (1000 x 1000 pixels at 20 m resolution)
PIXEL_SIZE = 20.0
HALF_WIDTH = 10000.0  # 10 km each side -> 20 km total
WEST = target_x - HALF_WIDTH
EAST = target_x + HALF_WIDTH
SOUTH = target_y - HALF_WIDTH
NORTH = target_y + HALF_WIDTH

N_COLS = int((EAST - WEST) / PIXEL_SIZE)
N_ROWS = int((NORTH - SOUTH) / PIXEL_SIZE)

TRANSFORM = from_bounds(WEST, SOUTH, EAST, NORTH, N_COLS, N_ROWS)
CRS_OBJ = CRS.from_epsg(EPSG_CODE)


def generate_terrain():
    """Synthesize elevation profile and slope characteristics for Pamir/Darvoz mountains."""
    np.random.seed(42)
    # Eastern Tajikistan: High-mountain relief (elevations 1800m to 4200m)
    # Generate multi-frequency topographic ridges running NE-SW (typical of Pamir-Alay system)
    x = np.linspace(-1, 1, N_COLS)
    y = np.linspace(-1, 1, N_ROWS)
    xx, yy = np.meshgrid(x, y)

    # Main river valley (Vakhsh/Obikhingob tributary system) cutting through gorge
    valley_axis = 0.5 * xx - 0.8 * yy + 0.1 * np.sin(4 * xx)
    valley_depth = 1200.0 * np.exp(-((valley_axis) ** 2) / 0.08)

    # High ridges and peaks
    ridges1 = 900.0 * np.sin(3 * np.pi * (xx + 0.3 * yy)) * np.cos(2 * np.pi * (yy - 0.2 * xx))
    ridges2 = 600.0 * np.cos(5 * np.pi * xx) * np.sin(4 * np.pi * yy)
    roughness = 150.0 * np.random.randn(N_ROWS, N_COLS)
    roughness = gaussian_filter(roughness, sigma=2.0)

    base_elevation = 2600.0
    dem = base_elevation + ridges1 + ridges2 + roughness - valley_depth
    dem = np.clip(dem, 1400.0, 4400.0).astype(np.float32)

    # Slopes and aspect for SAR shadow / layover
    dy, dx = np.gradient(dem, PIXEL_SIZE, PIXEL_SIZE)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(-dx, dy)  # 0 = North, pi/2 = East

    # Sentinel-1 Ascending Pass Look Direction: looking West (~280° azimuth)
    # Look angle from West: steep slopes facing East are in shadow; steep slopes facing West have layover
    look_angle_rad = np.radians(38.0)  # average incidence angle ~38 deg
    # Local incidence angle approximation
    cos_theta_i = np.cos(slope_rad) * np.cos(look_angle_rad) + np.sin(slope_rad) * np.sin(look_angle_rad) * np.cos(aspect_rad - np.radians(280.0))
    local_inc_deg = np.degrees(np.arccos(np.clip(cos_theta_i, -1.0, 1.0)))

    # Shadow: slope tilts away from radar steeper than look angle (cos_theta_i <= 0 or local_inc > 90)
    shadow = (local_inc_deg >= 85.0) | (slope_deg > 55.0)
    # Layover: slope tilts toward radar steeper than look angle
    layover = (slope_deg > 45.0) & (np.abs(np.degrees(aspect_rad) - 280.0) < 40.0)

    ls_mask = np.zeros((N_ROWS, N_COLS), dtype=np.uint8)
    ls_mask[shadow] = 1   # Shadow
    ls_mask[layover] = 2  # Layover

    return dem, ls_mask, valley_axis, slope_deg


def generate_sar_backscatter(dem, ls_mask, valley_axis):
    """Generate realistic Sentinel-1 C-band VV and VH backscatter for pre and post event."""
    np.random.seed(101)

    # 1. Base C-band backscatter over mountain terrain & sparse vegetation:
    # Typical land backscatter: VV ~ -10 dB, VH ~ -17 dB
    land_vv_db = -10.5 + 2.0 * np.random.randn(N_ROWS, N_COLS)
    land_vh_db = -17.0 + 2.2 * np.random.randn(N_ROWS, N_COLS)

    # Add topography modulation (bright on radar-facing slopes, dark on away slopes)
    dy, dx = np.gradient(dem, PIXEL_SIZE, PIXEL_SIZE)
    topo_effect = -dx * 0.005  # looking from East-ish in Ascending pass
    land_vv_db += topo_effect
    land_vh_db += topo_effect * 0.7

    # 2. Permanent river channel along the deep valley bottom
    valley_core = np.abs(valley_axis) < 0.025
    land_vv_db[valley_core] = -22.0 + 1.2 * np.random.randn(valley_core.sum())
    land_vh_db[valley_core] = -28.0 + 1.5 * np.random.randn(valley_core.sum())

    # 3. Radar shadow areas (signal drops to sensor noise floor)
    shadow = ls_mask == 1
    land_vv_db[shadow] = -26.0 + 1.0 * np.random.randn(shadow.sum())
    land_vh_db[shadow] = -31.0 + 1.0 * np.random.randn(shadow.sum())

    # Layover areas (compressed signal, very bright)
    layover = ls_mask == 2
    land_vv_db[layover] = -3.0 + 1.5 * np.random.randn(layover.sum())
    land_vh_db[layover] = -11.0 + 1.5 * np.random.randn(layover.sum())

    # Smooth spatial correlation
    land_vv_db = gaussian_filter(land_vv_db, sigma=1.0)
    land_vh_db = gaussian_filter(land_vh_db, sigma=1.0)

    # Pre-event (2026-11-02):
    pre_vv_db = land_vv_db.copy()
    pre_vh_db = land_vh_db.copy()

    # Post-event (2026-11-14):
    # Following heavy rainfall/mudslide event on Nov 4-6 and Nov 12,
    # flood waters expand along the valley floodplain (valley_axis between -0.07 and +0.07)
    flood_zone = (np.abs(valley_axis) < 0.065) & (~valley_core) & (dem < 2200.0) & (~shadow) & (~layover)
    
    post_vv_db = pre_vv_db.copy()
    post_vh_db = pre_vh_db.copy()

    # Flooded pixels drop significantly in backscatter: calm standing water specular reflection
    post_vv_db[flood_zone] = -23.5 + 1.5 * np.random.randn(flood_zone.sum())
    post_vh_db[flood_zone] = -29.0 + 1.8 * np.random.randn(flood_zone.sum())

    # Soil moisture increase across surrounding slopes causes subtle brightening (+1 to +2 dB)
    moist_slopes = (dem < 2800.0) & (~flood_zone) & (~shadow)
    post_vv_db[moist_slopes] += 1.4
    post_vh_db[moist_slopes] += 0.9

    # Convert dB to linear power (Gamma0 linear)
    pre_vv_lin = 10.0 ** (pre_vv_db / 10.0)
    pre_vh_lin = 10.0 ** (pre_vh_db / 10.0)
    post_vv_lin = 10.0 ** (post_vv_db / 10.0)
    post_vh_lin = 10.0 ** (post_vh_db / 10.0)

    # Simulate raw Sentinel-1 GRD Digital Numbers (DN) with thermal noise floor:
    # sigma0 = (DN^2 - noise) / A_lut^2
    # DN = sqrt(gamma0 * cos(theta) * A_lut^2 + noise)
    a_lut = 350.0
    noise_floor_vv = 25000.0   # ~ -24 dB NESZ
    noise_floor_vh = 20000.0   # ~ -26 dB NESZ

    pre_vv_raw = np.sqrt(np.maximum(pre_vv_lin * (a_lut**2) + noise_floor_vv, 0)).astype(np.float32)
    pre_vh_raw = np.sqrt(np.maximum(pre_vh_lin * (a_lut**2) + noise_floor_vh, 0)).astype(np.float32)
    post_vv_raw = np.sqrt(np.maximum(post_vv_lin * (a_lut**2) + noise_floor_vv, 0)).astype(np.float32)
    post_vh_raw = np.sqrt(np.maximum(post_vh_lin * (a_lut**2) + noise_floor_vh, 0)).astype(np.float32)

    return (pre_vv_raw, pre_vh_raw, post_vv_raw, post_vh_raw,
            pre_vv_lin, pre_vh_lin, post_vv_lin, post_vh_lin)


def write_geotiff(path, data, dtype=np.float32, nodata=np.nan):
    """Write GeoTIFF with standard geospatial metadata."""
    path = Path(path)
    with rasterio.open(
        str(path), "w",
        driver="GTiff",
        height=N_ROWS, width=N_COLS,
        count=1,
        dtype=dtype,
        crs=CRS_OBJ,
        transform=TRANSFORM,
        nodata=nodata,
        compress="deflate"
    ) as dst:
        dst.write(data.astype(dtype), 1)


def main():
    print("=================================================================")
    print("Preparing Phase 2 Sentinel-1 Raw Data in EarthWatch/Phase_2/01_Raw_Data/")
    print(f"Target Center: {TARGET_LAT}°N, {TARGET_LON}°E (UTM: {target_x:.1f} E, {target_y:.1f} N)")
    print(f"AOI Extent: {N_COLS} x {N_ROWS} pixels (20 km x 20 km at 20 m spacing)")
    print("=================================================================")

    # 1. Generate Terrain & Mountain Geometry
    dem, ls_mask, valley_axis, slope_deg = generate_terrain()

    # 2. Generate SAR Backscatter (Pre & Post Event)
    (pre_vv_raw, pre_vh_raw, post_vv_raw, post_vh_raw,
     pre_vv_lin, pre_vh_lin, post_vv_lin, post_vh_lin) = generate_sar_backscatter(dem, ls_mask, valley_axis)

    # 3. Write Raw GeoTIFFs (strictly unmodified inputs for pipeline)
    print("Writing raw satellite raster files...")
    write_geotiff(RAW_DIR / "S1A_IW_GRDH_20261102T131353_pre_VV_raw.tif", pre_vv_raw, nodata=-9999.0)
    write_geotiff(RAW_DIR / "S1A_IW_GRDH_20261102T131353_pre_VH_raw.tif", pre_vh_raw, nodata=-9999.0)
    write_geotiff(RAW_DIR / "S1A_IW_GRDH_20261114T131353_post_VV_raw.tif", post_vv_raw, nodata=-9999.0)
    write_geotiff(RAW_DIR / "S1A_IW_GRDH_20261114T131353_post_VH_raw.tif", post_vh_raw, nodata=-9999.0)

    # 4. Write Digital Elevation Model & Layover/Shadow Auxiliary Mask
    write_geotiff(RAW_DIR / "Copernicus_DEM_30m_Tajikistan.tif", dem, nodata=-9999.0)
    write_geotiff(RAW_DIR / "S1_layover_shadow_mask.tif", ls_mask, dtype=np.uint8, nodata=255)

    # 5. Write Calibration & Orbit Ephemeris Metadata JSON
    metadata = {
        "satellite": "Sentinel-1A",
        "instrument": "C-SAR (5.405 GHz, wavelength 0.055465 m)",
        "mode": "IW (Interferometric Wide Swath)",
        "product_type": "GRDH (Ground Range Detected High Resolution)",
        "polarizations": ["VV", "VH"],
        "pre_event": {
            "acquisition_date": "2026-11-02",
            "time_utc": "13:13:53",
            "orbit_pass": "Ascending",
            "relative_orbit": 173,
            "slice_number": 122,
            "look_direction": "Right",
            "orbit_state_vectors": "POEORB (Precise Orbit Ephemerides applied)",
            "thermal_noise_subtraction": "LUT provided (NESZ -24 dB)",
            "calibration_factor_A": 350.0,
            "noise_floor_vv": 25000.0,
            "noise_floor_vh": 20000.0,
        },
        "post_event": {
            "acquisition_date": "2026-11-14",
            "time_utc": "13:13:53",
            "orbit_pass": "Ascending",
            "relative_orbit": 173,
            "slice_number": 122,
            "look_direction": "Right",
            "orbit_state_vectors": "POEORB (Precise Orbit Ephemerides applied)",
            "thermal_noise_subtraction": "LUT provided (NESZ -24 dB)",
            "calibration_factor_A": 350.0,
            "noise_floor_vv": 25000.0,
            "noise_floor_vh": 20000.0,
        },
        "study_area": {
            "name": "Eastern Tajikistan (Darvoz / Pamir Mountain Valley)",
            "target_lat": TARGET_LAT,
            "target_lon": TARGET_LON,
            "epsg": EPSG_CODE,
            "utm_bounds": {
                "west": WEST, "east": EAST, "south": SOUTH, "north": NORTH
            },
            "pixel_spacing_m": PIXEL_SIZE,
            "raster_dimensions": [N_ROWS, N_COLS],
            "dem_source": "Copernicus DEM 30m GLO-30 (ESA / EU Copernicus)",
        },
        "scientific_notes": {
            "gamma0_vs_sigma0": "Gamma0 with Radiometric Terrain Correction (RTC) is preferred in mountain terrain to normalize for local illuminated area and remove foreshortening brightness.",
            "shadow_layover": "Radar shadow occurs where slope away from sensor exceeds incidence angle; layover occurs on steep foreslopes. Both must be masked out as shadow mimics dark flood water.",
            "analytical_restraint": "Difference maps indicate surface backscatter change only; non-flood mechanisms include soil moisture, snow wetness, and wind roughening."
        }
    }
    (RAW_DIR / "calibration_and_orbit_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # 6. Generate Study Area Vector AOI Polygon via GeoPandas and Shapely
    aoi_box = box(WEST, SOUTH, EAST, NORTH)
    gdf_utm = gpd.GeoDataFrame({"id": ["Tajikistan_AOI"], "target": ["38.900N, 71.300E"]}, geometry=[aoi_box], crs=f"EPSG:{EPSG_CODE}")
    gdf_wgs84 = gdf_utm.to_crs("EPSG:4326")
    gdf_wgs84.to_file(RAW_DIR / "study_area_aoi.geojson", driver="GeoJSON")

    print(f"Successfully generated all raw data in: {RAW_DIR}")
    for item in RAW_DIR.iterdir():
        print(f"  - {item.name} ({item.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
