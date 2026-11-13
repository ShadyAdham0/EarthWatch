"""Runner script for EarthWatch Phase 2 Sentinel-1 SAR Pipeline.

Usage:
  python scripts/run_phase2_pipeline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.sentinel1_pipeline import Sentinel1Pipeline, CONFIG


def main():
    raw_dir = ROOT / "Phase_2" / "01_Raw_Data"
    processed_dir = ROOT / "Phase_2" / "04_Processed_Data"
    viz_dir = ROOT / "Phase_2" / "05_Visualizations"

    print("=" * 70)
    print("EARTHWATCH — SENTINEL-1 SAR PREPROCESSING & FLOOD PIPELINE")
    print("NASA Space Apps Cairo 2026 — 'Dancing with the SARs'")
    print("=" * 70)

    # Check that raw data exists, otherwise run the generator
    if not list(raw_dir.glob("*.tif")):
        print("Raw data not found in Phase_2/01_Raw_Data/. Preparing raw data first...")
        from scripts import prepare_phase2_raw_data
        prepare_phase2_raw_data.main()

    pipeline = Sentinel1Pipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        viz_dir=viz_dir,
        config=CONFIG,
    )
    summary = pipeline.run()

    print("\n" + "=" * 70)
    print("PIPELINE DELIVERABLES GENERATED:")
    print("=" * 70)
    print("\nProcessed GeoTIFF Rasters in EarthWatch/Phase_2/04_Processed_Data/:")
    for tif in sorted(processed_dir.glob("*.tif")):
        size_mb = tif.stat().st_size / (1024 * 1024)
        print(f"  • {tif.name:36s} ({size_mb:.2f} MB)")

    print(f"\nxarray Datacube in EarthWatch/Phase_2/04_Processed_Data/:")
    nc = processed_dir / "sentinel1_processed_datacube.nc"
    if nc.exists():
        print(f"  • {nc.name:36s} ({nc.stat().st_size / (1024*1024):.2f} MB)")

    print("\nVisualizations in EarthWatch/Phase_2/05_Visualizations/:")
    for png in sorted(viz_dir.glob("*.png")):
        size_kb = png.stat().st_size / 1024
        print(f"  • {png.name:36s} ({size_kb:.1f} KB)")

    print("\nSummary Metrics:")
    print(f"  • Valid Analysis Area:      {summary['metrics']['valid_study_area_km2']} km²")
    print(f"  • Mountain Shadow/Layover:  {summary['metrics']['shadow_layover_area_km2']} km² (Masked)")
    print(f"  • Preliminary Flood Area:   {summary['metrics']['preliminary_flood_area_km2']} km²")
    print(f"  • Flood Candidate Percent:  {summary['metrics']['flood_candidate_percent']}% of valid area")
    print(f"  • Otsu Change Threshold:    {summary['metrics']['otsu_difference_threshold_db']} dB")
    print("=" * 70)


if __name__ == "__main__":
    main()
