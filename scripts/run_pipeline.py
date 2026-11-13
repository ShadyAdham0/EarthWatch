"""Run the whole chain: read → preprocess → detect → evaluate → export. Owner: AI3.

Usage:
  python scripts/run_pipeline.py --before data/raw/oct30/<oct30>.h5 --after data/raw/nov11/<nov11>.h5 --event tajikistan-baseline

Or with default file discovery:
  python scripts/run_pipeline.py --event tajikistan-baseline

The script identifies .h5 files by filename date, not by folder location.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import detect, evaluate, export, io, preprocess  # noqa: E402
from pipeline.export import write_geotiff  # noqa: E402


def find_h5_files(data_dir: Path) -> tuple[Path, Path]:
    """Auto-discover the Oct 30 and Nov 11 .h5 files from data/raw/.

    Files are identified by their filename (containing the acquisition date),
    NOT by folder name (the delivered zip had them swapped).

    Returns
    -------
    tuple[Path, Path]
        (before_path, after_path) — Oct 30 first, Nov 11 second.
    """
    h5_files = list(data_dir.rglob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found under {data_dir}")

    before = after = None
    for f in h5_files:
        name = f.name
        if "20261030T" in name:
            before = f
        elif "20261111T" in name:
            after = f

    if before is None:
        raise FileNotFoundError("Could not find Oct 30 file (containing '20261030T' in name)")
    if after is None:
        raise FileNotFoundError("Could not find Nov 11 file (containing '20261111T' in name)")

    return before, after


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--before", type=Path, default=None,
                    help="Path to pre-event .h5 file (auto-discovered if omitted)")
    ap.add_argument("--after", type=Path, default=None,
                    help="Path to post-event .h5 file (auto-discovered if omitted)")
    ap.add_argument("--event", required=True,
                    help="event_id, used as output folder name")
    ap.add_argument("--out", type=Path, default=Path("outputs"),
                    help="Base output directory (default: outputs/)")
    ap.add_argument("--processed-dir", type=Path, default=None,
                    help="Where to save GeoTIFFs (default: Phase_2/04_Processed_Data/)")
    ap.add_argument("--viz-dir", type=Path, default=None,
                    help="Where to save visualizations (default: Phase_2/05_Visualizations/)")
    ap.add_argument("--step", type=int, default=1,
                    help="Sub-sampling factor for faster processing (default: 1 = full res)")
    ap.add_argument("--no-speckle", action="store_true",
                    help="Skip speckle filter (for debugging)")
    ap.add_argument("--skip-ml", action="store_true",
                    help="Skip ML detection (faster)")
    args = ap.parse_args()

    # Resolve project root
    project_root = Path(__file__).resolve().parents[1]

    # Auto-discover files if not provided
    if args.before is None or args.after is None:
        data_dir = project_root / "data" / "raw"
        before_path, after_path = find_h5_files(data_dir)
    else:
        before_path, after_path = args.before, args.after

    # Output directories
    processed_dir = args.processed_dir or (project_root / "Phase_2" / "04_Processed_Data")
    viz_dir = args.viz_dir or (project_root / "Phase_2" / "05_Visualizations")
    processed_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # =========================================================================
    # Step 1: Read GCOV data
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 1: Reading GCOV data (step={args.step})")
    print(f"{'='*60}")
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}")

    scene_before = io.read_gcov(before_path, step=args.step)
    print(f"  → Before: {scene_before.date}, shape={scene_before.hh.shape}, "
          f"EPSG:{scene_before.epsg}")

    scene_after = io.read_gcov(after_path, step=args.step)
    print(f"  → After:  {scene_after.date}, shape={scene_after.hh.shape}, "
          f"EPSG:{scene_after.epsg}")

    # =========================================================================
    # Step 2: Preprocess (mask, filter, dB, align)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 2: Preprocessing (mask, speckle filter, dB, align)")
    print(f"{'='*60}")

    prep_before, prep_after = preprocess.prepare_pair(
        scene_before, scene_after,
        apply_speckle_filter=not args.no_speckle,
    )

    n_valid = int(prep_before.valid.sum())
    n_total = prep_before.valid.size
    print(f"  → Common shape: {prep_before.hh_db.shape}")
    print(f"  → Valid pixels: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")

    # =========================================================================
    # Step 3: Compute change map
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 3: Computing change map")
    print(f"{'='*60}")

    change_db = np.full(prep_before.hh_db.shape, np.nan, dtype=np.float32)
    valid = prep_before.valid & prep_after.valid
    change_db[valid] = prep_after.hh_db[valid] - prep_before.hh_db[valid]

    change_valid = change_db[np.isfinite(change_db)]
    print(f"  → Mean change: {change_valid.mean():.2f} dB")
    print(f"  → Std. dev.:   {change_valid.std():.2f} dB")
    print(f"  → Darker > 3dB: {100*(change_valid < -3).mean():.1f}%")
    print(f"  → Brighter > 3dB: {100*(change_valid > 3).mean():.1f}%")

    # =========================================================================
    # Step 4: Baseline flood detection
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 4: Baseline flood detection (log-ratio + Otsu)")
    print(f"{'='*60}")

    flood_baseline = detect.baseline_flood_mask(prep_before, prep_after)
    n_flood = int(flood_baseline.sum())
    print(f"  → Flood candidate pixels: {n_flood:,}")

    # =========================================================================
    # Step 5: Evaluate
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 5: Evaluating results")
    print(f"{'='*60}")

    # Pixel area: 20 m × 20 m at full resolution, scaled by step²
    pixel_area = (20.0 * args.step) ** 2
    stats_baseline = evaluate.summarize(flood_baseline, prep_before.valid, pixel_area)
    print(f"  → Flooded area: {stats_baseline['flooded_area_km2']} km²")
    print(f"  → Valid area:   {stats_baseline['valid_area_km2']} km²")
    print(f"  → Flooded %:    {stats_baseline['flooded_pct']}%")

    # =========================================================================
    # Step 5b: ML flood detection (optional)
    # =========================================================================
    stats_ml = None
    flood_ml = None
    if not args.skip_ml:
        print(f"\n{'='*60}")
        print(f"STEP 5b: ML flood detection (GMM clustering)")
        print(f"{'='*60}")
        try:
            from pipeline import ml
            flood_ml = ml.ml_flood_mask(prep_before, prep_after)
            stats_ml = evaluate.summarize(flood_ml, prep_before.valid, pixel_area)
            print(f"  → ML flood candidates: {int(flood_ml.sum()):,}")
            print(f"  → ML flooded area:     {stats_ml['flooded_area_km2']} km²")
        except ImportError as e:
            print(f"  → Skipping ML: {e}")
        except Exception as e:
            print(f"  → ML failed: {e}")

    # =========================================================================
    # Step 6: Export outputs (dashboard format)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 6: Exporting results")
    print(f"{'='*60}")

    event_info = {
        "event_id": args.event,
        "title": f"SAR Change Analysis: {scene_before.date} → {scene_after.date}",
        "location": "Eastern Tajikistan (~38.9°N, 71.3°E)",
        "satellite": "NISAR L2 GCOV, L-band (dual-pol HH+HV)",
        "method": "log-ratio + Otsu threshold (baseline)",
        "notes": (
            "CAUTION: The difference image is NOT a final flood map. "
            "A change in backscatter is not automatically a flood. "
            f"Both observations ({scene_before.date} and {scene_after.date}) "
            "are pre-event — this is a baseline analysis showing natural "
            "surface change without flooding."
        ),
    }

    event_dir = export.write_outputs(
        args.out, event_info, prep_before, prep_after,
        change_db, flood_baseline, stats_baseline,
    )
    print(f"  → Dashboard outputs: {event_dir}")

    # =========================================================================
    # Step 7: Export GeoTIFFs (Phase 2 deliverables)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 7: Writing GeoTIFFs to {processed_dir}")
    print(f"{'='*60}")

    # HH dB — before
    tif_path = processed_dir / f"{scene_before.date}_HH_dB.tif"
    write_geotiff(tif_path, prep_before.hh_db, prep_before.x, prep_before.y,
                  prep_before.epsg, "HH_backscatter_dB")
    print(f"  → {tif_path.name}")

    # HH dB — after
    tif_path = processed_dir / f"{scene_after.date}_HH_dB.tif"
    write_geotiff(tif_path, prep_after.hh_db, prep_after.x, prep_after.y,
                  prep_after.epsg, "HH_backscatter_dB")
    print(f"  → {tif_path.name}")

    # HV dB — before
    tif_path = processed_dir / f"{scene_before.date}_HV_dB.tif"
    write_geotiff(tif_path, prep_before.hv_db, prep_before.x, prep_before.y,
                  prep_before.epsg, "HV_backscatter_dB")
    print(f"  → {tif_path.name}")

    # HV dB — after
    tif_path = processed_dir / f"{scene_after.date}_HV_dB.tif"
    write_geotiff(tif_path, prep_after.hv_db, prep_after.x, prep_after.y,
                  prep_after.epsg, "HV_backscatter_dB")
    print(f"  → {tif_path.name}")

    # Change map
    tif_path = processed_dir / f"change_HH_dB_{scene_before.date}_to_{scene_after.date}.tif"
    write_geotiff(tif_path, change_db, prep_before.x, prep_before.y,
                  prep_before.epsg, "HH_change_dB")
    print(f"  → {tif_path.name}")

    # Baseline flood mask
    tif_path = processed_dir / f"flood_mask_baseline_{args.event}.tif"
    mask_float = flood_baseline.astype(np.float32)
    mask_float[~prep_before.valid] = np.nan
    write_geotiff(tif_path, mask_float, prep_before.x, prep_before.y,
                  prep_before.epsg, "baseline_flood_mask")
    print(f"  → {tif_path.name}")

    # Valid data mask
    tif_path = processed_dir / f"valid_mask_{args.event}.tif"
    write_geotiff(tif_path, prep_before.valid.astype(np.float32),
                  prep_before.x, prep_before.y, prep_before.epsg, "valid_data_mask")
    print(f"  → {tif_path.name}")

    # ML mask if available
    if flood_ml is not None:
        tif_path = processed_dir / f"flood_mask_ml_{args.event}.tif"
        ml_float = flood_ml.astype(np.float32)
        ml_float[~prep_before.valid] = np.nan
        write_geotiff(tif_path, ml_float, prep_before.x, prep_before.y,
                      prep_before.epsg, "ml_flood_mask")
        print(f"  → {tif_path.name}")

    # =========================================================================
    # Step 8: Generate visualizations
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 8: Generating visualizations → {viz_dir}")
    print(f"{'='*60}")

    generate_visualizations(
        prep_before, prep_after, change_db,
        flood_baseline, flood_ml,
        viz_dir, args.event,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE ({elapsed:.1f} s)")
    print(f"{'='*60}")
    print(f"  Dashboard:      {event_dir}")
    print(f"  GeoTIFFs:       {processed_dir}")
    print(f"  Visualizations: {viz_dir}")

    # Save processing log
    log = {
        "event_id": args.event,
        "before_file": str(before_path),
        "after_file": str(after_path),
        "before_date": scene_before.date,
        "after_date": scene_after.date,
        "step": args.step,
        "speckle_filter": not args.no_speckle,
        "shape": list(prep_before.hh_db.shape),
        "n_valid_pixels": n_valid,
        "pixel_area_m2": pixel_area,
        "baseline_stats": stats_baseline,
        "ml_stats": stats_ml,
        "change_stats": {
            "mean_db": float(change_valid.mean()),
            "std_db": float(change_valid.std()),
            "pct_darker_3db": float(100 * (change_valid < -3).mean()),
            "pct_brighter_3db": float(100 * (change_valid > 3).mean()),
        },
        "processing_settings": {
            "speckle_filter": preprocess.SPECKLE_FILTER,
            "speckle_window": preprocess.SPECKLE_WINDOW,
            "shadow_rtc_min": preprocess.SHADOW_RTC_MIN,
            "layover_rtc_max": preprocess.LAYOVER_RTC_MAX,
            "darkness_threshold_db": detect.DARKNESS_THRESHOLD_DB,
            "min_patch_pixels": detect.MIN_PATCH_PIXELS,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    log_path = event_dir / "processing_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"  Processing log: {log_path}")


def generate_visualizations(
    before: preprocess.Prepared,
    after: preprocess.Prepared,
    change_db: np.ndarray,
    flood_baseline: np.ndarray,
    flood_ml: np.ndarray | None,
    viz_dir: Path,
    event_id: str,
):
    """Generate all Phase 2 visualizations.

    Saves:
      - Pre-event HH image
      - Post-event HH image
      - Pre-event HV image
      - Post-event HV image
      - Side-by-side before/after comparison
      - Preliminary difference map
      - Flood mask overlays
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir.mkdir(parents=True, exist_ok=True)

    # Common extent (in km for readability)
    extent_km = [
        before.x[0] / 1e3, before.x[-1] / 1e3,
        before.y[-1] / 1e3, before.y[0] / 1e3,
    ]

    # Masked arrays for clean visualization
    def masked(arr, valid):
        return np.where(valid, arr, np.nan)

    # ---- 1. Pre-event HH ----
    print("  → Pre-event HH...")
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    im = ax.imshow(masked(before.hh_db, before.valid), extent=extent_km,
                   cmap="gray", vmin=-25, vmax=0, interpolation="nearest")
    ax.set_facecolor("#bbbbbb")
    ax.set_title(f"Pre-event HH Backscatter (Gamma0, dB)\n{before.date} — NISAR L-band",
                 fontsize=11)
    ax.set_xlabel("Easting (km, UTM 42N)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im, ax=ax, label="HH [dB]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(viz_dir / f"{event_id}_pre_HH.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 2. Post-event HH ----
    print("  → Post-event HH...")
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    im = ax.imshow(masked(after.hh_db, after.valid), extent=extent_km,
                   cmap="gray", vmin=-25, vmax=0, interpolation="nearest")
    ax.set_facecolor("#bbbbbb")
    ax.set_title(f"Post-event HH Backscatter (Gamma0, dB)\n{after.date} — NISAR L-band",
                 fontsize=11)
    ax.set_xlabel("Easting (km, UTM 42N)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im, ax=ax, label="HH [dB]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(viz_dir / f"{event_id}_post_HH.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 3. Pre-event HV ----
    print("  → Pre-event HV...")
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    im = ax.imshow(masked(before.hv_db, before.valid), extent=extent_km,
                   cmap="gray", vmin=-32, vmax=-8, interpolation="nearest")
    ax.set_facecolor("#bbbbbb")
    ax.set_title(f"Pre-event HV Backscatter (Gamma0, dB)\n{before.date} — NISAR L-band",
                 fontsize=11)
    ax.set_xlabel("Easting (km, UTM 42N)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im, ax=ax, label="HV [dB]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(viz_dir / f"{event_id}_pre_HV.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 4. Post-event HV ----
    print("  → Post-event HV...")
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    im = ax.imshow(masked(after.hv_db, after.valid), extent=extent_km,
                   cmap="gray", vmin=-32, vmax=-8, interpolation="nearest")
    ax.set_facecolor("#bbbbbb")
    ax.set_title(f"Post-event HV Backscatter (Gamma0, dB)\n{after.date} — NISAR L-band",
                 fontsize=11)
    ax.set_xlabel("Easting (km, UTM 42N)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im, ax=ax, label="HV [dB]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(viz_dir / f"{event_id}_post_HV.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 5. Side-by-side comparison ----
    print("  → Side-by-side comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=120)

    im1 = ax1.imshow(masked(before.hh_db, before.valid), extent=extent_km,
                     cmap="gray", vmin=-25, vmax=0, interpolation="nearest")
    ax1.set_facecolor("#bbbbbb")
    ax1.set_title(f"Before: {before.date}", fontsize=11)
    ax1.set_xlabel("Easting (km)")
    ax1.set_ylabel("Northing (km)")
    plt.colorbar(im1, ax=ax1, label="HH [dB]", shrink=0.7)

    im2 = ax2.imshow(masked(after.hh_db, after.valid), extent=extent_km,
                     cmap="gray", vmin=-25, vmax=0, interpolation="nearest")
    ax2.set_facecolor("#bbbbbb")
    ax2.set_title(f"After: {after.date}", fontsize=11)
    ax2.set_xlabel("Easting (km)")
    ax2.set_ylabel("Northing (km)")
    plt.colorbar(im2, ax=ax2, label="HH [dB]", shrink=0.7)

    fig.suptitle("HH Backscatter Before vs. After — NISAR L-band (Gamma0, dB)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(viz_dir / f"{event_id}_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 6. Difference map ----
    print("  → Difference map...")
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    im = ax.imshow(masked(change_db, before.valid), extent=extent_km,
                   cmap="RdBu_r", vmin=-6, vmax=6, interpolation="nearest")
    ax.set_facecolor("#bbbbbb")
    ax.set_title(f"HH Change (dB): {after.date} minus {before.date}\n"
                 "Blue = darker (potential water/wet), Red = brighter\n"
                 "⚠ This is NOT a flood map — see notes", fontsize=10)
    ax.set_xlabel("Easting (km, UTM 42N)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im, ax=ax, label="Δ HH [dB]", shrink=0.8)
    fig.tight_layout()
    fig.savefig(viz_dir / f"{event_id}_difference_map.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 7. Flood mask overlay ----
    print("  → Flood mask overlay...")
    fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
    ax.imshow(masked(after.hh_db, after.valid), extent=extent_km,
              cmap="gray", vmin=-25, vmax=0, interpolation="nearest")

    # Overlay flood candidates in semi-transparent red
    flood_overlay = np.ma.masked_where(~flood_baseline, flood_baseline.astype(float))
    ax.imshow(flood_overlay, extent=extent_km, cmap="Reds", alpha=0.6,
              vmin=0, vmax=1, interpolation="nearest")

    ax.set_facecolor("#bbbbbb")
    ax.set_title(f"Baseline Flood Candidates (red) on Post-event HH\n"
                 f"{after.date} — ⚠ PRELIMINARY, not a confirmed flood map",
                 fontsize=10)
    ax.set_xlabel("Easting (km, UTM 42N)")
    ax.set_ylabel("Northing (km)")
    fig.tight_layout()
    fig.savefig(viz_dir / f"{event_id}_flood_overlay.png", bbox_inches="tight")
    plt.close(fig)

    # ---- 8. ML flood mask overlay (if available) ----
    if flood_ml is not None and flood_ml.any():
        print("  → ML flood mask overlay...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=120)

        ax1.imshow(masked(after.hh_db, after.valid), extent=extent_km,
                   cmap="gray", vmin=-25, vmax=0, interpolation="nearest")
        overlay1 = np.ma.masked_where(~flood_baseline, flood_baseline.astype(float))
        ax1.imshow(overlay1, extent=extent_km, cmap="Reds", alpha=0.6,
                   vmin=0, vmax=1, interpolation="nearest")
        ax1.set_title("Baseline (log-ratio + Otsu)", fontsize=11)
        ax1.set_xlabel("Easting (km)")
        ax1.set_ylabel("Northing (km)")
        ax1.set_facecolor("#bbbbbb")

        ax2.imshow(masked(after.hh_db, after.valid), extent=extent_km,
                   cmap="gray", vmin=-25, vmax=0, interpolation="nearest")
        overlay2 = np.ma.masked_where(~flood_ml, flood_ml.astype(float))
        ax2.imshow(overlay2, extent=extent_km, cmap="Blues", alpha=0.6,
                   vmin=0, vmax=1, interpolation="nearest")
        ax2.set_title("ML (GMM clustering)", fontsize=11)
        ax2.set_xlabel("Easting (km)")
        ax2.set_ylabel("Northing (km)")
        ax2.set_facecolor("#bbbbbb")

        fig.suptitle("Flood Candidates: Baseline vs. ML\n"
                     "⚠ PRELIMINARY — not confirmed flood maps", fontsize=12, y=1.01)
        fig.tight_layout()
        fig.savefig(viz_dir / f"{event_id}_baseline_vs_ml.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  → All visualizations saved to {viz_dir}")


if __name__ == "__main__":
    import numpy as np  # ensure numpy is in scope for main
    main()
