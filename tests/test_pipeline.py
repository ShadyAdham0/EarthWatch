"""Unit and integration test suite for EarthWatch processing pipeline.

Tests the agreed contract (docs/CONTRACT.md) across:
  - pipeline.preprocess (to_db, Enhanced Lee filter, prepare_pair)
  - pipeline.detect (baseline_flood_mask on synthetic arrays)
  - pipeline.evaluate (summarize area and IoU/F1 accuracy vs reference)
  - pipeline.export (write_outputs reproducing contract format and meta.json)
  - pipeline.ml (ml_flood_mask GMM clustering)
  - pipeline.sentinel1_pipeline (Phase 2 Sentinel-1 end-to-end pipeline)
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline import detect, evaluate, export, preprocess
from pipeline.io import Scene
from pipeline.preprocess import Prepared


class TestPreprocess(unittest.TestCase):
    """Test radiometric conversions and preprocessing functions."""

    def test_to_db_valid_values(self):
        # 1.0 linear = 0.0 dB, 0.1 linear = -10.0 dB, 0.01 linear = -20.0 dB
        linear = np.array([1.0, 0.1, 0.01], dtype=np.float32)
        db = preprocess.to_db(linear)
        np.testing.assert_allclose(db, [0.0, -10.0, -20.0], atol=1e-5)

    def test_to_db_nonpositive_and_nan(self):
        # Zero, negative, and NaN should become NaN
        linear = np.array([0.0, -1.0, np.nan, 10.0], dtype=np.float32)
        db = preprocess.to_db(linear)
        self.assertTrue(np.isnan(db[0]))
        self.assertTrue(np.isnan(db[1]))
        self.assertTrue(np.isnan(db[2]))
        self.assertAlmostEqual(db[3], 10.0, places=4)

    def test_prepare_pair_synthetic_scenes(self):
        shape = (50, 50)
        x = np.linspace(500000, 501000, 50)
        y = np.linspace(4301000, 4300000, 50)

        # Scene 1: Before
        hh_b = np.full(shape, 0.15, dtype=np.float32)  # ~ -8.2 dB
        hv_b = np.full(shape, 0.03, dtype=np.float32)  # ~ -15.2 dB
        mask_b = np.ones(shape, dtype=np.uint8)
        mask_b[0:5, :] = 0  # not focused edge
        scene_b = Scene(hh_b, hv_b, x, y, 32642, "2026-10-30", "scene_b", mask_b)

        # Scene 2: After
        hh_a = np.full(shape, 0.15, dtype=np.float32)
        hv_a = np.full(shape, 0.03, dtype=np.float32)
        mask_a = np.ones(shape, dtype=np.uint8)
        mask_a[:, 0:5] = 255  # outside swath
        scene_a = Scene(hh_a, hv_a, x, y, 32642, "2026-11-11", "scene_a", mask_a)

        prep_b, prep_a = preprocess.prepare_pair(scene_b, scene_a, apply_speckle_filter=False)

        # Check alignment
        self.assertEqual(prep_b.hh_db.shape, shape)
        self.assertEqual(prep_a.hh_db.shape, shape)
        self.assertEqual(prep_b.date, "2026-10-30")
        self.assertEqual(prep_a.date, "2026-11-11")

        # Check common validity mask: row 0:5 and col 0:5 must be invalid
        self.assertFalse(prep_b.valid[2, 25])
        self.assertFalse(prep_b.valid[25, 2])
        self.assertTrue(prep_b.valid[25, 25])


class TestDetection(unittest.TestCase):
    """Test baseline flood detection on synthetic change scenarios."""

    def test_baseline_detection_synthetic_water_patch(self):
        shape = (100, 100)
        x = np.linspace(0, 2000, 100)
        y = np.linspace(2000, 0, 100)

        # Land baseline: -8 dB (before)
        before_db = np.full(shape, -8.0, dtype=np.float32)
        before_hv = np.full(shape, -15.0, dtype=np.float32)

        # After: land stays at -8 dB, but a 20x20 patch becomes open water (-22 dB)
        after_db = before_db.copy()
        after_hv = before_hv.copy()
        after_db[40:60, 40:60] = -22.0  # -14 dB decrease (water drop)
        after_hv[40:60, 40:60] = -28.0

        valid = np.ones(shape, dtype=bool)

        prep_b = Prepared(before_db, before_hv, valid, x, y, 32642, "2026-10-30")
        prep_a = Prepared(after_db, after_hv, valid, x, y, 32642, "2026-11-11")

        mask = detect.baseline_flood_mask(prep_b, prep_a)

        # The synthetic flooded box should be detected (True)
        flood_center = mask[45:55, 45:55]
        self.assertTrue(flood_center.all(), "Synthetic flooded core must be detected as True")

        # The surrounding land should be non-flood (False)
        self.assertFalse(mask[10:30, 10:30].any(), "Unchanged land must be False")

    def test_baseline_detection_rejects_brightening(self):
        shape = (80, 80)
        x = np.linspace(0, 1600, 80)
        y = np.linspace(1600, 0, 80)
        valid = np.ones(shape, dtype=bool)

        # Soil moisture / vegetation change causes brightening (+4 dB)
        before_db = np.full(shape, -10.0, dtype=np.float32)
        after_db = np.full(shape, -6.0, dtype=np.float32)  # brighter, not darker!
        before_hv = np.full(shape, -17.0, dtype=np.float32)
        after_hv = np.full(shape, -14.0, dtype=np.float32)

        prep_b = Prepared(before_db, before_hv, valid, x, y, 32642, "2026-10-30")
        prep_a = Prepared(after_db, after_hv, valid, x, y, 32642, "2026-11-11")

        mask = detect.baseline_flood_mask(prep_b, prep_a)
        self.assertEqual(int(mask.sum()), 0, "Brightened pixels must never be classified as flood")


class TestEvaluation(unittest.TestCase):
    """Test area calculations and evaluation metrics."""

    def test_summarize_without_reference(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 20:40] = True  # 400 pixels flooded
        valid = np.ones((100, 100), dtype=bool)
        pixel_area_m2 = 400.0  # 20m x 20m = 400 m²

        stats = evaluate.summarize(mask, valid, pixel_area_m2)

        # 400 pixels * 400 m² = 160,000 m² = 0.16 km²
        self.assertEqual(stats["flooded_area_km2"], 0.16)
        # 10,000 pixels * 400 m² = 4.0 km²
        self.assertEqual(stats["valid_area_km2"], 4.0)
        self.assertEqual(stats["flooded_pct"], 4.0)
        self.assertIsNone(stats["iou"])
        self.assertIsNone(stats["f1"])

    def test_summarize_with_perfect_reference(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True
        reference = mask.copy()
        valid = np.ones((50, 50), dtype=bool)

        stats = evaluate.summarize(mask, valid, 400.0, reference=reference)
        self.assertEqual(stats["iou"], 1.0)
        self.assertEqual(stats["f1"], 1.0)
        self.assertEqual(stats["precision"], 1.0)
        self.assertEqual(stats["recall"], 1.0)


class TestExport(unittest.TestCase):
    """Test dashboard handoff export matching docs/CONTRACT.md."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_outputs_contract_structure(self):
        shape = (40, 40)
        x = np.linspace(699000, 700000, 40)
        y = np.linspace(4309000, 4308000, 40)
        valid = np.ones(shape, dtype=bool)

        before_db = np.full(shape, -9.0, dtype=np.float32)
        after_db = np.full(shape, -9.0, dtype=np.float32)
        after_db[15:25, 15:25] = -21.0
        change_db = after_db - before_db

        flood_mask = np.zeros(shape, dtype=bool)
        flood_mask[15:25, 15:25] = True

        prep_b = Prepared(before_db, before_db, valid, x, y, 32642, "2026-10-30")
        prep_a = Prepared(after_db, after_db, valid, x, y, 32642, "2026-11-11")

        event = {
            "event_id": "test-event",
            "title": "Unit Test Event",
            "location": "Tajikistan",
            "satellite": "NISAR L2 GCOV",
            "method": "log-ratio + Otsu",
        }
        stats = {"flooded_area_km2": 0.04, "valid_area_km2": 0.64, "iou": None, "f1": None}

        out_path = export.write_outputs(self.temp_dir, event, prep_b, prep_a, change_db, flood_mask, stats)

        # Check all required files are present
        self.assertTrue((out_path / "before_hh.png").exists())
        self.assertTrue((out_path / "after_hh.png").exists())
        self.assertTrue((out_path / "change.png").exists())
        self.assertTrue((out_path / "flood_mask.png").exists())
        self.assertTrue((out_path / "meta.json").exists())

        # Verify meta.json schema
        meta = json.loads((out_path / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["event_id"], "test-event")
        self.assertEqual(len(meta["bounds"]), 2)
        self.assertIn("layers", meta)
        self.assertEqual(meta["layers"]["before"], "before_hh.png")
        self.assertEqual(meta["layers"]["flood"], "flood_mask.png")


class TestML(unittest.TestCase):
    """Test machine learning clustering flood detection."""

    def test_ml_detection_water_patch(self):
        from pipeline import ml

        shape = (80, 80)
        x = np.linspace(0, 1600, 80)
        y = np.linspace(1600, 0, 80)
        valid = np.ones(shape, dtype=bool)

        # Baseline: rough terrain -8 dB (HH), -15 dB (HV)
        before_db = -8.0 + 1.0 * np.random.randn(*shape).astype(np.float32)
        before_hv = -15.0 + 1.0 * np.random.randn(*shape).astype(np.float32)

        after_db = before_db.copy()
        after_hv = before_hv.copy()

        # Add a water patch in the center
        after_db[30:50, 30:50] = -22.0 + 0.5 * np.random.randn(20, 20).astype(np.float32)
        after_hv[30:50, 30:50] = -28.0 + 0.5 * np.random.randn(20, 20).astype(np.float32)

        prep_b = Prepared(before_db, before_hv, valid, x, y, 32642, "2026-10-30")
        prep_a = Prepared(after_db, after_hv, valid, x, y, 32642, "2026-11-11")

        mask = ml.ml_flood_mask(prep_b, prep_a)
        self.assertEqual(mask.shape, shape)
        # Check that the water patch is largely detected
        patch_detected = mask[35:45, 35:45].mean()
        self.assertGreater(patch_detected, 0.5, "ML should detect majority of the synthetic water patch")


class TestSentinel1Pipeline(unittest.TestCase):
    """Test Phase 2 Sentinel-1 pipeline modules."""

    def test_pipeline_deliverables_exist(self):
        root = Path(__file__).resolve().parents[1]
        raw_dir = root / "Phase_2" / "01_Raw_Data"
        proc_dir = root / "Phase_2" / "04_Processed_Data"
        viz_dir = root / "Phase_2" / "05_Visualizations"

        # Check raw files
        self.assertTrue((raw_dir / "S1A_IW_GRDH_20261102T131353_pre_VV_raw.tif").exists())
        self.assertTrue((raw_dir / "S1A_IW_GRDH_20261114T131353_post_VV_raw.tif").exists())
        self.assertTrue((raw_dir / "Copernicus_DEM_30m_Tajikistan.tif").exists())

        # Check processed rasters
        self.assertTrue((proc_dir / "pre_event_VV_gamma0_dB.tif").exists())
        self.assertTrue((proc_dir / "post_event_VV_gamma0_dB.tif").exists())
        self.assertTrue((proc_dir / "difference_VV_dB.tif").exists())
        self.assertTrue((proc_dir / "shadow_layover_mask.tif").exists())
        self.assertTrue((proc_dir / "preliminary_flood_candidates.tif").exists())
        self.assertTrue((proc_dir / "sentinel1_processed_datacube.nc").exists())
        self.assertTrue((proc_dir / "processing_summary.json").exists())

        # Check visualizations
        self.assertTrue((viz_dir / "01_pre_event_VV.png").exists())
        self.assertTrue((viz_dir / "02_post_event_VV.png").exists())
        self.assertTrue((viz_dir / "05_before_after_comparison.png").exists())
        self.assertTrue((viz_dir / "06_preliminary_difference_map.png").exists())
        self.assertTrue((viz_dir / "07_flood_candidates_overlay.png").exists())


if __name__ == "__main__":
    unittest.main()
