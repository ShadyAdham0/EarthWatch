# EarthWatch — Data Contract

The agreed formats between pipeline stages and the dashboard. If you need to change anything here, open a PR that edits this file and tell the people on both sides first.

## 1. Reading GCOV (AI1 → everyone)

`pipeline.io.read_gcov(path, frequency="A", step=1) -> Scene`

| Field | Type | Meaning |
|---|---|---|
| `hh` | 2-D float32 | HHHH backscatter, **linear power** (as stored in the file) |
| `hv` | 2-D float32 | HVHV backscatter, linear power |
| `x`, `y` | 1-D float64 | Pixel-centre coordinates in the product CRS (metres) |
| `epsg` | int | Product CRS, e.g. `32642` (UTM 42N) |
| `date` | str `YYYY-MM-DD` | Acquisition date (UTC) |
| `granule` | str | Source file name |

`step > 1` reads every n-th pixel (fast previews). Files are always opened read-only.

## 2. Preprocessing → Detection (AI1 → AI2, AI3)

`pipeline.preprocess.prepare_pair(before: Scene, after: Scene) -> (Prepared, Prepared)`

Both returned objects are **on the same pixel grid** (same shape, same `x`/`y`).

| Field | Type | Meaning |
|---|---|---|
| `hh_db` | 2-D float32 | HHHH in dB: `10 * log10(linear)` |
| `hv_db` | 2-D float32 | HVHV in dB |
| `valid` | 2-D bool | `True` where usable (not fill, NaN, ≤ 0, shadow/layover) |
| `x`, `y`, `epsg`, `date` | as above | Shared grid |

## 3. Detection (AI2, AI3)

- `pipeline.detect.baseline_flood_mask(before, after) -> np.ndarray[bool]`
- `pipeline.ml.ml_flood_mask(before, after) -> np.ndarray[bool]`
- `pipeline.evaluate.summarize(mask, valid, pixel_area_m2, reference=None) -> dict`
  returns `flooded_area_km2`, `valid_area_km2`, and `iou` / `f1` when a reference is given (else `None`).

Masks are the same shape as the prepared arrays. `True` = flood (new open water).

## 4. Outputs → Dashboard (AI3 → SW)

`pipeline.export.write_outputs(...)` writes one folder per event:

```
outputs/<event_id>/
  before_hh.png      greyscale backscatter, before date
  after_hh.png       greyscale backscatter, after date
  change.png         change map (colour-coded, transparent where invalid)
  flood_mask.png     transparent PNG, flooded pixels coloured
  meta.json
```

All PNGs cover exactly `bounds` and are in **EPSG:4326 (lat/lon)** orientation: north up, row 0 = north edge.

`meta.json`:

```json
{
  "event_id": "example-event",
  "title": "Example flood event",
  "location": "Region, Country",
  "bounds": [[37.15, 69.13], [39.99, 72.83]],
  "before_date": "2026-10-30",
  "after_date": "2026-11-11",
  "satellite": "NISAR L2 GCOV, L-band",
  "method": "log-ratio + Otsu threshold",
  "flooded_area_km2": 0.0,
  "valid_area_km2": 0.0,
  "metrics": { "iou": null, "f1": null },
  "layers": {
    "before": "before_hh.png",
    "after": "after_hh.png",
    "change": "change.png",
    "flood": "flood_mask.png"
  },
  "notes": "Free text: limitations, caveats"
}
```

- `bounds` = `[[south, west], [north, east]]` — pass straight to Leaflet `L.imageOverlay(url, bounds)`.
- `metrics` values are `null` when no reference map is available.
- A working example lives in [`sample/example-event/`](../sample/example-event/) (fake data).
