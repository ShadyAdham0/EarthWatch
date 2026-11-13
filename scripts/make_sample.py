"""Generate FAKE outputs in the contract format (sample/example-event/) for frontend work.

The images are synthetic, not real satellite data.
  python scripts/make_sample.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample" / "example-event"
H, W = 300, 390  # rows (north→south), cols (west→east)


def save_grey(path: Path, db: np.ndarray) -> None:
    plt.imsave(path, db, cmap="gray", vmin=-25, vmax=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]

    before = -9 + 3 * np.sin(xx / 25) * np.cos(yy / 30) + rng.normal(0, 1.5, (H, W))
    river = np.abs(yy - (150 + 40 * np.sin(xx / 60))) < 6
    before[river] = -22 + rng.normal(0, 1, river.sum())

    flood = (np.abs(yy - (150 + 40 * np.sin(xx / 60))) < 30) & (xx > 120) & (xx < 280)
    after = before.copy()
    after[flood] = -21 + rng.normal(0, 1.2, flood.sum())

    change = after - before
    save_grey(OUT / "before_hh.png", before)
    save_grey(OUT / "after_hh.png", after)

    rgba = plt.get_cmap("RdBu")(np.clip((change + 10) / 20, 0, 1))
    rgba[np.abs(change) < 2, 3] = 0.0
    plt.imsave(OUT / "change.png", rgba)

    mask = np.zeros((H, W, 4))
    mask[flood & ~river] = [0.10, 0.45, 0.95, 0.75]
    plt.imsave(OUT / "flood_mask.png", mask)

    pixel_km2 = (2.84 * 111 / H) * (3.70 * 87 / W)  # rough size of one sample pixel
    meta = {
        "event_id": "example-event",
        "title": "Example flood event (FAKE SAMPLE DATA)",
        "location": "Sample region",
        "bounds": [[37.15, 69.13], [39.99, 72.83]],
        "before_date": "2026-10-30",
        "after_date": "2026-11-11",
        "satellite": "NISAR L2 GCOV, L-band",
        "method": "log-ratio + Otsu threshold",
        "flooded_area_km2": round(float((flood & ~river).sum() * pixel_km2), 1),
        "valid_area_km2": round(float(H * W * pixel_km2), 1),
        "metrics": {"iou": 0.71, "f1": 0.83},
        "layers": {"before": "before_hh.png", "after": "after_hh.png",
                   "change": "change.png", "flood": "flood_mask.png"},
        "notes": "Synthetic sample for dashboard development. Not real data.",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote sample outputs to {OUT}")


if __name__ == "__main__":
    main()
