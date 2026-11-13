"""Collect Phase 1 facts from two GCOV files and compare their grids (read-only).

  python scripts/phase1_facts.py <before.h5> <after.h5> [--out outputs/phase1/facts.json]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

G = "/science/LSAR/GCOV"
ID = "/science/LSAR/identification"


def val(f, path):
    if path not in f:
        return None
    v = f[path][()]
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        if v.size > 8:
            return f"<array {v.shape}>"
        return [x.decode(errors="replace") if isinstance(x, bytes) else x.item() for x in v]
    return v.item() if hasattr(v, "item") else v


def scalar_params(f, group):
    """All scalar (non-array) datasets under a group, as {relative_path: value}."""
    out = {}
    if group not in f:
        return out

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.size <= 4:
            out[name] = val(f, f"{group}/{name}")
    f[group].visititems(visit)
    return out


def facts(path):
    with h5py.File(path, "r") as f:
        ga = f"{G}/grids/frequencyA"
        x = f[f"{ga}/xCoordinates"][()]
        y = f[f"{ga}/yCoordinates"][()]
        d = {
            "file": Path(path).name,
            "identification": scalar_params(f, ID),
            "params": scalar_params(f, f"{G}/metadata/processingInformation/parameters"),
            "algorithms": scalar_params(f, f"{G}/metadata/processingInformation/algorithms"),
            "card": scalar_params(f, f"{G}/metadata/ceosAnalysisReadyData"),
            "orbitType": val(f, f"{G}/metadata/orbit/orbitType"),
            "attitudeType": val(f, f"{G}/metadata/attitude/attitudeType"),
            "freqA": {
                "epsg": val(f, f"{ga}/projection"),
                "shape": list(f[f"{ga}/HHHH"].shape),
                "dx": val(f, f"{ga}/xCoordinateSpacing"),
                "dy": val(f, f"{ga}/yCoordinateSpacing"),
                "x0": float(x[0]), "x1": float(x[-1]),
                "y0": float(y[0]), "y1": float(y[-1]),
                "terms": val(f, f"{ga}/listOfCovarianceTerms"),
            },
        }
        d["identification"].pop("boundingPolygon", None)
        return d, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out", default="outputs/phase1/facts.json")
    a = ap.parse_args()

    fb, xb, yb = facts(a.before)
    fa, xa, ya = facts(a.after)
    dx = fb["freqA"]["dx"]
    dy = fb["freqA"]["dy"]
    off_x = (xa[0] - xb[0]) / dx
    off_y = (ya[0] - yb[0]) / dy
    sub_x = abs(off_x - round(off_x))
    sub_y = abs(off_y - round(off_y))
    ox, oy = int(round(off_x)), int(round(off_y))
    # overlap in pixel indices of the BEFORE grid
    cx0, cx1 = max(0, ox), min(len(xb), ox + len(xa))
    cy0, cy1 = max(0, oy), min(len(yb), oy + len(ya))
    align = {
        "same_epsg": fb["freqA"]["epsg"] == fa["freqA"]["epsg"],
        "same_spacing": (fb["freqA"]["dx"], fb["freqA"]["dy"]) == (fa["freqA"]["dx"], fa["freqA"]["dy"]),
        "same_shape": fb["freqA"]["shape"] == fa["freqA"]["shape"],
        "offset_px_x": float(off_x), "offset_px_y": float(off_y),
        "subpixel_residual_x": float(sub_x), "subpixel_residual_y": float(sub_y),
        "integer_shift": bool(sub_x < 1e-6 and sub_y < 1e-6),
        "overlap_cols_in_before": [cx0, cx1], "overlap_rows_in_before": [cy0, cy1],
        "overlap_shape": [cy1 - cy0, cx1 - cx0],
    }
    # parameters that differ between the two files
    diff = {}
    for sec in ("identification", "params", "algorithms", "card"):
        for k in sorted(set(fb[sec]) | set(fa[sec])):
            if fb[sec].get(k) != fa[sec].get(k):
                diff[f"{sec}/{k}"] = [fb[sec].get(k), fa[sec].get(k)]

    out = {"before": fb, "after": fa, "alignment": align, "differences": diff}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"alignment": align, "differences": diff}, indent=2, default=str))


if __name__ == "__main__":
    main()
