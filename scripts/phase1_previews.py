"""Phase 1 previews: HHHH / HVHV in dB for each date, plus a change preview (read-only).

  python scripts/phase1_previews.py <before.h5> <after.h5> [--step 10] [--out outputs/phase1]

Reads every `step`-th pixel. Invalid pixels (NaN, Inf, <= 0, outside swath, not fully focused)
are masked and shown transparent/grey.
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

GA = "/science/LSAR/GCOV/grids/frequencyA"


def load(path, step):
    with h5py.File(path, "r") as f:
        gid = f["/science/LSAR/identification/granuleId"][()].decode()
        date = f["/science/LSAR/identification/zeroDopplerStartTime"][()].decode()[:10]
        x = f[f"{GA}/xCoordinates"][::step]
        y = f[f"{GA}/yCoordinates"][::step]
        mask = f[f"{GA}/mask"][::step, ::step]
        layers = {t: f[f"{GA}/{t}"][::step, ::step] for t in ("HHHH", "HVHV")}
    in_swath = mask != 255
    focused = (mask != 0) & in_swath
    out = {"gid": gid, "date": date, "x": x, "y": y, "in_swath": in_swath, "focused": focused}
    for t, lin in layers.items():
        ok = np.isfinite(lin) & (lin > 0) & focused
        db = np.full(lin.shape, np.nan, np.float32)
        db[ok] = 10 * np.log10(lin[ok])
        out[t] = db
        out[f"{t}_stats"] = {
            "nan_or_inf_in_swath_pct": float(100 * (~np.isfinite(lin) & in_swath).sum() / in_swath.sum()),
            "nonpositive_pct": float(100 * (np.isfinite(lin) & (lin <= 0)).sum() / in_swath.sum()),
            "valid_pct_of_swath": float(100 * ok.sum() / in_swath.sum()),
            "p2_db": float(np.nanpercentile(db, 2)), "median_db": float(np.nanmedian(db)),
            "p98_db": float(np.nanpercentile(db, 98)),
        }
    out["swath_pct_of_grid"] = float(100 * in_swath.mean())
    out["not_focused_pct_of_swath"] = float(100 * (in_swath & ~focused).sum() / in_swath.sum())
    return out


def show(ax, img, extent, title, cmap, vmin, vmax, label):
    im = ax.imshow(img, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_facecolor("#bbbbbb")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Easting (km, UTM 42N)")
    ax.set_ylabel("Northing (km)")
    plt.colorbar(im, ax=ax, label=label, shrink=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--out", default="outputs/phase1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    scenes = [load(a.before, a.step), load(a.after, a.step)]
    stats = {}
    ranges = {"HHHH": (-25, 0), "HVHV": (-32, -8)}
    for s in scenes:
        extent = [s["x"][0] / 1e3, s["x"][-1] / 1e3, s["y"][-1] / 1e3, s["y"][0] / 1e3]
        stats[s["date"]] = {"granule": s["gid"], "swath_pct_of_grid": s["swath_pct_of_grid"],
                            "not_focused_pct_of_swath": s["not_focused_pct_of_swath"]}
        for t in ("HHHH", "HVHV"):
            stats[s["date"]][t] = s[f"{t}_stats"]
            fig, ax = plt.subplots(figsize=(8, 7.5), dpi=110)
            show(ax, s[t], extent, f"NISAR L2 GCOV {t} (gamma0, dB) — {s['date']}\n{s['gid']}",
                 "gray", *ranges[t], f"{t} backscatter [dB]")
            fig.tight_layout()
            fig.savefig(out / f"{s['date']}_{t}.png")
            plt.close(fig)

    b, af = scenes
    extent = [b["x"][0] / 1e3, b["x"][-1] / 1e3, b["y"][-1] / 1e3, b["y"][0] / 1e3]
    diff = af["HHHH"] - b["HHHH"]
    fig, ax = plt.subplots(figsize=(8, 7.5), dpi=110)
    show(ax, diff, extent, f"HHHH change (dB): {af['date']} minus {b['date']}\n"
         "Blue = darker (possible new water / wet snow), red = brighter",
         "RdBu_r", -6, 6, "Δ HHHH [dB]")
    fig.tight_layout()
    fig.savefig(out / "change_HHHH.png")
    plt.close(fig)
    d = diff[np.isfinite(diff)]
    stats["change_HHHH_db"] = {"mean": float(d.mean()), "std": float(d.std()),
                               "pct_below_minus3db": float(100 * (d < -3).mean()),
                               "pct_above_plus3db": float(100 * (d > 3).mean())}

    (out / "preview_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
