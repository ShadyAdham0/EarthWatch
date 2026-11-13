"""Print the structure of a NISAR HDF5 file (read-only).

  python scripts/inspect_h5.py <file.h5> [--values] [--group /science/LSAR/identification]

--values prints the contents of small datasets (strings, scalars, short arrays).
"""

import argparse

import h5py
import numpy as np


def fmt_attrs(obj) -> str:
    parts = []
    for k in ("units", "description", "long_name", "_FillValue", "valid_min", "valid_max"):
        if k in obj.attrs:
            v = obj.attrs[k]
            if isinstance(v, bytes):
                v = v.decode(errors="replace")
            elif isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            parts.append(f"{k}={v}")
    return "  [" + "; ".join(str(p) for p in parts) + "]" if parts else ""


def small_value(ds):
    if ds.size > 16:
        return None
    v = ds[()]
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray):
        return [x.decode(errors="replace") if isinstance(x, bytes) else x for x in v.tolist()]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--values", action="store_true")
    ap.add_argument("--group", default="/")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        def visit(name, obj):
            full = f"{args.group.rstrip('/')}/{name}"
            if isinstance(obj, h5py.Dataset):
                line = f"{full}  shape={obj.shape} dtype={obj.dtype}"
                if obj.compression:
                    line += f" comp={obj.compression}"
                line += fmt_attrs(obj)
                if args.values:
                    v = small_value(obj)
                    if v is not None:
                        line += f"  = {v!r}"
                print(line)
            else:
                print(f"{full}/")
        f[args.group].visititems(visit)


if __name__ == "__main__":
    main()
