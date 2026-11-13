"""Download the NISAR granules listed in data/granules.txt from ASF.

Credentials are never stored in the repo. Provide ONE of:
  - EARTHDATA_TOKEN env var (Earthdata Login -> Generate Token), or
  - EARTHDATA_USERNAME + EARTHDATA_PASSWORD env vars, or
  - a ~/.netrc (_netrc on Windows) entry for urs.earthdata.nasa.gov

Usage:
  python scripts/download_data.py              # list + download main files
  python scripts/download_data.py --dry-run    # only check granules exist
  python scripts/download_data.py --all-files  # also XML metadata / QA files
"""

import argparse
import os
import sys
from pathlib import Path

import asf_search as asf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = ROOT / "data" / "granules.txt"
DEFAULT_OUT = ROOT / "data" / "raw"


def read_granules(path: Path) -> dict[str, str]:
    """Return {label: granule_name}. Label defaults to the granule name."""
    granules = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            label, name = (p.strip() for p in line.split("|", 1))
        else:
            label = name = line
        if "..." in name:
            sys.exit(f"Placeholder granule name in {path}: {name!r}. Replace it first.")
        granules[label] = name
    if not granules:
        sys.exit(f"No granules listed in {path}.")
    return granules


def make_session() -> asf.ASFSession:
    session = asf.ASFSession()
    token = os.environ.get("EARTHDATA_TOKEN")
    user = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if token:
        return session.auth_with_token(token)
    if user and password:
        return session.auth_with_creds(user, password)
    print("No EARTHDATA_* env vars set; relying on .netrc for authentication.")
    return session


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=DEFAULT_LIST, help="granule list file")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="download directory")
    ap.add_argument("--dry-run", action="store_true", help="search only, no download")
    ap.add_argument("--all-files", action="store_true", help="download additional files (XML, QA) too")
    args = ap.parse_args()

    granules = read_granules(args.list)
    results = asf.granule_search(list(granules.values()))

    found = {p.properties.get("sceneName") or p.properties.get("fileID") for p in results}
    missing = [n for n in granules.values() if not any(n in (f or "") for f in found)]
    for p in results:
        props = p.properties
        size_mb = (props.get("bytes") or 0) / 1e6
        print(f"{props.get('sceneName')}  {props.get('startTime')}  "
              f"track={props.get('pathNumber')} frame={props.get('frameNumber')}  {size_mb:,.0f} MB")
    if missing:
        sys.exit(f"Not found on ASF: {missing}")
    if args.dry_run:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    file_type = asf.FileDownloadType.ALL_FILES if args.all_files else asf.FileDownloadType.DEFAULT_FILE
    results.download(path=str(args.out), session=make_session(), processes=2, fileType=file_type)
    print(f"Done. Files in {args.out}")


if __name__ == "__main__":
    main()
