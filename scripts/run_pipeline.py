"""Run the whole chain: read → preprocess → detect → evaluate → export. Owner: AI3.

Usage (once implemented):
  python scripts/run_pipeline.py --before data/raw/<oct30>.h5 --after data/raw/<nov11>.h5 --event example-event
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import detect, evaluate, export, io, preprocess  # noqa: E402,F401


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--after", type=Path, required=True)
    ap.add_argument("--event", required=True, help="event_id, used as output folder name")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.parse_args()
    raise NotImplementedError("AI3: wire io.read_gcov → preprocess.prepare_pair → detect → evaluate → export")


if __name__ == "__main__":
    main()
