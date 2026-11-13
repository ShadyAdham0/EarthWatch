# EarthWatch

NASA Space Apps Challenge 2026 — *Dancing with the SARs*. Surface-change and flood detection from NISAR L-band SAR (L2 GCOV).

**Team: read [docs/WORKFLOW.md](docs/WORKFLOW.md) first** — roles, repo structure, file contract and git workflow.

## Quick start

### 1. Get access and clone

The repo is private. Accept the GitHub invite you were sent (ask the repo owner if you don't have one), then:

```bash
git clone https://github.com/ShadyAdham0/EarthWatch.git
cd EarthWatch
```

### 2. Set up Python (3.10+)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check it works:

```bash
python -c "import h5py, numpy, matplotlib; print('ok')"
```

### 3. Pick your role

Open [docs/WORKFLOW.md § 2](docs/WORKFLOW.md#2-roles--pick-yours), choose a role, claim it in the team chat and put your name in the table.

### 4. Get the data (only if your role needs it)

**AI1, AI2, AI3 and G1 need the data. SW and G2 can start without it** (SW uses [`sample/`](sample/)).

For now the two NISAR observations (Oct 30 and Nov 11, 2026) are shared as one zip (~4 GB) in the team Google Drive — the link is in the team chat. Extract it into `data/raw/`:

```
data/raw/<files from the zip>.h5
```

`data/raw/` is git-ignored, so nothing you put there gets committed. **Never commit `.h5` files.**

Once the granule names are filled in [`data/granules.txt`](data/granules.txt), you can instead download straight from NASA — see [Downloading from ASF](#downloading-from-asf).

### 5. Start working

```bash
git switch -c <your-id>/<topic>      # e.g. ai1/preprocess, sw/map-view
```

Then follow your role's first steps below, and the git workflow in [docs/WORKFLOW.md § 5](docs/WORKFLOW.md#5-git-workflow).

## First steps per role

| Role | Start here |
|---|---|
| **AI1** — preprocessing | Open one `.h5` file with `h5py` and list its groups and datasets. Fill in `pipeline/io.py` → `read_gcov()`. |
| **AI2** — baseline detection | Build `pipeline/detect.py` → `baseline_flood_mask()` on small synthetic arrays first (see the docstring). |
| **AI3** — ML & export | Fill in `pipeline/export.py` → `write_outputs()` so it reproduces the files in `sample/example-event/`. Then plan the ML step in `pipeline/ml.py`. |
| **SW** — dashboard | Create the React app in `frontend/` and display `sample/example-event/` on a Leaflet map using `meta.json` bounds. |
| **G1** — data & event | Find the two granule names in [ASF Vertex](https://search.asf.alaska.edu/) and put them in `data/granules.txt`. Start `docs/research/event.md`. |
| **G2** — research & pitch | Start `docs/research/method.md`: how radar sees water, HH vs. HV, limitations. Collect sources. |

The function signatures in `pipeline/` are the agreed contract — see [docs/CONTRACT.md](docs/CONTRACT.md). Fill in the bodies; ask before changing a signature.

## Repository layout

```
data/granules.txt     granule list (committed)
data/metadata/        XML + QA CSV (committed, small)
data/reference/       reference flood / water maps (small, committed)
data/raw/             downloaded .h5 (git-ignored)
pipeline/             processing package: io, preprocess, detect, evaluate, ml, export
scripts/              download_data.py, run_pipeline.py, make_sample.py
sample/               fake outputs in the contract format, for frontend work
frontend/             React + Leaflet dashboard (SW creates it)
docs/                 WORKFLOW.md, CONTRACT.md, reports, research
outputs/              real pipeline results (git-ignored)
```

## Downloading from ASF

1. Create a free account at [Earthdata Login](https://urs.earthdata.nasa.gov/).
2. Set credentials in your shell (never commit them). PowerShell:
   ```powershell
   $env:EARTHDATA_TOKEN = "<your token>"
   ```
   or `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, or a `_netrc` entry for `urs.earthdata.nasa.gov`.
3. Check the granules exist, then download into `data/raw/`:
   ```bash
   python scripts/download_data.py --dry-run
   python scripts/download_data.py --all-files
   ```

To add a granule: find it in [ASF Vertex](https://search.asf.alaska.edu/) (Dataset: NISAR, product GCOV), copy the granule name, and add a line `label | granule_name` to `data/granules.txt`.

## Rules (short version)

- Never commit `.h5` files, `.env` files, tokens or passwords.
- Never modify the original `.h5` files — open them read-only.
- Work on your own branch, open small pull requests, keep `main` working.
