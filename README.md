# OrbitWatch

NASA Space Apps Challenge 2026 — *Dancing with the SARs*. Surface-change and flood detection from NISAR L-band SAR (L2 GCOV).

## Data

The NISAR `.h5` files are several GB each, so they are **not stored in git** (`.gitignore` blocks `*.h5`). The repo only records which granules we use, in [`data/granules.txt`](data/granules.txt). Small files (XML metadata, QA CSVs) can be committed under `data/metadata/`.

### Get the data

1. Create a free account at [Earthdata Login](https://urs.earthdata.nasa.gov/).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set credentials in your shell (never commit them). PowerShell:
   ```powershell
   $env:EARTHDATA_TOKEN = "<your token>"
   ```
   or `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, or a `_netrc` entry for `urs.earthdata.nasa.gov`.
4. Check the granules exist, then download into `data/raw/`:
   ```bash
   python scripts/download_data.py --dry-run
   python scripts/download_data.py --all-files
   ```

### Adding a granule

Find it in [ASF Vertex](https://search.asf.alaska.edu/) (Dataset: NISAR, product GCOV), copy the granule name, and add a line `label | granule_name` to `data/granules.txt`.

## Layout

```
data/granules.txt     granule list (committed)
data/metadata/        XML + QA CSV (committed, small)
data/raw/             downloaded .h5 (ignored)
scripts/              download + processing scripts
outputs/              generated previews/reports (ignored)
```
