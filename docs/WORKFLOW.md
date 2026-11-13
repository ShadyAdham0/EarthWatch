# EarthWatch — Team Workflow

How the six of us work in parallel without waiting on each other, and how the pieces come together into one demo.

> **TL;DR**
> 1. Everyone owns separate files and works on their own branch.
> 2. Parts connect only through agreed file formats (the **contract**, section 4), not through each other's code.
> 3. The dashboard is built against sample files from hour 1; real outputs replace them later.
> 4. Small pull requests into `main`, merged often. `main` must always run.
> 5. Never commit `.h5` files, credentials or tokens.

---

## 1. The pipeline at a glance

```
 M1 data ──► M3 preprocessing ──► M4 detection ──► outputs/<event>/ ──► M5 dashboard
   (.h5)       (dB arrays,          (flood mask,      (PNGs + meta.json)     (React map)
               masks, aligned)       stats, ML)
                                         ▲
 M2 event research + reference map ──────┘ (used to evaluate accuracy)

 M6 glues it together: run_pipeline.py, merges PRs, tests, presentation
```

Each arrow is a **handoff**. As long as the handoff format is respected, each person can build and test their part on their own.

## 2. Roles and what each person delivers

| Member | Area | Owns these files | Delivers |
|---|---|---|---|
| **M1** | Data acquisition | `data/granules.txt`, `data/metadata/`, `scripts/download_data.py` | Before/after NISAR GCOV files, downloadable by anyone |
| **M2** | Event & reference | `data/reference/`, `docs/research/` | Event details, reference flood/water map, cited sources |
| **M3** | SAR preprocessing | `pipeline/io.py`, `pipeline/preprocess.py` | Aligned, masked dB arrays for both dates |
| **M4** | Detection & ML | `pipeline/detect.py`, `pipeline/export.py` | Flood mask, affected area, accuracy vs. reference |
| **M5** | Dashboard | `frontend/` | Working map with before/after, flood layer, stats |
| **M6** | Integration & pitch | `scripts/run_pipeline.py`, `sample/`, `docs/`, merges | One-command pipeline, final demo, slides |

Roles can shift depending on strengths. If you finish early, help the person whose output you depend on.

## 3. Repository structure

```
data/
  granules.txt          M1  NISAR granule names we use
  metadata/             M1  XML + QA CSV files (small, committed)
  reference/            M2  reference flood map / permanent-water mask
  raw/                  git-ignored — downloaded .h5 files live here
pipeline/               Python package — the processing chain
  io.py                 M3  read GCOV .h5 → numpy arrays + georeferencing
  preprocess.py         M3  dB conversion, invalid-pixel masks, crop, alignment
  detect.py             M4  baseline threshold + ML step
  export.py             M4  write outputs in the contract format
scripts/
  download_data.py      M1  download granules from ASF (exists)
  run_pipeline.py       M6  run the whole chain with one command
frontend/               M5  React + Leaflet dashboard
sample/                 M6  small fake outputs for frontend development
docs/
  WORKFLOW.md           this file
  CONTRACT.md           exact output formats (summary in section 4)
  phase1_report.md      Phase 1 findings
  research/             M2  research notes and sources
notebooks/              personal exploration: <name>_<topic>.ipynb
outputs/                git-ignored — real pipeline results
```

Folders that don't exist yet get created by their owner in their first pull request.

## 4. The contract — how parts connect

### 4.1 Preprocessing → Detection (M3 → M4)

`pipeline/preprocess.py` exposes a function that returns, **for each date, on the same pixel grid**:

| Item | Type | Meaning |
|---|---|---|
| `hh_db` | 2-D float32 array | HHHH backscatter in dB |
| `hv_db` | 2-D float32 array | HVHV backscatter in dB |
| `valid` | 2-D bool array | `True` where the pixel is usable (not fill/NaN/shadow) |
| `transform`, `crs` | georeferencing | Same for both dates once aligned |
| `date` | string `YYYY-MM-DD` | Acquisition date |

M4 can start immediately using small synthetic arrays of this shape.

### 4.2 Detection → Dashboard (M4 → M5)

The pipeline writes one folder per event:

```
outputs/<event_id>/
  before_hh.png      greyscale backscatter, before date
  after_hh.png       greyscale backscatter, after date
  change.png         change map (colour-coded)
  flood_mask.png     transparent PNG, flooded pixels coloured
  meta.json          everything the dashboard needs to place and describe the layers
```

`meta.json` example:

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
  "notes": "Free text: limitations, caveats"
}
```

`bounds` is `[[south, west], [north, east]]` in latitude/longitude, which is what Leaflet's image overlays expect. All PNGs share those bounds.

**Rule:** if you need to change the contract, open a PR that edits `docs/CONTRACT.md` and tell the people on both sides of that arrow first.

## 5. Git workflow

### First-time setup

```bash
git clone https://github.com/ShadyAdham0/EarthWatch.git
cd EarthWatch
pip install -r requirements.txt
```

### Every piece of work

1. Get the latest `main`:
   ```bash
   git switch main
   git pull
   ```
2. Create your branch, named `<member>/<topic>`:
   ```bash
   git switch -c m3/preprocess
   ```
3. Work, then commit small, meaningful steps:
   ```bash
   git add pipeline/preprocess.py
   git commit -m "Add dB conversion and invalid-pixel mask"
   ```
4. Push and open a pull request into `main`:
   ```bash
   git push -u origin m3/preprocess
   ```
   Then open the PR on GitHub. Describe what it does and how you tested it.
5. M6 (or another teammate) reviews and merges. Delete the branch after merging.
6. If `main` moved while you worked, update your branch before the PR:
   ```bash
   git switch main
   git pull
   git switch m3/preprocess
   git merge main
   ```

### Rules

- **Small PRs, merged often.** Aim for at least one merge every few hours. Big PRs at the end of the day cause conflicts.
- **Stay in your own files.** If you need to change someone else's file, tell them first.
- **`main` always works.** Don't merge code that breaks the pipeline or the dashboard.
- **Use `git add <file>`**, not `git add .`, so you don't commit data or junk by accident.

## 6. Data rules

- **Never commit `.h5` or other large data files.** GitHub rejects files over 100 MB. `.gitignore` blocks them, but double-check before pushing.
- **Never commit credentials.** Earthdata tokens and passwords go in environment variables only. See the README.
- **Never modify the original `.h5` files.** Open them read-only. Write results to `outputs/`.
- Raw data goes in `data/raw/`. For now the team's data is shared as a zip on Google Drive; once granule names are in `data/granules.txt`, anyone can run `python scripts/download_data.py`.
- Small files (XML metadata, QA CSVs, reference maps under a few MB) can be committed.

## 7. Code and notebook guidelines

- **Notebooks are personal.** Name them `<name>_<topic>.ipynb` and don't edit anyone else's. Notebooks merge badly in git.
- When something in a notebook works, move it into a function in `pipeline/` so others can use it.
- Use clear names and a short docstring on each function: what goes in, what comes out, units (dB vs. linear!).
- Keep paths relative to the repo root. Don't hard-code `C:\Users\...`.

## 8. Timeline and sync points

| When | Milestone | Who |
|---|---|---|
| Before kickoff | Data downloaded, opens in Python; this workflow agreed | M1, M3, all |
| Day 1, hour 1 | `sample/` outputs committed so M5 can start | M6 |
| Day 1, hours 0–3 | Event confirmed, reference map found; granules verified | M1, M2 |
| Day 1, hours 2–5 | Preprocessing returns aligned dB arrays | M3 |
| Day 1, hours 5–8 | Baseline flood mask + area statistics | M4 |
| **End of Day 1** | **Sync:** first real `outputs/<event>/` produced end to end | All |
| Day 2, hours 0–3 | Dashboard shows real outputs; ML step added | M4, M5 |
| Day 2, hours 3–5 | Accuracy vs. reference in dashboard | M4, M5, M2 |
| Day 2, hours 5–7 | Testing, fixes, polish; demo rehearsal | All, led by M6 |
| Final hour | Presentation: problem, data, method, results, limitations, live demo | M6 + all |

**Short stand-ups** at the start of each block: each person says what they finished, what they're doing next, and what they're blocked on (2 minutes each, max).

## 9. Definition of done

A part is done when:

- It's merged into `main`.
- Someone else can run it from a fresh clone by following the README.
- Its output matches the contract.
- Limitations and assumptions are written down (in the PR description or `docs/`).

## 10. Scientific honesty

- Don't claim a flood is detected without evidence. A change in backscatter is not automatically a flood.
- Separate **confirmed findings** from **assumptions** in every report.
- If something can't be verified, say what is missing.
- Cite sources for every claim about the event, the data or the method.

## 11. Help and questions

- Blocked for more than 20 minutes? Ask in the team chat. Don't wait silently.
- Unsure where something belongs? Ask M6.
- Found a data or security problem (e.g., a credential committed by mistake)? Tell the team immediately.
