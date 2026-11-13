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
 G1 data + event ──► AI1 preprocessing ──► AI2 baseline detection ──► AI3 ML + export ──► outputs/<event>/ ──► SW dashboard
   (.h5, reference)    (dB arrays, masks,      (threshold mask,            (ML mask, PNGs,       (PNGs + meta.json)     (React map)
                        aligned grids)          evaluation)                 run_pipeline.py)

 G2 research, testing and presentation ── works alongside everyone
```

Each arrow is a **handoff**. As long as the handoff format is respected, each person can build and test their part on their own.

## 2. Roles — pick yours

There are six roles: three AI, one software and two general. **Nobody is assigned yet — each person picks the role they want.**

**How to pick:**
1. Read the roles below and choose the one that fits you best.
2. Claim it in the team chat ("I'm taking AI2"). First come, first served.
3. Write your name in the **Taken by** column (edit this file on GitHub, or ask SW to do it).
4. If two people want the same role, talk it through; whoever doesn't get it picks from what's left.
5. Not sure? Pick a **G** role. They need no heavy coding and matter just as much for the judging.

| ID | Suggested background | Area | Owns these files | Delivers | Taken by |
|---|---|---|---|---|---|
| **AI1** | AI / data | SAR preprocessing | `pipeline/io.py`, `pipeline/preprocess.py` | Aligned, masked dB arrays for both dates; HDF5 inspection and alignment check | _open_ |
| **AI2** | AI / data | Baseline detection & evaluation | `pipeline/detect.py` (baseline), `pipeline/evaluate.py` | Threshold flood mask, affected area, accuracy (IoU, F1) vs. reference | _open_ |
| **AI3** | AI / ML | ML model & pipeline output | `pipeline/ml.py`, `pipeline/export.py`, `scripts/run_pipeline.py` | ML flood mask compared with baseline; outputs in contract format; one-command pipeline | _open_ |
| **SW** | Software | Dashboard & repo | `frontend/`, `sample/`; reviews and merges PRs | Working map with before/after, flood layer, stats; `main` always runnable | _open_ |
| **G1** | General | Data & event | `data/granules.txt`, `data/metadata/`, `data/reference/`, `docs/research/event.md` | Granules found in ASF Vertex and downloaded; event verified with sources; reference map | _open_ |
| **G2** | General | Research, QA, testing & pitch | `docs/research/method.md`, `docs/phase1_report.md`, slides | Flood-detection science write-up; Phase 1 report; app testing; presentation and demo script | _open_ |

**Why these roles:** the three AI members cover the three processing stages in order, so each has one clear input and one clear output. The software member owns everything the user sees and keeps `main` working. The two general members cover the work that needs no heavy coding but is just as important for judging: real data, a verified event, sound science and a clear pitch.

### Pairs (who helps whom)

- **G1 ↔ AI1:** G1 finds and downloads the data; AI1 confirms it opens and is usable. G1 documents the QA findings AI1 uncovers.
- **G2 ↔ AI2:** G2 researches how radar sees water and what limits apply; AI2 turns that into the baseline method and evaluation.
- **G2 ↔ SW:** G2 tests the dashboard as a first-time user and writes its explanatory text.
- **AI3 ↔ SW:** AI3's `export.py` must produce exactly what the dashboard reads. Agree on `meta.json` in the first hour.

### Phase 1 tasks → owners

| Phase 1 task | Owner | Support |
|---|---|---|
| 1. Inspect HDF5 structure | AI1 | G1 |
| 2. Investigate QA failure | AI1 | G1 writes it up |
| 3. Visualize HH/HV previews | AI2 | — |
| 4. Verify spatial alignment | AI1 | — |
| 5. Flood-detection feasibility research | G2 | AI2 |
| 6. Event & temporal coverage | G1 | G2 |
| Phase 1 final report | G2 | everyone sends findings |

Roles can shift depending on strengths. If you finish early, help the person whose output you depend on.

## 3. Repository structure

```
data/
  granules.txt          G1   NISAR granule names we use
  metadata/             G1   XML + QA CSV files (small, committed)
  reference/            G1   reference flood map / permanent-water mask
  raw/                  git-ignored — downloaded .h5 files live here
pipeline/               Python package — the processing chain
  io.py                 AI1  read GCOV .h5 → numpy arrays + georeferencing
  preprocess.py         AI1  dB conversion, invalid-pixel masks, crop, alignment
  detect.py             AI2  baseline threshold (log-ratio + Otsu)
  evaluate.py           AI2  area statistics, IoU / F1 vs. reference
  ml.py                 AI3  ML step (clustering / random forest)
  export.py             AI3  write outputs in the contract format
scripts/
  download_data.py      G1   download granules from ASF (exists)
  run_pipeline.py       AI3  run the whole chain with one command
frontend/               SW   React + Leaflet dashboard
sample/                 SW   small fake outputs for frontend development
docs/
  WORKFLOW.md           this file
  CONTRACT.md           exact output formats (summary in section 4)
  phase1_report.md      G2   Phase 1 findings
  research/             G1, G2  event research, method research, sources
notebooks/              personal exploration: <name>_<topic>.ipynb
outputs/                git-ignored — real pipeline results
```

`pipeline/` already contains stub functions with the agreed signatures (owners fill in the bodies), and `sample/example-event/` contains fake outputs so the dashboard can start right away. `frontend/` and `data/reference/` are created by their owners in their first pull request.

## 4. The contract — how parts connect

### 4.1 Preprocessing → Detection (AI1 → AI2, AI3)

`pipeline/preprocess.py` exposes a function that returns, **for each date, on the same pixel grid**:

| Item | Type | Meaning |
|---|---|---|
| `hh_db` | 2-D float32 array | HHHH backscatter in dB |
| `hv_db` | 2-D float32 array | HVHV backscatter in dB |
| `valid` | 2-D bool array | `True` where the pixel is usable (not fill/NaN/shadow) |
| `transform`, `crs` | georeferencing | Same for both dates once aligned |
| `date` | string `YYYY-MM-DD` | Acquisition date |

AI2 and AI3 can start immediately using small synthetic arrays of this shape.

### 4.2 Detection → Dashboard (AI3 → SW)

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
2. Create your branch, named `<your-id>/<topic>` (e.g. `ai1/preprocess`, `sw/map-view`, `g1/event-research`):
   ```bash
   git switch -c ai1/preprocess
   ```
3. Work, then commit small, meaningful steps:
   ```bash
   git add pipeline/preprocess.py
   git commit -m "Add dB conversion and invalid-pixel mask"
   ```
4. Push and open a pull request into `main`:
   ```bash
   git push -u origin ai1/preprocess
   ```
   Then open the PR on GitHub. Describe what it does and how you tested it.
5. SW reviews and merges (for SW's own PRs, any AI member reviews). Delete the branch after merging.
6. If `main` moved while you worked, update your branch before the PR:
   ```bash
   git switch main
   git pull
   git switch ai1/preprocess
   git merge main
   ```

**New to git?** (G1, G2) Docs and data files can also be added straight on GitHub: open the folder, click **Add file → Upload files** or edit a file with the pencil icon, choose **"Create a new branch… and start a pull request"**, and submit. SW will merge it.

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
| Before kickoff | Data downloaded and opens in Python; this workflow agreed | G1, AI1, all |
| Day 1, hour 1 | `sample/` outputs committed; `meta.json` agreed | SW, AI3 |
| Day 1, hours 0–3 | Event confirmed, reference map found; Phase 1 checks done | G1, AI1, G2 |
| Day 1, hours 0–3 | Dashboard skeleton runs on sample data | SW |
| Day 1, hours 2–5 | Preprocessing returns aligned dB arrays | AI1 |
| Day 1, hours 2–5 | ML approach chosen and prototyped on synthetic data | AI3 |
| Day 1, hours 5–8 | Baseline flood mask + area statistics + evaluation | AI2 |
| **End of Day 1** | **Sync:** first real `outputs/<event>/` produced end to end | All |
| Day 2, hours 0–3 | ML step on real data, compared with baseline; dashboard shows real outputs | AI3, AI2, SW |
| Day 2, hours 3–5 | Accuracy vs. reference shown in dashboard; method text written | AI2, SW, G2 |
| Day 2, hours 5–7 | Testing, fixes, polish; demo rehearsal | All, testing led by G2 |
| Final hour | Presentation: problem, data, method, results, limitations, live demo | G2 presents with all |

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
- Unsure where something belongs or how to use git? Ask SW.
- Found a data or security problem (e.g., a credential committed by mistake)? Tell the team immediately.
