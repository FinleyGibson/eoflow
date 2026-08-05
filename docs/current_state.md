# EOFlow — Current State (Handover)

_Last updated: 2026-07-29_

## 1. Overview

EOFlow models water-quality determinands (currently turbidity) at Environment Agency
sampling sites in Devon from Earth Observation, terrain, soil, and rainfall data. For each
water-quality observation, the pipeline delineates the upstream catchment, computes a set
of spatial layers over that catchment, reduces them to scalar features, and fits a
regression model predicting the measured determinand value.

Core library code: `src/eoflow/`. CLI entry points for every pipeline stage: `scripts/`.
Interactive/exploratory counterparts and one-off analyses: `notebooks/`. The end-to-end
sequence and exact commands are documented in `docs/PIPELINE.md` — this file is a higher-level
map of what exists and why, not a replacement for it.

## 2. The Core Pipeline

### 2.1 Data collation

Data comes from independent sources, each with its own client module and a query-by-geometry
pattern (point or polygon in, matching data out):

| Type                                       | Source                                                        | Module                                       | Notes                                                                                                                                                        |
| ------------------------------------------ | ------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Water-quality observations (target)        | Environment Agency Water Quality API                          | `eoflow/ea.py`                               | Official monitoring; paginated API client, determinand + area filtering                                                                                      |
| Water-quality observations (supplementary) | Citizen-scientist ArcGIS FeatureServer                        | `eoflow/cs.py`                               | Community monitoring data; same polygon-query shape as EA. Not yet merged with EA data in the main pipeline (source-annotated combination is a listed to-do) |
| Terrain (static)                           | DEM GeoTIFF (OpenTopography)                                  | `eoflow/topography.py`, `scripts/get_dem.py` | Elevation, slope, aspect rasters                                                                                                                             |
| Soil (static)                              | DEFRA/SEARG 12 soil-structure groups (ArcGIS Feature Service) | `eoflow/soil.py`                             | Fractional catchment coverage per group + BFI/SPR (hydrological soil properties)                                                                             |
| Rainfall (dynamic)                         | NIMROD 1 km composite radar (CEDA archive)                    | `eoflow/rainfall.py`                         | Used for training data — see §5 for current coverage                                                                                                         |
| Earth Observation indices (dynamic)        | Sentinel-2 L2A via openEO                                     | `eoflow/eo.py`                               | Currently NDVI/NDWI, plus raw band cubes for computing further indices; non-interactive OIDC auth supported for headless use                                 |

"Static" layers (terrain, soil) are computed once per catchment geometry; "dynamic" layers
(rainfall, EO indices) are computed per-observation over a date window ending at that
observation's sample date (`--rainfall-days` / `--eo-days` control the window length).

### 2.2 Dataset building — catchment delineation

`eoflow/catchment.py` wraps `pysheds` for D8 watershed delineation: fills depressions,
computes flow direction/accumulation, snaps each sample's coordinates to the nearest stream
cell (`--flow-acc-threshold` controls how aggressively), and extracts the upstream catchment
polygon. `scripts/delineate_catchments.py` runs this across every unique lat/lon in a
water-quality CSV, caching per-location results and checkpointing so a batch run can be
interrupted and resumed. Output is a GeoPackage of catchment polygons + snap metadata
(`delineate_catchment.py` is the single-site equivalent for inspection/debugging).

### 2.3 Sample assembly

`eoflow/samples.py` defines the objects the rest of the pipeline is built around:

- **`Sample`** — one water-quality observation plus (optionally) its delineated catchment
  polygon. Exposes `compute_layers()` (terrain + soil + rainfall in one call) and
  `fetch_ndvi()`/`fetch_ndwi()` (openEO), then `save()`/`load()` to/from disk.
- **`CatchmentLayers`** — container for a sample's computed rasters/tables: `topography`,
  `slope`, `aspect`, `soil_type`, `soil_polygons`, `rainfall`, `ndvi`, `ndwi`, `eo_bands`.
- **`CatchmentDataset`** — loads every row of a delineated-catchments GeoPackage into a list
  of `Sample` objects (`CatchmentDataset.from_gpkg(..., only_delineated=True)`).

`scripts/prepare_samples.py` iterates a `CatchmentDataset`, calls `compute_layers` +
`fetch_ndvi`/`fetch_ndwi` per sample, and saves each fully-populated sample to
`data/sample_instances/<tag>/`. It's resumable (skips a sample if its output already exists)
and saves incrementally, so a crash partway through loses nothing already written.

**Bugs fixed in this repo this session, worth knowing about if you see old data, old commits,
or older revisions of `docs/PIPELINE.md` referencing them:**

1. **Tagging collision.** The output tag was originally just `notation` (the _site_ ID), but
   the same site is visited many times on different dates — so the script kept only the
   first-encountered date per site and silently discarded every repeat visit as "already
   exists". Fixed by tagging `<notation>_<sample_id>` instead, using the per-observation ID
   embedded in the GPKG's `id` URL column. If you find sample directories named with a bare
   notation (no trailing `_<digits>`) from before this fix, they only represent one arbitrary
   date for that site. `docs/PIPELINE.md` step 8 has been updated to match.
2. **Memory leak on long batch runs.** `CatchmentDataset` keeps every `Sample` object (and
   its computed layers — some EO cubes are several hundred MB) alive for the script's entire
   run. On a 1000-row run this eventually exhausts RAM regardless of any individual sample's
   size (confirmed via an OOM-kill at sample 642/1000 on a 15 GB, no-swap box). Fixed by
   resetting `sample.layers` to empty immediately after each save.
3. **NIMROD download/process handoff didn't actually work as `PIPELINE.md`'s own commands
   implied.** `nimrod_download_script.sh` uses `wget --mirror`, which nests the downloaded
   tars several directories deep rather than flat in the target directory — but
   `process_nimrod_local`'s tar discovery (`eoflow/rainfall.py`) only globbed the top level of
   `--input`, so pointing it straight at the download directory (as `PIPELINE.md`'s own
   example showed) silently found zero files. Fixed by searching recursively (`rglob`
   instead of `glob`). Separately, the two convenience wrapper scripts
   (`download_all_nimrod.sh` / `process_all_nimrod.sh`) didn't agree with each other on the
   per-year directory naming (`data/nimrod_<year>` vs. `data/nimrod_raw/nimrod_<year>`),
   and `process_all_nimrod.sh`'s default output directory was one level too shallow
   (`data/nimrod_processed/<year>` instead of `data/nimrod_processed/raw/<year>`) — both
   fixed to agree with each other and with `PIPELINE.md`'s documented convention
   (`data/nimrod_raw/<year>` → `data/nimrod_processed/raw/<year>`).

### 2.4 Feature extraction

`eoflow/features.py` (`scripts/extract_features.py`) reduces each saved sample's layers to
~49 scalar, catchment-level features, grouped as:

- **Geometry** — area, perimeter, Polsby–Popper compactness
- **Terrain** — elevation mean/range/std, slope mean/std/p90, fraction steeper than two
  configurable thresholds (default 10°/15°), aspect northness/eastness
- **Earth Observation** — currently two Sentinel-2 indices, computed and grouped as a single
  extensible category (`_ndvi_features`/`_ndwi_features` in `eoflow/features.py`; adding
  another index means adding one more `_<index>_features` function alongside these two, same
  pattern):
    - _NDVI_ (vegetation) — mean, std, p10, temporal amplitude, bare-fraction (<0.20) and
      dense-fraction (>0.60)
    - _NDWI_ (surface water/moisture) — mean, wet-fraction (>0.00)
- **Soil** — fractional coverage of each of the 12 SEARG groups, area-weighted BFI/SPR, Shannon
  entropy (soil-mosaic diversity), and an "erodible" fraction (sum of four structurally
  erodible groups)
- **Rainfall** — mean/max intensity, total depth, spatial coefficient of variation
- **Interaction terms** — hand-designed products hypothesised to matter for
  turbidity/erosion (e.g. `rainfall × bare-fraction`, `bare-fraction × steep-slope-fraction`)

Output is a CSV (optionally Parquet). `scripts/feature_report.py` gives summary stats and a
per-group data-completeness chart for a given extract.

### 2.5 Modelling — a hot-swappable model interface

`eoflow/models/base.py` defines `BaseModel`, an abstract interface every model implements:
`fit(X, y)`, `predict(X)`, `save(path)`, `load(path)` (abstract), plus `score(X, y)` →
`{r2, rmse, mae}` and `is_fitted` provided for free. `split_features_target(df)` is the shared
helper that turns a features CSV into `(X, y)` by dropping the identity columns
(`notation`, `site_name`, `date`, `result`, `unit`) and keeping every numeric predictor.

Because every model consumes the same `(X, y)` shape and exposes the same four methods, a new
modelling approach (e.g. the Graph Gaussian Process or GNN discussed in `todo.md`) is a new
`BaseModel` subclass — nothing upstream (feature extraction, sample assembly, data collation)
needs to change to swap it in.

Currently one concrete implementation exists: **`LinearRegression`**
(`eoflow/models/LinearRegression.py`) — a thin `sklearn` OLS wrapper that drops non-numeric
columns and NaN rows on `fit` (but not on `predict`/`score` — callers are expected to have
already dropped incomplete rows, since a model can't sensibly predict from an incomplete
feature row). Demonstrated end-to-end in
`notebooks/model_analysis/linear_regression_turbidity.ipynb`: load a features CSV → drop
incomplete rows → train/test split → fit → holdout accuracy → leave-one-out CV (more stable
than a single split on a small sample) → coefficient inspection → a reference table
explaining what each of the largest-magnitude coefficients actually measures and how it's
calculated.

## 3. Extras

### 3.1 River network graph extraction

`eoflow/rivers.py` queries OpenStreetMap's Overpass API for waterway `LineString`s within an
area of interest (with retry/fallback across multiple Overpass endpoints), then builds a
topological `networkx` graph — nodes at intersections and endpoints, edges as river segments
with lengths (Haversine for geographic CRS). `notebooks/script_development/graph_with_nodes.ipynb`
and `river_graph_building.ipynb` cover this. Per `todo.md`, this exists to eventually support
graph-aware modelling (flow-distance/directional weighting between a sample and upstream
influences, rather than plain Euclidean distance) — a Graph Gaussian Process is the specific
approach named there. **Not yet wired into the main delineation → features → model pipeline.**
An earlier HydroSHEDS-based alternative and its exploratory notebooks were dropped from this
branch as unused (OSM is the source actually used).

### 3.2 Live/current Met Office rainfall

An earlier Met Office UKV 2 km rainfall pipeline (`download_rainfall.py`, `convert_rainfall.py`,
`get_rainfall_for_polygon`) was removed from this branch and preserved on the
`met-office-rainfall-legacy` branch. NIMROD (1 km, via CEDA) is used for training data
instead because it has much greater historical depth — this session extended that archive
back from 2020 to 2016 specifically for this reason (§5). The Met Office product would be the
one to reach for if genuinely up-to-the-hour current rainfall is ever needed (e.g. a live
prediction/monitoring mode), which is presumably why it's kept on a legacy branch rather than
deleted outright.

## 4. Supporting infrastructure

- **API** (`eoflow/api.py`, `scripts/run_api.py`, `docs/API.md`, `docs/API_QUICKSTART.md`) — a
  FastAPI service exposing EA + citizen-scientist water-quality samples by point or polygon.
  Run with `uvicorn eoflow.api:app`.
- **Config** (`eoflow/config.py`) — JSON config at a platform-specific directory
  (`platformdirs`), auto-created with defaults on first use.
- **Credentials** (`.env` in repo root): `CEDA_TOKEN` (NIMROD — short-lived, expect to refresh
  every few days on a multi-day run), `OPENTOPOGRAPHY_API_KEY` (DEM), `OPENEO_AUTH_CLIENT_ID`
  / `OPENEO_AUTH_CLIENT_SECRET` (non-interactive Sentinel-2 auth — without these, openEO falls
  back to an interactive browser device-flow prompt).

## 5. Data currently on disk

| Path                                                                         | Contents                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/dems/`                                                                 | Devon DEMs (SRTM + Copernicus 30 m) and a UK-wide DEM                                                                                                                                                                                                                                                                                                                                                                                   |
| `data/shapefiles/devon_county/`                                              | Study-area boundary used for NIMROD cropping and EA filtering                                                                                                                                                                                                                                                                                                                                                                           |
| `data/ea_water_quality/`                                                     | Raw + cleaned EA turbidity CSVs, 2010–2026 and 2020–2026 windows                                                                                                                                                                                                                                                                                                                                                                        |
| `data/nimrod_processed/concatenated/{2016..2024}/`                           | Per-day consolidated NIMROD rainfall, **3,226 day-files, 49 G**, continuous 2016–2024. Known gaps: 2022 stops entirely at Dec 14 (confirmed via CEDA's own directory listing — genuinely absent from their archive, not recoverable) and a handful of individually-corrupted radar frames scattered across other years (well under 1%). Per-timestep raw data is _not_ kept once a year is consolidated — see the retention note below. |
| `data/delineated_catchments/devon_turbidity_sites_2010-10_2026-06_1000.gpkg` | 1000 EA turbidity observations across 154 unique sites, dates 2010-01-05 to 2026-05-22, all successfully delineated                                                                                                                                                                                                                                                                                                                     |
| `data/sample_instances/`                                                     | Prepared `Sample`s (terrain/soil/rainfall/EO layers) — **650/1000 done as of writing, run still in progress** (see §6)                                                                                                                                                                                                                                                                                                                  |
| `data/eo_cubes/`                                                             | EO cube cache (currently empty)                                                                                                                                                                                                                                                                                                                                                                                                         |
| `outputs/catchment_features_devon_turbidity_preview.csv`                     | 173-row snapshot of extracted features, taken mid-run — stale relative to the in-progress `sample_instances` count above                                                                                                                                                                                                                                                                                                                |
| `outputs/catchment_features_2020_sample5.csv`                                | Original 5-row demo dataset the notebook shipped with                                                                                                                                                                                                                                                                                                                                                                                   |

**Retention policy** (adopted mid-backfill, not the original design): once a year's NIMROD
per-timestep raw data is confirmed fully represented in its per-day consolidated output, the
raw copy is deleted — keeping both was not sustainable on this box's disk. The same applies
to downloaded source tars once processed. If re-deriving per-timestep data is ever needed,
it must be re-downloaded and re-processed from CEDA; only the consolidated per-day files are
kept long-term.

## 6. In-flight work / open issues

- **`prepare_samples.py` full 1000-observation run** is in progress in a detached `screen`
  session (`screen -r prepare_samples` to attach), log at `logs/prepare_samples_1000.log`.
  650/1000 done, zero hard failures. Next steps once it finishes: re-run
  `scripts/extract_features.py` against the complete `data/sample_instances/`, then re-execute
  `linear_regression_turbidity.ipynb` against that full CSV.
- **Data coverage ceiling.** NIMROD only covers 2016–2024 and Sentinel-2 only ~2015+, but the
  1000-observation GPKG spans 2010–2026. Roughly half the observations will end up with
  incomplete rainfall/EO features for reasons that can't be fixed short of finding another
  data source — this is expected, not a bug, and the model-fitting code already drops
  incomplete rows.
- **Model accuracy is currently poor/overfit** on the partial data seen so far (train R²≈1.0,
  test/leave-one-out R² strongly negative) — expected with 49 features against only a few
  dozen complete rows; should improve once the full run's completeness is available, but 49
  features is still a lot relative to the ~300–450 complete rows this dataset can plausibly
  yield, so feature selection/regularisation is worth considering before trusting this model.
- **No swap configured** on this box (15 GB RAM total) — large unattended batch jobs need
  active memory management; see the fix in §2.3.

## Current Issues

### 1. Delineated catchments are not properly nested

Catchment delineation (`eoflow/catchment.py`, §2.2) is based purely on theoretical DEM-derived
D8 flow-accumulation lines. This produces a real topological problem: for two sample sites on
the same watercourse where one is upstream of the other, the upstream site's catchment should
always be fully contained within (subsumed by) the downstream site's catchment — but in
practice it often isn't. Confirmed visually via `scripts/visualise_catchments.py`'s output
(see a worked example at `outputs/devon_turbidity_unique_catchments.html`, generated on a local
machine, not this one).

Two suspected causes, not yet distinguished:

- **`--flow-acc-threshold` miscalibration.** This parameter controls how aggressively a sample's
  coordinates get snapped to the nearest stream cell (see §2.2); if it's snapping different
  nearby sites onto different, non-nested flow paths, that alone could explain the problem.
- **Theoretical flow lines diverging from real waterway geometry.** DEM-derived D8 flow
  accumulation is an approximation — it can disagree with where the actual river channel is,
  especially on flat terrain, floodplains, or where DEM resolution can't resolve a narrow
  channel. Augmenting/constraining the flow-accumulation-derived network with real waterway
  geometry (the OSM extraction in `eoflow/rivers.py`, §3.1 — currently unused by the
  delineation step) is one plausible fix, snapping pour points to actual mapped rivers rather
  than a purely theoretical flow line.

This matters beyond a visual/QA nicety: every terrain, soil, rainfall, and EO feature in
§2.4 is aggregated over the catchment polygon, so an incorrectly-shaped or wrongly-nested
catchment directly biases every feature computed for that sample — not just the geometry ones.
Not yet root-caused; worth investigating before trusting model results built on the current
`data/delineated_catchments/*.gpkg` files.

N.B. Part of the pipeline which builds the dataset will need re-running once this issue has been resolved.
