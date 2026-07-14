# Data Pipeline Checklist

## Prerequisites

- [x] Obtain an OpenTopography API key and set `OPENTOPOGRAPHY_API_KEY`
- [x] Obtain a CEDA account token and set `CEDA_TOKEN`
- [x] Create a Copernicus Data Space account (required if using EO features)
- [x] Create a `.env` file in the project root containing:

```env
OPENTOPOGRAPHY_API_KEY=your_key_here
CEDA_TOKEN=your_token_here
```

- [x] Obtain a study-area shapefile (or use the provided East Devon shapefile)

### DEM Preparation

- [x] Download the DEM:

```bash
python -m scripts.get_dem --country England --output data/dems/dem.tif
```

- [x] (Optional) Verify DEM coverage:

```bash
python -m scripts.visualise_dem
```

---

> **Note:** Stream A's CSV steps were originally run on a separate (local) machine; the clean CSV and delineated GeoPackage have since been transferred here (`scp`) and are present at the paths below.

# Stream A — EA Water Quality Samples

- [x] Download EA water quality observations (`get_ea_water_quality.py` or `ge_ea_water_quality_by_shapefile.py`) — `data/ea_water_quality/turbidity_2010-01_2026-06.csv`
- [x] Convert and clean the raw EA CSV (`convert_ea_csv.py`) — `data/ea_water_quality/turbidity_2010-01_2026-06_clean.csv`
- [ ] (Optional) Verify sample locations (`visualise_ea_samples.py`)
- [x] Delineate catchments (`delineate_catchments.py`) — `data/delineated_catchments/devon_turbidity_sites_2010-10_2026-06.gpkg` (4189 rows, all delineated)
- [ ] (Optional) Inspect delineation quality (`visualise_devon_catchments.py`)
- [x] (Optional) `gpkg_report.py` / `downsample_gpkg.py` used to QA and produce smaller working subsets — `..._1000.gpkg` (1000-sample general subset) and `..._2020_sample5.gpkg` (5-sample, 2020-only, used for the Compute Sample Layers test run below)
- [x] Transfer delineated GeoPackage to this machine

---

# Stream B — NIMROD Rainfall Data

- [x] (Optional) Verify CEDA authentication _(implied — downloads below succeeded)_
- [x] Download NIMROD rainfall data — `data/nimrod_processed/raw/` covers 2020–2026
- [x] Process and crop rainfall data to the study area — per-timestep files present in the expected layout
- [ ] (Optional) Visualise rainfall outputs
- [~] (Optional) Consolidate timesteps into per-day NetCDFs, via `scripts/consolidate_nimrod.sh` — **partial, currently stalled**: 2020 (359/366 days), 2021 (365/365), 2022 (327/365), 2023 (355/365), 2024 (238/366, stopped at 2024-08-25). The run isn't currently active — it exited on the `nco_inq_varid()`/`transverse_mercator` schema-mismatch error described in `PIPELINE.md` step 7 and hasn't been rerun; 2025–2026 haven't been started at all. **2020 is complete enough that it was used directly for the Compute Sample Layers test run below.** Rerun with `-f` after investigating the offending 2024-08-25 raw file to continue.

---

# Compute Sample Layers

- [x] Run `prepare_samples.py` — **test run only**: 5 samples (`devon_turbidity_sites_2020_sample5.gpkg`, all 2020, `--nimrod-dir data/nimrod_processed/concatenated/2020 --rainfall-days 10 --eo-days 30`), all 5 saved successfully to `data/sample_instances/`. Not yet run for the full/downsampled dataset.
- [x] Ran with EO included (not skipped) — `.env`'s `OPENEO_AUTH_CLIENT_ID`/`SECRET` allow non-interactive client-credentials auth, so `--skip-eo` wasn't needed even headless
- [x] Fixed a real bug hit during this run: `Sample.save()` called `.to_json()` on a row that still carried a live `geometry` object, which isn't JSON-serialisable and crashed every sample with "Maximum recursion level reached". Fixed in `src/eoflow/samples.py` (`geometry` is dropped before serialising — the polygon is already saved separately as `catchment.wkt`).

> **Note:** the two January 2020 samples' rainfall windows are truncated (10-day lookback runs into December 2019, which predates the NIMROD archive) — no NaNs resulted, but their rainfall stats cover fewer days than the other samples.

---

# Validation and Feature Extraction

- [ ] Run `test_nimrod_rainfall.py`
- [x] Run `extract_features.py` — `outputs/catchment_features_2020_sample5.csv` (5 rows × 54 cols, 0 NaN). `--parquet` was requested but skipped (`pyarrow` not installed)
- [x] (Optional) Run `feature_report.py` (new script, added this session) — `outputs/catchment_features_2020_sample5_completeness.png`
- [ ] (Optional) Run `full_sample_flow.py`

---

# Expected Outputs

- [x] `data/dems/devon_dem_cop30.tif` (and `devon_dem.tif`)
- [x] `data/ea_water_quality/turbidity_2010-01_2026-06_clean.csv`
- [x] `data/delineated_catchments/devon_turbidity_sites_2010-10_2026-06.gpkg`
- [x] `data/nimrod_processed/raw/` (2020–2026)
- [~] `data/nimrod_processed/concatenated/{year}/{YYYYMMDD}.nc` — 2020–2023 mostly done, 2024 partial, 2025–2026 not started (see Stream B note above)
- [x] `data/sample_instances/` — 5 samples (2020 test run)
- [x] Feature CSV — `outputs/catchment_features_2020_sample5.csv`; Parquet not generated (`pyarrow` missing)
