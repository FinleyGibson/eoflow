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

> **Note:** Stream A was completed on a separate (local) machine; the catchment GeoPackage still needs to be transferred (e.g. via `scp`) to this machine before Compute Sample Layers can run here.

# Stream A — EA Water Quality Samples

- [x] Download EA water quality observations (`get_ea_water_quality.py` or `ge_ea_water_quality_by_shapefile.py`)
- [x] Convert and clean the raw EA CSV (`convert_ea_csv.py`)
- [ ] (Optional) Verify sample locations (`visualise_ea_samples.py`)
- [x] Delineate catchments (`delineate_catchments.py`)
- [ ] (Optional) Inspect delineation quality (`visualise_devon_catchments.py`)
- [ ] Transfer `outputs/catchments.gpkg` to this machine

---

# Stream B — NIMROD Rainfall Data

- [x] (Optional) Verify CEDA authentication _(implied — downloads below succeeded)_
- [x] Download NIMROD rainfall data — `data/nimrod_processed/raw/` covers 2020–2026
- [x] Process and crop rainfall data to the study area — per-timestep files present in the expected layout
- [ ] (Optional) Visualise rainfall outputs
- [~] (Optional) Consolidate timesteps into per-day NetCDFs — **in progress**, via `scripts/consolidate_nimrod.sh` (see `PIPELINE.md`); full backfill running as of 2026-07-13

---

# Compute Sample Layers

- [ ] Run `prepare_samples.py`
- [ ] (Optional) Run with `--skip-eo` to omit Sentinel-2 EO processing

---

# Validation and Feature Extraction

- [ ] Run `test_nimrod_rainfall.py`
- [ ] Run `extract_features.py`
- [ ] (Optional) Run `full_sample_flow.py`

---

# Expected Outputs

- [x] `data/dems/dem.tif`
- [x] `data/ea_samples.csv`
- [x] `outputs/catchments.gpkg` _(on local machine — pending transfer)_
- [x] `data/nimrod_processed/raw/`
- [~] `data/nimrod_processed/concatenated/{year}/{YYYYMMDD}.nc` (optional, in progress)
- [ ] `data/sample_instances/`
- [ ] Feature CSV/Parquet outputs
