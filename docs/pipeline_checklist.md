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

# Stream A — EA Water Quality Samples

- [ ] Download EA water quality observations (`get_ea_water_quality.py` or `ge_ea_water_quality_by_shapefile.py`)
- [ ] Convert and clean the raw EA CSV (`convert_ea_csv.py`)
- [ ] (Optional) Verify sample locations (`visualise_ea_samples.py`)
- [ ] Delineate catchments (`delineate_catchments.py`)
- [ ] (Optional) Inspect delineation quality (`visualise_devon_catchments.py`)

---

# Stream B — NIMROD Rainfall Data

- [ ] (Optional) Verify CEDA authentication
- [ ] Download NIMROD rainfall data
- [ ] Process and crop rainfall data to the study area
- [ ] (Optional) Visualise rainfall outputs
- [ ] (Optional) Consolidate timesteps into a single NetCDF

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

- [ ] `data/dems/dem.tif`
- [ ] `data/ea_samples.csv`
- [ ] `outputs/catchments.gpkg`
- [ ] `data/nimrod_data/`
- [ ] `data/nimrod_2023.nc` (optional)
- [ ] `data/sample_instances/`
- [ ] Feature CSV/Parquet outputs
