# Dataset Builder

Steps to go from a raw Environment Agency water-quality API response to a
GeoPackage of sample sites with delineated catchment polygons, and then to
visualise the result on an interactive map.

---

## Prerequisites

```bash
pip install -e .
```

The workflow also requires:

- A **DEM GeoTIFF** covering the area of interest (see step 2).  
  The Devon Copernicus GLO-30 DEM is already at `data/dems/devon_dem_cop30.tif`.
- A **region shapefile** for geographic filtering (e.g. `data/shapefiles/devon_county/`).

---

## Step 1 — Fetch water-quality data from the EA API

Download turbidity (or any other determinand) observations for an area of
interest. Results are written to a CSV and the run is **checkpointed by
month**, so it is safe to interrupt and resume.

There are two scripts for this step depending on how you want to filter
geographically:

### Option A — Filter by shapefile (recommended)

`ge_ea_water_quality_by_shapefile.py` fetches data from the EA API without
a geographic filter and then clips results client-side to a polygon
extracted from your shapefile. This avoids 400 errors that some precanned
EA area codes can produce.

```bash
python -m scripts.ge_ea_water_quality_by_shapefile \
    --shapefile   data/shapefiles/devon_county \
    --determinand 0076 \
    --start-date  2023-01-01 \
    --end-date    2023-12-31 \
    --out         data/wq_samples/devon_turbidity_2023.csv
```

| Flag                          | Description                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| `--shapefile`                 | **(required)** Path to a shapefile directory or `.shp` file defining the region of interest            |
| `--determinand`               | EA determinand code — `0076` Temperature, `0077` Conductivity, `0180` Orthophosphate, `6396` Turbidity |
| `--start-date` / `--end-date` | Inclusive date window (`YYYY-MM-DD`)                                                                   |
| `--out`                       | Output CSV path (default: `water_quality.csv`)                                                         |

### Option B — Filter by EA area code

`get_ea_water_quality.py` uses the EA's built-in precanned area codes for
server-side filtering. This can be faster but some area codes return
400 errors.

```bash
python -m scripts.get_ea_water_quality \
    --determinand 0076 \
    --start-date  2023-01-01 \
    --end-date    2023-12-31 \
    --area        environment_agency,DCS \
    --out         data/wq_samples/devon_turbidity_2023.csv
```

| Flag                          | Description                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `--determinand`               | EA determinand code                                                             |
| `--start-date` / `--end-date` | Inclusive date window (`YYYY-MM-DD`)                                            |
| `--area`                      | Precanned EA area code (e.g. `environment_agency,DCS`); omit to fetch all areas |
| `--out`                       | Output CSV path                                                                 |

Both scripts produce a CSV with the raw EA schema including a `result`
column and BNG easting/northing coordinates.

---

## Step 2 — (Optional) Download a DEM

Skip this step if `data/dems/devon_dem_cop30.tif` already exists.

You need a free [OpenTopography API key](https://opentopography.org/).

```bash
export OPENTOPOGRAPHY_API_KEY=<your_key>

python -m scripts.get_devon_dem \
    --output data/dems/devon_dem_cop30.tif
```

The script clips the downloaded DEM to the Devon county polygon so only
the relevant area is stored.

---

## Step 3 — Build the catchment dataset

Run `dataset_builder.py` to iterate over every sample in the CSV, delineate
its upstream catchment from the DEM using D8 flow routing, and assemble the
results into a GeoPackage.

```bash
python -m scripts.dataset_builder \
    --csv  data/wq_samples/devon_turbidity_2023.csv \
    --dem  data/dems/devon_dem_cop30.tif \
    --out  outputs/devon_catchments.gpkg
```

The script:

1. Loads the CSV and drops rows with missing coordinates.
2. Deduplicates by `(lat, lon)` — identical locations share a single cached
   catchment polygon so no work is repeated.
3. For each unique location, calls `eoflow.catchment.delineate_catchment` to
   snap the pour point to the nearest stream cell and trace the watershed.
4. Writes a checkpoint to the output `.gpkg` every `--checkpoint-every` new
   delineations (default 10). On restart the script skips rows that already
   have a catchment so runs are safely resumable after a crash.
5. Stores the results as a `geopandas.GeoDataFrame` with columns:
    - all original CSV columns
    - `geometry` — the catchment `Polygon` in WGS-84
    - `delineation_status` — `"ok"` or `"failed"`
    - `delineation_error` — error message when status is `"failed"`

| Flag                   | Default                         | Description                                     |
| ---------------------- | ------------------------------- | ----------------------------------------------- |
| `--csv`                | _(required)_                    | Input CSV path                                  |
| `--dem`                | _(required)_                    | GeoTIFF DEM path                                |
| `--out`                | `outputs/devon_catchments.gpkg` | Output GeoPackage (also the checkpoint)         |
| `--lat-col`            | `lat`                           | Latitude column name in the CSV                 |
| `--lon-col`            | `long`                          | Longitude column name in the CSV                |
| `--flow-acc-threshold` | `1000`                          | Upstream-cell threshold for pour-point snapping |
| `--checkpoint-every`   | `10`                            | Save frequency (number of new delineations)     |
| `--log-level`          | `INFO`                          | `DEBUG` / `INFO` / `WARNING`                    |
| `--log-file`           | _(none)_                        | Optional path to write logs to a file           |

> **Note on the CSV format.** The raw EA CSV uses `latitude` / `longitude`
> columns (added by the water quality scripts). Pass `--lat-col latitude
--lon-col longitude` if the default `lat` / `long` names are not present.

---

## Step 4 — Visualise sample points

`visualise_ea_samples.py` reads the CSV (raw or cleaned) and plots each
observation as a colour-coded circle marker on an interactive Folium map.
The map is written to an HTML file and opened automatically in the browser.

```bash
python -m scripts.visualise_ea_samples \
    --input data/wq_samples/devon_turbidity_2023.csv
```

Markers are coloured by the `result` column (or the first non-metadata
column). A log scale is applied automatically when the value range spans
more than one order of magnitude.

**Override the value column:**

```bash
python -m scripts.visualise_ea_samples \
    --input       data/wq_samples/ea_water_quality_clean.csv \
    --value-column "Turbidity (NEPHELOMETRIC TURBIDITY UNITS)"
```

**Save the HTML to a specific path instead of a temp file:**

```bash
python -m scripts.visualise_ea_samples \
    --input  data/wq_samples/devon_turbidity_2023.csv \
    --output outputs/devon_turbidity_map.html
```

| Flag             | Default                      | Description                 |
| ---------------- | ---------------------------- | --------------------------- |
| `--input`        | `ea_water_quality_clean.csv` | CSV to read                 |
| `--value-column` | _(auto-detected)_            | Column to colour markers by |
| `--output`       | _(temp file)_                | HTML output path            |

---

## Step 5 -- Prepare sample instances (compute layers + fetch EO data)

Before extracting features, each sample from the GeoPackage produced in
Step 3 needs its spatial layers computed (topography, slope, aspect, soil
type, rainfall) and its Earth-observation indices fetched (NDVI, NDWI via
openEO). The results are saved as individual Sample directories under
`data/sample_instances/`.

`prepare_samples.py` handles this as a batch operation:

```bash
python -m scripts.prepare_samples \
    --gpkg    outputs/devon_catchments.gpkg \
    --dem     data/dems/devon_dem_cop30.tif \
    --out-dir data/sample_instances \
    --rainfall-start 2023-01-01 \
    --rainfall-end   2023-12-31 \
    --eo-start       2023-01-01 \
    --eo-end         2023-12-31
```

**Skip the slow openEO fetch** (useful when you only need terrain/soil/rainfall):

```bash
python -m scripts.prepare_samples --skip-eo
```

| Flag                   | Default                             | Description                                                         |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------------- |
| `--gpkg`               | `outputs/devon_catchments.gpkg`     | Input GeoPackage from dataset_builder.py                            |
| `--dem`                | `data/dems/devon_dem_cop30.tif`     | DEM GeoTIFF for topography / slope / aspect                        |
| `--out-dir`            | `data/sample_instances`             | Output directory for saved Sample subdirectories                    |
| `--rainfall-dir`       | `data/temp/rainfall`                | Cache directory for Met Office rainfall downloads                   |
| `--rainfall-start`     | _(none)_                            | Start date for rainfall (YYYY-MM-DD). Omit to skip rainfall        |
| `--rainfall-end`       | _(none)_                            | End date for rainfall (YYYY-MM-DD). Omit to skip rainfall          |
| `--eo-start`           | _(none)_                            | Start date for NDVI/NDWI queries. Omit for per-sample defaults     |
| `--eo-end`             | _(none)_                            | End date for NDVI/NDWI queries. Omit for per-sample defaults       |
| `--skip-eo`            | _(off)_                             | Skip the openEO NDVI/NDWI fetch step entirely                      |
| `--flow-acc-threshold` | `5000`                              | Flow-accumulation threshold for pour-point snapping                 |

The script is safe to re-run -- it skips sample directories that already
contain a `metadata.json`. If a run is interrupted, just re-execute and it
will resume from where it left off.

> **openEO authentication.** The first call to `Sample.connect_openeo()`
> will open a browser window for authentication. Make sure you have an
> account with the configured openEO backend (e.g. Copernicus Data Space).

**Alternatively**, you can do this step with a Python snippet for more control:

```python
from pathlib import Path
from datetime import datetime
from eoflow.samples import CatchmentDataset, Sample
from eoflow.utils import DATA_DIR, PROJECT_ROOT

# -- Config ----------------------------------------------------------------
GPKG_PATH       = PROJECT_ROOT / "outputs/devon_catchments.gpkg"
DEM_PATH        = DATA_DIR / "dems/devon_dem_cop30.tif"
SAMPLES_OUT_DIR = DATA_DIR / "sample_instances"
RAINFALL_DIR    = DATA_DIR / "temp/rainfall"
RAINFALL_START  = datetime(2023, 1, 1)
RAINFALL_END    = datetime(2023, 12, 31)
EO_START_DATE   = "2023-01-01"
EO_END_DATE     = "2023-12-31"

# -- Load the GeoPackage into a CatchmentDataset ---------------------------
ds = CatchmentDataset.from_gpkg(GPKG_PATH, only_delineated=True)
print(ds)  # e.g. CatchmentDataset(142 samples, 142 with catchments)

# -- Connect to openEO once (reused for all samples) ----------------------
conn = Sample.connect_openeo()

# -- Process each sample ---------------------------------------------------
for i, sample in enumerate(ds):
    tag = sample.notation or f"sample_{i:04d}"
    out_dir = SAMPLES_OUT_DIR / tag

    # Skip if already saved
    if (out_dir / "metadata.json").exists():
        print(f"  skip {tag} (already exists)")
        continue

    # 5a. Compute terrain + soil + rainfall layers
    sample.compute_layers(
        dem_path=DEM_PATH,
        rainfall_start=RAINFALL_START,
        rainfall_end=RAINFALL_END,
        rainfall_download_dir=RAINFALL_DIR,
    )

    # 5b. Fetch NDVI and NDWI from openEO / Sentinel-2
    try:
        sample.fetch_ndvi(conn, start_date=EO_START_DATE, end_date=EO_END_DATE)
        sample.fetch_ndwi(conn, start_date=EO_START_DATE, end_date=EO_END_DATE)
    except Exception as exc:
        print(f"  warning: EO fetch failed for {tag}: {exc}")

    # 5c. Save the fully-populated Sample to disk
    sample.save(out_dir)
    print(f"  saved {tag}  ({i+1}/{len(ds)})")
```

Each saved Sample directory contains:

| File                          | Content                                                |
| ----------------------------- | ------------------------------------------------------ |
| `metadata.json`               | Observation row (site, date, result, coordinates, ...) |
| `catchment.wkt`               | Catchment polygon as WKT                               |
| `layers/topography.nc`        | Elevation raster (NetCDF)                              |
| `layers/slope.nc`             | Slope raster (NetCDF)                                  |
| `layers/aspect.nc`            | Aspect raster (NetCDF)                                 |
| `layers/soil_type.json`       | Soil-type series (JSON)                                |
| `layers/soil_polygons.geojson`| Soil polygons (GeoJSON)                                |
| `layers/rainfall.nc`          | Rainfall time-series raster (NetCDF)                   |
| `layers/ndvi.nc`              | NDVI time-series raster (NetCDF)                       |
| `layers/ndwi.nc`              | NDWI time-series raster (NetCDF)                       |

> **Tip:** The snippet above is safe to re-run -- it skips directories that
> already contain a `metadata.json`. If a run is interrupted, just re-execute
> and it will resume from where it left off.

> **openEO authentication.** The first call to `Sample.connect_openeo()`
> will open a browser window for authentication. Make sure you have an
> account with the configured openEO backend (e.g. Copernicus Data Space).

---

## Step 6 -- Extract features

`extract_features.py` reads the saved Sample instances produced in Step 5,
computes catchment-scale regression features from each sample's spatial
layers, and writes the resulting feature matrix to a CSV (and optionally
Parquet).

```bash
python -m scripts.extract_features \
    --samples-dir data/sample_instances \
    --output      outputs/catchment_features.csv
```

The script produces one row per sample and groups columns into: identity /
target, geometry, terrain, NDVI, NDWI, soil, rainfall, and interaction
features. A summary report is printed to the console after extraction.

**Also save a Parquet file** (preserves types better than CSV):

```bash
python -m scripts.extract_features \
    --samples-dir data/sample_instances \
    --output      outputs/catchment_features.csv \
    --parquet
```

**Adjust classification thresholds:**

```bash
python -m scripts.extract_features \
    --samples-dir data/sample_instances \
    --output      outputs/catchment_features.csv \
    --bare-ndvi   0.25 \
    --dense-ndvi  0.55 \
    --wet-ndwi    0.05 \
    --steep-low   8.0 \
    --steep-high  12.0
```

| Flag             | Default                             | Description                                                          |
| ---------------- | ----------------------------------- | -------------------------------------------------------------------- |
| `--samples-dir`  | `data/sample_instances`              | Directory containing saved Sample subdirectories                     |
| `--output`       | `outputs/catchment_features.csv`     | Destination CSV path                                                 |
| `--parquet`      | _(off)_                             | Also save a `.parquet` file alongside the CSV                        |
| `--bare-ndvi`    | `0.20`                              | NDVI threshold below which a pixel is classified as bare/sparse      |
| `--dense-ndvi`   | `0.60`                              | NDVI threshold above which a pixel is classified as dense vegetation |
| `--wet-ndwi`     | `0.00`                              | NDWI threshold above which a pixel is classified as open water       |
| `--steep-low`    | `10.0`                              | Primary steep-slope threshold (degrees)                              |
| `--steep-high`   | `15.0`                              | Secondary steep-slope threshold (degrees)                            |

> **Note:** Samples that fail to load or extract are skipped with a warning.
> The script requires at least one successful extraction to produce output.

---

## Full pipeline (one-liner summary)

```
ge_ea_water_quality_by_shapefile -> dataset_builder -> prepare_samples -> extract_features -> visualise_ea_samples
         (API fetch)              (DEM delineation)   (layers + EO)    (feature CSV)        (interactive map)
```

```bash
# 1 -- fetch (using shapefile filter)
python -m scripts.ge_ea_water_quality_by_shapefile \
    --shapefile data/shapefiles/devon_county \
    --determinand 0076 --start-date 2023-01-01 --end-date 2023-12-31 \
    --out data/wq_samples/devon_turbidity_2023.csv

# 2 -- build catchments
python -m scripts.dataset_builder \
    --csv  data/wq_samples/devon_turbidity_2023.csv \
    --dem  data/dems/devon_dem_cop30.tif \
    --out  outputs/devon_catchments.gpkg \
    --lat-col latitude --lon-col longitude

# 3 -- prepare sample instances (compute layers + fetch EO data)
python -m scripts.prepare_samples \
    --gpkg outputs/devon_catchments.gpkg \
    --dem  data/dems/devon_dem_cop30.tif \
    --rainfall-start 2023-01-01 --rainfall-end 2023-12-31 \
    --eo-start 2023-01-01 --eo-end 2023-12-31

# 4 -- extract features
python -m scripts.extract_features \
    --samples-dir data/sample_instances \
    --output      outputs/catchment_features.csv \
    --parquet

# 5 -- visualise
python -m scripts.visualise_ea_samples \
    --input  data/wq_samples/devon_turbidity_2023.csv \
    --output outputs/devon_turbidity_map.html
```


---

## One-command alternative: run_pipeline.sh

A bash script that runs the entire pipeline end-to-end is provided at the
project root:

```bash
./run_pipeline.sh
```

Each step only runs after the previous one completes successfully (`set -e`).
Steps are automatically skipped when their output files already exist.

**Useful flags:**

| Flag              | Effect                                            |
| ----------------- | ------------------------------------------------- |
| `--skip-eo`       | Skip the slow openEO NDVI/NDWI fetch              |
| `--skip-fetch`    | Skip the EA water-quality API download             |
| `--skip-dem`      | Skip the DEM download                              |
| `--skip-vis`      | Skip the visualisation step                        |
| `--no-parquet`    | Do not save a Parquet copy of the features         |
| `--determinand N` | Override the EA determinand code (default `0076`)  |

All paths and dates are configurable via environment variables:

```bash
START_DATE=2024-01-01 END_DATE=2024-06-30 DETERMINAND=6396 ./run_pipeline.sh --skip-eo
```
