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

## Full pipeline (one-liner summary)

```
ge_ea_water_quality_by_shapefile  →  dataset_builder  →  visualise_ea_samples
           (API fetch)                (DEM delineation)       (interactive map)
```

```bash
# 1 — fetch (using shapefile filter)
python -m scripts.ge_ea_water_quality_by_shapefile \
    --shapefile data/shapefiles/devon_county \
    --determinand 0076 --start-date 2023-01-01 --end-date 2023-12-31 \
    --out data/wq_samples/devon_turbidity_2023.csv

# 2 — build catchments
python -m scripts.dataset_builder \
    --csv  data/wq_samples/devon_turbidity_2023.csv \
    --dem  data/dems/devon_dem_cop30.tif \
    --out  outputs/devon_catchments.gpkg \
    --lat-col latitude --lon-col longitude

# 3 — visualise
python -m scripts.visualise_ea_samples \
    --input  data/wq_samples/devon_turbidity_2023.csv \
    --output outputs/devon_turbidity_map.html
```
