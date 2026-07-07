# Data Pipeline

The two main source streams — EA water quality samples and NIMROD rainfall — are independent and can be prepared in parallel. Both require a DEM.

---

## Prerequisites

- **DEM** (`get_dem.py`) — download a GeoTIFF elevation model from OpenTopography; requires a free `OPENTOPOGRAPHY_API_KEY` available from [opentopography.org](https://portal.opentopography.org/login).

    ```
    python -m scripts.get_dem --country England --output data/dems/dem.tif
    ```

    > _Optional QA:_ `visualise_dem.py` — render the DEM as an interactive map to check coverage before proceeding

- **Study-area shapefile** — a polygon boundary for your area of interest, used by `process_nimrod_local.py` (to crop rainfall) and `ge_ea_water_quality_by_shapefile.py` (to filter samples). Administrative boundaries can be downloaded from the [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/), or one has been provided for the East Devon case study area of interest.
- **CEDA credentials** — a `CEDA_TOKEN` env var is required for NIMROD downloads (Stream B) and can be created at [ceda.ac.uk](https://accounts.ceda.ac.uk/realms/ceda/account/#/).
- **Copernicus Data Space account** — a free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu) is required for Sentinel-2 EO data; authentication is via OIDC device-flow (interactive browser prompt) triggered automatically by `prepare_samples.py` unless `--skip-eo` is passed

This has been accessed through the openeo [Python client](https://openeo.org/documentation/1.0/python/), which should facilitate switching to an alternative EO data provider, (such as [Earth Observation Hub](https://earthobservationhub.org/) or [Google Earth Engine](https://developers.google.com/earth-engine)) should that be required at a later date.

The env-var credentials can be stored in a `.env` file in the project root:

```
OPENTOPOGRAPHY_API_KEY=your_key_here
CEDA_TOKEN=your_token_here
```

---

## Stream A — EA Water Quality Samples

1. **`get_ea_water_quality.py`** _(or `ge_ea_water_quality_by_shapefile.py` to filter by geography)_
   Download water quality observations from the EA API → raw CSV

2. **`convert_ea_csv.py`**
   Clean the raw CSV: rename columns, pivot determinands, add WGS-84 lat/lon → clean CSV

    > _Optional QA:_ `visualise_ea_samples.py` — plot sampling locations on an interactive map to check spatial coverage and distribution of values

3. **`delineate_catchments.py`** — _needs: clean CSV + DEM_
   For each unique sampling location: snap to the nearest stream, run pysheds D8 delineation, and store the catchment polygon → GeoPackage (`.gpkg`)
    - Caches results per unique lat/lon so repeated locations are only computed once
    - Checkpoints periodically so a crash can be resumed
    - `delineate_catchment.py` is the single-point equivalent, useful for inspecting one site

    ```
    python -m scripts.delineate_catchments \
        --csv  data/ea_samples.csv \
        --dem  data/dems/dem.tif \
        --out  outputs/catchments.gpkg
    ```

    > _Optional QA:_ `visualise_devon_catchments.py` — overlay the delineated catchment polygons, DEM, drainage network, and sample points on an interactive map to verify delineation quality

---

## Stream B — NIMROD Rainfall Data

4. **`nimrod_authentication_check.sh`** _(optional)_
   Verify `CEDA_TOKEN` can reach the CEDA restricted archive

5. **`nimrod_download_script.sh`**
   Bulk-download one year of NIMROD 1 km composite radar tar files from CEDA → local `.tar` files

    ```
    bash scripts/nimrod_download_script.sh 2023 data/nimrod_raw/
    ```

6. **`process_nimrod_local.py`** — _needs: tar files + a shapefile for the study area_
   Unpack each tar, crop every 5-minute timestep to the study-area boundary, write per-timestep NetCDF files:

    ```
    data/nimrod_data/{year}/{YYYYMMDD}/{YYYYMMDD_HHMMSS}.nc
    ```

    > _Optional QA:_ `visualise_nimrod.py` — overlay a single timestep or an aggregated (mean/max/sum) view of the processed NetCDF files on an interactive map to verify spatial coverage and values

7. **`consolidate_nimrod.py`** _(optional)_
   Merge all per-timestep files into a single time-concatenated NetCDF — faster for repeated loading across many samples
    ```
    python -m scripts.consolidate_nimrod \
        --input  data/nimrod_data/2023 \
        --output data/nimrod_2023.nc
    ```

---

## Compute All Layers

8. **`prepare_samples.py`** — _needs: GeoPackage (A) + DEM + NIMROD dir (B)_
   Iterates over every delineated sample in the GeoPackage and:
    - Computes terrain layers (topography, slope, aspect) from the DEM
    - Queries soil type from the SEARG API
    - Loads NIMROD rainfall from disk via `compute_rainfall`
    - _(optional)_ Fetches NDVI / NDWI from Sentinel-2 via openEO (`--skip-eo` to omit)
    - Saves each fully-populated `Sample` to disk → `data/sample_instances/<notation>/`

    ```
    python -m scripts.prepare_samples \
        --gpkg          outputs/catchments.gpkg \
        --dem           data/dems/dem.tif \
        --nimrod-dir    data/nimrod_data \
        --rainfall-days 10 \
        --eo-days       30
    ```

---

## Use

| Script                    | Purpose                                                                                         | Output        |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ------------- |
| `test_nimrod_rainfall.py` | Smoke-test: load one sample, pull rainfall from disk, print statistics                          | stdout        |
| `extract_features.py`     | Batch feature extraction across all saved samples                                               | CSV / Parquet |
| `full_sample_flow.py`     | _(Optional QA)_ Visualise all layers for one sample as static subplots + interactive Folium map | PNG + HTML    |

---

## Scripts vs Notebooks

Several scripts are the non-interactive equivalents of a notebook. The table below maps them and summarises what each pair covers.

| Script                          | Notebook                            | What it covers                                                                                                                                  |
| ------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_ea_water_quality.py`       | `environment_agency_api_demo.ipynb` | Querying the EA Water Quality API; filtering observations by area or geography                                                                  |
| `delineate_catchments.py`            | `delineate_catchments_demo.ipynb`        | Batch catchment delineation from a CSV of sampling points — DEM loading, pour-point snapping, D8 watershed extraction, GeoPackage checkpointing |
| `delineate_catchment.py`        | `catchment_delineation.ipynb`       | Single-site walkthrough of the full delineation pipeline — pit filling, flow direction, accumulation, and polygon extraction                    |
| `visualise_ea_samples.py`       | `environment_agency_api_demo.ipynb` | Interactive map of EA sampling locations with values colour-coded by determinand                                                                |
| `visualise_dem.py`              | `topography.ipynb`                  | DEM rendered as a colour-coded raster overlay; elevation, slope, and aspect across the study area                                               |
| `visualise_devon_catchments.py` | `catchment_delineation.ipynb`       | Delineated catchment polygons overlaid on the DEM and drainage network, with sample-point markers                                               |
| `visualise_nimrod.py`           | `nimrod_viewer.ipynb`               | NIMROD radar rainfall data — single timestep or time-aggregated (mean/max/sum) — overlaid on a Leaflet map                                      |
| `prepare_samples.py`            | `sample_demo.ipynb`                 | Full per-sample layer computation: terrain, soil type, NIMROD rainfall, and Sentinel-2 NDVI/NDWI via openEO                                     |
| `full_sample_flow.py`           | `full_sample_flow.ipynb`            | End-to-end single-sample walkthrough from raw CSV to computed layers, feature extraction, and interactive + static visualisation                |

Notebooks without a script equivalent:

| Notebook                     | What it covers                                                                                                    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `MET_office_rainfall.ipynb`  | Legacy Met Office UKV rainfall download and per-catchment analysis (superseded by the NIMROD disk-based approach) |
| `soil_types.ipynb`           | DEFRA Soil Structure Groups — county-wide and per-catchment soil-type fractional coverage                         |
| `graph_with_nodes.ipynb`     | River network graph construction with node/edge attributes                                                        |
| `river_graph_building.ipynb` | Building topological river graphs from flow-direction rasters                                                     |
