# Data Pipeline

The two main source streams — EA water quality samples and NIMROD rainfall — are independent and can be prepared in parallel. Both require a DEM.

---

## Prerequisites

- **DEM** (`get_dem.py`) — download a GeoTIFF elevation model from OpenTopography; requires a free `OPENTOPOGRAPHY_API_KEY` available from [opentopography.org](https://portal.opentopography.org/login).

    ```
    python -m scripts.get_dem --country England --output data/dems/dem.tif
    ```

    > _Optional QA:_ `visualise_dem.py` — render the DEM as an interactive map to check coverage before proceeding

- **Study-area shapefile** — a polygon boundary for your area of interest, used by `nimrod_process_local.py` (to crop rainfall) and `get_ea_water_quality_by_shapefile.py` (to filter samples). Administrative boundaries can be downloaded from the [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/), or one has been provided for the East Devon case study area of interest.

    > _Devon case study convenience:_ `get_devon_shapefile.py` extracts the Devon county boundary from an ONS "Counties and Unitary Authorities" download, and `get_devon_dem.py` combines that with `get_dem.py` to fetch a DEM pre-clipped to Devon — these produced the shapefile/DEM already present under `data/`.
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

1. **`get_ea_water_quality.py`** _(or `get_ea_water_quality_by_shapefile.py` to filter by geography)_
   Download water quality observations from the EA API → raw CSV

2. **`convert_ea_csv.py`**
   Clean the raw CSV: rename columns, pivot determinands, add WGS-84 lat/lon → clean CSV

    > _Optional QA:_ `visualise_ea_samples.py` — plot sampling locations on an interactive map to check spatial coverage and distribution of values
    >
    > `sample_report.py` — print quick summary stats (row count, unique locations, date/lat/lon coverage) for a samples CSV:
    > ```
    > python -m scripts.sample_report data/ea_water_quality/turbidity_clean.csv
    > ```

3. **`delineate_catchments.py`** — _needs: clean CSV + DEM_
   For each unique sampling location: snap to the nearest stream, run pysheds D8 delineation, and store the catchment polygon → GeoPackage (`.gpkg`)
    - Caches results per unique lat/lon so repeated locations are only computed once
    - Checkpoints periodically so a crash can be resumed
    - `delineate_catchment.py` is the single-point equivalent, useful for inspecting one site

    ```
    python -m scripts.delineate_catchments \
        --csv  data/ea_water_quality/turbidity_clean.csv \
        --dem  data/dems/devon_dem_cop30.tif \
        --out  data/delineated_catchments/catchments.gpkg
    ```

    > _Optional QA:_ `visualise_catchments.py` — overlay the delineated catchment polygons, DEM, drainage network, and sample points on an interactive map to verify delineation quality
    >
    > `gpkg_report.py` — print summary stats (row/site/date counts, delineation success rate) for a delineated GeoPackage and plot per-site sample-date violin plots:
    > ```
    > python -m scripts.gpkg_report data/delineated_catchments/catchments.gpkg
    > ```

    > _Optional downsampling:_ `downsample_gpkg.py` — reduce a large delineated GeoPackage to a smaller, diverse subset (spread across sites and time) — useful for quick/cheap pipeline runs before committing to the full dataset:
    > ```
    > python -m scripts.downsample_gpkg \
    >     --gpkg       data/delineated_catchments/catchments.gpkg \
    >     --n-samples  1000 \
    >     --output     data/delineated_catchments/catchments_1000.gpkg
    > ```

---

## Stream B — NIMROD Rainfall Data

4. **`nimrod_authentication_check.sh`** _(optional)_
   Verify `CEDA_TOKEN` can reach the CEDA restricted archive

5. **`nimrod_download_script.sh`**
   Bulk-download one year of NIMROD 1 km composite radar tar files from CEDA → local `.tar` files

    ```
    bash scripts/nimrod_download_script.sh 2023 data/nimrod_raw/2023
    ```

    `wget --mirror` nests the downloaded tar files several directories deep
    (e.g. `data/nimrod_raw/2023/dap.ceda.ac.uk/badc/ukmo-nimrod/data/composite/uk-1km/2023/*.tar`)
    rather than flat in the given directory — this is expected, and step 6 finds
    them regardless of depth.

    The script also validates every downloaded tar and pre-flight-checks the
    token before starting (`CEDA_TOKEN` is short-lived; a token valid at the
    start of a multi-year run can expire partway through, which otherwise
    surfaces as CEDA silently serving an HTML login page in place of the file).

    > _Convenience wrapper:_ `download_all_nimrod.sh` — loops the above over 2016–2025 into `data/nimrod_raw/<year>/`

6. **`nimrod_process_local.py`** — _needs: tar files + a shapefile for the study area_
   Unpack each tar (found recursively under `--input`, at any depth), crop every 5-minute timestep to the study-area boundary, write per-timestep NetCDF files:

    ```
    data/nimrod_processed/raw/{year}/{YYYYMMDD}/{YYYYMMDD_HHMMSS}.nc
    ```

    > _Optional QA:_ `visualise_nimrod.py` — overlay a single timestep or an aggregated (mean/max/sum) view of the processed NetCDF files on an interactive map to verify spatial coverage and values

    > _Convenience wrapper:_ `process_all_nimrod.sh` — loops the above over every year directory found under `data/nimrod_raw/`, writing to `data/nimrod_processed/raw/<year>` by default

7. **`consolidate_nimrod.sh`** _(optional)_
   Merge per-timestep files into per-day NetCDFs — faster for repeated loading across many samples, and required before a day's data can be opened as a single time series at all.

    NIMROD files store `time`, `forecast_reference_time`, and `forecast_period` as **scalar** coordinates rather than a record dimension, so a naive `ncrcat`/`cdo mergetime` either fails outright or (worse) silently keeps only one timestep. This script instead stacks each day's files with `ncecat` and repairs the time coordinates from the timestamps encoded in the filenames.

    > An earlier `xarray`/`dask`-based merge (`consolidate_nimrod.py`) has been removed: its `open_mfdataset(parallel=True)` approach did not finish in practice on the full multi-year archive (tens of thousands of small per-timestep files) — it also turns out to be unsafe at any file count, since `xarray`'s file-handle cache isn't thread-safe.

    > `ncecat` requires every input file for a day to share the same variable set; if a raw per-timestep file is missing an expected variable (schema drift somewhere upstream — seen in practice for at least one 2024 file missing `transverse_mercator`), that day fails with `nco_inq_varid(): ... is not defined in file` and the run stops rather than silently skipping it. Rerunning with `-f` after removing/fixing the offending file resumes from there.

    ```
    # Merge one year
    bash scripts/consolidate_nimrod.sh \
        data/nimrod_processed/raw/2023 \
        data/nimrod_processed/concatenated/2023

    # Merge everything found under raw/, creating year subdirs under target
    bash scripts/consolidate_nimrod.sh \
        data/nimrod_processed/raw \
        data/nimrod_processed/concatenated

    # -f forces reprocessing of days that already have output; -h for full usage
    ```

    Output layout mirrors the input:

    ```
    data/nimrod_processed/concatenated/{year}/{YYYYMMDD}.nc
    ```

    Each day file already has a proper `time` record dimension, so if you want a single whole-year file matching the shape `get_nimrod_rainfall_for_polygon` expects from a consolidated file, a plain `ncrcat` over a year's day files is fast and sufficient:

    ```
    ncrcat data/nimrod_processed/concatenated/2023/*.nc data/nimrod_processed/concatenated/2023.nc
    ```

    > **Disk retention:** once a year's raw per-timestep tree is confirmed fully
    > represented in its consolidated output (every `YYYYMMDD` day directory under
    > `raw/<year>` has a matching `YYYYMMDD.nc` under `concatenated/<year>`), the raw
    > copy — and the downloaded source tars from step 5 — are not worth keeping
    > alongside the consolidated one; delete them to reclaim disk. Re-deriving
    > per-timestep data later means re-downloading and re-processing from CEDA.

---

## Compute All Layers

8. **`prepare_samples.py`** — _needs: GeoPackage (A) + DEM + NIMROD dir (B)_
   Iterates over every delineated sample in the GeoPackage and:
    - Computes terrain layers (topography, slope, aspect) from the DEM
    - Queries soil type from the SEARG API
    - Loads NIMROD rainfall from disk via `compute_rainfall`
    - _(optional)_ Fetches NDVI / NDWI from Sentinel-2 via openEO (`--skip-eo` to omit)
    - Saves each fully-populated `Sample` to disk → `data/sample_instances/<notation>_<sample_id>/`
      (`<sample_id>` is the observation ID embedded in the source CSV/GPKG's `id` column —
      `notation` alone identifies the *site*, not the observation, and the same site is
      typically sampled on many different dates)

    `--nimrod-dir` accepts the raw per-timestep directory tree (step 6's output), the per-day consolidated directory tree (step 7's output), or a single `.nc` file covering an arbitrary range (e.g. a whole year merged with `ncrcat`) — consolidation is a speed optimisation, not a requirement.

    ```
    python -m scripts.prepare_samples \
        --gpkg          data/delineated_catchments/catchments.gpkg \
        --dem           data/dems/devon_dem_cop30.tif \
        --nimrod-dir    data/nimrod_processed/concatenated/2020 \
        --rainfall-days 10 \
        --eo-days       30
    ```

    openEO authentication is non-interactive (no browser prompt) when `OPENEO_AUTH_CLIENT_ID`/`OPENEO_AUTH_CLIENT_SECRET` are set in `.env` — useful on a headless/remote machine, since the device-flow prompt described in Prerequisites otherwise needs an interactive browser.

---

## Use

| Script                    | Purpose                                                                                         | Output        |
| ------------------------- | ----------------------------------------------------------------------------------------------- | ------------- |
| `test_nimrod_rainfall.py` | Smoke-test: load one sample, pull rainfall from disk, print statistics                          | stdout        |
| `extract_features.py`     | Batch feature extraction across all saved samples                                               | CSV / Parquet |
| `feature_report.py`       | _(Optional QA)_ Summary stats + per-group data-completeness bar chart for an extracted-features CSV | stdout + PNG  |
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
| `visualise_catchments.py`       | `catchment_delineation.ipynb`       | Delineated catchment polygons overlaid on the DEM and drainage network, with sample-point markers                                               |
| `visualise_nimrod.py`           | `nimrod_viewer.ipynb`               | NIMROD radar rainfall data — single timestep or time-aggregated (mean/max/sum) — overlaid on a Leaflet map                                      |
| `prepare_samples.py`            | `sample_demo.ipynb`                 | Full per-sample layer computation: terrain, soil type, NIMROD rainfall, and Sentinel-2 NDVI/NDWI via openEO                                     |
| `full_sample_flow.py`           | `full_sample_flow.ipynb`            | End-to-end single-sample walkthrough from raw CSV to computed layers, feature extraction, and interactive + static visualisation                |

Notebooks without a script equivalent:

| Notebook                     | What it covers                                                                                                    |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `soil_types.ipynb`           | DEFRA Soil Structure Groups — county-wide and per-catchment soil-type fractional coverage                         |
| `graph_with_nodes.ipynb`     | River network graph construction with node/edge attributes                                                        |
| `river_graph_building.ipynb` | Building topological river graphs from flow-direction rasters                                                     |

> The Met Office UKV 2 km rainfall pipeline (`download_rainfall.py`, `convert_rainfall.py`, `get_rainfall_for_polygon`, `MET_office_rainfall.ipynb`) has been removed from this branch in favour of the NIMROD disk-based approach above. It's preserved on the `met-office-rainfall-legacy` branch if needed again.
