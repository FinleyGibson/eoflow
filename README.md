# EOFlow

Pipeline for modeling water quality form Earth Observation data.

## To Do

- [ ] API - build a RESTful API to sever data and outputs on request.
- [ ] Feature Extraction
    - [ ] EO data -> Maze coverage
    - [ ] EO data -> Plastic sheeting
    - [ ] EO data -> Turnip cover
    - [ ] Rainfall data
- [ ] Water quality graph representation
    - [ ] Build Graph representation of rivers
    - [ ] Implement distance measurement metrics
    - [ ] Get Citizen Scientist Data
    - [x] Get Environment Agency Data
        - [ ] Cache based on Union-Interseciton + Time (this must have been done before!)
    - [ ] Add
- [ ] Model water quality form EO data.
    - [ ] Graph Gaussian process
        - [ ] Produce directional graph
            - [ ] Extract River Network as a Graph
                - [x] Define Area of interest (AOI)
                    - [x] Watershed delineation
                - [x] Get geo tif mask of river networks
                    - [x] Use open source maps for this?
                        - [x] [HydroSHEDS](https://www.hydrosheds.org/hydrosheds-core-downloads)
                            - [x] Downsample to AOI
                            - [x] Render visualisation of
                        - [x] [openStreetMap]()
                            - [x] Downsample to AOI
                            - [x] Render visualisation of
                    - [ ] Use thresholded NDVI data to produce Mask
                - [ ] Transform river networks data into graph representation
                    - [ ] [RivGraph](https://github.com/VeinsOfTheEarth/RivGraph)
                    - [ ] implement metrics for graph "distances"
                        - [ ] Distances
                        - [ ] Volume
                    - [ ] Add water quality sample locations to graphs
                        - [ ] CS collection sites
                        - [ ] EA collection sites
        - [ ] Implement Gaussian process over Graph in accordance with [Borovitskiy Et al.](https://proceedings.mlr.press/v130/borovitskiy21a.html)

## Devon Catchment Visualisation — Quick Start

This pipeline fetches EA water-quality data for Devon, downloads a DEM,
delineates catchments for each sampling location, and produces an interactive
HTML map with elevation, flow-accumulation, and catchment overlays.

### Prerequisites

| Requirement            | Notes                                                |
| ---------------------- | ---------------------------------------------------- |
| Python environment     | See [Installation](#installation-instructions) below |
| Devon county shapefile | See step 1                                           |
| OpenTopography API key | Free — register at <https://opentopography.org/>     |

### Step 1 — Devon county shapefile

Download the _Counties and Unitary Authorities_ boundary dataset from the
ONS Open Geography Portal and place it somewhere on disk. Then run:

```shell
# Edit the path inside the script to point at your downloaded shapefile dir,
# then run:
python -m scripts.get_devon_shapefile
# Writes: data/devon_county/devon_county.shp (and sidecar files)
```

### Step 2 — Download the Devon DEM

Requires the `OPENTOPOGRAPHY_API_KEY` environment variable (or `--api-key`).

```shell
export OPENTOPOGRAPHY_API_KEY=<your_key>

python -m scripts.get_devon_dem \
    --output data/devon_dem.tif \
    --dem-type SRTMGL1        # 30 m resolution (default)
# Writes: data/devon_dem.tif
```

### Step 3 — Fetch Devon water-quality samples

Fetches Environment Agency data month-by-month and filters to the Devon
boundary. The example below uses **turbidity** (determinand `6396`) for 2023.

```shell
python -m scripts.get_devon_water_quality \
    --determinand 6396 \
    --start-date 2023-01-01 \
    --end-date   2023-12-31 \
    --out data/devon_turbidity_2023.csv
# Writes: data/devon_turbidity_2023.csv
```

Other useful determinand codes: `0076` temperature · `0077` conductivity ·
`0180` orthophosphate.

### Step 4 — Delineate catchments

Runs pysheds watershed delineation for every unique sampling location and
saves results (catchment polygons, snapped pour points, flow-accumulation
values) to a GeoPackage. Progress is checkpointed so the run can be safely
interrupted and resumed.

```shell
python -m scripts.delineate_catchments \
    --csv  data/devon_turbidity_2023.csv \
    --dem  data/devon_dem.tif \
    --out  data/devon_water_quality_dataset.gpkg \
    --flow-acc-threshold 100   # lower = snap to smaller streams
# Writes: data/devon_water_quality_dataset.gpkg
```

> **Threshold guidance** — `--flow-acc-threshold` controls how far the pour
> point is snapped to the nearest stream cell. Lower values (50–200) keep
> snaps close to the sampling location and include small tributaries. Higher
> values (500–1000) snap to larger channels but may produce large offsets.

### Step 5 — Visualise

Produces an interactive Folium HTML map and opens it in the default browser.
The map includes:

- **DEM elevation** overlay (hillshaded terrain)
- **Flow-accumulation** overlay (blue, log-scale — highlights drainage network)
- **Catchment polygons** colour-coded by the chosen measurement value
- **Sample-point markers** with popups (value, snap offset, flow accumulation)
- **Snapped pour-point** markers joined to sample locations by a dashed line
- **Devon county boundary**

```shell
python -m scripts.visualise_catchments \
    --gpkg      data/devon_water_quality_dataset.gpkg \
    --dem       data/devon_dem.tif \
    --shapefile data/devon_county \
    --value-column result \
    --output    data/devon_catchments_map.html
# Writes and opens: data/devon_catchments_map.html
```

Additional display options:

```shell
    --dem-opacity        0.45   # DEM overlay transparency (0–1)
    --flow-acc-threshold 100    # hide cells with accumulation ≤ N
    --flow-acc-opacity   0.7    # flow-accumulation layer transparency (0–1)
    --no-flow-acc               # disable the flow-accumulation overlay
    --max-pixels         2048   # down-sample DEM/flow-acc rasters if larger
```

---

## Installation instructions

### Dev

```shell
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
