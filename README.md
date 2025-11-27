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
                - [ ] Get geo tif mask of river networks
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
            - [ ] Add CS collection sites as sites on this graph.
        - [ ] Implement Gaussian process over Graph in accordance with [Borovitskiy Et al.](https://proceedings.mlr.press/v130/borovitskiy21a.html)

## Installation instructions

### Dev

```shell
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
