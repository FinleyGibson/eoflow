# EOFlow

Pipeline for modeling water quality form Earth Observation data.

## To Do

- [ ] API - build a RESTful API to sever data and outputs on request.
- [ ] Model water quality form EO data.o
    - [ ] Graph Gaussian process
        - [ ] Produce directional graph
            - [ ] Extract River Network as a Graph
                - [ ] Define Area of interest (AOI)
                    - [ ] [HydroSHEDS](https://www.hydrosheds.org/hydrosheds-core-downloads)
                    - [ ] Watershed delineation
                - [ ] Get geo tif mask of river networks
                    - [ ] Use open source maps for this?
                    - [ ] Use thresholded NDVI data to produce Mask
                - [ ] [RivGraph](https://github.com/VeinsOfTheEarth/RivGraph)
            - [ ] Add CS collection sites as sites on this graph.
        - [ ] Implement Gaussian process over Graph in accordance with [Borovitskiy Et al.](https://proceedings.mlr.press/v130/borovitskiy21a.html)
