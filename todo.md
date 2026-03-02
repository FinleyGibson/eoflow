# To Do:

## Overall Plan

- [ ] Water sample data
    - [ ] Citizen Scientist Data (real)
        - [ ] Load and visualise from ArcGIS API
        - [ ] Make available via API
    - [x] EA Data
        - [x] Load and visualise
        - [x] Make available via API
    - [ ] Combine data sources with source annotation (labelled)
- [ ] River Graph
    - [x] Generate river graph from maps
    - [x] Add Water sample data to graph as nodes
    - [ ] Convert to Directed graph
- [ ] River Tree
    - [ ] Move to tree data structure
- [ ] Modelling
    - [x] Script to build sample-catchment dataset.
    - [ ] Basic interpolation model
    - [ ] Directed interpolation model
    - [ ] Weighted interpolation model
    - [ ] Weighted directed interpolation model
    - [ ] Graph Neural Network
    - [ ] Graph Gaussian Process
    - [ ] Conformal prediction | Conformal Neural Network
        - [ ] [MAPIE](https://mapie.readthedocs.io/en/stable/) Python library for conformal prediction
            - Practical [guide](https://algotrading101.com/learn/conformal-prediction-guide/) to MAPIE
    - [ ] Look for modelling on tree structures
- [ ] API
    - [x] Split tests for EA data into API amd non API
        - [x] Ensure passes all tests.
    - [x] Add CS samples to water samples in API.
    - [ ] Package API for Benjamin.
        - [ ] New repo
        - [ ] Transplant required code only
        - [ ] Rebuild minimalist uv build

## For next week

- **Focus initial work on Environment Agency (EA) data**
    - [x] Compile and structure the EA dataset.
    - [x] Begin feature extraction from this dataset.

- **Prepare model inputs**
    - [ ] Normalise variables by:
        - [ ] Catchment area
        - [ ] Rainfall
        - [ ] Time of year
        - [ ] Number of dry days prior
    - [ ] Incorporate soil type classification:
        - [ ] Obtain soil data from Cranfield, BGS Soil Observatory, and FEH/CEH sources.
        - [ ] Calculate fractional coverage of the 26 soil types per catchment.

- **Integrate rainfall data**
    - [ ] Use Met Office spatial rainfall data via AWS S3.
    - [ ] Run Albert’s scripts (`run.sh`, `hdf5toasci`) to extract rainfall for catchment areas.

- **Feature engineering priorities**
    - [ ] NDVI
    - [ ] ΔNDWI
    - [ ] Impervious surfaces (tarmac/concrete)
    - [ ] Maize crop presence
    - [ ] Plastic sheeting
    - [ ] Other erosion-relevant land features

- **Model development**
    - [ ] Build model linking environmental features + rainfall + catchment characteristics → turbidity.
    - [ ] Validate model results using citizen science data.

- **Background reading & coordination**
    - [ ] Review Albert’s thesis for methodology alignment.
    - [ ] Monitor opportunities from the Environmental Intelligence conference (IDSAI, Exeter; contact: Hywel Williams).

## Next Steps

- [x] Get SOME EA data using script
- [x] Script to convert to desired format
- [x] Prune EA data to a few examples (10) from within Devon
- [x] Prune the Topography data to Devon only
- [x] Update tests to use Devon only data.
- [x] Run script to get catchments
- [x] Check outputs
- [x] Visualise outputs
- [x] Build output class
- [x] port methods to get EO data into class
- [x] Calculate catchment area

- [ ] Fix Watershed delineation
    - [ ] Use pygis
        - Get the method from Albert's teams message.
- [ ] Modify Albert's scripts to get rainfall data
    - [ ] Fix aws command line tool on my local machine
    - [ ] Run in command line.
    - [ ] Run from python.
    - [ ] Add method to Sample class.
- [ ] Incorporate soil type classification:
    - leave this until the model pipeline is working.
    - [ ] Obtain soil data from Cranfield, BGS Soil Observatory, and FEH/CEH sources.
    - [ ] Calculate fractional coverage of the 26 soil types per catchment.
