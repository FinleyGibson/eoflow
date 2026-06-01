# To Do

## Data

- [ ] CS Data: load and visualise from ArcGIS API, make available via API
- [ ] Combine EA + CS sources with source annotation (labelled)

## Graph & Tree

- [ ] Convert river graph to directed graph
- [ ] Move to tree data structure

## Historic Rainfall

- [ ] 2026 data
    - [x] Download raw data
    - [x] Crop to Devon area
    - [ ] Break into watersheds
    - [ ] Stack watersheds temporally
        - [ ] Issue slow stacking
    - [ ] Issue: inversion fix
- [ ] Update to correct area of interest

## Feature Extraction

- [ ] Define desired metrics: slope, aspect, direction, land use, bare earth, NDVI, ΔNDWI, impervious surfaces, maize, plastic sheeting
- [ ] Implement individual components
- [ ] Implement overlaps

## Model Inputs & Temporal

- [ ] Normalise variables: catchment area, rainfall, time of year, dry days prior
- [ ] Met Office spatial rainfall via AWS S3
- [ ] Temporal: investigate resolution, ensure full data computed, define window strategy
- [ ] Weightings: Euclidean distance and graph flow distance from sample

## Modelling

- [ ] Basic interpolation model
- [ ] Directed interpolation model
- [ ] Weighted interpolation model
- [ ] Weighted directed interpolation model
- [ ] Graph Neural Network
- [ ] Graph Gaussian Process
- [ ] Conformal prediction / Conformal Neural Network — [MAPIE](https://mapie.readthedocs.io/en/stable/) ([guide](https://algotrading101.com/learn/conformal-prediction-guide/))
- [ ] Look for modelling approaches on tree structures
- [ ] Build and validate model: features + rainfall + catchment characteristics → turbidity

## API

- [ ] Package API for Benjamin: new repo, transplant required code, rebuild with uv

## Restructure

To be done alongside Docker integration. Two modes: local provided data vs. live update.

- [ ] Query whole CS shapefile (incl. rainfall)
- [ ] Local data mode: use data referenced from Albert's file system
- [ ] Live update mode

## Background & Coordination

- [ ] Review Albert's thesis for methodology alignment
- [ ] Monitor IDSAI Environmental Intelligence conference (contact: Hywel Williams, Exeter)

---

## Done

- EA data: loaded, visualised, available via API
- River graph: generated from maps, water sample nodes added
- Sample-catchment dataset build script
- Watershed delineation fixed and tested
- Soil type classification: data obtained, fractional coverage per catchment calculated
- Rainfall: Albert's scripts adapted, AWS CLI fixed, integrated into `Sample` class, notebook demo made
- All components combined into `Datapoint` class (topography, slope, aspect, soil, EO, rainfall)
- Basic feature extraction from `Sample` objects
- API: EA tests split (API vs. non-API), CS samples added
