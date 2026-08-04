# To Do

- [ ] Combine EA + CS sources with source annotation (labelled)
- [ ] Fix issue with catchments
- [ ] Move modelling prediction ability into API/Docker
- [ ] Set up periodic script to manage pipeline for incoming data as a scheduled task
    - Engineer robustness into pipeline
- [ ] Set up periodic model retraining
- [ ] Figure out how data will be stored between myself and Benjamin
    - Currently I have handled data storage in house (this only needs to happen if we are retraining the model periodically)
- [ ] Clear separation of static catchment features from dynamic
    - Do not need to be querying Soil types every time, for example. Once for catchment would do.
- [ ] Enhanced feature extraction.

## Optional:

- [ ] Efficient storage method for requested data: rasterio for a Ge
- [ ] River network interpolation model
- [ ] Combined river network interpolation and model prediction

## Current state:

- [ ] We have a full modelling workflow from start to finish that can make predictions and report results.

## Current issues:

- [ ] Sample catchments are not taking real-life hydrology into account.
    - Recompute with a higher accumulation threshold
    - Enhance flow accumulation lines with real extraceed
