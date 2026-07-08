## Issues

- [ ] attribute 'in1d'

## Confirm states

- [x] runs locally - Works!
- [x] debugs locally- Works!
- [x] runs remotely - "module 'numpy' has no attribute 'in1d'"
- [x] debugs remotely- "module 'numpy' has no attribute 'in1d'"

## Version checks

- [ ] Python match- Synced
- [ ] numpy match- Synced
- [ ] pysheds match

## Local debug issues

- Some snap offsets work
- Some fail: 2026-06-25 15:22:22,545 - eoflow.scripts.delineate_catchments - WARNING - Could not snap point to stream network: `nodata` value not representable in dtype of array.
- All areas are 0.00

## To Complete pipeline

- [x] Split into static and dynamic data
- [ ] Compute static
- [ ] Compute dynamic datacubes
    - [ ] test get\_[dynamic]\_by_poly
        - [ ] EO
        - [ ] Rainfall
    - [ ] Write index by poly
    - [ ]
- [ ] Index static
- [ ] Index datacubes

- Static
    - Topographic data
    - Soil data

- Dynamic
    - EO
        - Bands
        - Indices
    - Rainfall: Nimrod
