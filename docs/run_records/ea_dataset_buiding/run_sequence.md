## Steps

### 1. Acquired EA turbidity samples using the following command with the Devon shapefile. This was performed on local.

```bash
uv run ./scripts/ge_ea_water_quality_by_shapefile.py --shapefile data/shapefiles/devon_county/ --determinand 6396 --start-date 2010-01-01 --end-date 2023-12-31 --out out/ea_turbidity_2010-2024.csv --checkpoint-dir .checkpoints_get_ea_turbidity
```

output:

```bash
============================================================
Download complete
  Months fetched this run : 168
  Months skipped (cached) : 0
  Records in region       : 3558
  Output file             : out/ea_turbidity_2010-2024.csv
============================================================
```

### 2. Process data to include longitude and latitude

```bash
uv run -m scripts.convert_ea_csv \
    --input out/ea_turbidity_2010-2024.csv \
    --output out/ea_turbidity_2010-2024_clean.csv
```

#### 2.1 Visualise points

Simple visualisation of point on a folium map.

```bash
uv run scripts/visualise_ea_samples.py --input out/ea_turbidity_2010-2024_clean.csv
```

### 3. Transferred data to remote server

Copied to remote servers assets directory via SCP.

```bash
scp out/ea_turbidity_2010-2024_clean.csv fjg205@10.121.4.88:/home/fjg205/projects/eoflow/assets/turbidity_samples/
```

4. Delinated catchments

```bash
sh -c 'uv run scripts/delineate_catchments.py --csv assets/turbidity_samples/ea_turbidity_2010-2024_clean.csv --out assets/sample_catchments/ea_turbidity_2010-2024_catchments_$1.gpkg --DEM assets/devon_dem_cop30.tif --lat-col latitude --lon-col longitude --log-file logs/ea_turbidity_2010-2014_$1.log' _ 00
```

Note: Wrapped in` sh -c` to set $1 to 00 in case it needs repeating.
Note: This ignores duplicate locations

## 5. Compute layers

```bash
uv run scripts/prepare_samples.py --gpkg assets/sample_catchments/ea_turbidity_2010-2024_catchments_00.gpkg --dem assets/devon_dem_cop30.tif --out-dir outputs/layered_samples/ea_samples_2010-2024_catchments_00.gpkg --rainfall-start 2010-01-01 --rainfall-end 2023-12-31 --eo-start 2010-01-01 --eo-end 2023-12-31
```
