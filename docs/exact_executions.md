# get GA data for Devon

## Get basic turbidity data

```sh
uv run scripts/get_ea_water_quality_by_shapefile.py   --shapefile data/shapefiles/devon_county/devon_county.shp   --determinand 6396   --start-date 2010-01-01   --end-date 2026-06-01   --out data/ea_water_quality/turbidity_2010-01_2026-06.csv
```

## Clarify csv

```sh
uv run scripts/convert_ea_csv.py --input data/ea_water_quality/turbidity_2010-01_2026-06.csv --output data/ea_water_quality/turbidity_2010-01_2026-06_clean.csv
```

## check data

```sh
uv run scripts/sample_report.py data/ea_water_quality/turbidity_2010-01_2026-06_clean.csv
```

output:

```sh
Reporting on data/ea_water_quality/turbidity_2010-01_2026-06_clean.csv
Total samples: 4189
Unique sampling locations: 149
Date coverage: 2010-01-05T11:50:00 to 2026-05-22T08:11:00
Lat coverage: 50.22533205136931 to 51.108059073985956
Lon coverage: -3.018156480191549 to -4.445961323481872

GDF head:
                                                  id  ...            longitude
0  https://environment.data.gov.uk/water-quality/...  ...  -3.0471997671377955
1  https://environment.data.gov.uk/water-quality/...  ...   -3.293450721357024
2  https://environment.data.gov.uk/water-quality/...  ...  -3.3029429923425027
3  https://environment.data.gov.uk/water-quality/...  ...  -3.3029429923425027
4  https://environment.data.gov.uk/water-quality/...  ...  -3.3029429923425027
```
