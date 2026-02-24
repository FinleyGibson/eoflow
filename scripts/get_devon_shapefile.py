from pathlib import Path

from eoflow.utils import PROJECT_ROOT, load_shapefile

shapefile_dir = Path(
    "/home/finley/Work/RDS/projects/enforce/data/Counties_and_Unitary_Authorities_December_2023_Boundaries"
)


gdf = load_shapefile(path=shapefile_dir)
devon_gdf = gdf[gdf["CTYUA23NM"] == "Devon"]

# write shapefile to disk
devon_gdf.to_file(PROJECT_ROOT / "data/devon_county/devon_county.shp")
