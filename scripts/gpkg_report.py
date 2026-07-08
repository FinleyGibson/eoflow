"""
Print a quick summary report for a water-quality samples CSV.

Usage
-----
    python -m scripts.sample_report <input.gpkg>
"""

import sys
from pathlib import Path

import geopandas as gpd

from eoflow.log_utils import get_logger

logger = get_logger(__file__)


def main() -> None:
    if len(sys.argv) < 2:
        logger.error("Usage: python -m scripts.gpkg_report <input.gpkg>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.is_file():
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)
    if input_file.suffix != ".gpkg" and input_file.suffix != ".pkg":
        logger.error("Input file must be a GeoPackage file: %s", input_file)
        sys.exit(1)

    logger.info("Reporting on %s", input_file)
    gdf = gpd.read_file(input_file)

    n_polygons = len(gdf["geometry"].unique())
    n_locations = len(gdf.groupby(["latitude", "longitude"]))

    # The report itself is the script's intended stdout output, so it is
    # printed directly rather than routed through the logger.
    print(f"Reporting on {input_file}")
    print(f"Total samples: {len(gdf)}")
    print(f"Unique sampling locations: {len(gdf.groupby(['latitude', 'longitude']))}")
    print(f"Date coverage: {gdf['phenomenonTime'].min()} to {gdf['phenomenonTime'].max()}")
    print(f"Lat coverage: {gdf['latitude'].min()} to {gdf['latitude'].max()}")
    print(f"Lon coverage: {gdf['longitude'].min()} to {gdf['longitude'].max()}")
    print(f"Catchments delineated: {n_polygons}/{n_locations}")
    print(
        f"Samples delineated: {gdf['geometry'].notnull().sum()}/{gdf.shape[0]} ({gdf['geometry'].notnull().sum() / gdf.shape[0] * 100:.2f}%)"
    )
    print("")
    print("GDF head:")
    print(f"{gdf.head(5)}")
    print("GDF tail:")
    print(f"{gdf.tail(5)}")


if __name__ == "__main__":
    main()
