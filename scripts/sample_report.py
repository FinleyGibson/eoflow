import sys
from pathlib import Path

import geopandas as gpd


def main():

    ## Get input file

    input_file = Path(sys.argv[1])
    assert input_file.is_file(), f"Input file not found: {input_file}"
    assert input_file.suffix == ".csv", f"Input file must be a CSV file: {input_file}"

    print(f"Reporting on {input_file}")

    gdf = gpd.read_file(input_file)

    print(f"Total samples: {len(gdf)}")
    print(f"Unique sampling locations: {len(gdf.groupby(['latitude', 'longitude']))}")
    print(f"Date coverage: {gdf['phenomenonTime'].min()} to {gdf['phenomenonTime'].max()}")
    print(f"Lat coverage: {gdf['latitude'].min()} to {gdf['latitude'].max()}")
    print(f"Lon coverage: {gdf['longitude'].min()} to {gdf['longitude'].max()}")
    print("")
    print("GDF head:")
    print(f"{gdf.head(5)}")


if __name__ == "__main__":
    main()
