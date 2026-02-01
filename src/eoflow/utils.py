from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def load_shapefile(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Load a shapefile or directory containing shapefile components.

    Accepts either a path to a `.shp` file or a directory containing shapefile
    files. Returns a GeoDataFrame.
    """
    p = Path(path)

    if p.is_dir():
        # try to find a .shp file in the directory
        shp_files = list(p.glob("*.shp"))
        if not shp_files:
            raise FileNotFoundError(f"No .shp files found in directory: {p}")
        shp_path = shp_files[0]
    else:
        shp_path = p

    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    # Read with geopandas; let geopandas handle encodings/CRS
    gdf = gpd.read_file(str(shp_path))

    return gdf
