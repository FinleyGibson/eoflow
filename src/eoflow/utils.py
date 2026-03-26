from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
from numpy.typing import ArrayLike
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Lazily-initialised transformer: British National Grid → WGS 84
_BNG_TO_WGS84: Transformer | None = None


def _get_bng_transformer() -> Transformer:
    """Return a cached ``Transformer`` from EPSG:27700 to EPSG:4326."""
    global _BNG_TO_WGS84
    if _BNG_TO_WGS84 is None:
        _BNG_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    return _BNG_TO_WGS84


def easting_northing_to_latlon(
    easting: float | ArrayLike,
    northing: float | ArrayLike,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert British National Grid coordinates to WGS 84 latitude/longitude.

    Parameters
    ----------
    easting : float or array-like
        Easting value(s) in EPSG:27700 (British National Grid).
    northing : float or array-like
        Northing value(s) in EPSG:27700 (British National Grid).

    Returns
    -------
    latitude, longitude : tuple[float | np.ndarray, float | np.ndarray]
        Corresponding WGS 84 coordinates. When array inputs are provided
        the outputs are NumPy arrays of the same shape.
    """
    transformer = _get_bng_transformer()
    # always_xy=True means input order is (x, y) == (easting, northing)
    # and output order is (longitude, latitude)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


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
