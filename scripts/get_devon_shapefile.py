"""
Extract the Devon county boundary from the ONS "Counties and Unitary
Authorities" shapefile and save it as a standalone shapefile.

Usage
-----
    python -m scripts.get_devon_shapefile
"""

from __future__ import annotations

from pathlib import Path

from eoflow.log_utils import get_logger
from eoflow.utils import PROJECT_ROOT, load_shapefile

logger = get_logger(__file__)

_SHAPEFILE_DIR = Path(
    "/home/finley/Work/RDS/projects/enforce/data/Counties_and_Unitary_Authorities_December_2023_Boundaries"
)
_OUTPUT_PATH = PROJECT_ROOT / "data/devon_county/devon_county.shp"


def main() -> None:
    logger.info("Loading shapefile from %s", _SHAPEFILE_DIR)
    gdf = load_shapefile(path=_SHAPEFILE_DIR)
    logger.info("Loaded %d feature(s)", len(gdf))

    devon_gdf = gdf[gdf["CTYUA23NM"] == "Devon"]
    logger.info("Filtered to %d Devon feature(s)", len(devon_gdf))

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    devon_gdf.to_file(_OUTPUT_PATH)
    logger.info("Devon shapefile written to %s", _OUTPUT_PATH)


if __name__ == "__main__":
    main()
