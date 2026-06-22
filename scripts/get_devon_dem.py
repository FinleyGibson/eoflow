"""
Download a GeoTIFF Digital Elevation Model (DEM) clipped to the Devon county
boundary from the `OpenTopography API <https://opentopography.org>`__.

This script combines the DEM download approach from :mod:`scripts.get_dem`
with the Devon shapefile loading from :mod:`scripts.get_devon_water_quality`
to produce a DEM that covers only the area inside the Devon county polygon.

The workflow is:

1. Load the Devon county shapefile and reproject to WGS 84 (EPSG:4326).
2. Compute the bounding box of the Devon boundary (with optional padding).
3. Download the DEM for that bounding box via the OpenTopography API.
4. Clip/mask the DEM to the Devon polygon so pixels outside Devon are set
   to nodata (−32 768).

The output GeoTIFF is written in **EPSG:4326** with int16 elevation values
and a nodata value of −32 768, matching the format expected by
:func:`eoflow.catchment.delineate_catchment` and the ``--dem`` argument in
:mod:`scripts.delineate_catchments`.

Prerequisites
-------------
You need a **free** OpenTopography API key:

1. Register at https://opentopography.org/
2. Go to *My Account → My OpenTopography Authorisation Token*
3. Copy the token and either:

   * set the environment variable ``OPENTOPOGRAPHY_API_KEY``, **or**
   * pass it via ``--api-key`` on the command line.

Usage
-----
Using the default Devon shapefile::

    python -m scripts.get_devon_dem \\
        --output data/devon_dem.tif

With options::

    python -m scripts.get_devon_dem \\
        --output data/devon_dem_90m.tif \\
        --dem-type SRTMGL3 \\
        --pad 0.05 \\
        --shapefile data/devon_county

Supported DEM types
-------------------
* ``SRTMGL1``  – SRTM GL1 30 m  *(default)*
* ``SRTMGL3``  – SRTM GL3 90 m
* ``COP30``    – Copernicus GLO-30 m
* ``COP90``    – Copernicus GLO-90 m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask

from eoflow.log_utils import get_logger
from eoflow.utils import load_shapefile

# Re-use the download machinery from get_dem
from scripts.get_dem import (
    _DTYPE,
    _NODATA,
    _VALID_DEM_TYPES,
    _confirm_large_download,
    download_dem,
)

logger = get_logger(__file__)

# Default shapefile path (same as get_devon_water_quality.py)
_DEFAULT_SHAPEFILE = Path(__file__).resolve().parent.parent / "data" / "devon_county"


# ---------------------------------------------------------------------------
# Shapefile helpers
# ---------------------------------------------------------------------------


def _load_devon_gdf(shapefile_path: Path) -> gpd.GeoDataFrame:
    """Load the Devon shapefile and reproject to EPSG:4326.

    Parameters
    ----------
    shapefile_path : Path
        Path to the shapefile directory or ``.shp`` file.

    Returns
    -------
    gpd.GeoDataFrame
        Devon boundary in WGS 84.
    """
    gdf = load_shapefile(shapefile_path)
    logger.info("Loaded shapefile with %d feature(s)", len(gdf))

    if gdf.crs is not None and gdf.crs != "EPSG:4326":
        logger.info("Reprojecting from %s to EPSG:4326", gdf.crs)
        gdf = gdf.to_crs("EPSG:4326")

    return gdf


# ---------------------------------------------------------------------------
# Clip / mask
# ---------------------------------------------------------------------------


def _clip_dem_to_polygon(
    dem_path: Path,
    gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> Path:
    """Mask a DEM GeoTIFF to the geometries in *gdf*.

    Pixels outside the polygon(s) are set to nodata (−32 768).  The
    raster is cropped to the geometry extent so the file size is minimal.

    Parameters
    ----------
    dem_path : Path
        Input DEM GeoTIFF (bounding-box download).
    gdf : gpd.GeoDataFrame
        Devon boundary polygons in the same CRS as the DEM.
    output_path : Path
        Destination for the clipped GeoTIFF.

    Returns
    -------
    Path
        *output_path* after writing.
    """
    geometries = gdf.geometry.values

    with rasterio.open(dem_path) as src:
        # Ensure CRS alignment
        if gdf.crs != src.crs:
            gdf_reprojected = gdf.to_crs(src.crs)
            geometries = gdf_reprojected.geometry.values

        out_image, out_transform = rasterio.mask.mask(
            src,
            geometries,
            crop=True,
            nodata=_NODATA,
            filled=True,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "dtype": np.dtype(_DTYPE).name,
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": _NODATA,
                "compress": "lzw",
                "tiled": True,
            }
        )

    # Ensure output is int16
    out_data = out_image.astype(_DTYPE)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_data)

    # Sanity check
    with rasterio.open(output_path) as check:
        valid_pixels = np.count_nonzero(check.read(1) != _NODATA)
        total_pixels = check.height * check.width
        logger.info(
            "  Clipped DEM: %s  shape=(%d, %d)  CRS=%s  nodata=%s  dtype=%s",
            output_path.name,
            check.height,
            check.width,
            check.crs,
            check.nodata,
            check.dtypes[0],
        )
        logger.info(
            "  Valid pixels: %d / %d  (%.1f%%)",
            valid_pixels,
            total_pixels,
            100.0 * valid_pixels / total_pixels if total_pixels else 0,
        )

    return output_path


# ---------------------------------------------------------------------------
# Main download + clip pipeline
# ---------------------------------------------------------------------------


def get_devon_dem(
    output_path: Path,
    *,
    shapefile_path: Path = _DEFAULT_SHAPEFILE,
    dem_type: str = "COP30",
    pad: float = 0.0,
    api_key: str | None = None,
) -> Path:
    """Download a DEM and clip it to the Devon county boundary.

    Parameters
    ----------
    output_path : Path
        Destination GeoTIFF path.
    shapefile_path : Path
        Path to the Devon county shapefile (directory or ``.shp``).
    dem_type : str
        OpenTopography DEM dataset (``SRTMGL1``, ``SRTMGL3``, ``COP30``,
        ``COP90``).
    pad : float
        Padding in degrees added to each side of the bounding box before
        downloading.  The final output is still clipped to the exact
        polygon, so this only affects how much data is fetched.
    api_key : str or None
        OpenTopography API key.  Falls back to
        ``OPENTOPOGRAPHY_API_KEY`` env var.

    Returns
    -------
    Path
        Path to the clipped GeoTIFF.
    """
    # --- Step 1: Load shapefile -------------------------------------------
    print(f"\nStep 1: Loading Devon boundary from {shapefile_path} …")
    gdf = _load_devon_gdf(shapefile_path)

    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy) → (west, south, east, north)
    west, south, east, north = bounds

    if pad:
        south -= pad
        north += pad
        west -= pad
        east += pad

    print(f"  ✓ Devon boundary loaded ({len(gdf)} feature(s))")
    print(f"  Bounding box (padded {pad}°):")
    print(f"    Longitude: [{west:.4f}, {east:.4f}]")
    print(f"    Latitude:  [{south:.4f}, {north:.4f}]")

    # --- Step 2: Download DEM for the bounding box ------------------------
    print(f"\nStep 2: Downloading {dem_type} DEM from OpenTopography …")

    output_path = Path(output_path)
    bbox_path = output_path.with_name(output_path.stem + "_bbox.tif")

    download_dem(
        south=south,
        west=west,
        north=north,
        east=east,
        output_path=bbox_path,
        dem_type=dem_type,
        api_key=api_key,
    )
    print(f"  ✓ Bounding-box DEM saved to {bbox_path}")

    # --- Step 3: Clip to Devon polygon ------------------------------------
    print("\nStep 3: Clipping DEM to Devon boundary …")

    _clip_dem_to_polygon(bbox_path, gdf, output_path)
    print(f"  ✓ Clipped DEM saved to {output_path}")

    # Clean up the intermediate bounding-box file
    bbox_path.unlink(missing_ok=True)
    logger.info("Removed intermediate file %s", bbox_path)

    print(f"\n{'=' * 60}")
    print(f"Devon DEM ready: {output_path}")
    print(f"{'=' * 60}\n")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download a GeoTIFF DEM clipped to the Devon county boundary "
            "from the OpenTopography API."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m scripts.get_devon_dem --output data/devon_dem.tif\n"
            "  python -m scripts.get_devon_dem --output data/devon_dem_90m.tif "
            "--dem-type SRTMGL3\n"
            "\n"
            "DEM types:\n"
            "  SRTMGL1  SRTM GL1 30 m   (default)\n"
            "  SRTMGL3  SRTM GL3 90 m\n"
            "  COP30    Copernicus 30 m\n"
            "  COP90    Copernicus 90 m\n"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GeoTIFF path.",
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        default=_DEFAULT_SHAPEFILE,
        help=(
            "Path to the Devon county shapefile (directory or .shp file). (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--dem-type",
        type=str,
        default="SRTMGL1",
        choices=sorted(_VALID_DEM_TYPES),
        help="DEM dataset to download (default: %(default)s).",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help=(
            "Padding in degrees around the Devon bounding box. The final "
            "output is still clipped to the polygon. (default: %(default)s)"
        ),
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "OpenTopography API key. If not provided, the "
            "OPENTOPOGRAPHY_API_KEY environment variable is used."
        ),
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Skip the confirmation prompt for large downloads.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Load shapefile early to compute bounds for the size check
    try:
        gdf = _load_devon_gdf(args.shapefile)
    except Exception as exc:
        print(f"\nError loading shapefile: {exc}", file=sys.stderr)
        sys.exit(1)

    bounds = gdf.total_bounds
    west, south, east, north = bounds
    if args.pad:
        south -= args.pad
        north += args.pad
        west -= args.pad
        east += args.pad

    # Confirm large downloads
    if not args.yes:
        if not _confirm_large_download(south, west, north, east, args.dem_type):
            print("Aborted.", file=sys.stderr)
            sys.exit(0)

    # Download and clip
    try:
        get_devon_dem(
            output_path=args.output,
            shapefile_path=args.shapefile,
            dem_type=args.dem_type,
            pad=args.pad,
            api_key=args.api_key,
        )
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
