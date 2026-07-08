"""
Download a GeoTIFF Digital Elevation Model (DEM) for a given country or
bounding box from the `OpenTopography API <https://opentopography.org>`__.

The output GeoTIFF is written in **EPSG:4326** (WGS 84) with int16 elevation
values and a nodata value of -32768, matching the format expected by
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
By country name (bounding box looked up via Nominatim)::

    python -m scripts.get_dem \\
        --country England \\
        --output data/england_dem.tif

By explicit bounding box (south, west, north, east)::

    python -m scripts.get_dem \\
        --bbox 50.5,-4.0,51.0,-3.1 \\
        --output data/devon_dem.tif

Optional flags::

    --dem-type   SRTMGL1          # DEM dataset (see list below)
    --pad        0.05             # degrees of padding around the bbox
    --api-key    <token>          # if not using the env var

Supported DEM types
-------------------
* ``SRTMGL1``  – SRTM GL1 30 m  *(default – matches existing project DEMs)*
* ``SRTMGL3``  – SRTM GL3 90 m
* ``COP30``    – Copernicus GLO-30 m
* ``COP90``    – Copernicus GLO-90 m
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
import requests

from eoflow.log_utils import get_logger

logger = get_logger(__file__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

_VALID_DEM_TYPES = {"SRTMGL1", "SRTMGL3", "COP30", "COP90"}

_USER_AGENT = "eoflow/0.1.0 (https://github.com/eoflow)"

# Target nodata and dtype to match the project's existing DEMs.
_NODATA: int = -32768
_DTYPE = np.int16

# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------


def _lookup_country_bbox(country: str, pad: float = 0.0) -> Tuple[float, float, float, float]:
    """Return *(south, west, north, east)* for *country* via OSM Nominatim.

    Parameters
    ----------
    country : str
        Free-text country or region name, e.g. ``"England"``.
    pad : float
        Padding in degrees added to each side of the bounding box.

    Returns
    -------
    tuple[float, float, float, float]
        ``(south, west, north, east)`` in decimal degrees (EPSG:4326).

    Raises
    ------
    ValueError
        If no results are returned by Nominatim.
    """
    logger.info("Looking up bounding box for '%s' via Nominatim …", country)
    resp = requests.get(
        _NOMINATIM_URL,
        params={"q": country, "format": "json", "limit": "1"},
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()

    if not results:
        raise ValueError(
            f"Nominatim returned no results for '{country}'.  "
            "Try a different spelling or use --bbox instead."
        )

    # Nominatim returns [south, north, west, east] as strings.
    bb = results[0]["boundingbox"]
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])

    if pad:
        south -= pad
        north += pad
        west -= pad
        east += pad

    logger.info(
        "  Bounding box: south=%.4f, west=%.4f, north=%.4f, east=%.4f",
        south,
        west,
        north,
        east,
    )
    return south, west, north, east


def _parse_bbox(bbox_str: str, pad: float = 0.0) -> Tuple[float, float, float, float]:
    """Parse a ``south,west,north,east`` string into a 4-tuple of floats."""
    parts = [s.strip() for s in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"--bbox must be four comma-separated numbers (south,west,north,east), got: {bbox_str}"
        )
    south, west, north, east = (float(p) for p in parts)

    if south >= north:
        raise ValueError(f"south ({south}) must be less than north ({north})")
    if west >= east:
        raise ValueError(f"west ({west}) must be less than east ({east})")

    if pad:
        south -= pad
        north += pad
        west -= pad
        east += pad

    return south, west, north, east


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _estimate_size_bytes(
    south: float, west: float, north: float, east: float, dem_type: str
) -> int:
    """Return a rough estimate of the download size in bytes."""
    arcsec = {"SRTMGL1": 1, "COP30": 1, "SRTMGL3": 3, "COP90": 3}.get(dem_type, 1)
    n_lat = int((north - south) * 3600 / arcsec)
    n_lon = int((east - west) * 3600 / arcsec)
    # int16 = 2 bytes per pixel, GeoTIFF overhead is small.
    return n_lat * n_lon * 2


def _fmt_size(raw_bytes: int, n_lat: int | None = None, n_lon: int | None = None) -> str:
    """Format *raw_bytes* as a human-readable string."""
    dims = f" ({n_lat}×{n_lon} pixels)" if n_lat is not None and n_lon is not None else ""
    if raw_bytes < 1024**2:
        return f"~{raw_bytes / 1024:.0f} KB{dims}"
    if raw_bytes < 1024**3:
        return f"~{raw_bytes / 1024**2:.0f} MB{dims}"
    return f"~{raw_bytes / 1024**3:.1f} GB{dims}"


def _estimate_size(south: float, west: float, north: float, east: float, dem_type: str) -> str:
    """Return a rough human-readable estimate of the download size."""
    arcsec = {"SRTMGL1": 1, "COP30": 1, "SRTMGL3": 3, "COP90": 3}.get(dem_type, 1)
    n_lat = int((north - south) * 3600 / arcsec)
    n_lon = int((east - west) * 3600 / arcsec)
    raw_bytes = _estimate_size_bytes(south, west, north, east, dem_type)
    return _fmt_size(raw_bytes, n_lat, n_lon)


# 500 MB threshold for the interactive confirmation prompt.
_LARGE_DOWNLOAD_BYTES = 500 * 1024 * 1024


def _confirm_large_download(
    south: float,
    west: float,
    north: float,
    east: float,
    dem_type: str,
) -> bool:
    """Prompt the user for confirmation when the estimated size exceeds 500 MB.

    Returns ``True`` if the download should proceed, ``False`` otherwise.
    """
    est = _estimate_size_bytes(south, west, north, east, dem_type)
    if est < _LARGE_DOWNLOAD_BYTES:
        return True

    human = _estimate_size(south, west, north, east, dem_type)
    print(
        f"\n⚠  The requested area is large – estimated download size: {human}.\n"
        f"   Consider using a coarser DEM type (e.g. SRTMGL3 or COP90) for\n"
        f"   large regions.\n"
    )
    try:
        answer = input("Proceed with download? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def download_dem(
    south: float,
    west: float,
    north: float,
    east: float,
    output_path: Path,
    *,
    dem_type: str = "SRTMGL1",
    api_key: str | None = None,
) -> Path:
    """Download a DEM GeoTIFF from OpenTopography and save it to *output_path*.

    The file is post-processed so that:

    * CRS is EPSG:4326
    * dtype is int16
    * nodata is -32768

    Parameters
    ----------
    south, west, north, east : float
        Bounding box in decimal degrees (WGS 84).
    output_path : Path
        Destination file path.
    dem_type : str
        One of ``SRTMGL1``, ``SRTMGL3``, ``COP30``, ``COP90``.
    api_key : str or None
        OpenTopography API key.  Falls back to the
        ``OPENTOPOGRAPHY_API_KEY`` environment variable.

    Returns
    -------
    Path
        The path to the written GeoTIFF file.
    """
    if dem_type not in _VALID_DEM_TYPES:
        raise ValueError(f"Invalid dem_type '{dem_type}'.  Choose from: {_VALID_DEM_TYPES}")

    key = api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not key:
        raise RuntimeError(
            "No OpenTopography API key found.  Either pass --api-key or set "
            "the OPENTOPOGRAPHY_API_KEY environment variable.\n"
            "Get a free key at https://opentopography.org/"
        )

    size_est = _estimate_size(south, west, north, east, dem_type)
    logger.info("Requesting %s DEM from OpenTopography  (%s) …", dem_type, size_est)
    logger.info("  Bounding box: S=%.4f  W=%.4f  N=%.4f  E=%.4f", south, west, north, east)

    params = {
        "demtype": dem_type,
        "south": f"{south:.6f}",
        "north": f"{north:.6f}",
        "west": f"{west:.6f}",
        "east": f"{east:.6f}",
        "outputFormat": "GTiff",
        "API_Key": key,
    }

    resp = requests.get(_OPENTOPO_URL, params=params, stream=True, timeout=300)

    # OpenTopography returns 200 with a JSON error body for some failures.
    content_type = resp.headers.get("Content-Type", "")
    if resp.status_code != 200 or "application/json" in content_type or "text/" in content_type:
        try:
            body = resp.json() if "json" in content_type else resp.text
        except Exception:
            body = resp.text[:500]
        raise RuntimeError(f"OpenTopography API error (HTTP {resp.status_code}): {body}")

    # Stream the response to a temporary file then post-process.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = output_path.with_suffix(".raw.tif")

    downloaded_bytes = 0
    with open(raw_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            fh.write(chunk)
            downloaded_bytes += len(chunk)

    logger.info("  Downloaded %d bytes → %s", downloaded_bytes, raw_path)

    # --- Post-process: ensure int16, nodata=-32768, CRS=4326 -------------
    _normalise_geotiff(raw_path, output_path)
    raw_path.unlink(missing_ok=True)

    logger.info("DEM saved to %s", output_path)
    return output_path


def _normalise_geotiff(src_path: Path, dst_path: Path) -> None:
    """Re-write *src_path* as a clean int16 GeoTIFF at *dst_path*.

    This ensures the output always has:
    * CRS  = EPSG:4326
    * dtype = int16
    * nodata = -32768
    * LZW compression
    """
    with rasterio.open(src_path) as src:
        data = src.read(1)
        src_nodata = src.nodata
        transform = src.transform
        crs = src.crs

        # Replace source nodata → target nodata
        if src_nodata is not None:
            mask = np.isclose(data, src_nodata) | np.isnan(data.astype(float))
        else:
            mask = np.isnan(data.astype(float))

        out = data.astype(_DTYPE)
        out[mask] = _NODATA

        profile = {
            "driver": "GTiff",
            "dtype": np.dtype(_DTYPE).name,
            "width": src.width,
            "height": src.height,
            "count": 1,
            "crs": crs or "EPSG:4326",
            "transform": transform,
            "nodata": _NODATA,
            "compress": "lzw",
            "tiled": True,
        }

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(out, 1)

    # Quick sanity check
    with rasterio.open(dst_path) as check:
        logger.info(
            "  Normalised: %s  shape=%s  CRS=%s  nodata=%s  dtype=%s",
            dst_path.name,
            (check.height, check.width),
            check.crs,
            check.nodata,
            check.dtypes[0],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download a GeoTIFF DEM for a country or bounding box from the "
            "OpenTopography API.  The result is ready for use with "
            "scripts.delineate_catchments --dem."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m scripts.get_dem --country England --output data/england_dem.tif\n"
            "  python -m scripts.get_dem --bbox 50.5,-4.0,51.0,-3.1 --output data/devon_dem.tif\n"
            "\n"
            "DEM types:\n"
            "  SRTMGL1  SRTM GL1 30 m   (default)\n"
            "  SRTMGL3  SRTM GL3 90 m\n"
            "  COP30    Copernicus 30 m\n"
            "  COP90    Copernicus 90 m\n"
        ),
    )

    location = p.add_mutually_exclusive_group(required=True)
    location.add_argument(
        "--country",
        type=str,
        default=None,
        help=(
            "Country or region name (e.g. 'England', 'Wales').  "
            "The bounding box is resolved via OSM Nominatim."
        ),
    )
    location.add_argument(
        "--bbox",
        type=str,
        default=None,
        help=(
            "Explicit bounding box as four comma-separated values: "
            "south,west,north,east  (decimal degrees, EPSG:4326)."
        ),
    )

    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GeoTIFF path.",
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
        help="Padding in degrees to add around the bounding box (default: %(default)s).",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "OpenTopography API key.  If not provided, the "
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

    # --- Resolve bounding box ---------------------------------------------
    try:
        if args.country:
            south, west, north, east = _lookup_country_bbox(args.country, pad=args.pad)
        else:
            south, west, north, east = _parse_bbox(args.bbox, pad=args.pad)
    except Exception as exc:
        logger.error("Error resolving bounding box: %s", exc)
        sys.exit(1)

    # --- Confirm large downloads ------------------------------------------
    if not args.yes:
        if not _confirm_large_download(south, west, north, east, args.dem_type):
            print("Aborted.", file=sys.stderr)
            sys.exit(0)

    # --- Download ---------------------------------------------------------
    try:
        result_path = download_dem(
            south=south,
            west=west,
            north=north,
            east=east,
            output_path=args.output,
            dem_type=args.dem_type,
            api_key=args.api_key,
        )
        print(f"\nDEM written to {result_path}")
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
