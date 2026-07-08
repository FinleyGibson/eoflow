"""
Download Sentinel-2 band data and/or spectral indices (NDVI, NDWI, EVI, ...)
for a region defined by a user-supplied shapefile, over a specified time
window.

This script uses the low-level building blocks in :mod:`eoflow.eo` directly
— the same functions used internally by :class:`eoflow.samples.Sample`'s
``query_*`` / ``fetch_*`` methods (see :mod:`eoflow.samples`) — to construct
and download openEO datacubes for an arbitrary shapefile-defined region,
rather than for a single sample's delineated catchment polygon.

The workflow is:

1. Load the shapefile and compute its bounding box in WGS 84 (optionally
   padded), following the same approach as
   :mod:`scripts.get_ea_water_quality_by_shapefile`.
2. Connect to the openEO backend (:func:`eoflow.eo.connect`).
3. For each requested band group / spectral index, build the datacube
   (:func:`eoflow.eo.build_band_cube` / :func:`eoflow.eo.build_index_cube`),
   download it synchronously (:func:`eoflow.eo.download_cube_as_xarray`),
   and write it to its own NetCDF file in the output directory.

Each band group and each index is saved as a **separate datacube** file
(e.g. ``bands.nc``, ``ndvi.nc``, ``ndwi.nc``, ``evi.nc``).

Usage
-----
    python -m scripts.get_eo_data_by_shapefile \\
        --shapefile data/devon_county \\
        --start-date 2024-03-01 \\
        --end-date 2024-04-30 \\
        --bands B02 B03 B04 B08 \\
        --indices NDVI NDWI EVI \\
        --output-dir data/eo_cubes/devon

    # Indices only, no raw bands
    python -m scripts.get_eo_data_by_shapefile \\
        --shapefile data/my_region \\
        --start-date 2024-06-01 \\
        --end-date 2024-06-30 \\
        --indices NDVI NDWI \\
        --output-dir data/eo_cubes/my_region

    # Skip OIDC login (public collections only)
    python -m scripts.get_eo_data_by_shapefile \\
        --shapefile data/my_region \\
        --start-date 2024-06-01 \\
        --end-date 2024-06-30 \\
        --bands B04 B08 \\
        --output-dir data/eo_cubes/my_region \\
        --no-auth

Supported indices
------------------
``NDVI``, ``NDWI``, ``EVI``, ``MNDWI``, ``NDRE`` (case-insensitive) — see
:data:`eoflow.eo.SUPPORTED_INDICES`.

Notes
-----
* Requires the optional ``openeo`` and ``xarray`` dependencies (see
  :mod:`eoflow.eo`).
* Downloading large spatial/temporal extents synchronously can be slow or
  fail outright — keep bounding boxes and date ranges modest. For bigger
  requests, extend this script to use openEO batch jobs instead.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from eoflow import eo
from eoflow.log_utils import get_logger
from eoflow.utils import load_shapefile

if TYPE_CHECKING:
    import openeo  # type: ignore[import-untyped]

logger = get_logger(__file__)


# ---------------------------------------------------------------------------
# Shapefile -> spatial extent
# ---------------------------------------------------------------------------


def load_spatial_extent(shapefile_path: Path, *, pad: float = 0.0) -> Dict[str, float]:
    """Load a shapefile and return an openEO ``spatial_extent`` dict.

    Parameters
    ----------
    shapefile_path : Path
        Path to a shapefile (directory or ``.shp`` file).
    pad : float
        Degrees of padding added to each side of the bounding box.

    Returns
    -------
    dict
        ``{"west": ..., "south": ..., "east": ..., "north": ...}`` in
        WGS 84 decimal degrees, as expected by
        :func:`eoflow.eo.build_band_cube` / :func:`eoflow.eo.build_index_cube`.
    """
    gdf = load_shapefile(shapefile_path)
    logger.info("Loaded shapefile with %d feature(s)", len(gdf))

    if gdf.crs is not None and gdf.crs != "EPSG:4326":
        logger.info("Reprojecting from %s to EPSG:4326", gdf.crs)
        gdf = gdf.to_crs("EPSG:4326")

    west, south, east, north = gdf.total_bounds
    if pad:
        west, south, east, north = west - pad, south - pad, east + pad, north + pad

    extent = {
        "west": float(west),
        "south": float(south),
        "east": float(east),
        "north": float(north),
    }
    logger.info(
        "Spatial extent: west=%.5f south=%.5f east=%.5f north=%.5f",
        extent["west"],
        extent["south"],
        extent["east"],
        extent["north"],
    )
    return extent


# ---------------------------------------------------------------------------
# Datacube download helpers
# ---------------------------------------------------------------------------


def fetch_bands_cube(
    connection: "openeo.Connection",
    spatial_extent: Dict[str, float],
    temporal_extent: Tuple[str, str],
    bands: Sequence[str],
    output_path: Path,
    *,
    collection: str = eo.SENTINEL2_COLLECTION,
    max_cloud_cover: int = 85,
) -> Path:
    """Download a multi-band Sentinel-2 datacube and save it as NetCDF.

    Parameters
    ----------
    connection : openeo.Connection
        An authenticated (or public) openEO connection.
    spatial_extent : dict
        Bounding-box dict (``west``, ``south``, ``east``, ``north``).
    temporal_extent : tuple of str
        ``(start_date, end_date)`` as ISO-8601 strings.
    bands : sequence of str
        Band names to download together, e.g. ``["B02", "B03", "B04", "B08"]``.
    output_path : Path
        Destination NetCDF file.
    collection : str
        openEO collection ID.
    max_cloud_cover : int
        Maximum cloud cover percentage (0-100).

    Returns
    -------
    Path
        *output_path* after writing.
    """
    logger.info("Building band cube for bands=%s …", list(bands))
    cube = eo.build_band_cube(
        connection,
        spatial_extent,
        temporal_extent,
        bands,
        collection=collection,
        max_cloud_cover=max_cloud_cover,
    )
    ds = eo.download_cube_as_xarray(cube)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output_path)
    logger.info("Saved band cube (%s) to %s", ", ".join(bands), output_path)
    return output_path


def fetch_index_cube(
    connection: "openeo.Connection",
    index: str,
    spatial_extent: Dict[str, float],
    temporal_extent: Tuple[str, str],
    output_path: Path,
    *,
    collection: str = eo.SENTINEL2_COLLECTION,
    max_cloud_cover: int = 85,
) -> Path:
    """Download a named spectral index datacube and save it as NetCDF.

    Parameters
    ----------
    connection : openeo.Connection
        An authenticated (or public) openEO connection.
    index : str
        Spectral index name (see :data:`eoflow.eo.SUPPORTED_INDICES`).
    spatial_extent : dict
        Bounding-box dict (``west``, ``south``, ``east``, ``north``).
    temporal_extent : tuple of str
        ``(start_date, end_date)`` as ISO-8601 strings.
    output_path : Path
        Destination NetCDF file.
    collection : str
        openEO collection ID.
    max_cloud_cover : int
        Maximum cloud cover percentage (0-100).

    Returns
    -------
    Path
        *output_path* after writing.
    """
    logger.info("Building %s cube …", index.upper())
    cube = eo.build_index_cube(
        connection,
        index,
        spatial_extent,
        temporal_extent,
        collection=collection,
        max_cloud_cover=max_cloud_cover,
    )
    ds = eo.download_cube_as_xarray(cube)
    da = eo.extract_dataarray(ds, name=index.lower())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    da.to_netcdf(output_path)
    logger.info("Saved %s cube to %s", index.upper(), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def fetch_eo_data(
    shapefile_path: Path,
    start_date: str,
    end_date: str,
    output_dir: Path,
    *,
    bands: Optional[Sequence[str]] = None,
    indices: Optional[Sequence[str]] = None,
    pad: float = 0.0,
    collection: str = eo.SENTINEL2_COLLECTION,
    max_cloud_cover: int = 85,
    backend: str = eo.OPENEO_BACKEND,
    authenticate: bool = True,
) -> List[Path]:
    """Download the requested Sentinel-2 bands/indices for a shapefile region.

    Parameters
    ----------
    shapefile_path : Path
        Shapefile (directory or ``.shp`` file) defining the region of
        interest.
    start_date, end_date : str
        Temporal extent in ``YYYY-MM-DD`` format.
    output_dir : Path
        Directory where NetCDF datacubes are written — one file for the
        combined band group (``bands.nc``) and one per requested index
        (``{index}.nc``).
    bands : sequence of str, optional
        Band names to download together as a single multi-band datacube
        (e.g. ``["B02", "B03", "B04", "B08"]``). Skipped when empty/None.
    indices : sequence of str, optional
        Spectral index names to download, each saved as its own datacube.
        See :data:`eoflow.eo.SUPPORTED_INDICES` for supported names
        (``NDVI``, ``NDWI``, ``EVI``, ``MNDWI``, ``NDRE``; case-insensitive).
    pad : float
        Degrees of padding added around the shapefile bounding box.
    collection : str
        openEO collection ID. Defaults to ``SENTINEL2_L2A``.
    max_cloud_cover : int
        Maximum cloud cover percentage (0-100).
    backend : str
        openEO backend URL.
    authenticate : bool
        Whether to perform OIDC authentication when connecting. Pass
        *False* to use an unauthenticated connection (public collections
        only).

    Returns
    -------
    list of Path
        Paths to all NetCDF files successfully written.

    Raises
    ------
    ValueError
        If neither *bands* nor *indices* is provided.
    """
    bands = list(bands or [])
    indices = list(indices or [])

    if not bands and not indices:
        raise ValueError("Nothing to fetch: specify --bands and/or --indices.")

    logger.info("Loading region boundary from %s …", shapefile_path)
    spatial_extent = load_spatial_extent(shapefile_path, pad=pad)
    temporal_extent = (start_date, end_date)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to openEO backend %s …", backend)
    connection = eo.connect(backend, authenticate=authenticate)

    written: List[Path] = []

    if bands:
        band_path = output_dir / "bands.nc"
        try:
            fetch_bands_cube(
                connection,
                spatial_extent,
                temporal_extent,
                bands,
                band_path,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )
            written.append(band_path)
        except Exception as exc:
            logger.error("Failed to fetch bands %s: %s", bands, exc)

    for index in indices:
        index_path = output_dir / f"{index.lower()}.nc"
        try:
            fetch_index_cube(
                connection,
                index,
                spatial_extent,
                temporal_extent,
                index_path,
                collection=collection,
                max_cloud_cover=max_cloud_cover,
            )
            written.append(index_path)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", index.upper(), exc)

    n_requested = (1 if bands else 0) + len(indices)
    logger.info("Done: %d/%d datacube(s) written to %s", len(written), n_requested, output_dir)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Download Sentinel-2 band data and/or spectral indices "
            "(NDVI, NDWI, EVI, ...) for a region defined by a shapefile, "
            "over a specified time window. Each band group / index is "
            "saved as a separate NetCDF datacube."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--shapefile",
        type=Path,
        required=True,
        help="Path to a shapefile defining the region of interest (directory or .shp file).",
    )
    p.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    p.add_argument(
        "--end-date",
        required=True,
        help="End date in YYYY-MM-DD format.",
    )
    p.add_argument(
        "--bands",
        nargs="+",
        default=None,
        metavar="BAND",
        help=(
            "Band names to download together as a single multi-band datacube "
            "(e.g. --bands B02 B03 B04 B08), saved as bands.nc."
        ),
    )
    p.add_argument(
        "--indices",
        nargs="+",
        default=None,
        metavar="INDEX",
        type=str.upper,
        choices=sorted(eo.SUPPORTED_INDICES),
        help=(
            "Spectral index names to download, each saved as its own "
            f"datacube (e.g. --indices NDVI NDWI EVI). "
            f"Supported: {', '.join(sorted(eo.SUPPORTED_INDICES))}."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write NetCDF datacubes into.",
    )
    p.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Degrees of padding added around the shapefile bounding box.",
    )
    p.add_argument(
        "--collection",
        default=eo.SENTINEL2_COLLECTION,
        help="openEO collection ID.",
    )
    p.add_argument(
        "--max-cloud-cover",
        type=int,
        default=85,
        help="Maximum cloud cover percentage (0-100) used to pre-filter scenes.",
    )
    p.add_argument(
        "--backend",
        default=eo.OPENEO_BACKEND,
        help="openEO backend URL.",
    )
    p.add_argument(
        "--no-auth",
        dest="authenticate",
        action="store_false",
        default=True,
        help="Skip OIDC authentication (only publicly accessible collections will work).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Validate dates
    for label, val in [("start-date", args.start_date), ("end-date", args.end_date)]:
        try:
            datetime.strptime(val, "%Y-%m-%d")
        except ValueError:
            logger.error("Invalid %s: %s  (expected YYYY-MM-DD)", label, val)
            sys.exit(1)

    if not args.bands and not args.indices:
        logger.error("Nothing to fetch: specify --bands and/or --indices.")
        sys.exit(1)

    if not args.shapefile.exists():
        logger.error("Shapefile not found: %s", args.shapefile)
        sys.exit(1)

    try:
        fetch_eo_data(
            shapefile_path=args.shapefile,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            bands=args.bands,
            indices=args.indices,
            pad=args.pad,
            collection=args.collection,
            max_cloud_cover=args.max_cloud_cover,
            backend=args.backend,
            authenticate=args.authenticate,
        )
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(130)
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
